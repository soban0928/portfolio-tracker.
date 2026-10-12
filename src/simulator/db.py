"""SQLite persistence for portfolios, trades and orders, plus CSV export.

The trade log is the source of truth. Loading a portfolio replays its trades
through :meth:`Portfolio.from_records`, so cash, positions, cost basis and
realised P&L can never drift out of sync with the stored transactions.
Saving is a full, atomic sync of one portfolio (rows are replaced inside a
single transaction), which is simple and fast at this scale.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from simulator.config import DEFAULT_DB_PATH, STARTING_CASH, FeeConfig
from simulator.orders import Order, OrderStatus, OrderType, Side
from simulator.portfolio import Portfolio, Trade

SCHEMA = """
CREATE TABLE IF NOT EXISTS portfolios (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL UNIQUE,
    starting_cash REAL    NOT NULL,
    commission    REAL    NOT NULL,
    slippage_bps  REAL    NOT NULL,
    clock         TEXT,
    created_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS trades (
    id              TEXT    PRIMARY KEY,
    portfolio_id    INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    seq             INTEGER NOT NULL,
    order_id        TEXT,
    date            TEXT    NOT NULL,
    ticker          TEXT    NOT NULL,
    side            TEXT    NOT NULL CHECK (side IN ('BUY', 'SELL')),
    quantity        INTEGER NOT NULL CHECK (quantity > 0),
    price           REAL    NOT NULL,
    currency        TEXT    NOT NULL,
    fx_rate         REAL    NOT NULL,
    commission      REAL    NOT NULL,
    reference_price REAL,
    order_type      TEXT    NOT NULL,
    realised_pnl    REAL    NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_trades_portfolio ON trades(portfolio_id, seq);

CREATE TABLE IF NOT EXISTS orders (
    id           TEXT    PRIMARY KEY,
    portfolio_id INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    seq          INTEGER NOT NULL,
    ticker       TEXT    NOT NULL,
    side         TEXT    NOT NULL,
    quantity     INTEGER NOT NULL,
    created      TEXT    NOT NULL,
    order_type   TEXT    NOT NULL,
    limit_price  REAL,
    stop_price   REAL,
    status       TEXT    NOT NULL,
    reason       TEXT    NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_orders_portfolio ON orders(portfolio_id, seq);
"""


class Database:
    """Thread-safe SQLite repository for portfolios.

    Streamlit serves each session on its own thread, so a single connection is
    shared with ``check_same_thread=False`` and guarded by a lock.
    """

    def __init__(self, path: str | Path = DEFAULT_DB_PATH) -> None:
        """Open (and if needed create) the database at ``path``.

        Args:
            path: File path, or ``":memory:"`` for a throwaway database.
        """
        self.path = str(path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        """Run statements in one transaction (commit on success, else roll back)."""
        with self._lock, self._conn:
            yield self._conn

    # ------------------------------------------------------------ portfolios
    def create_portfolio(
        self,
        name: str,
        starting_cash: float = STARTING_CASH,
        fees: FeeConfig | None = None,
    ) -> Portfolio:
        """Create and persist a new, empty portfolio.

        Raises:
            ValueError: If the name is blank or already used.
        """
        name = name.strip()
        if not name:
            raise ValueError("Portfolio name is required")
        fees = fees or FeeConfig()
        pf = Portfolio(name=name, starting_cash=starting_cash, fees=fees)
        try:
            with self._tx() as c:
                cur = c.execute(
                    "INSERT INTO portfolios (name, starting_cash, commission, slippage_bps, "
                    "created_at) VALUES (?, ?, ?, ?, ?)",
                    (name, starting_cash, fees.commission, fees.slippage_bps, _now()),
                )
        except sqlite3.IntegrityError:
            raise ValueError(f"A portfolio named {name!r} already exists") from None
        pf.id = cur.lastrowid
        return pf

    def list_portfolios(self) -> pd.DataFrame:
        """All portfolios with id, name, starting cash, fees and trade count."""
        with self._lock:
            return pd.read_sql_query(
                "SELECT p.id, p.name, p.starting_cash, p.commission, p.slippage_bps, "
                "p.clock, p.created_at, COUNT(t.id) AS trades "
                "FROM portfolios p LEFT JOIN trades t ON t.portfolio_id = p.id "
                "GROUP BY p.id ORDER BY p.id",
                self._conn,
            )

    def load_portfolio(self, portfolio_id: int) -> Portfolio:
        """Load a portfolio by id, rebuilding state by replaying its trades.

        Raises:
            KeyError: If no portfolio has that id.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM portfolios WHERE id = ?", (portfolio_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"No portfolio with id {portfolio_id}")
            trade_rows = self._conn.execute(
                "SELECT * FROM trades WHERE portfolio_id = ? ORDER BY seq", (portfolio_id,)
            ).fetchall()
            order_rows = self._conn.execute(
                "SELECT * FROM orders WHERE portfolio_id = ? ORDER BY seq", (portfolio_id,)
            ).fetchall()
        return Portfolio.from_records(
            trades=[_row_to_trade(r) for r in trade_rows],
            orders=[_row_to_order(r) for r in order_rows],
            clock=row["clock"],
            id=row["id"],
            name=row["name"],
            starting_cash=row["starting_cash"],
            fees=FeeConfig(commission=row["commission"], slippage_bps=row["slippage_bps"]),
        )

    def save_portfolio(self, pf: Portfolio) -> None:
        """Persist a portfolio's clock, trades and orders atomically.

        Raises:
            ValueError: If the portfolio has never been created in this database.
        """
        if pf.id is None:
            raise ValueError("Portfolio has no id; create it with create_portfolio() first")
        clock = pf.clock.date().isoformat() if pf.clock is not None else None
        with self._tx() as c:
            updated = c.execute(
                "UPDATE portfolios SET clock = ?, name = ? WHERE id = ?", (clock, pf.name, pf.id)
            ).rowcount
            if not updated:
                raise ValueError(f"No portfolio with id {pf.id}")
            c.execute("DELETE FROM trades WHERE portfolio_id = ?", (pf.id,))
            c.execute("DELETE FROM orders WHERE portfolio_id = ?", (pf.id,))
            c.executemany(
                "INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        t.id, pf.id, i, t.order_id, t.date.date().isoformat(), t.ticker,
                        str(t.side), t.quantity, t.price, t.currency, t.fx_rate,
                        t.commission, t.reference_price, str(t.order_type), t.realised_pnl,
                    )
                    for i, t in enumerate(pf.trades)
                ],
            )  # fmt: skip
            c.executemany(
                "INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        o.id, pf.id, i, o.ticker, str(o.side), o.quantity,
                        o.created.date().isoformat(), str(o.order_type), o.limit_price,
                        o.stop_price, str(o.status), o.reason,
                    )
                    for i, o in enumerate(pf.orders)
                ],
            )  # fmt: skip

    def delete_portfolio(self, portfolio_id: int) -> None:
        """Delete a portfolio and (via cascade) its trades and orders."""
        with self._tx() as c:
            c.execute("DELETE FROM portfolios WHERE id = ?", (portfolio_id,))

    def reset_portfolio(self, portfolio_id: int) -> Portfolio:
        """Clear all trades/orders, returning the portfolio to its starting cash."""
        pf = self.load_portfolio(portfolio_id)
        fresh = Portfolio(
            name=pf.name, starting_cash=pf.starting_cash, fees=pf.fees, id=portfolio_id
        )
        self.save_portfolio(fresh)
        return fresh


# ---------------------------------------------------------------- export


def export_trades_csv(pf: Portfolio, path: str | Path | None = None) -> str:
    """Trade history as CSV text; also written to ``path`` if given."""
    csv = pf.trades_frame().to_csv(index=False)
    if path is not None:
        Path(path).write_text(csv, encoding="utf-8")
    return csv


# ---------------------------------------------------------------- helpers


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _row_to_trade(r: sqlite3.Row) -> Trade:
    return Trade(
        id=r["id"],
        order_id=r["order_id"] or "",
        date=pd.Timestamp(r["date"]),
        ticker=r["ticker"],
        side=Side(r["side"]),
        quantity=int(r["quantity"]),
        price=float(r["price"]),
        currency=r["currency"],
        fx_rate=float(r["fx_rate"]),
        commission=float(r["commission"]),
        reference_price=r["reference_price"],
        order_type=OrderType(r["order_type"]),
        realised_pnl=float(r["realised_pnl"]),
    )


def _row_to_order(r: sqlite3.Row) -> Order:
    return Order(
        id=r["id"],
        ticker=r["ticker"],
        side=Side(r["side"]),
        quantity=int(r["quantity"]),
        created=pd.Timestamp(r["created"]),
        order_type=OrderType(r["order_type"]),
        limit_price=r["limit_price"],
        stop_price=r["stop_price"],
        status=OrderStatus(r["status"]),
        reason=r["reason"],
    )

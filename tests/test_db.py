import pandas as pd
import pytest

from simulator.config import FeeConfig
from simulator.db import Database, export_trades_csv
from simulator.orders import OrderStatus, make_order
from simulator.portfolio import Portfolio

FEES = FeeConfig(commission=5, slippage_bps=10)


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test.db")
    yield d
    d.close()


def trade_some(pf, store):
    pf.place_order(make_order("XYZ.L", "buy", 100, "2024-01-02"), store)
    pf.place_order(make_order("ABC", "buy", 10, "2024-01-02"), store)
    pf.place_order(make_order("XYZ.L", "sell", 40, "2024-01-03"), store)
    pf.place_order(
        make_order("ABC", "sell", 10, "2024-01-03", "stop", stop_price=50), store
    )  # stays pending
    return pf


def test_create_and_list(db):
    pf = db.create_portfolio("Alpha", 5_000, FEES)
    assert pf.id is not None and pf.cash == 5_000
    db.create_portfolio("Beta")
    df = db.list_portfolios()
    assert list(df["name"]) == ["Alpha", "Beta"]
    assert list(df["trades"]) == [0, 0]
    with pytest.raises(ValueError, match="already exists"):
        db.create_portfolio("Alpha")
    with pytest.raises(ValueError):
        db.create_portfolio("   ")


def test_round_trip_survives_restart(tmp_path, store):
    path = tmp_path / "persist.db"
    db1 = Database(path)
    pf = trade_some(db1.create_portfolio("Main", fees=FEES), store)
    db1.save_portfolio(pf)
    db1.close()

    db2 = Database(path)  # simulate an app restart
    loaded = db2.load_portfolio(pf.id)
    assert loaded.name == "Main" and loaded.fees == FEES
    assert loaded.cash == pytest.approx(pf.cash)
    assert loaded.realised_pnl == pytest.approx(pf.realised_pnl)
    assert loaded.clock == pf.clock == pd.Timestamp("2024-01-03")
    assert {t: p.quantity for t, p in loaded.positions.items()} == {"XYZ.L": 60, "ABC": 10}
    assert loaded.positions["XYZ.L"].cost_basis == pytest.approx(pf.positions["XYZ.L"].cost_basis)
    assert [t.id for t in loaded.trades] == [t.id for t in pf.trades]
    assert len(loaded.pending_orders) == 1
    assert loaded.pending_orders[0].stop_price == 50
    assert db2.list_portfolios()["trades"].iloc[0] == 3
    db2.close()


def test_resave_is_idempotent_and_reflects_changes(db, store):
    pf = trade_some(db.create_portfolio("P"), store)
    db.save_portfolio(pf)
    db.save_portfolio(pf)
    pf.cancel_order(pf.pending_orders[0].id)
    db.save_portfolio(pf)
    loaded = db.load_portfolio(pf.id)
    assert len(loaded.trades) == 3
    assert loaded.orders[-1].status is OrderStatus.CANCELLED


def test_load_missing_and_save_unknown(db):
    with pytest.raises(KeyError):
        db.load_portfolio(999)
    with pytest.raises(ValueError, match="no id"):
        db.save_portfolio(Portfolio())
    with pytest.raises(ValueError, match="No portfolio"):
        db.save_portfolio(Portfolio(id=999))


def test_delete_cascades(db, store):
    pf = trade_some(db.create_portfolio("Gone"), store)
    db.save_portfolio(pf)
    db.delete_portfolio(pf.id)
    assert db.list_portfolios().empty
    with db._tx() as c:
        assert c.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 0
        assert c.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0


def test_reset(db, store):
    pf = trade_some(db.create_portfolio("R", 7_000), store)
    db.save_portfolio(pf)
    fresh = db.reset_portfolio(pf.id)
    assert fresh.cash == 7_000 and not fresh.trades
    assert db.load_portfolio(pf.id).cash == 7_000


def test_in_memory_database(store):
    with Database(":memory:") as db:
        pf = db.create_portfolio("M")
        pf.place_order(make_order("XYZ.L", "buy", 1, "2024-01-02"), store)
        db.save_portfolio(pf)
        assert len(db.load_portfolio(pf.id).trades) == 1


def test_export_trades_csv(tmp_path, store):
    pf = trade_some(Portfolio(fees=FEES), store)
    out = tmp_path / "trades.csv"
    text = export_trades_csv(pf, out)
    assert out.read_text() == text
    df = pd.read_csv(out)
    assert len(df) == 3
    assert {
        "date",
        "ticker",
        "side",
        "quantity",
        "price",
        "commission_gbp",
        "realised_pnl_gbp",
    } <= set(df.columns)
    assert export_trades_csv(Portfolio()).startswith("id,order_id,date")

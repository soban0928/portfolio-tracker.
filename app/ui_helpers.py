"""Shared Streamlit plumbing: cached resources, sidebar portfolio picker, formatting.

No business logic lives here; it only wires the UI to the ``simulator`` package.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# Allow running without `pip install -e .` (e.g. a bare `streamlit run`).
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from simulator.config import (  # noqa: E402
    DEFAULT_DB_PATH,
    DOWNLOADED_DATA_DIR,
    STARTING_CASH,
    FeeConfig,
)
from simulator.data import PriceStore  # noqa: E402
from simulator.db import Database  # noqa: E402
from simulator.portfolio import Portfolio  # noqa: E402

DB_ENV_VAR = "SIMULATOR_DB"


@st.cache_resource
def get_store() -> PriceStore:
    """Price data, loaded once per server process."""
    return PriceStore.default()


@st.cache_resource
def _db(path: str) -> Database:
    return Database(path)


def get_db() -> Database:
    """Shared database (path from the ``SIMULATOR_DB`` env var, else the default)."""
    return _db(os.environ.get(DB_ENV_VAR, str(DEFAULT_DB_PATH)))


def using_synthetic_data() -> bool:
    """True when no downloaded (real) price files are present."""
    return not any(DOWNLOADED_DATA_DIR.glob("*.csv"))


def page_setup(title: str, icon: str = "📈") -> None:
    """Common page config and the synthetic-data notice."""
    st.set_page_config(page_title=f"{title} · Trading Simulator", page_icon=icon, layout="wide")
    st.title(title)


def data_notice() -> None:
    """Explain the data source in the sidebar."""
    if using_synthetic_data():
        st.sidebar.caption(
            "⚠️ Using bundled **synthetic** sample prices. Run "
            "`python scripts/download_data.py` for real data."
        )


def sidebar_portfolio() -> Portfolio:
    """Sidebar: choose or create a portfolio. Returns the active portfolio.

    The portfolio is reloaded from SQLite on every rerun, so the database is
    always the source of truth.
    """
    db = get_db()
    st.sidebar.header("Portfolio")
    portfolios = db.list_portfolios()
    if portfolios.empty:
        db.create_portfolio("My Portfolio")
        portfolios = db.list_portfolios()

    ids = portfolios["id"].tolist()
    names = dict(zip(portfolios["id"], portfolios["name"], strict=True))
    current = st.session_state.get("portfolio_id")
    index = ids.index(current) if current in ids else 0
    chosen = st.sidebar.selectbox(
        "Active portfolio", ids, index=index, format_func=lambda i: names[i], key="pf_select"
    )
    st.session_state["portfolio_id"] = chosen

    with (
        st.sidebar.expander("➕ New portfolio"),
        st.form("new_portfolio", clear_on_submit=True),
    ):
        name = st.text_input("Name")
        cash = st.number_input("Starting cash (£)", 100.0, 10_000_000.0, STARTING_CASH, 500.0)
        commission = st.number_input("Commission per trade (£)", 0.0, 100.0, 5.0, 0.5)
        slippage = st.number_input("Slippage / spread (bps)", 0.0, 500.0, 10.0, 1.0)
        if st.form_submit_button("Create"):
            try:
                pf = db.create_portfolio(name, cash, FeeConfig(commission, slippage))
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.session_state["portfolio_id"] = pf.id
                st.session_state.pop("pf_select", None)
                st.rerun()

    pf = db.load_portfolio(chosen)
    st.sidebar.caption(
        f"Fees: £{pf.fees.commission:g} commission + {pf.fees.slippage_bps:g} bps slippage"
    )
    st.sidebar.caption(
        f"Simulation date: **{pf.clock.date() if pf.clock is not None else 'not started'}**"
    )
    data_notice()
    return pf


def gbp(x: float) -> str:
    """Format as pounds, e.g. ``£1,234.56`` / ``-£12.00``."""
    return f"-£{abs(x):,.2f}" if x < 0 else f"£{x:,.2f}"


def pct(x: float, signed: bool = True) -> str:
    """Format a fraction as a percentage (``nan`` -> em dash)."""
    if x != x:  # NaN
        return "—"
    return f"{x:+.2%}" if signed else f"{x:.2%}"


def num(x: float, digits: int = 2) -> str:
    """Format a plain number (``nan`` -> em dash)."""
    return "—" if x != x else f"{x:.{digits}f}"

"""Smoke tests for the Streamlit dashboard using streamlit.testing.AppTest."""

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))  # pages import ui_helpers/charts from app/

PAGES = ["1_Trade.py", "2_Portfolio.py", "3_Transactions.py", "4_Analytics.py", "5_Backtest.py"]


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    path = tmp_path / "app.db"
    monkeypatch.setenv("SIMULATOR_DB", str(path))
    return path


def run(page: str) -> AppTest:
    script = APP / page if page == "Home.py" else APP / "pages" / page
    at = AppTest.from_file(str(script), default_timeout=60)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


def seed_portfolio(db_path):
    from simulator.data import PriceStore
    from simulator.db import Database
    from simulator.orders import make_order

    store = PriceStore.default()
    with Database(db_path) as db:
        pf = db.create_portfolio("Seeded")
        pf.place_order(make_order("AAPL", "buy", 20, "2020-01-02"), store)
        pf.place_order(make_order("VOD.L", "buy", 500, "2020-01-03"), store)
        pf.place_order(make_order("AAPL", "sell", 5, "2020-03-02"), store)
        pf.place_order(
            make_order("AAPL", "sell", 5, "2020-03-02", "limit", limit_price=10_000), store
        )
        pf.process_pending(store, "2020-12-31")
        db.save_portfolio(pf)


@pytest.mark.parametrize("page", ["Home.py", *PAGES])
def test_pages_render_with_empty_portfolio(db_path, page):
    at = run(page)
    assert at.title[0].value


@pytest.mark.parametrize("page", ["Home.py", *PAGES])
def test_pages_render_with_trades(db_path, page):
    seed_portfolio(db_path)
    at = run(page)
    if page == "2_Portfolio.py":
        assert len(at.get("plotly_chart")) == 3
    if page == "4_Analytics.py":
        assert any(m.label == "Sharpe ratio" for m in at.metric)
    if page == "3_Transactions.py":
        assert at.get("download_button")


def test_place_market_order_through_ui(db_path):
    at = run("1_Trade.py")
    next(n for n in at.number_input if n.label == "Quantity (shares)").set_value(3)
    button(at, "Place order").click().run()
    assert not at.exception
    assert any("Filled: BUY 3 AAPL" in s.value for s in at.success)

    from simulator.db import Database

    with Database(db_path) as db:
        pf = db.load_portfolio(int(db.list_portfolios()["id"].iloc[0]))
    assert pf.shares_held("AAPL") == 3


def test_rejected_order_shows_error(db_path):
    at = run("1_Trade.py")
    at.radio[0].set_value("SELL").run()
    button(at, "Place order").click().run()
    assert any("Order rejected" in e.value for e in at.error)


def test_run_backtest_through_ui(db_path):
    at = run("5_Backtest.py")
    button(at, "Run backtest").click().run()
    assert not at.exception
    assert len(at.get("plotly_chart")) == 2
    assert at.dataframe

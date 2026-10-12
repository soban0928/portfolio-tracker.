"""Smoke tests: every dashboard page renders from the committed artefacts."""

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))
PAGES = [
    "1_Match_Predictor.py",
    "2_Tournament_Probabilities.py",
    "3_Simulated_Bracket.py",
    "4_Model_Evaluation.py",
    "5_Predicted_vs_Actual.py",
]


def run(page: str) -> AppTest:
    path = APP / page if page == "Home.py" else APP / "pages" / page
    at = AppTest.from_file(str(path), default_timeout=120)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at


@pytest.mark.parametrize("page", ["Home.py", *PAGES])
def test_page_renders(page):
    at = run(page)
    assert at.title[0].value


@pytest.mark.parametrize("year", [2022, 2018])
@pytest.mark.parametrize("page", [PAGES[1], PAGES[2], PAGES[4]])
def test_year_switch(page, year):
    at = run(page)
    at.segmented_control(key="year").set_value(year).run()
    assert not at.exception, [e.message for e in at.exception]


def test_match_predictor_interaction():
    at = run("1_Match_Predictor.py")
    at.selectbox[0].set_value("Brazil")
    at.selectbox[1].set_value("Brazil").run()
    assert any("different teams" in w.value for w in at.warning)
    at.selectbox[1].set_value("Japan").run()  # venue options depend on the teams
    at.selectbox[2].set_value("Japan at home").run()
    assert not at.exception
    assert len(at.get("plotly_chart")) == 3


def test_elo_only_toggle():
    at = run("2_Tournament_Probabilities.py")
    at.radio[0].set_value("Elo-only Dixon-Coles").run()
    assert not at.exception

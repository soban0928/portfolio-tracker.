"""Shared Streamlit plumbing: artefact loading, caching and formatting.

No modelling logic lives here; the app reads artefacts written by
``scripts/run_pipeline.py`` and calls into the ``wcpredict`` package.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:  # allow running without `pip install -e .`
    sys.path.insert(0, str(_SRC))

from wcpredict.config import ARTEFACT_DIR  # noqa: E402
from wcpredict.predict import Predictor, load_predictor  # noqa: E402

YEARS = [2026, 2022, 2018]
STAGE_LABELS = {
    "R32": "Round of 32",
    "R16": "Round of 16",
    "QF": "Quarter-final",
    "SF": "Semi-final",
    "FINAL": "Final",
    "WINNER": "Champion",
    "GROUP": "Group stage",
}


def page(title: str, icon: str = "⚽") -> None:
    """Common page setup."""
    st.set_page_config(page_title=f"{title} · World Cup Predictor", page_icon=icon, layout="wide")
    st.title(title)


@st.cache_data
def summary() -> dict:
    """Pipeline summary (``artefacts/summary.json``)."""
    return json.loads((ARTEFACT_DIR / "summary.json").read_text())


@st.cache_data
def csv(relative: str, **kwargs: object) -> pd.DataFrame:
    """Read an artefact CSV."""
    return pd.read_csv(ARTEFACT_DIR / relative, **kwargs)


@st.cache_data
def bracket(year: int) -> list[dict]:
    """Bracket summary for a tournament."""
    return json.loads((ARTEFACT_DIR / str(year) / "bracket.json").read_text())


@st.cache_resource
def predictor() -> Predictor:
    """Current models and team states (loaded once per server)."""
    return load_predictor()


def pct(x: float, digits: int = 1) -> str:
    """Format a probability as a percentage."""
    return f"{100 * x:.{digits}f}%"


def year_picker(key: str = "year") -> int:
    """Tournament selector shared by several pages."""
    return st.segmented_control("Tournament", YEARS, default=2026, key=key) or 2026


FEATURE_LABELS = {
    "elo_diff": "Elo difference",
    "venue": "Home / neutral venue",
    "form5_diff": "Form diff (last 5)",
    "form10_diff": "Form diff (last 10)",
    "gf10_home": "Goals scored, team A (last 10)",
    "ga10_home": "Goals conceded, team A (last 10)",
    "gf10_away": "Goals scored, team B (last 10)",
    "ga10_away": "Goals conceded, team B (last 10)",
    "gd10_diff": "Goal-difference diff (last 10)",
    "elo_trend_diff": "Elo trend diff (last 10)",
    "rest_home": "Rest days, team A",
    "rest_away": "Rest days, team B",
    "exp_home": "Experience, team A",
    "exp_away": "Experience, team B",
    "is_friendly": "Friendly",
    "is_tournament": "Major tournament",
    "importance": "Match importance",
}

"""Shared fixtures."""

from __future__ import annotations

import pandas as pd
import pytest

from wcpredict.data import clean_matches, load_matches


def raw_frame(rows: list[tuple]) -> pd.DataFrame:
    """Rows of (date, home, away, hs, as, tournament, country, neutral)."""
    df = pd.DataFrame(
        rows,
        columns=["date", "home_team", "away_team", "home_score", "away_score",
                 "tournament", "country", "neutral"],
    )  # fmt: skip
    df["date"] = pd.to_datetime(df["date"])
    df["city"] = "X"
    return df


@pytest.fixture(scope="session")
def matches() -> pd.DataFrame:
    """The real, cleaned dataset (loaded once per test session)."""
    return load_matches()


@pytest.fixture
def toy_matches() -> pd.DataFrame:
    """A tiny hand-made history for unit tests."""
    rows = [
        ("2020-01-01", "A", "B", 2, 0, "Friendly", "A", False),
        ("2020-02-01", "B", "C", 1, 1, "FIFA World Cup qualification", "B", False),
        ("2020-03-01", "C", "A", 0, 3, "UEFA Euro", "D", True),
        ("2020-04-01", "A", "C", 1, 0, "FIFA World Cup", "A", False),
        ("2020-05-01", "B", "A", 2, 2, "Gulf Cup", "E", True),
        ("2020-06-01", "C", "B", 0, 1, "UEFA Nations League", "C", False),
    ]
    return clean_matches(raw_frame(rows))

"""Paths, seeds and tournament definitions shared across the package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ARTEFACT_DIR = PROJECT_ROOT / "artefacts"

#: Global seed used by models, CV and simulations.
SEED = 2026

#: Matches before this date are only used to warm up Elo/form, not for training.
TRAIN_START = pd.Timestamp("1994-01-01")


@dataclass(frozen=True)
class TournamentSpec:
    """A World Cup we predict and evaluate.

    Attributes:
        year: Tournament year.
        start: Date of the opening match; models only see matches before this.
        end: Date of the final.
        n_teams: Number of finalists (32 or 48).
    """

    year: int
    start: pd.Timestamp
    end: pd.Timestamp
    n_teams: int


TOURNAMENTS: dict[int, TournamentSpec] = {
    2018: TournamentSpec(2018, pd.Timestamp("2018-06-14"), pd.Timestamp("2018-07-15"), 32),
    2022: TournamentSpec(2022, pd.Timestamp("2022-11-20"), pd.Timestamp("2022-12-18"), 32),
    2026: TournamentSpec(2026, pd.Timestamp("2026-06-11"), pd.Timestamp("2026-07-19"), 48),
}

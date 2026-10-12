"""Predict any fixture with the current (all-data) models.

Loads ``artefacts/current`` written by ``scripts/run_pipeline.py``: the best
W/D/L classifier, the goals model and every active team's latest state. If a
saved model cannot be unpickled (e.g. a different XGBoost version), the
models are retrained from the committed raw data.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from wcpredict.config import ARTEFACT_DIR
from wcpredict.features import TeamState, fixture_features
from wcpredict.goals import GoalsModel, top_scorelines
from wcpredict.models import MatchModel

COMPETITIONS = {
    "World Cup match": "world_cup",
    "Continental final tournament": "continental_final",
    "Qualifier / Nations League": "qualifier",
    "Friendly": "friendly",
}


@dataclass
class MatchPrediction:
    """Everything the Match Predictor shows for one fixture."""

    team_a: str
    team_b: str
    classifier_probs: np.ndarray  # (3,) P(A win), P(draw), P(B win)
    goals_probs: np.ndarray  # (3,)
    expected_goals: tuple[float, float]
    score_matrix: np.ndarray  # (11, 11), rows = team A goals
    top_scorelines: list[tuple[int, int, float]]
    features: dict[str, float]


class Predictor:
    """Current models plus team states."""

    def __init__(self, classifier: MatchModel, goals: GoalsModel, states: pd.DataFrame,
                 meta: dict) -> None:  # fmt: skip
        """Wrap fitted models and the team-state table."""
        self.classifier = classifier
        self.goals = goals
        self.meta = meta
        self.as_of = pd.Timestamp(meta["as_of"])
        self._states = {r.team: _row_to_state(r) for r in states.itertuples()}

    @property
    def teams(self) -> list[str]:
        """Teams with a stored state, strongest (by Elo) first."""
        return sorted(self._states, key=lambda t: -self._states[t].elo)

    def state(self, team: str) -> TeamState:
        """Latest state of ``team``."""
        return self._states[team]

    def predict(
        self, team_a: str, team_b: str, venue: float = 0.0, competition: str = "world_cup"
    ) -> MatchPrediction:
        """Predict ``team_a`` v ``team_b`` (venue +1 if A hosts, -1 if B hosts, 0 neutral)."""
        if team_a == team_b:
            raise ValueError("Pick two different teams")
        feats = fixture_features(
            self.state(team_a), self.state(team_b), self.as_of, venue=venue,
            competition=competition, rest_home=4.0, rest_away=4.0,
        )  # fmt: skip
        X = pd.DataFrame([feats])
        matrix = self.goals.score_matrix(X)[0]
        lam, mu = self.goals.rates(X)
        return MatchPrediction(
            team_a, team_b,
            self.classifier.predict_proba(X)[0],
            self.goals.predict_proba(X)[0],
            (float(lam[0]), float(mu[0])),
            matrix,
            top_scorelines(matrix, 5),
            feats,
        )  # fmt: skip


def _row_to_state(r: object) -> TeamState:
    last = getattr(r, "last_match", None)
    return TeamState(
        team=r.team, elo=r.elo, pts5=r.pts5, pts10=r.pts10, gf10=r.gf10, ga10=r.ga10,
        trend10=r.trend10, n_played=r.n_played,
        last_match=None if last is None or pd.isna(last) else pd.Timestamp(last),
        rank=np.nan,
    )  # type: ignore[attr-defined]  # fmt: skip


def load_predictor(artefact_dir: Path = ARTEFACT_DIR) -> Predictor:
    """Load current models and states; retrain from raw data if unpickling fails."""
    cur = artefact_dir / "current"
    meta = json.loads((cur / "meta.json").read_text())
    states = pd.read_csv(cur / "team_states.csv")
    try:
        classifier = joblib.load(cur / "classifier.joblib")
        goals = joblib.load(cur / "goals.joblib")
    except Exception as exc:  # noqa: BLE001 - version mismatch etc.: fall back to retraining
        warnings.warn(f"Could not load saved models ({exc}); retraining.", stacklevel=2)
        classifier, goals = retrain_current()
    return Predictor(classifier, goals, states, meta)


def retrain_current() -> tuple[MatchModel, GoalsModel]:
    """Retrain the current classifier (XGBoost) and goals model on all data."""
    from wcpredict.data import load_matches
    from wcpredict.elo import compute_elo
    from wcpredict.evaluation import training_rows
    from wcpredict.features import BASE_FEATURES, build_features
    from wcpredict.models import encode, xgboost

    f = build_features(compute_elo(load_matches()))
    rows = training_rows(f, f["date"].max() + pd.Timedelta(days=1))
    return xgboost(BASE_FEATURES).fit(rows, encode(rows["result"])), GoalsModel().fit(rows)

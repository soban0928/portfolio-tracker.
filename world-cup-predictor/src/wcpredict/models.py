"""Win/draw/loss classifiers and baselines.

All models predict probabilities for the classes ``["H", "D", "A"]`` (first
team wins, draw, second team wins) and share one interface:
``fit(X, y)`` / ``predict_proba(X) -> (n, 3)``.

Training data is **mirrored**: each match is also added with the teams
swapped (features mirrored, H<->A). Predictions average a fixture with its
mirror, so P(A beats B) does not depend on which team is listed first.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from wcpredict.config import SEED
from wcpredict.features import swap_sides

CLASSES = ["H", "D", "A"]
LABEL = {c: i for i, c in enumerate(CLASSES)}


class MatchModel(Protocol):
    """Interface shared by every classifier and baseline."""

    name: str
    features: list[str]

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> MatchModel: ...  # noqa: D102

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...  # noqa: D102


def encode(results: pd.Series) -> np.ndarray:
    """``"H"/"D"/"A"`` -> 0/1/2."""
    return results.map(LABEL).to_numpy(dtype=int)


def flip_labels(y: np.ndarray) -> np.ndarray:
    """Swap H and A labels (for mirrored rows)."""
    return 2 - y


def flip_proba(p: np.ndarray) -> np.ndarray:
    """Swap H and A probability columns."""
    return p[:, ::-1]


def mirror(X: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Original rows plus their side-swapped mirror images."""
    return pd.concat([X, swap_sides(X)], ignore_index=True), np.concatenate([y, flip_labels(y)])


class SklearnModel:
    """Wraps any sklearn-style classifier with mirroring and symmetric prediction."""

    def __init__(self, name: str, estimator: object, features: list[str]) -> None:
        """Create a named model over ``features``."""
        self.name = name
        self.estimator = estimator
        self.features = list(features)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> SklearnModel:
        """Fit on mirrored data."""
        Xm, ym = mirror(X[self.features], y)
        self.estimator.fit(Xm, ym)  # type: ignore[attr-defined]
        return self

    def _raw(self, X: pd.DataFrame) -> np.ndarray:
        p = self.estimator.predict_proba(X[self.features])  # type: ignore[attr-defined]
        return np.asarray(p, dtype=np.float64)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Average of the fixture and its mirror image (exactly symmetric)."""
        p = self._raw(X)
        p_mirror = flip_proba(self._raw(swap_sides(X[self.features])))
        out = 0.5 * (p + p_mirror)
        return out / out.sum(axis=1, keepdims=True)


class BaseRateModel:
    """Predicts the training-set class frequencies (symmetrised) for every match."""

    name = "Base rates"
    features: list[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> BaseRateModel:
        """Store mirrored class frequencies."""
        _, ym = mirror(X.iloc[:, :0], y)
        self.p_ = np.bincount(ym, minlength=3) / len(ym)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Same probabilities for every row."""
        return np.tile(self.p_, (len(X), 1))


class HigherRatedModel:
    """Always picks the higher-rated team (Elo incl. home advantage) to win.

    Never predicts a draw. Probabilities are one-hot, so log loss is infinite;
    this baseline is reported for accuracy only.
    """

    name = "Higher-rated team"
    features = ["elo_diff", "venue"]

    def __init__(self, home_advantage: float = 100.0) -> None:
        """Use ``home_advantage`` Elo points for the host."""
        self.home_advantage = home_advantage

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> HigherRatedModel:
        """Nothing to learn."""
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """One-hot on the favourite (ties go to the first team)."""
        edge = X["elo_diff"].to_numpy() + self.home_advantage * X["venue"].to_numpy()
        p = np.zeros((len(X), 3))
        p[edge >= 0, 0] = 1.0
        p[edge < 0, 2] = 1.0
        return p


def elo_only(features: list[str] | None = None) -> SklearnModel:
    """Baseline: multinomial logistic regression on Elo difference and venue only."""
    return SklearnModel(
        "Elo-only (logistic)",
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        features or ["elo_diff", "venue"],
    )


def logistic(features: list[str]) -> SklearnModel:
    """Multinomial logistic regression on all features."""
    return SklearnModel(
        "Logistic regression",
        make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
        features,
    )


def random_forest(features: list[str], **params: object) -> SklearnModel:
    """Random Forest; defaults favour smooth probabilities (large leaves)."""
    defaults = dict(n_estimators=300, min_samples_leaf=40, max_features=0.5, n_jobs=-1,
                    random_state=SEED)  # fmt: skip
    return SklearnModel("Random Forest", RandomForestClassifier(**{**defaults, **params}), features)


def xgboost(features: list[str], **params: object) -> SklearnModel:
    """Gradient-boosted trees with a multi-class soft-probability objective."""
    defaults = dict(
        n_estimators=300, learning_rate=0.03, max_depth=3, min_child_weight=20,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
        objective="multi:softprob", eval_metric="mlogloss", n_jobs=4, random_state=SEED,
    )  # fmt: skip
    return SklearnModel("XGBoost", XGBClassifier(**{**defaults, **params}), features)


def model_zoo(features: list[str]) -> list[MatchModel]:
    """All models compared in the evaluation, baselines first."""
    return [
        BaseRateModel(),
        HigherRatedModel(),
        elo_only(),
        logistic(features),
        random_forest(features),
        xgboost(features),
    ]

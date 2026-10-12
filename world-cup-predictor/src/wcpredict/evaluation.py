"""Metrics, time-series cross-validation and calibration curves."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd

from wcpredict.config import TRAIN_START
from wcpredict.models import MatchModel, encode

EPS = 1e-15


def one_hot(y: np.ndarray, k: int = 3) -> np.ndarray:
    """Integer labels -> one-hot matrix."""
    out = np.zeros((len(y), k))
    out[np.arange(len(y)), y] = 1.0
    return out


def accuracy(y: np.ndarray, p: np.ndarray) -> float:
    """Share of matches where the most likely outcome happened."""
    return float((p.argmax(axis=1) == y).mean())


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    """Mean negative log probability of the observed outcome (natural log).

    Returns ``inf`` if any observed outcome had probability 0.
    """
    probs = p[np.arange(len(y)), y]
    if (probs <= 0).any():
        return float("inf")
    return float(-np.log(np.clip(probs, EPS, 1)).mean())


def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Multi-class Brier score: mean over matches of sum_k (p_k - o_k)^2 (range 0-2)."""
    return float(((p - one_hot(y, p.shape[1])) ** 2).sum(axis=1).mean())


def rps(y: np.ndarray, p: np.ndarray) -> float:
    """Ranked probability score for the ordered outcomes H < D < A (range 0-1).

    RPS = 1/(K-1) * sum_k (cumP_k - cumO_k)^2, averaged over matches. It rewards
    putting probability near the observed outcome (a draw is "closer" to a
    home win than an away win is).
    """
    cp = np.cumsum(p, axis=1)[:, :-1]
    co = np.cumsum(one_hot(y, p.shape[1]), axis=1)[:, :-1]
    return float(((cp - co) ** 2).sum(axis=1).mean() / (p.shape[1] - 1))


def score(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """All metrics for one set of predictions."""
    return {
        "accuracy": accuracy(y, p),
        "log_loss": log_loss(y, p),
        "brier": brier(y, p),
        "rps": rps(y, p),
        "n": float(len(y)),
    }


def training_rows(features: pd.DataFrame, before: pd.Timestamp | str) -> pd.DataFrame:
    """Rows usable for training a model applied from ``before`` onwards."""
    f = features
    return f[(f["date"] >= TRAIN_START) & (f["date"] < pd.Timestamp(before))]


def time_series_cv(
    model_factory: Callable[[], MatchModel],
    features: pd.DataFrame,
    test_years: Iterable[int],
    subset: Callable[[pd.DataFrame], pd.Series] | None = None,
) -> pd.DataFrame:
    """Expanding-window CV: for each year Y, train on [TRAIN_START, Y) and test on Y.

    Args:
        model_factory: Creates a fresh, unfitted model.
        features: Output of :func:`wcpredict.features.build_features`.
        test_years: Years used as successive test folds.
        subset: Optional row filter for the *test* fold (e.g. competitive only).

    Returns:
        One row per fold with the metrics and the model name.
    """
    rows = []
    for year in test_years:
        start = pd.Timestamp(year=year, month=1, day=1)
        train = training_rows(features, start)
        test = features[features["date"].dt.year == year]
        if subset is not None:
            test = test[subset(test)]
        model = model_factory()
        model.fit(train, encode(train["result"]))
        p = model.predict_proba(test)
        rows.append({"model": model.name, "year": year, **score(encode(test["result"]), p)})
    return pd.DataFrame(rows)


def summarise_cv(cv: pd.DataFrame) -> pd.DataFrame:
    """Match-weighted mean of each metric per model."""
    metrics = ["accuracy", "log_loss", "brier", "rps"]
    w = cv["n"]
    out = cv[metrics].mul(w, axis=0).groupby(cv["model"]).sum().div(w.groupby(cv["model"]).sum(),
                                                                     axis=0)  # fmt: skip
    out["n"] = cv.groupby("model")["n"].sum()
    return out


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Reliability data for each outcome class (equal-width probability bins).

    Returns:
        Rows of (outcome, bin_mid, mean_predicted, observed_rate, count).
    """
    rows = []
    edges = np.linspace(0, 1, bins + 1)
    for k, label in enumerate(["H", "D", "A"]):
        idx = np.clip(np.digitize(p[:, k], edges[1:-1]), 0, bins - 1)
        for b in range(bins):
            mask = idx == b
            if mask.sum() == 0:
                continue
            rows.append(
                {
                    "outcome": label,
                    "bin_mid": (edges[b] + edges[b + 1]) / 2,
                    "mean_predicted": float(p[mask, k].mean()),
                    "observed_rate": float((y[mask] == k).mean()),
                    "count": int(mask.sum()),
                }
            )
    return pd.DataFrame(rows)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Count-weighted mean |predicted - observed| over all classes and bins."""
    t = calibration_table(y, p, bins)
    return float((t["count"] * (t["mean_predicted"] - t["observed_rate"]).abs()).sum()
                 / t["count"].sum())  # fmt: skip

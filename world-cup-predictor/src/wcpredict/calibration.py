"""Post-hoc probability calibration (isotonic or Platt) for W/D/L models.

Calibrators are learned on a held-out, *later* slice of the training window:
the base model is fitted on matches before the calibration window, its
predictions on the window train the calibrators, and the base model is then
refitted on all training data. Calibration is symmetric: one calibrator is
shared by the win classes (H and A, using mirrored predictions) and one is
used for draws, followed by renormalisation.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from wcpredict.models import MatchModel, encode, flip_labels, flip_proba

EPS = 1e-6


class _Platt:
    def fit(self, p: np.ndarray, y: np.ndarray) -> _Platt:
        z = np.log(np.clip(p, EPS, 1 - EPS) / np.clip(1 - p, EPS, 1)).reshape(-1, 1)
        self.lr = LogisticRegression(C=1e6).fit(z, y)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        z = np.log(np.clip(p, EPS, 1 - EPS) / np.clip(1 - p, EPS, 1)).reshape(-1, 1)
        return self.lr.predict_proba(z)[:, 1]


class _Isotonic:
    def fit(self, p: np.ndarray, y: np.ndarray) -> _Isotonic:
        self.iso = IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds="clip").fit(p, y)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        return self.iso.predict(p)


class CalibratedModel:
    """Wraps a model factory with isotonic or Platt calibration."""

    def __init__(
        self,
        factory: Callable[[], MatchModel],
        method: str = "isotonic",
        calib_years: int = 4,
    ) -> None:
        """Create a calibrated model.

        Args:
            factory: Creates an unfitted base model.
            method: ``"isotonic"`` or ``"platt"``.
            calib_years: Length of the held-out calibration window (most recent years).
        """
        if method not in ("isotonic", "platt"):
            raise ValueError("method must be 'isotonic' or 'platt'")
        self.factory = factory
        self.method = method
        self.calib_years = calib_years
        base = factory()
        self.name = f"{base.name} + {method}"
        self.features = base.features

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> CalibratedModel:
        """Fit base on early data, calibrators on the latest window, then refit base."""
        cut = X["date"].max() - pd.DateOffset(years=self.calib_years)
        early, late = X[X["date"] <= cut], X[X["date"] > cut]
        holdout_model = self.factory().fit(early, encode(early["result"]))
        p = holdout_model.predict_proba(late)
        yl = encode(late["result"])
        p_all = np.vstack([p, flip_proba(p)])
        y_all = np.concatenate([yl, flip_labels(yl)])
        make = _Isotonic if self.method == "isotonic" else _Platt
        self.win_cal_ = make().fit(p_all[:, 0], (y_all == 0).astype(int))
        self.draw_cal_ = make().fit(p_all[:, 1], (y_all == 1).astype(int))
        self.base_ = self.factory().fit(X, y)
        return self

    def calibrate(self, p: np.ndarray) -> np.ndarray:
        """Apply the fitted calibrators to raw probabilities and renormalise."""
        out = np.column_stack(
            [self.win_cal_.predict(p[:, 0]), self.draw_cal_.predict(p[:, 1]),
             self.win_cal_.predict(p[:, 2])]
        )  # fmt: skip
        out = np.clip(out, EPS, None)
        return out / out.sum(axis=1, keepdims=True)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated W/D/L probabilities."""
        return self.calibrate(self.base_.predict_proba(X))

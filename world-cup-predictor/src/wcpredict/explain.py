"""SHAP explanations for the W/D/L classifiers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from wcpredict.models import CLASSES, SklearnModel


def _shap_values(model: SklearnModel, X: pd.DataFrame, background: pd.DataFrame) -> np.ndarray:
    """SHAP values with shape (n, features, classes) for tree or linear models."""
    est = model.estimator
    if hasattr(est, "steps"):  # sklearn pipeline: scaler + logistic regression
        scaler, clf = est[:-1], est[-1]
        explainer = shap.LinearExplainer(clf, scaler.transform(background))
        values = explainer.shap_values(scaler.transform(X))
    else:
        values = shap.TreeExplainer(est).shap_values(X)
    values = np.asarray(values)
    if values.ndim == 3 and values.shape[0] == len(CLASSES) and values.shape[1] == len(X):
        values = np.transpose(values, (1, 2, 0))  # older SHAP: list per class
    return values


def shap_summary(
    model: SklearnModel, X: pd.DataFrame, n: int = 1000, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Global SHAP importance and a per-row sample for beeswarm-style plots.

    Explains the underlying fitted estimator (trained on mirrored rows) on a
    random sample of ``n`` matches.

    Returns:
        ``importance``: mean |SHAP| per feature (overall and per class), sorted.
        ``sample``: long table (feature, value, shap_H, shap_D, shap_A) for the sample.
    """
    Xs = X[model.features].sample(min(n, len(X)), random_state=seed)
    background = X[model.features].sample(min(200, len(X)), random_state=seed + 1)
    values = _shap_values(model, Xs, background)  # (n, f, 3)
    imp = pd.DataFrame(
        {f"mean_abs_{c}": np.abs(values[:, :, k]).mean(axis=0) for k, c in enumerate(CLASSES)},
        index=model.features,
    )
    imp["mean_abs"] = imp.mean(axis=1)
    imp = imp.sort_values("mean_abs", ascending=False)
    sample = []
    for fi, feat in enumerate(model.features):
        sample.append(
            pd.DataFrame(
                {
                    "feature": feat,
                    "value": Xs[feat].to_numpy(),
                    **{f"shap_{c}": values[:, fi, k] for k, c in enumerate(CLASSES)},
                }
            )  # fmt: skip
        )
    return imp, pd.concat(sample, ignore_index=True)

import numpy as np
import pandas as pd
import pytest

from wcpredict.evaluation import (
    accuracy,
    brier,
    calibration_table,
    expected_calibration_error,
    log_loss,
    rps,
    score,
    summarise_cv,
    time_series_cv,
    training_rows,
)
from wcpredict.features import BASE_FEATURES, swap_sides
from wcpredict.models import (
    BaseRateModel,
    HigherRatedModel,
    elo_only,
    encode,
    flip_proba,
    logistic,
    model_zoo,
    random_forest,
    xgboost,
)

P = np.array([[0.5, 0.3, 0.2]])


# ---------------------------------------------------------------- metrics
def test_metrics_hand_calc():
    assert log_loss(np.array([0]), P) == pytest.approx(np.log(2))
    assert brier(np.array([0]), P) == pytest.approx(0.25 + 0.09 + 0.04)
    # cumulative p = [0.5, 0.8]; observed H -> [1, 1]; (0.25 + 0.04) / 2
    assert rps(np.array([0]), P) == pytest.approx(0.145)
    # observed A -> [0, 0]; (0.25 + 0.64) / 2
    assert rps(np.array([2]), P) == pytest.approx(0.445)
    assert rps(np.array([1]), P) == pytest.approx((0.25 + 0.04) / 2)
    assert accuracy(np.array([0, 1]), np.vstack([P, P])) == 0.5


def test_log_loss_infinite_for_zero_probability():
    assert log_loss(np.array([1]), np.array([[1.0, 0.0, 0.0]])) == float("inf")


def test_perfect_and_uniform_scores():
    y = np.array([0, 1, 2])
    s = score(y, np.eye(3))
    assert s["log_loss"] == pytest.approx(0, abs=1e-9) and s["brier"] == 0 and s["rps"] == 0
    u = score(y, np.full((3, 3), 1 / 3))
    assert u["log_loss"] == pytest.approx(np.log(3))


def test_calibration_of_calibrated_predictions():
    rng = np.random.default_rng(0)
    p = rng.dirichlet([2, 1, 2], size=20_000)
    y = np.array([rng.choice(3, p=row) for row in p])
    assert expected_calibration_error(y, p) < 0.02
    t = calibration_table(y, p)
    assert set(t["outcome"]) == {"H", "D", "A"}
    overconfident = np.clip(p * 1.6 - 0.2, 0, None)
    overconfident /= overconfident.sum(axis=1, keepdims=True)
    assert expected_calibration_error(y, overconfident) > 0.03


# ---------------------------------------------------------------- models
@pytest.fixture(scope="module")
def small(features):
    train = features[(features["date"] >= "2008-01-01") & (features["date"] < "2014-01-01")]
    test = features[features["date"].dt.year == 2014]
    return train, encode(train["result"]), test


@pytest.mark.parametrize(
    "factory",
    [elo_only, lambda: logistic(BASE_FEATURES),
     lambda: random_forest(BASE_FEATURES, n_estimators=30),
     lambda: xgboost(BASE_FEATURES, n_estimators=50)],
)  # fmt: skip
def test_models_symmetric_and_normalised(small, factory):
    train, y, test = small
    model = factory().fit(train, y)
    p = model.predict_proba(test)
    assert p.shape == (len(test), 3)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    p_swapped = flip_proba(model.predict_proba(swap_sides(test[model.features])))
    np.testing.assert_allclose(p, p_swapped, atol=1e-6)  # XGBoost computes in float32


def test_stronger_team_more_likely_to_win(small):
    train, y, _ = small
    model = elo_only().fit(train, y)
    X = pd.DataFrame({"elo_diff": [-300.0, 0.0, 300.0], "venue": [0.0, 0.0, 0.0]})
    p = model.predict_proba(X)
    assert p[0, 0] < p[1, 0] < p[2, 0]
    assert p[1, 0] == pytest.approx(p[1, 2])  # equal teams, neutral: symmetric
    assert p[1, 1] == p[:, 1].max()  # draws likeliest for evenly matched teams


def test_baselines(small):
    train, y, test = small
    base = BaseRateModel().fit(train, y)
    p = base.predict_proba(test)
    assert p[0, 0] == pytest.approx(p[0, 2])  # mirrored -> symmetric
    assert np.allclose(p, p[0])
    hr = HigherRatedModel().fit(train, y).predict_proba(test)
    assert set(np.unique(hr)) == {0.0, 1.0} and (hr[:, 1] == 0).all()
    assert len(model_zoo(BASE_FEATURES)) == 6


def test_time_series_cv_trains_only_on_past(features, monkeypatch):
    seen = []

    class Spy(BaseRateModel):
        name = "spy"

        def fit(self, X, y):
            seen.append(X["date"].max())
            return super().fit(X, y)

    cv = time_series_cv(Spy, features, [2015, 2016], subset=lambda d: d["match_type"] != "friendly")
    assert seen[0] < pd.Timestamp("2015-01-01") and seen[1] < pd.Timestamp("2016-01-01")
    assert list(cv["year"]) == [2015, 2016]
    summary = summarise_cv(cv)
    assert summary.loc["spy", "n"] == cv["n"].sum()
    assert training_rows(features, "2000-01-01")["date"].min() >= pd.Timestamp("1994-01-01")


@pytest.mark.slow
def test_full_models_beat_base_rates_out_of_sample(features):
    cv = pd.concat(
        [time_series_cv(f, features, [2016, 2017])
         for f in (BaseRateModel, elo_only, lambda: xgboost(BASE_FEATURES))]
    )  # fmt: skip
    s = summarise_cv(cv)
    assert s.loc["Elo-only (logistic)", "log_loss"] < s.loc["Base rates", "log_loss"] - 0.1
    assert s.loc["XGBoost", "log_loss"] < s.loc["Base rates", "log_loss"] - 0.1


def test_pickling_stores_xgboost_in_portable_format(small):
    """Saved XGBoost models must not depend on the installed XGBoost version."""
    import pickle

    from wcpredict.models import _PortableXGBoost

    train, y, test = small
    model = xgboost(BASE_FEATURES, n_estimators=20).fit(train, y)
    state = model.__getstate__()
    assert isinstance(state["estimator"], _PortableXGBoost)
    assert state["estimator"].model_json.startswith(b"{")  # XGBoost JSON model format
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict_proba(test), model.predict_proba(test))
    hyper = ["objective", "n_estimators", "learning_rate", "max_depth", "min_child_weight",
             "subsample", "colsample_bytree", "reg_lambda", "random_state"]  # fmt: skip
    for key in hyper:
        assert restored.estimator.get_params()[key] == model.estimator.get_params()[key], key

    unfitted = pickle.loads(pickle.dumps(xgboost(BASE_FEATURES)))
    assert not unfitted.estimator.__sklearn_is_fitted__()
    lr = logistic(BASE_FEATURES).fit(train, y)
    assert lr.__getstate__()["estimator"] is lr.estimator  # non-XGBoost: pickled as usual

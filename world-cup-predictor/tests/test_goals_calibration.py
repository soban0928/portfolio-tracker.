import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from wcpredict.calibration import CalibratedModel
from wcpredict.evaluation import expected_calibration_error
from wcpredict.goals import (
    GoalsModel,
    design,
    outcome_probs,
    score_matrix,
    tau,
    top_scorelines,
)
from wcpredict.models import BaseRateModel, encode

TRUE_BETA = np.array([0.1, 0.15, 0.2, 0.2, 0.3, -0.05, 0.05])


def simulated_matches(n: int = 6000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "date": pd.Timestamp("2010-01-01") + pd.to_timedelta(rng.integers(0, 3000, n), "D"),
            "elo_diff": rng.normal(0, 200, n),
            "venue": rng.choice([-1.0, 0.0, 1.0], n),
            "gf10_home": rng.gamma(4, 0.35, n),
            "ga10_home": rng.gamma(4, 0.35, n),
            "gf10_away": rng.gamma(4, 0.35, n),
            "ga10_away": rng.gamma(4, 0.35, n),
            "is_friendly": rng.integers(0, 2, n).astype(float),
            "is_tournament": rng.integers(0, 2, n).astype(float),
        }
    )
    X["home_score"] = rng.poisson(np.exp(design(X, "home") @ TRUE_BETA))
    X["away_score"] = rng.poisson(np.exp(design(X, "away") @ TRUE_BETA))
    gd = X["home_score"] - X["away_score"]
    X["result"] = np.select([gd > 0, gd < 0], ["H", "A"], "D")
    return X.sort_values("date").reset_index(drop=True)


def test_score_matrix_normalised_and_poisson_when_rho_zero():
    m = score_matrix(np.array([1.3]), np.array([0.9]), rho=0.0)[0]
    assert m.sum() == pytest.approx(1.0)
    expected = np.outer(poisson.pmf(range(11), 1.3), poisson.pmf(range(11), 0.9))
    np.testing.assert_allclose(m, expected / expected.sum())


def test_tau_hand_values():
    lam, mu, rho = 1.2, 0.8, -0.1
    assert tau(np.array(0), np.array(0), lam, mu, rho) == pytest.approx(1 - lam * mu * rho)
    assert tau(np.array(1), np.array(1), lam, mu, rho) == pytest.approx(1 + 0.1)
    assert tau(np.array(2), np.array(1), lam, mu, rho) == 1
    # negative rho inflates draws 0-0 and 1-1 relative to independence
    p0 = outcome_probs(score_matrix(np.array([1.2]), np.array([1.2]), 0.0))[0, 1]
    p1 = outcome_probs(score_matrix(np.array([1.2]), np.array([1.2]), -0.1))[0, 1]
    assert p1 > p0


def test_outcome_probs_symmetric_for_equal_rates():
    p = outcome_probs(score_matrix(np.array([1.1]), np.array([1.1]), -0.05))[0]
    assert p.sum() == pytest.approx(1.0)
    assert p[0] == pytest.approx(p[2])


def test_goals_model_recovers_parameters():
    X = simulated_matches()
    model = GoalsModel(half_life_years=1e9, ridge=0.0).fit(X)
    np.testing.assert_allclose(model.beta_, TRUE_BETA, atol=0.08)
    assert abs(model.rho_) < 0.06  # data generated without DC correlation
    coefs = model.coefficients()
    assert list(coefs.index)[:2] == ["const", "elo"] and (coefs["std_err"] > 0).all()


def test_goals_model_predictions_monotone_and_symmetric():
    X = simulated_matches(2000)
    model = GoalsModel().fit(X)
    row = X.iloc[[0]].copy()
    row[["venue", "is_friendly", "is_tournament"]] = 0.0
    row[["gf10_home", "ga10_home", "gf10_away", "ga10_away"]] = 1.4
    probs = []
    for d in (-200.0, 0.0, 200.0):
        row["elo_diff"] = d
        probs.append(model.predict_proba(row)[0])
    assert probs[0][0] < probs[1][0] < probs[2][0]
    assert probs[1][0] == pytest.approx(probs[1][2])
    lam, mu = model.rates(row)
    assert lam[0] > mu[0]


def test_top_scorelines_and_sampling():
    m = score_matrix(np.array([2.0]), np.array([0.5]), 0.0)[0]
    top = top_scorelines(m, 3)
    assert top[0][:2] in [(1, 0), (2, 0)]
    assert top[0][2] >= top[1][2] >= top[2][2]
    model = GoalsModel().fit(simulated_matches(1500))
    draws = model.sample_betas(4000, np.random.default_rng(1))
    assert draws.shape == (4000, 7)
    np.testing.assert_allclose(draws.mean(axis=0), model.beta_, atol=0.01)


class Overconfident(BaseRateModel):
    """Uses the true probabilities but sharpens them (systematically overconfident)."""

    name = "overconfident"

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        p = X[["p_h", "p_d", "p_a"]].to_numpy() ** 3
        return p / p.sum(axis=1, keepdims=True)


@pytest.mark.parametrize("method", ["isotonic", "platt"])
def test_calibration_fixes_overconfidence(method):
    rng = np.random.default_rng(0)
    n = 30_000
    p_h = rng.uniform(0.15, 0.7, n)
    p_d = np.full(n, 0.25)
    p = np.column_stack([p_h, p_d, 1 - p_h - p_d])
    p = np.where(rng.random(n)[:, None] < 0.5, p, p[:, ::-1])  # symmetric population
    y = np.array([rng.choice(3, p=row) for row in p])
    X = pd.DataFrame({"p_h": p[:, 0], "p_d": p[:, 1], "p_a": p[:, 2],
                      "date": pd.date_range("2000-01-01", periods=n, freq="6h"),
                      "result": np.array(["H", "D", "A"])[y]})  # fmt: skip
    model = CalibratedModel(Overconfident, method=method, calib_years=3).fit(X, encode(X["result"]))
    before = expected_calibration_error(y, Overconfident().predict_proba(X))
    after = expected_calibration_error(y, model.predict_proba(X))
    assert after < before / 2
    assert model.name == f"overconfident + {method}"


def test_calibration_rejects_unknown_method():
    with pytest.raises(ValueError):
        CalibratedModel(BaseRateModel, method="magic")

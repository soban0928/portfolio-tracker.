from itertools import permutations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from wcpredict.config import TOURNAMENTS
from wcpredict.data import world_cup_matches
from wcpredict.elo import compute_elo, ratings_before
from wcpredict.goals import GoalsModel, score_matrix
from wcpredict.simulation import SimulationInputs, _sample_scores, knockout_win_matrix, simulate
from wcpredict.tournament import structure_from_results


@pytest.fixture(scope="module")
def structures(matches):
    e = compute_elo(matches)
    out = {}
    for year in (2018, 2026):
        elo = ratings_before(e, TOURNAMENTS[year].start).to_dict()
        out[year] = structure_from_results(world_cup_matches(matches, year), elo)
    return out


def stub_inputs(structure, strength: np.ndarray, elo_coef: float = 0.3) -> SimulationInputs:
    """Goals depend only on a chosen strength vector (in units of 100 Elo)."""
    model = GoalsModel()
    model.beta_ = np.array([np.log(1.3), elo_coef, 0, 0, 0, 0, 0])
    model.cov_ = np.eye(7) * 1e-8
    model.rho_ = -0.05
    T = len(structure.teams)
    pairs = list(permutations(range(T), 2))
    pf = pd.DataFrame(
        {
            "i": [i for i, _ in pairs],
            "j": [j for _, j in pairs],
            "elo_diff": [100 * (strength[i] - strength[j]) for i, j in pairs],
            "venue": 0.0,
            **{c: 0.5 for c in ["gf10_home", "ga10_home", "gf10_away", "ga10_away"]},
            "is_friendly": 0.0,
            "is_tournament": 0.0,
        }
    )
    return SimulationInputs(structure, model, pf, elo=1500 + 100 * strength)


def test_knockout_win_matrix_properties():
    lam = np.array([[1.0, 2.0, 1.2], [0.8, 1.0, 1.2], [1.2, 1.2, 1.0]])
    beat, pens = knockout_win_matrix(lam, -0.05)
    np.testing.assert_allclose(beat + beat.T, 1.0)
    assert beat[0, 1] > 0.5  # 2.0 v 0.8 expected goals
    assert beat[2, 1] == pytest.approx(0.5)  # equal rates
    assert 0.15 < pens[2, 1] < 0.35  # evenly matched ties often go to penalties


def test_sampled_scores_match_distribution():
    m = score_matrix(np.array([1.6]), np.array([0.7]), -0.05)
    hg, ag = _sample_scores(m.reshape(1, -1).cumsum(axis=1), np.random.default_rng(0), 200_000)
    g = np.arange(11)
    assert hg.mean() == pytest.approx((m[0].sum(axis=1) * g).sum(), abs=0.01)
    assert ag.mean() == pytest.approx((m[0].sum(axis=0) * g).sum(), abs=0.01)
    assert (hg == ag).mean() == pytest.approx(np.trace(m[0]), abs=0.005)


@pytest.mark.parametrize("year", [2018, 2026])
def test_stage_probabilities_sum_correctly(structures, year):
    s = structures[year]
    T = len(s.teams)
    r = simulate(stub_inputs(s, np.linspace(-2, 2, T)), n_sims=4000, batch_size=500, seed=1)
    p = r.stages
    expected = {"p_R16": 16, "p_QF": 8, "p_SF": 4, "p_FINAL": 2, "p_WINNER": 1}
    if year == 2026:
        expected["p_R32"] = 32
    for col, total in expected.items():
        assert p[col].sum() == pytest.approx(total)
    np.testing.assert_allclose(r.positions.sum(axis=1), 1.0)
    np.testing.assert_allclose(r.positions.sum(axis=0), len(s.groups))
    # probabilities are monotone along the stages for every team
    cols = [c for c in p.columns if c.startswith("p_")]
    assert (np.diff(p[cols].to_numpy(), axis=1) <= 1e-12).all()
    assert r.meta["n_sims"] == 4000 and r.meta["n_batches"] == 8
    assert sum(c.sum() for c in r.winner_counts[-1]) == 4000


def test_stronger_teams_win_more_often(structures):
    s = structures[2026]
    rng = np.random.default_rng(0)
    strength = rng.normal(0, 1.5, len(s.teams))
    r = simulate(stub_inputs(s, strength), n_sims=20_000, seed=3)
    p_win = r.stages["p_WINNER"].reindex(s.teams).to_numpy()
    assert spearmanr(strength, p_win).statistic > 0.9
    assert s.teams[int(np.argmax(strength))] == r.stages.index[0]


def test_equal_teams_have_equal_chances(structures):
    s = structures[2026]
    T = len(s.teams)
    r = simulate(stub_inputs(s, np.zeros(T), elo_coef=0.0), n_sims=48_000, batch_size=4000,
                 seed=5)  # fmt: skip
    p = r.stages
    # With equal teams each knockout match is 50/50, so P(win) = P(reach R32)/32 = 1/48.
    assert np.abs(p["p_WINNER"] - 1 / 48).max() < 5 * p["se_binom_WINNER"].max()
    assert p["p_R32"].mean() == pytest.approx(2 / 3)
    assert np.abs(p["p_R32"] - 2 / 3).max() < 0.03


def test_reproducible_with_seed(structures):
    s = structures[2018]
    inp = stub_inputs(s, np.linspace(-1, 1, len(s.teams)))
    a = simulate(inp, n_sims=2000, seed=11).stages
    b = simulate(inp, n_sims=2000, seed=11).stages
    c = simulate(inp, n_sims=2000, seed=12).stages
    pd.testing.assert_frame_equal(a, b)
    assert not a["p_WINNER"].equals(c["p_WINNER"])


def test_mcse_estimators_agree_without_parameter_sampling(structures):
    s = structures[2018]
    inp = stub_inputs(s, np.linspace(-1, 1, len(s.teams)))
    r = simulate(inp, n_sims=40_000, batch_size=1000, seed=2, sample_params=False)
    top = r.stages.iloc[:5]
    ratio = top["se_WINNER"] / top["se_binom_WINNER"]
    assert ((ratio > 0.6) & (ratio < 1.5)).all()
    single = simulate(inp, n_sims=500, batch_size=1000, seed=2)
    assert single.stages["se_WINNER"].isna().all()

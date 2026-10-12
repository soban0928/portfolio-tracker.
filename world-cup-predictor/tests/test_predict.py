import numpy as np
import pytest

from wcpredict.predict import COMPETITIONS, load_predictor


@pytest.fixture(scope="module")
def predictor():
    return load_predictor()


def test_predictor_loads_and_ranks_teams(predictor):
    teams = predictor.teams
    assert len(teams) > 150
    assert predictor.state(teams[0]).elo >= predictor.state(teams[-1]).elo
    assert {"Spain", "Argentina", "San Marino"} <= set(teams)


def test_prediction_properties(predictor):
    p = predictor.predict("Spain", "San Marino")
    assert p.classifier_probs.sum() == pytest.approx(1.0)
    assert p.goals_probs.sum() == pytest.approx(1.0)
    assert p.classifier_probs[0] > 0.8 and p.goals_probs[0] > 0.8
    assert p.expected_goals[0] > p.expected_goals[1]
    assert p.score_matrix.sum() == pytest.approx(1.0)
    assert p.top_scorelines[0][0] > p.top_scorelines[0][1]


def test_prediction_symmetry_and_venue(predictor):
    ab = predictor.predict("France", "England")
    ba = predictor.predict("England", "France")
    np.testing.assert_allclose(ab.classifier_probs, ba.classifier_probs[::-1], atol=1e-6)
    np.testing.assert_allclose(ab.goals_probs, ba.goals_probs[::-1], atol=1e-9)
    home = predictor.predict("France", "England", venue=1.0)
    assert home.goals_probs[0] > ab.goals_probs[0]
    with pytest.raises(ValueError):
        predictor.predict("France", "France")
    assert set(COMPETITIONS.values()) <= {"world_cup", "continental_final", "qualifier",
                                          "friendly"}  # fmt: skip


def test_retrain_fallback(monkeypatch, tmp_path):
    import shutil

    from wcpredict.config import ARTEFACT_DIR

    cur = tmp_path / "current"
    shutil.copytree(ARTEFACT_DIR / "current", cur)
    (cur / "classifier.joblib").write_bytes(b"not a pickle")
    with pytest.warns(UserWarning, match="retraining"):
        p = load_predictor(tmp_path)
    assert p.predict("Brazil", "Peru").classifier_probs.sum() == pytest.approx(1.0)

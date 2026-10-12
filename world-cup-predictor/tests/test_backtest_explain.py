import json

import numpy as np
import pandas as pd
import pytest

from wcpredict.backtest import (
    advance_prob_goals,
    exit_stage_probs,
    match_predictions,
    real_fixture_frame,
    score_match_predictions,
    tournament_scores,
)
from wcpredict.config import ARTEFACT_DIR
from wcpredict.elo import compute_elo
from wcpredict.evaluation import oof_predictions, summarise_predictions, training_rows
from wcpredict.explain import shap_summary
from wcpredict.features import BASE_FEATURES
from wcpredict.goals import GoalsModel
from wcpredict.models import BaseRateModel, elo_only, encode, logistic, xgboost
from wcpredict.pipeline import build_context


@pytest.fixture(scope="module")
def ctx2022(matches):
    return build_context(matches, compute_elo(matches), 2022)


@pytest.fixture(scope="module")
def fitted(features, ctx2022):
    train = training_rows(features, ctx2022.start)
    train = train[train["date"] >= "2010-01-01"]
    y = encode(train["result"])
    return train, [BaseRateModel().fit(train, y), elo_only().fit(train, y)], GoalsModel().fit(train)


def toy_stages():
    teams = ["A", "B", "C", "D"]
    return pd.DataFrame(
        {"p_SF": [1.0, 1.0, 1.0, 1.0], "p_FINAL": [0.8, 0.6, 0.4, 0.2],
         "p_WINNER": [0.5, 0.3, 0.15, 0.05]},
        index=teams,
    )  # fmt: skip


def test_exit_stage_probs_sum_to_one():
    exits = exit_stage_probs(toy_stages())
    np.testing.assert_allclose(exits.sum(axis=1), 1.0)
    assert exits.loc["A", "WINNER"] == 0.5 and exits.loc["A", "FINAL"] == pytest.approx(0.3)


def test_tournament_scores_ranks():
    progress = pd.Series({"A": "SF", "B": "WINNER", "C": "FINAL", "D": "SF"})
    s = tournament_scores(toy_stages(), progress)
    assert s["champion"] == "B" and s["champion_rank"] == 2
    assert s["favourite"] == "A"
    assert s["finalist_ranks"] == {"B": 2, "C": 3}
    # exit probabilities: A at SF 1-0.8, B wins 0.3, C final 0.4-0.15, D at SF 1-0.2
    expected = np.mean(np.log([0.2, 0.3, 0.25, 0.8]))
    assert s["exit_log_score"] == pytest.approx(expected)
    assert s["stage_brier"]["WINNER"]["model"] == pytest.approx(
        np.mean([0.25, 0.49, 0.0225, 0.0025])
    )


def test_real_fixture_frame_uses_frozen_states(ctx2022):
    ff = real_fixture_frame(ctx2022)
    assert len(ff) == 64
    final = ff[ff["stage"] == "FINAL"].iloc[0]
    expected = ctx2022.elo[final["home_team"]] - ctx2022.elo[final["away_team"]]
    assert final["elo_diff"] == pytest.approx(expected)  # not updated during the tournament
    qatar = ff[(ff["home_team"] == "Qatar") | (ff["away_team"] == "Qatar")]
    assert (qatar["venue"] != 0).all()


def test_match_predictions_and_scores(ctx2022, fitted):
    _, models, goals = fitted
    preds = match_predictions(models, goals, real_fixture_frame(ctx2022))
    assert set(preds["model"]) == {m.name for m in models} | {goals.name}
    ko = preds[preds["stage"] != "GROUP"]
    assert ko["p_advance"].between(0, 1).all() and ko["home_advanced"].isin([0, 1]).all()
    assert preds["p_advance"][preds["stage"] == "GROUP"].isna().all()
    scores = score_match_predictions(preds)
    assert (scores["n_group"] == 48).all() and (scores["n_ko"] == 16).all()
    assert scores.loc["Base rates", "ko_log_loss"] == pytest.approx(np.log(2))


def test_advance_prob_goals_equal_teams(fitted):
    _, _, goals = fitted
    X = pd.DataFrame({"elo_diff": [0.0], "venue": [0.0], "gf10_home": [1.4], "ga10_home": [1.4],
                      "gf10_away": [1.4], "ga10_away": [1.4], "is_friendly": [0.0],
                      "is_tournament": [1.0]})  # fmt: skip
    assert advance_prob_goals(goals, X)[0] == pytest.approx(0.5)


def test_oof_predictions(features):
    oof = oof_predictions(BaseRateModel, features, [2016, 2017])
    assert set(oof["year"]) == {2016, 2017}
    s = summarise_predictions(oof)
    assert "ece" in s.columns and s.loc["Base rates", "n"] == len(oof)


@pytest.mark.parametrize(
    "factory", [lambda: logistic(BASE_FEATURES), lambda: xgboost(BASE_FEATURES, n_estimators=40)]
)
def test_shap_summary(fitted, factory):
    train, _, _ = fitted
    model = factory().fit(train, encode(train["result"]))
    imp, sample = shap_summary(model, train, n=150)
    assert list(imp.index)[0] == "elo_diff"  # Elo difference dominates
    assert set(imp.index) == set(BASE_FEATURES)
    assert len(sample) == 150 * len(BASE_FEATURES)
    assert {"shap_H", "shap_D", "shap_A"} <= set(sample.columns)


def test_committed_artefacts_consistent():
    summary = json.loads((ARTEFACT_DIR / "summary.json").read_text())
    champions = {y: t["tournament"]["full"]["champion"] for y, t in summary["tournaments"].items()}
    assert champions == {"2018": "France", "2022": "Argentina", "2026": "Spain"}
    stages = pd.read_csv(ARTEFACT_DIR / "2026" / "stages_full.csv", index_col="team")
    assert stages["p_WINNER"].sum() == pytest.approx(1.0)
    assert stages["p_R32"].sum() == pytest.approx(32.0)
    assert summary["n_sims"] == 100_000

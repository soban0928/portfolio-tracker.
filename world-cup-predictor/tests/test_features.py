import numpy as np
import pandas as pd
import pytest

from tests.conftest import raw_frame
from wcpredict.data import clean_matches
from wcpredict.elo import compute_elo
from wcpredict.features import (
    BASE_FEATURES,
    PRIORS,
    REST_CAP,
    build_features,
    fifa_rank_table,
    fixture_features,
    state_before,
    swap_sides,
    team_states,
)


@pytest.fixture
def toy_features(toy_matches):
    return build_features(compute_elo(toy_matches))


def test_first_match_uses_priors(toy_features):
    first = toy_features.iloc[0]
    assert first["form5_diff"] == 0
    assert first["gf10_home"] == PRIORS["gf10"]
    assert first["rest_home"] == REST_CAP
    assert first["exp_home"] == 0


def test_features_hand_calc(toy_features):
    # Match 3 (2020-03-01): C v A. A previously beat B 2-0 (3 pts, 31 days earlier).
    # C previously drew 1-1 with B (1 pt, 29 days earlier).
    row = toy_features.iloc[2]
    assert (row["home_team"], row["away_team"]) == ("C", "A")
    assert row["form5_diff"] == pytest.approx(1 - 3)
    assert row["gf10_home"] == 1 and row["ga10_home"] == 1
    assert row["gf10_away"] == 2 and row["ga10_away"] == 0
    assert row["gd10_diff"] == pytest.approx(0 - 2)
    assert row["rest_home"] == 29 and row["rest_away"] == 60  # 60 days, capped at 60
    assert row["exp_home"] == pytest.approx(np.log(2))


def test_all_feature_columns_present_and_finite(toy_features):
    assert set(BASE_FEATURES) <= set(toy_features.columns)
    assert np.isfinite(toy_features[BASE_FEATURES].to_numpy()).all()


def test_no_leakage_future_score_change(toy_matches):
    """Changing a later result must not change any earlier match's features."""
    base = build_features(compute_elo(toy_matches))
    changed = toy_matches.copy()
    changed.loc[changed.index[3], ["home_score", "away_score"]] = [0, 7]
    alt = build_features(compute_elo(changed))
    pd.testing.assert_frame_equal(base.loc[:3, BASE_FEATURES], alt.loc[:3, BASE_FEATURES])
    assert not base.loc[4:, BASE_FEATURES].equals(alt.loc[4:, BASE_FEATURES])


def test_no_leakage_on_real_data(matches):
    """Features for matches before a cut-off are identical when later data is removed."""
    cutoff = pd.Timestamp("2022-11-20")
    full = build_features(compute_elo(matches))
    truncated = build_features(compute_elo(matches[matches["date"] < cutoff]))
    before = full[full["date"] < cutoff]
    pd.testing.assert_frame_equal(
        before[BASE_FEATURES].reset_index(drop=True),
        truncated[BASE_FEATURES].reset_index(drop=True),
    )


def test_state_before_matches_training_features(matches):
    """Inference-time state equals the training-time features for a real fixture."""
    feats = build_features(compute_elo(matches))
    states = team_states(compute_elo(matches))
    row = feats[(feats["tournament"] == "FIFA World Cup") & (feats["date"] == "2022-12-18")].iloc[0]
    h = state_before(states, row["home_team"], row["date"])
    a = state_before(states, row["away_team"], row["date"])
    fx = fixture_features(h, a, row["date"], venue=row["venue"], competition=row["competition"])
    for col in BASE_FEATURES:
        assert fx[col] == pytest.approx(row[col]), col


def test_state_before_unknown_team(toy_matches):
    states = team_states(compute_elo(toy_matches))
    s = state_before(states, "Nowhere", "2021-01-01")
    assert s.elo == 1500 and s.n_played == 0 and s.last_match is None
    fx = fixture_features(s, s, "2021-01-01", rest_home=3, rest_away=100)
    assert fx["rest_home"] == 3 and fx["rest_away"] == REST_CAP


def test_swap_sides_is_involution(toy_features):
    X = toy_features[BASE_FEATURES]
    pd.testing.assert_frame_equal(swap_sides(swap_sides(X)), X)
    sw = swap_sides(X)
    assert (sw["elo_diff"] == -X["elo_diff"]).all()
    assert (sw["gf10_home"] == X["gf10_away"]).all()


def test_fifa_rank_attached_before_match_only():
    raw = raw_frame(
        [
            ("2020-01-10", "A", "B", 1, 0, "Friendly", "A", False),
            ("2020-02-10", "A", "B", 1, 0, "Friendly", "A", False),
        ]
    )
    fifa = pd.DataFrame(
        {
            "team": ["A", "B", "A", "B"],
            "total_points": [1000, 1500, 1600, 900],
            "date": ["2020-01-01", "2020-01-01", "2020-02-10", "2020-02-10"],
        }
    )
    f = build_features(compute_elo(clean_matches(raw)), fifa)
    # both matches use the 2020-01-01 ranking (A 2nd, B 1st); same-day ranking excluded
    assert list(f["rank_home"]) == [2, 2]
    assert f["log_rank_ratio"].iloc[0] == pytest.approx(np.log(1) - np.log(2))


def test_real_fifa_ranking_names_map(matches):
    from wcpredict.config import RAW_DIR

    ranks = fifa_rank_table(pd.read_csv(RAW_DIR / "fifa_ranking.csv"))
    wc_teams = set(matches.loc[(matches["tournament"] == "FIFA World Cup")
                               & (matches["date"].dt.year == 2022), "home_team"])  # fmt: skip
    assert wc_teams <= set(ranks["team"])

import numpy as np
import pandas as pd
import pytest

from tests.conftest import raw_frame
from wcpredict.data import clean_matches
from wcpredict.elo import (
    EloParams,
    compute_elo,
    expected_score,
    goal_diff_multiplier,
    ratings_before,
    venue_indicator,
)


def one_match(hs, as_, tournament="Friendly", neutral=True, country="Z"):
    raw = raw_frame([("2020-01-01", "A", "B", hs, as_, tournament, country, neutral)])
    return compute_elo(clean_matches(raw)).iloc[0]


def test_expected_score_properties():
    assert expected_score(1500, 1500) == pytest.approx(0.5)
    assert expected_score(1600, 1500) + expected_score(1500, 1600) == pytest.approx(1.0)
    # 400 points -> 10:1 odds
    assert expected_score(1900, 1500) == pytest.approx(10 / 11)
    assert expected_score(1500, 1500, venue=1, home_adv=100) == pytest.approx(
        1 / (1 + 10 ** (-0.25))
    )


@pytest.mark.parametrize(("gd", "g"), [(0, 1.0), (1, 1.0), (-1, 1.0), (2, 1.5), (3, 1.75),
                                       (-4, 1.875)])  # fmt: skip
def test_goal_diff_multiplier(gd, g):
    assert goal_diff_multiplier(gd) == g


def test_venue_indicator():
    assert venue_indicator(True, False) == 1
    assert venue_indicator(False, True) == -1
    assert venue_indicator(False, False) == 0


@pytest.mark.parametrize(
    ("hs", "as_", "delta"),
    [
        (1, 0, 10.0),  # friendly K=20, E=0.5, W=1 -> 20 * 0.5
        (2, 0, 15.0),  # x1.5
        (3, 0, 17.5),  # x(11+3)/8
        (0, 0, 0.0),
        (0, 1, -10.0),
    ],
)
def test_single_update_hand_calc(hs, as_, delta):
    row = one_match(hs, as_)
    assert row["elo_home"] == 1500 and row["elo_away"] == 1500
    assert row["elo_delta"] == pytest.approx(delta)


def test_importance_and_home_advantage():
    wc = one_match(1, 0, tournament="FIFA World Cup")
    assert wc["elo_delta"] == pytest.approx(30.0)  # K=60
    home_draw = one_match(0, 0, neutral=False, country="A")
    e = 1 / (1 + 10 ** (-100 / 400))
    assert home_draw["venue"] == 1
    assert home_draw["elo_delta"] == pytest.approx(20 * (0.5 - e))  # -2.80


def test_ablation_flags():
    raw = raw_frame([("2020-01-01", "A", "B", 3, 0, "FIFA World Cup", "Z", True)])
    m = clean_matches(raw)
    flat = compute_elo(m, EloParams(use_goal_diff=False, use_importance=False)).iloc[0]
    assert flat["elo_delta"] == pytest.approx(40 * 0.5)


def test_pre_match_ratings_chain_and_zero_sum(toy_matches):
    e = compute_elo(toy_matches)
    # each team's pre-match rating equals its post-match rating from its last game
    post: dict[str, float] = {}
    for _, r in e.iterrows():
        assert r["elo_home"] == pytest.approx(post.get(r["home_team"], 1500))
        assert r["elo_away"] == pytest.approx(post.get(r["away_team"], 1500))
        post[r["home_team"]] = r["elo_home"] + r["elo_delta"]
        post[r["away_team"]] = r["elo_away"] - r["elo_delta"]
    assert sum(post.values()) == pytest.approx(1500 * len(post))
    final = ratings_before(e, "2030-01-01")
    for team, rating in post.items():
        assert final[team] == pytest.approx(rating)


def test_ratings_before_ignores_same_day_and_later(toy_matches):
    e = compute_elo(toy_matches)
    before = ratings_before(e, "2020-04-01")  # excludes the 2020-04-01 match
    row = e[e["date"] == pd.Timestamp("2020-04-01")].iloc[0]
    assert before[row["home_team"]] == pytest.approx(row["elo_home"])
    assert before[row["away_team"]] == pytest.approx(row["elo_away"])


def test_future_results_do_not_change_past_ratings(toy_matches):
    base = compute_elo(toy_matches)
    changed = toy_matches.copy()
    changed.loc[changed.index[-1], ["home_score", "away_score"]] = [9, 0]
    alt = compute_elo(changed)
    cols = ["elo_home", "elo_away"]
    np.testing.assert_allclose(base[cols].to_numpy(), alt[cols].to_numpy())
    assert base["elo_delta"].iloc[-1] != alt["elo_delta"].iloc[-1]


def test_real_data_sanity(matches):
    e = compute_elo(matches)
    top = ratings_before(e, "2022-11-20")
    assert {"Brazil", "Argentina"} <= set(top.index[:6])
    assert top["San Marino"] < top.median()
    # higher-rated side wins more often than it loses in competitive games since 1994
    recent = e[(e["date"] >= "1994-01-01") & (e["match_type"] != "friendly")]
    fav_home = recent["elo_exp_home"] > 0.5
    fav_wins = np.where(fav_home, recent["result"] == "H", recent["result"] == "A")
    fav_loses = np.where(fav_home, recent["result"] == "A", recent["result"] == "H")
    assert fav_wins.mean() > 2 * fav_loses.mean()

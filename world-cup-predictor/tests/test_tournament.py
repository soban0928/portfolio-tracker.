from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from wcpredict.config import TOURNAMENTS
from wcpredict.data import infer_groups, world_cup_matches
from wcpredict.elo import compute_elo, ratings_before
from wcpredict.tournament import (
    GROUP_PAIRS,
    actual_progress,
    assign_thirds,
    group_goal_arrays,
    rank_groups,
    rank_thirds,
    real_standings,
    structure_from_results,
)

EQUAL_ELO = np.zeros(4)


def order(hg, ag, rules, elo=EQUAL_ELO):
    return list(rank_groups(np.array(hg), np.array(ag), elo, rules).order)


def test_group_pairs_cover_all_fixtures():
    assert GROUP_PAIRS == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def test_simple_points_order_and_stats():
    # A beats everyone, B beats C and D, C beats D
    t = rank_groups(np.array([1, 1, 1, 1, 1, 1]), np.zeros(6), EQUAL_ELO)
    assert list(t.order) == [0, 1, 2, 3]
    assert list(t.points) == [9, 6, 3, 0]
    assert list(t.goal_diff) == [3, 1, -1, -3]
    assert t.stat_of_position(t.points, 1) == 9


def test_head_to_head_before_goal_difference_2026_only():
    # A-B 1-0, A-C 0-3, A-D 1-0, B-C 3-0, B-D 3-0, C-D 0-0
    # A and B on 6 pts; A won the head-to-head, B has the better GD (+5 v -1).
    hg, ag = [1, 0, 1, 3, 3, 0], [0, 3, 0, 0, 0, 0]
    assert order(hg, ag, "2026") == [0, 1, 2, 3]
    assert order(hg, ag, "2018") == [1, 0, 2, 3]


def test_head_to_head_reapplied_to_remaining_subset():
    # A-B 2-1, A-C 0-1, A-D 1-0, B-C 1-0, B-D 3-0, C-D 1-0: A, B, C all on 6 pts.
    # Mini-league: all 3 pts, GD 0; goals A 2, B 2, C 1 -> C third of the three.
    # Re-applied to {A, B}: A beat B -> A first, although B's overall GD (+3) beats A's (+1).
    hg, ag = [2, 0, 1, 1, 3, 1], [1, 1, 0, 0, 0, 0]
    assert order(hg, ag, "2026") == [0, 1, 2, 3]
    # 2018 rules: overall GD first -> B (+3), then A and C (+1) split on goals (3 v 2).
    assert order(hg, ag, "2018") == [1, 0, 2, 3]


def test_elo_is_final_tiebreaker():
    zeros = np.zeros(6)
    assert order(zeros, zeros, "2026", elo=np.array([1500, 1900, 1700, 1600])) == [1, 2, 3, 0]


def test_vectorised_batch_matches_single():
    rng = np.random.default_rng(0)
    hg, ag = rng.poisson(1.3, (500, 6)), rng.poisson(1.1, (500, 6))
    elo = rng.normal(1700, 100, (500, 4))
    batch = rank_groups(hg, ag, elo).order
    for k in range(0, 500, 50):
        assert list(batch[k]) == list(rank_groups(hg[k], ag[k], elo[k]).order)
    assert sorted(batch[0]) == [0, 1, 2, 3]


def test_invalid_rules():
    with pytest.raises(ValueError):
        rank_groups(np.zeros(6), np.zeros(6), EQUAL_ELO, rules="1930")


def test_rank_thirds():
    pts = np.array([4, 4, 3, 4])
    gd = np.array([0, 1, 5, 1])
    gf = np.array([2, 3, 6, 4])
    elo = np.array([1, 2, 3, 4])
    assert list(rank_thirds(pts, gd, gf, elo)) == [3, 1, 0, 2]


def test_assign_thirds_properties():
    winners = [4, 8, 0, 11, 6, 2, 1, 10]
    real = [2, 5, 4, 10, 8, 1, 9, 11]
    assert assign_thirds(tuple(sorted(real)), winners, real) == real
    for combo in combinations(range(12), 8):
        a = assign_thirds(combo, winners, real)
        assert sorted(a) == list(combo)
        assert all(g != w for g, w in zip(a, winners, strict=True))
    with pytest.raises(ValueError):
        assign_thirds((0, 1), winners, real)


@pytest.fixture(scope="module")
def elo_matches(matches):
    return compute_elo(matches)


@pytest.mark.parametrize("year", [2018, 2022, 2026])
def test_real_structures(matches, elo_matches, year):
    wc = world_cup_matches(matches, year)
    elo = ratings_before(elo_matches, TOURNAMENTS[year].start).to_dict()
    s = structure_from_results(wc, elo)  # raises if qualifiers don't match reality
    n_first = 16 if year == 2026 else 8
    assert len(s.first_round) == n_first
    assert [len(r) for r in s.later_rounds] == [n_first // 2 ** k for k in range(1, 5) if
                                               n_first // 2 ** k >= 1]  # fmt: skip
    assert len(s.teams) == len(set(s.teams)) == TOURNAMENTS[year].n_teams
    progress = actual_progress(wc)
    assert (progress == "WINNER").sum() == 1
    assert (progress == "GROUP").sum() == 16
    if year == 2026:
        assert s.n_best_thirds == 8 and len(s.third_lookup) == 495
        assert sum(b[0] == "third" for _, b in s.first_round) == 8
    else:
        assert all(a[2] == 1 and b[2] == 2 for a, b in s.first_round)


def test_2018_group_h_needs_fair_play(matches, elo_matches):
    """Japan and Senegal were level on everything but fair-play points in 2018."""
    wc = world_cup_matches(matches, 2018)
    elo = ratings_before(elo_matches, TOURNAMENTS[2018].start).to_dict()
    group = next(g for g in infer_groups(wc[wc["stage"] == "GROUP"]) if "Japan" in g)
    hg, ag = group_goal_arrays(wc[wc["stage"] == "GROUP"], group)
    t = rank_groups(hg, ag, np.array([elo[x] for x in group]), "2018")
    japan, senegal = group.index("Japan"), group.index("Senegal")
    assert t.points[japan] == t.points[senegal] and t.goal_diff[japan] == t.goal_diff[senegal]
    standings = real_standings(wc, elo, "2018").set_index("team")
    assert standings.loc["Japan", "position"] == 2  # observed outcome respected
    assert standings.loc["Senegal", "position"] == 3


def test_group_goal_arrays_orientation():
    gm = pd.DataFrame(
        {"home_team": ["B", "A", "A", "C", "D", "C"], "away_team": ["A", "C", "D", "B", "B", "D"],
         "home_score": [2, 0, 1, 1, 0, 3], "away_score": [1, 0, 1, 1, 2, 0]}
    )  # fmt: skip
    hg, ag = group_goal_arrays(gm, ["A", "B", "C", "D"])
    assert (hg[0], ag[0]) == (1, 2)  # A v B listed as B 2-1 A
    with pytest.raises(ValueError):
        group_goal_arrays(gm.iloc[1:], ["A", "B", "C", "D"])

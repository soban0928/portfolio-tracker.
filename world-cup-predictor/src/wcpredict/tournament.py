"""World Cup formats: group tables, tiebreakers, best third-placed teams, bracket.

Group ranking is vectorised: it ranks any number of 4-team groups at once
(``(..., 6)`` arrays of goals for the six fixtures), so the same code ranks a
real group and 100,000 simulated tournaments.

Tiebreak rules
--------------
``"2026"`` (head-to-head first, as in the 2026 regulations):
    points; then among tied teams: head-to-head points, goal difference, goals;
    head-to-head criteria re-applied to any subset still tied; then overall goal
    difference, overall goals; then pre-tournament Elo.
``"2018"`` (also used for 2022):
    points, overall goal difference, overall goals; then among teams still tied:
    head-to-head points, goal difference, goals; then pre-tournament Elo.

Fair-play points and FIFA ranking/drawing of lots cannot be reproduced from
the data; pre-tournament Elo is used as the final, deterministic tiebreaker.

Bracket
-------
FIFA's bracket is **derived from the real fixtures**: the computed real group
tables tell us which group position each knockout participant occupied, so the
first knockout round becomes slots like ("pos", group, 1) vs ("pos", group, 2)
and later rounds become "winner of match i vs winner of match j". For 2026,
eight first-round slots face a third-placed team; FIFA assigns them through a
495-row table that is not in the dataset. :func:`assign_thirds` uses the real
assignment whenever a qualifying third's group matches it and otherwise fills
the remaining slots deterministically, never pairing a team with its own
group winner. This approximates FIFA's table and is exact for the real
tournament.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

from wcpredict.data import infer_groups, knockout_winner

#: The six fixtures of a 4-team group as (team_i, team_j) indices within the group.
GROUP_PAIRS: list[tuple[int, int]] = list(combinations(range(4), 2))
STAGE_ORDER = ["GROUP", "R32", "R16", "QF", "SF", "FINAL", "WINNER"]

Slot = tuple  # ("pos", group, position 1-3) or ("third", k)


def _pair_matrices(hg: np.ndarray, ag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Points and goals matrices (..., 4, 4): entry [i, j] is team i vs team j."""
    shape = hg.shape[:-1] + (4, 4)
    goals = np.zeros(shape)
    pts = np.zeros(shape)
    for p, (i, j) in enumerate(GROUP_PAIRS):
        gi, gj = hg[..., p], ag[..., p]
        goals[..., i, j], goals[..., j, i] = gi, gj
        pts[..., i, j] = np.where(gi > gj, 3, np.where(gi == gj, 1, 0))
        pts[..., j, i] = np.where(gj > gi, 3, np.where(gi == gj, 1, 0))
    return pts, goals


def _h2h(mask: np.ndarray, pts: np.ndarray, goals: np.ndarray):
    """Head-to-head points, goal difference and goals within ``mask`` (..., 4, 4)."""
    p = (mask * pts).sum(-1)
    gf = (mask * goals).sum(-1)
    ga = (mask * np.swapaxes(goals, -1, -2)).sum(-1)
    return p, gf - ga, gf


def _same(*keys: np.ndarray) -> np.ndarray:
    """Mask [i, j] true where teams i != j share every key value."""
    eq = np.ones(keys[0].shape + (4,), dtype=bool)
    for k in keys:
        eq &= k[..., :, None] == k[..., None, :]
    return eq & ~np.eye(4, dtype=bool)


@dataclass
class GroupTables:
    """Result of ranking groups.

    Attributes:
        order: (..., 4) team indices within the group, 1st to 4th.
        points, goal_diff, goals_for: (..., 4) stats per team (group index order).
    """

    order: np.ndarray
    points: np.ndarray
    goal_diff: np.ndarray
    goals_for: np.ndarray

    def stat_of_position(self, stat: np.ndarray, position: int) -> np.ndarray:
        """Value of ``stat`` for the team finishing in ``position`` (1-based)."""
        idx = self.order[..., position - 1 : position]
        return np.take_along_axis(stat, idx, axis=-1)[..., 0]


def rank_groups(
    hg: np.ndarray, ag: np.ndarray, elo: np.ndarray, rules: str = "2026"
) -> GroupTables:
    """Rank 4-team groups.

    Args:
        hg: (..., 6) goals of the first team in each ``GROUP_PAIRS`` fixture.
        ag: (..., 6) goals of the second team.
        elo: (..., 4) pre-tournament Elo (final tiebreaker).
        rules: ``"2026"`` or ``"2018"`` (see module docstring).

    Returns:
        :class:`GroupTables`.
    """
    hg, ag = np.asarray(hg, dtype=float), np.asarray(ag, dtype=float)
    elo = np.broadcast_to(np.asarray(elo, dtype=float), hg.shape[:-1] + (4,))
    pts_m, goals_m = _pair_matrices(hg, ag)
    pts = pts_m.sum(-1)
    gf = goals_m.sum(-1)
    gd = gf - goals_m.sum(-2)
    if rules == "2026":
        h1 = _h2h(_same(pts), pts_m, goals_m)
        h2 = _h2h(_same(pts, *h1), pts_m, goals_m)
        keys = [pts, *h1, *h2, gd, gf, elo]
    elif rules == "2018":
        h = _h2h(_same(pts, gd, gf), pts_m, goals_m)
        keys = [pts, gd, gf, *h, elo]
    else:
        raise ValueError("rules must be '2026' or '2018'")
    # np.lexsort: last key is primary; negate for descending order.
    order = np.lexsort([-k for k in reversed(keys)], axis=-1)
    return GroupTables(order, pts, gd, gf)


def rank_thirds(
    points: np.ndarray, goal_diff: np.ndarray, goals_for: np.ndarray, elo: np.ndarray
) -> np.ndarray:
    """Order third-placed teams across groups (best first) along the last axis.

    Criteria: points, goal difference, goals scored, then pre-tournament Elo.
    """
    return np.lexsort([-elo, -goals_for, -goal_diff, -points], axis=-1)


def assign_thirds(
    qualified: tuple[int, ...], slot_winner_groups: list[int], real_third_groups: list[int]
) -> list[int]:
    """Assign qualifying third-placed groups to the winner-vs-third slots.

    Depth-first search over slots in order. Each slot tries its real opponent's
    group first (if it qualified), then the other groups in ascending order,
    never its own winner's group. The real assignment is therefore reproduced
    exactly when the real set of thirds qualifies, and a valid assignment is
    always found (each slot excludes only one group).

    Args:
        qualified: Groups whose third-placed team advanced (len == number of slots).
        slot_winner_groups: Group of the winner in each slot.
        real_third_groups: Group of the third the slot faced in reality.

    Returns:
        Third-placed group assigned to each slot.
    """
    n = len(slot_winner_groups)
    assignment: list[int] = []

    def candidates(s: int, pool: set[int]) -> list[int]:
        ordered = sorted(pool)
        if real_third_groups[s] in pool:
            ordered.remove(real_third_groups[s])
            ordered.insert(0, real_third_groups[s])
        return [g for g in ordered if g != slot_winner_groups[s]]

    def solve(s: int, pool: set[int]) -> bool:
        if s == n:
            return True
        for g in candidates(s, pool):
            assignment.append(g)
            if solve(s + 1, pool - {g}):
                return True
            assignment.pop()
        return False

    if len(qualified) != n or not solve(0, set(qualified)):
        raise ValueError(f"No valid third-place assignment for {qualified}")
    return [int(g) for g in assignment]


@dataclass
class Structure:
    """Everything needed to simulate one World Cup.

    Attributes:
        year: Tournament year.
        teams: Team names; a team's id is its index.
        groups: (G, 4) team ids per group (groups numbered by first kick-off).
        rules: Tiebreak rule set.
        first_round: Slots for each first knockout round match.
        later_rounds: For each later round, pairs of indices into the previous round.
        round_names: Names of the knockout rounds, first to final.
        n_best_thirds: How many third-placed teams advance (0 or 8).
        third_slot_winner_groups: Winner's group for each winner-vs-third slot.
        third_slot_real_groups: Third's group in the real tournament for each slot.
        third_lookup: Maps a sorted tuple of qualifying groups to slot assignments.
    """

    year: int
    teams: list[str]
    groups: np.ndarray
    rules: str
    first_round: list[tuple[Slot, Slot]]
    later_rounds: list[list[tuple[int, int]]]
    round_names: list[str]
    n_best_thirds: int = 0
    third_slot_winner_groups: list[int] = field(default_factory=list)
    third_slot_real_groups: list[int] = field(default_factory=list)
    third_lookup: dict[tuple[int, ...], list[int]] = field(default_factory=dict)

    @property
    def team_index(self) -> dict[str, int]:
        """Team name -> id."""
        return {t: i for i, t in enumerate(self.teams)}

    def build_third_lookup(self) -> None:
        """Precompute slot assignments for every possible set of qualifying thirds."""
        n_groups = len(self.groups)
        self.third_lookup = {
            combo: assign_thirds(combo, self.third_slot_winner_groups, self.third_slot_real_groups)
            for combo in combinations(range(n_groups), self.n_best_thirds)
        }


def group_goal_arrays(
    group_matches: pd.DataFrame, group: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Real goals for one group in ``GROUP_PAIRS`` order."""
    hg, ag = np.zeros(6), np.zeros(6)
    for p, (i, j) in enumerate(GROUP_PAIRS):
        a, b = group[i], group[j]
        row = group_matches[
            ((group_matches["home_team"] == a) & (group_matches["away_team"] == b))
            | ((group_matches["home_team"] == b) & (group_matches["away_team"] == a))
        ]
        if len(row) != 1:
            raise ValueError(f"Expected one match between {a} and {b}, found {len(row)}")
        r = row.iloc[0]
        flipped = r["home_team"] == b
        hg[p] = r["away_score"] if flipped else r["home_score"]
        ag[p] = r["home_score"] if flipped else r["away_score"]
    return hg, ag


def real_standings(wc: pd.DataFrame, elo: dict[str, float], rules: str) -> pd.DataFrame:
    """Final group tables of a real tournament (team, group, position, pts, gd, gf).

    Ties the data cannot separate (e.g. 2018 Group H, Japan v Senegal, decided
    on fair-play points) are broken by who actually reached the knockout stage,
    then by Elo. Simulations never use this observed tiebreak.
    """
    group_matches = wc[wc["stage"] == "GROUP"]
    advanced = set(wc.loc[wc["stage"] != "GROUP", ["home_team", "away_team"]].to_numpy().ravel())
    rows = []
    for g, group in enumerate(infer_groups(group_matches)):
        hg, ag = group_goal_arrays(group_matches, group)
        final_key = np.array([1e6 * (x in advanced) + elo.get(x, 1500.0) for x in group])
        t = rank_groups(hg, ag, final_key, rules)
        for pos, idx in enumerate(t.order, start=1):
            rows.append(
                {"team": group[idx], "group": g, "position": pos, "points": t.points[idx],
                 "goal_diff": t.goal_diff[idx], "goals_for": t.goals_for[idx]}
            )  # fmt: skip
    return pd.DataFrame(rows)


def structure_from_results(wc: pd.DataFrame, elo: dict[str, float]) -> Structure:
    """Build the tournament structure (groups + bracket) from a real World Cup.

    Args:
        wc: Output of :func:`wcpredict.data.world_cup_matches`.
        elo: Pre-tournament Elo per team (tiebreaker).

    Raises:
        ValueError: If the computed group tables are inconsistent with who
            actually reached the knockout stage.
    """
    year = int(wc["date"].dt.year.iloc[0])
    rules = "2026" if year >= 2026 else "2018"
    groups = infer_groups(wc[wc["stage"] == "GROUP"])
    teams = [t for g in groups for t in g]
    index = {t: i for i, t in enumerate(teams)}
    table = real_standings(wc, elo, rules).set_index("team")

    ko = wc[wc["stage"] != "GROUP"].copy()
    ko["winner"] = ko.apply(knockout_winner, axis=1)
    round_names = [s for s in ["R32", "R16", "QF", "SF", "FINAL"] if s in set(ko["stage"])]
    first = ko[ko["stage"] == round_names[0]]

    def slot(team: str) -> tuple[str, int, int]:
        r = table.loc[team]
        return ("pos", int(r["group"]), int(r["position"]))

    first_round: list[tuple[Slot, Slot]] = []
    third_winner_groups, third_real_groups = [], []
    for _, m in first.iterrows():
        a, b = sorted([slot(m["home_team"]), slot(m["away_team"])], key=lambda x: x[2])
        if b[2] == 3:
            if a[2] != 1:
                raise ValueError(f"Third-placed team not facing a group winner: {a} v {b}")
            third_winner_groups.append(a[1])
            third_real_groups.append(b[1])
            b = ("third", len(third_winner_groups) - 1)
        first_round.append((a, b))

    later_rounds = []
    prev = first.reset_index(drop=True)
    for name in round_names[1:]:
        cur = ko[ko["stage"] == name].reset_index(drop=True)
        pairs = []
        for _, m in cur.iterrows():
            src = [
                int(prev.index[prev["winner"] == t][0]) for t in (m["home_team"], m["away_team"])
            ]
            pairs.append((src[0], src[1]))
        later_rounds.append(pairs)
        prev = cur

    n_thirds = len(third_winner_groups)
    s = Structure(
        year=year,
        teams=teams,
        groups=np.array([[index[t] for t in g] for g in groups]),
        rules=rules,
        first_round=first_round,
        later_rounds=later_rounds,
        round_names=round_names,
        n_best_thirds=n_thirds,
        third_slot_winner_groups=third_winner_groups,
        third_slot_real_groups=third_real_groups,
    )
    if n_thirds:
        s.build_third_lookup()
    _check_qualifiers(s, table, set(first[["home_team", "away_team"]].to_numpy().ravel()), elo)
    return s


def _check_qualifiers(s: Structure, table: pd.DataFrame, actual: set[str], elo: dict[str, float]):
    """Computed qualifiers must equal the teams that actually played the first KO round."""
    top_two = set(table.index[table["position"] <= 2])
    thirds = table[table["position"] == 3]
    best = set()
    if s.n_best_thirds:
        order = rank_thirds(
            thirds["points"].to_numpy(), thirds["goal_diff"].to_numpy(),
            thirds["goals_for"].to_numpy(), np.array([elo.get(t, 1500.0) for t in thirds.index]),
        )  # fmt: skip
        best = set(thirds.index[order[: s.n_best_thirds]])
    if top_two | best != actual:
        raise ValueError(
            f"Computed qualifiers differ from reality: missing {actual - top_two - best}, "
            f"extra {(top_two | best) - actual}"
        )


def actual_progress(wc: pd.DataFrame) -> pd.Series:
    """Furthest stage each team reached in a real tournament (WINNER for the champion)."""
    teams = pd.unique(wc[["home_team", "away_team"]].to_numpy().ravel())
    reached = dict.fromkeys(teams, "GROUP")
    rank = {s: i for i, s in enumerate(STAGE_ORDER)}
    for _, m in wc[~wc["stage"].isin(["GROUP", "THIRD"])].iterrows():
        for t in (m["home_team"], m["away_team"]):
            if rank[m["stage"]] > rank[reached[t]]:
                reached[t] = m["stage"]
    final = wc[wc["stage"] == "FINAL"].iloc[0]
    reached[knockout_winner(final)] = "WINNER"
    return pd.Series(reached, name="stage")

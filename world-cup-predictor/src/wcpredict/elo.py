"""Custom Elo ratings for national teams.

Update rule (per match, from the home side's perspective)::

    E      = 1 / (1 + 10 ** (-(R_home - R_away + venue * H) / 400))
    delta  = K_importance * G(goal_diff) * (W - E)
    R_home += delta ; R_away -= delta

* ``venue`` is +1 if the home side is the host, -1 if the away side is,
  0 on a truly neutral ground; ``H`` is the home advantage in Elo points.
* ``K_importance`` depends on the competition (World Cup 60 ... friendly 20).
* ``G`` scales by margin: 1 for a draw or one-goal game, 1.5 for two goals,
  (11 + |gd|) / 8 for three or more.
* ``W`` is 1 / 0.5 / 0 for a win / draw / loss.

Ratings stored on each match are the **pre-match** values, so they only
reflect earlier results.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EloParams:
    """Elo hyper-parameters.

    Attributes:
        initial: Rating of a team's first appearance.
        home_advantage: Elo points added to the host side.
        k_multiplier: Global scale applied to the importance K-factors.
        use_goal_diff: Apply the goal-difference multiplier.
        use_importance: Use competition K-factors (else a flat K of 40).
    """

    initial: float = 1500.0
    home_advantage: float = 100.0
    k_multiplier: float = 1.0
    use_goal_diff: bool = True
    use_importance: bool = True


def expected_score(r_home: float, r_away: float, venue: float = 0.0, home_adv: float = 100.0):
    """Expected score of the home side (works on scalars or arrays)."""
    return 1.0 / (1.0 + 10.0 ** (-(r_home - r_away + venue * home_adv) / 400.0))


def goal_diff_multiplier(goal_diff: int) -> float:
    """Margin-of-victory multiplier G."""
    gd = abs(int(goal_diff))
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


def venue_indicator(home_is_host: bool, away_is_host: bool) -> float:
    """+1 home side hosts, -1 away side hosts, 0 neutral."""
    if home_is_host and not away_is_host:
        return 1.0
    if away_is_host and not home_is_host:
        return -1.0
    return 0.0


def compute_elo(matches: pd.DataFrame, params: EloParams | None = None) -> pd.DataFrame:
    """Run Elo through ``matches`` in chronological order.

    Args:
        matches: Cleaned matches (see :func:`wcpredict.data.clean_matches`),
            sorted by date.
        params: Elo hyper-parameters.

    Returns:
        Copy of ``matches`` with ``elo_home``, ``elo_away`` (pre-match),
        ``venue``, ``elo_exp_home`` (expected score incl. venue) and
        ``elo_delta`` (rating change applied to the home side).
    """
    p = params or EloParams()
    ratings: dict[str, float] = {}
    n = len(matches)
    pre_h, pre_a, venue_arr, exp_arr, delta_arr = (np.empty(n) for _ in range(5))
    k_col = matches["importance_k"].to_numpy() if p.use_importance else np.full(n, 40.0)
    cols = zip(
        matches["home_team"], matches["away_team"], matches["home_score"],
        matches["away_score"], matches["home_is_host"], matches["away_is_host"], k_col,
        strict=True,
    )  # fmt: skip
    for i, (h, a, hs, as_, hh, ah, k) in enumerate(cols):
        rh = ratings.get(h, p.initial)
        ra = ratings.get(a, p.initial)
        v = venue_indicator(bool(hh), bool(ah))
        e = expected_score(rh, ra, v, p.home_advantage)
        w = 1.0 if hs > as_ else (0.0 if hs < as_ else 0.5)
        g = goal_diff_multiplier(hs - as_) if p.use_goal_diff else 1.0
        d = k * p.k_multiplier * g * (w - e)
        ratings[h], ratings[a] = rh + d, ra - d
        pre_h[i], pre_a[i], venue_arr[i], exp_arr[i], delta_arr[i] = rh, ra, v, e, d
    out = matches.copy()
    out["elo_home"], out["elo_away"] = pre_h, pre_a
    out["venue"], out["elo_exp_home"], out["elo_delta"] = venue_arr, exp_arr, delta_arr
    return out


def ratings_before(elo_matches: pd.DataFrame, date: pd.Timestamp | str) -> pd.Series:
    """Every team's rating going into ``date`` (only matches strictly before it).

    Args:
        elo_matches: Output of :func:`compute_elo`.
        date: Cut-off date.

    Returns:
        Series indexed by team, sorted descending.
    """
    past = elo_matches[elo_matches["date"] < pd.Timestamp(date)]
    home = (
        past[["date", "match_id", "home_team"]]
        .assign(post=past["elo_home"] + past["elo_delta"])
        .rename(columns={"home_team": "team"})
    )
    away = (
        past[["date", "match_id", "away_team"]]
        .assign(post=past["elo_away"] - past["elo_delta"])
        .rename(columns={"away_team": "team"})
    )
    both = pd.concat([home, away]).sort_values(["date", "match_id"])
    return both.groupby("team")["post"].last().sort_values(ascending=False)

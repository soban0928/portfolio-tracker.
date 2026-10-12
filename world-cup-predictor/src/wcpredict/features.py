"""Leakage-safe match features.

Every feature for a match is built only from matches played **before** it:

* Elo ratings are pre-match values (see :mod:`wcpredict.elo`).
* Team form is computed on a per-team timeline. :func:`team_states` stores
  each team's state *after* every match; a fixture takes the state from each
  team's **previous** match. Predictions for a hypothetical fixture on date D
  use the last state strictly before D (:func:`state_before`), so training and
  inference share one definition.

Feature list (``FEATURES``), all from the home/first team's perspective:

``elo_diff``        pre-match Elo difference
``venue``           +1 host / 0 neutral / -1 away side hosts
``form5_diff``      difference in points per game over the last 5 matches
``form10_diff``     same over the last 10 matches
``gf10_*, ga10_*``  goals scored / conceded per game over the last 10 matches
``gd10_diff``       difference in average goal difference over the last 10
``elo_trend_diff``  difference in Elo change over each team's last 10 matches
``rest_*``          days since each team's previous match (capped at 60)
``exp_*``           log(1 + matches played) — data scarcity for small nations
``is_friendly``, ``is_tournament``, ``importance``  match context
``log_rank_ratio``  optional: log(FIFA rank away / rank home), latest ranking
                    published before the match (NaN when unavailable)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wcpredict.data import standardise_team

REST_CAP = 60.0
#: Neutral priors used before a team has any history.
PRIORS = {"pts5": 1.35, "pts10": 1.35, "gf10": 1.4, "ga10": 1.4, "trend10": 0.0}

BASE_FEATURES = [
    "elo_diff",
    "venue",
    "form5_diff",
    "form10_diff",
    "gf10_home",
    "ga10_home",
    "gf10_away",
    "ga10_away",
    "gd10_diff",
    "elo_trend_diff",
    "rest_home",
    "rest_away",
    "exp_home",
    "exp_away",
    "is_friendly",
    "is_tournament",
    "importance",
]
FIFA_FEATURES = ["log_rank_ratio"]


def team_states(elo_matches: pd.DataFrame) -> pd.DataFrame:
    """Per-team state after each of its matches.

    Args:
        elo_matches: Output of :func:`wcpredict.elo.compute_elo`.

    Returns:
        Long DataFrame (one row per team per match) sorted by team and time,
        with rolling form columns that **include** that match.
    """
    m = elo_matches
    common = ["date", "match_id"]
    home = pd.DataFrame(
        {
            **{c: m[c] for c in common},
            "team": m["home_team"],
            "gf": m["home_score"],
            "ga": m["away_score"],
            "elo_post": m["elo_home"] + m["elo_delta"],
        }
    )
    away = pd.DataFrame(
        {
            **{c: m[c] for c in common},
            "team": m["away_team"],
            "gf": m["away_score"],
            "ga": m["home_score"],
            "elo_post": m["elo_away"] - m["elo_delta"],
        }
    )
    t = pd.concat([home, away], ignore_index=True).sort_values(["team", "date", "match_id"])
    t["pts"] = np.select([t["gf"] > t["ga"], t["gf"] < t["ga"]], [3.0, 0.0], 1.0)
    g = t.groupby("team", sort=False)
    t["pts5"] = g["pts"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    t["pts10"] = g["pts"].transform(lambda s: s.rolling(10, min_periods=1).mean())
    t["gf10"] = g["gf"].transform(lambda s: s.rolling(10, min_periods=1).mean())
    t["ga10"] = g["ga"].transform(lambda s: s.rolling(10, min_periods=1).mean())
    t["trend10"] = g["elo_post"].transform(lambda s: s - s.shift(10).fillna(s.iloc[0]))
    t["n_played"] = g.cumcount() + 1
    return t.reset_index(drop=True)


def _previous_state(states: pd.DataFrame) -> pd.DataFrame:
    """For each (team, match), the team's state after its previous match."""
    cols = ["pts5", "pts10", "gf10", "ga10", "trend10", "n_played", "date"]
    prev = states.groupby("team", sort=False)[cols].shift(1)
    out = states[["team", "match_id"]].copy()
    for c in ["pts5", "pts10", "gf10", "ga10", "trend10"]:
        out[c] = prev[c].fillna(PRIORS[c])
    out["n_prior"] = prev["n_played"].fillna(0)
    out["rest"] = (states["date"] - prev["date"]).dt.days.clip(upper=REST_CAP).fillna(REST_CAP)
    return out


def fifa_rank_table(fifa: pd.DataFrame) -> pd.DataFrame:
    """Normalise the FIFA ranking file to (team, date, rank).

    Rank is derived from points within each publication date, so it is
    consistent across the 2018 change of points formula.
    """
    f = fifa.dropna(subset=["total_points"]).copy()
    f["team"] = f["team"].astype(str).map(standardise_team)
    f["date"] = pd.to_datetime(f["date"])
    f["rank"] = f.groupby("date")["total_points"].rank(ascending=False, method="min")
    return f[["team", "date", "rank"]].sort_values("date").reset_index(drop=True)


def _attach_rank(df: pd.DataFrame, ranks: pd.DataFrame, team_col: str, out_col: str):
    """Latest rank published strictly before each match date (merge_asof)."""
    left = df[["match_id", "date", team_col]].rename(columns={team_col: "team"})
    left = left.sort_values("date")
    merged = pd.merge_asof(
        left, ranks, on="date", by="team", direction="backward", allow_exact_matches=False
    )
    return df.merge(merged[["match_id", "rank"]].rename(columns={"rank": out_col}),
                    on="match_id", how="left")  # fmt: skip


def build_features(elo_matches: pd.DataFrame, fifa: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach all model features to every match.

    Args:
        elo_matches: Output of :func:`wcpredict.elo.compute_elo`.
        fifa: Optional raw FIFA ranking table (adds ``log_rank_ratio``).

    Returns:
        ``elo_matches`` plus the feature columns (see module docstring).
    """
    states = team_states(elo_matches)
    prev = _previous_state(states).set_index(["match_id", "team"])
    df = elo_matches.copy()
    for side in ("home", "away"):
        key = pd.MultiIndex.from_arrays([df["match_id"], df[f"{side}_team"]])
        s = prev.reindex(key)
        for c in ["pts5", "pts10", "gf10", "ga10", "trend10", "rest", "n_prior"]:
            df[f"{c}_{side}"] = s[c].to_numpy()

    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    df["form5_diff"] = df["pts5_home"] - df["pts5_away"]
    df["form10_diff"] = df["pts10_home"] - df["pts10_away"]
    df["gf10_home"], df["ga10_home"] = df.pop("gf10_home"), df.pop("ga10_home")
    df["gd10_diff"] = (df["gf10_home"] - df["ga10_home"]) - (df["gf10_away"] - df["ga10_away"])
    df["elo_trend_diff"] = df["trend10_home"] - df["trend10_away"]
    df["exp_home"] = np.log1p(df["n_prior_home"])
    df["exp_away"] = np.log1p(df["n_prior_away"])
    df["is_friendly"] = (df["match_type"] == "friendly").astype(float)
    df["is_tournament"] = (df["match_type"] == "tournament").astype(float)
    df["importance"] = df["importance_k"].astype(float)

    if fifa is not None:
        ranks = fifa_rank_table(fifa)
        df = _attach_rank(df, ranks, "home_team", "rank_home")
        df = _attach_rank(df, ranks, "away_team", "rank_away")
        df["log_rank_ratio"] = np.log(df["rank_away"]) - np.log(df["rank_home"])
    return df


# ---------------------------------------------------------------- inference


@dataclass(frozen=True)
class TeamState:
    """A team's pre-fixture state as of a date (see :func:`state_before`)."""

    team: str
    elo: float
    pts5: float
    pts10: float
    gf10: float
    ga10: float
    trend10: float
    n_played: float
    last_match: pd.Timestamp | None
    rank: float = np.nan


def state_before(
    states: pd.DataFrame,
    team: str,
    date: pd.Timestamp | str,
    initial_elo: float = 1500.0,
    ranks: pd.DataFrame | None = None,
) -> TeamState:
    """A team's state using only matches strictly before ``date``."""
    ts = pd.Timestamp(date)
    hist = states[(states["team"] == team) & (states["date"] < ts)]
    rank = np.nan
    if ranks is not None:
        r = ranks[(ranks["team"] == team) & (ranks["date"] < ts)]
        rank = float(r["rank"].iloc[-1]) if len(r) else np.nan
    if hist.empty:
        return TeamState(team, initial_elo, *(PRIORS[k] for k in PRIORS), 0.0, None, rank)
    last = hist.iloc[-1]
    return TeamState(
        team, float(last["elo_post"]), float(last["pts5"]), float(last["pts10"]),
        float(last["gf10"]), float(last["ga10"]), float(last["trend10"]),
        float(last["n_played"]), last["date"], rank,
    )  # fmt: skip


def fixture_features(
    home: TeamState,
    away: TeamState,
    date: pd.Timestamp | str,
    venue: float = 0.0,
    competition: str = "world_cup",
    rest_home: float | None = None,
    rest_away: float | None = None,
) -> dict[str, float]:
    """Feature row for a hypothetical fixture (same definitions as training).

    Args:
        home: First team's state.
        away: Second team's state.
        date: Fixture date (used for default rest days).
        venue: +1 if ``home`` hosts, -1 if ``away`` hosts, 0 neutral.
        competition: Competition level (see :data:`wcpredict.data.IMPORTANCE_K`).
        rest_home: Override days of rest (else days since last match, capped).
        rest_away: Same for the second team.
    """
    from wcpredict.data import IMPORTANCE_K, match_type

    ts = pd.Timestamp(date)

    def rest(state: TeamState, override: float | None) -> float:
        if override is not None:
            return float(min(override, REST_CAP))
        if state.last_match is None:
            return REST_CAP
        return float(min((ts - state.last_match).days, REST_CAP))

    mtype = match_type(competition)
    return {
        "elo_diff": home.elo - away.elo,
        "venue": float(venue),
        "form5_diff": home.pts5 - away.pts5,
        "form10_diff": home.pts10 - away.pts10,
        "gf10_home": home.gf10,
        "ga10_home": home.ga10,
        "gf10_away": away.gf10,
        "ga10_away": away.ga10,
        "gd10_diff": (home.gf10 - home.ga10) - (away.gf10 - away.ga10),
        "elo_trend_diff": home.trend10 - away.trend10,
        "rest_home": rest(home, rest_home),
        "rest_away": rest(away, rest_away),
        "exp_home": float(np.log1p(home.n_played)),
        "exp_away": float(np.log1p(away.n_played)),
        "is_friendly": float(mtype == "friendly"),
        "is_tournament": float(mtype == "tournament"),
        "importance": IMPORTANCE_K[competition],
        "log_rank_ratio": float(np.log(away.rank) - np.log(home.rank)),
    }


SWAP_PAIRS = [
    ("gf10_home", "gf10_away"),
    ("ga10_home", "ga10_away"),
    ("rest_home", "rest_away"),
    ("exp_home", "exp_away"),
]
ANTISYMMETRIC = ["elo_diff", "venue", "form5_diff", "form10_diff", "gd10_diff",
                 "elo_trend_diff", "log_rank_ratio"]  # fmt: skip


def swap_sides(X: pd.DataFrame) -> pd.DataFrame:
    """Mirror feature rows so the away team becomes the first team."""
    out = X.copy()
    for a, b in SWAP_PAIRS:
        if a in out and b in out:
            out[a], out[b] = X[b].to_numpy(), X[a].to_numpy()
    for c in ANTISYMMETRIC:
        if c in out:
            out[c] = -X[c]
    return out

"""Glue: build per-tournament inputs using only data from before kick-off."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np
import pandas as pd

from wcpredict.config import TOURNAMENTS
from wcpredict.data import world_cup_matches
from wcpredict.elo import ratings_before
from wcpredict.features import TeamState, fixture_features, state_before, team_states
from wcpredict.goals import GoalsModel
from wcpredict.simulation import SimulationInputs
from wcpredict.tournament import Structure, structure_from_results

#: Typical rest between World Cup matches, used for hypothetical fixtures.
TOURNAMENT_REST_DAYS = 4.0


def tournament_hosts(wc: pd.DataFrame) -> set[str]:
    """Teams that played at least one match of the tournament at home (non-neutral)."""
    home = wc.loc[wc["home_is_host"] & ~wc["neutral"], "home_team"]
    away = wc.loc[wc["away_is_host"], "away_team"]
    return set(home) | set(away)


def final_host(wc: pd.DataFrame) -> set[str]:
    """The host nation of the final (knockout home advantage applies only to it)."""
    final = wc[wc["stage"] == "FINAL"].iloc[0]
    teams = set(wc["home_team"]) | set(wc["away_team"])
    return {final["country"]} & teams


def venue_for(a: str, b: str, hosts: set[str]) -> float:
    """+1 if only ``a`` is a host, -1 if only ``b`` is, else 0."""
    return float(a in hosts) - float(b in hosts)


@dataclass
class TournamentContext:
    """Pre-tournament state for one World Cup."""

    year: int
    start: pd.Timestamp
    matches: pd.DataFrame  # the real tournament fixtures (for evaluation only)
    structure: Structure
    hosts: set[str]
    ko_hosts: set[str]
    states: dict[str, TeamState]
    elo: dict[str, float]


def build_context(matches: pd.DataFrame, elo_matches: pd.DataFrame, year: int) -> TournamentContext:
    """Freeze every team's state at the tournament's opening date.

    Only matches strictly before ``TOURNAMENTS[year].start`` influence the
    states and ratings. The real fixtures are used for the tournament
    *structure* (who is in which group, bracket shape) and for evaluation.
    """
    spec = TOURNAMENTS[year]
    wc = world_cup_matches(matches, year)
    pre = elo_matches[elo_matches["date"] < spec.start]
    elo = ratings_before(pre, spec.start).to_dict()
    structure = structure_from_results(wc, elo)
    st = team_states(pre)
    states = {t: state_before(st, t, spec.start) for t in structure.teams}
    hosts = tournament_hosts(wc)
    return TournamentContext(
        year, spec.start, wc, structure, hosts, final_host(wc) & hosts, states, elo
    )


def fixture_frame(
    ctx: TournamentContext, pairs: list[tuple[str, str]], knockout: bool = False
) -> pd.DataFrame:
    """Pre-tournament feature rows for the given (team_a, team_b) fixtures.

    Group-stage fixtures give every host nation home advantage; knockout
    fixtures only the host of the final.
    """
    hosts = ctx.ko_hosts if knockout else ctx.hosts
    rows = [
        fixture_features(
            ctx.states[a],
            ctx.states[b],
            ctx.start,
            venue=venue_for(a, b, hosts),
            competition="world_cup",
            rest_home=TOURNAMENT_REST_DAYS,
            rest_away=TOURNAMENT_REST_DAYS,
        )
        for a, b in pairs
    ]
    df = pd.DataFrame(rows)
    df["team_a"], df["team_b"] = [a for a, _ in pairs], [b for _, b in pairs]
    return df


def simulation_inputs(ctx: TournamentContext, goals_model: GoalsModel) -> SimulationInputs:
    """Pair features for every ordered pair of teams plus the structure."""
    teams = ctx.structure.teams
    index = {t: i for i, t in enumerate(teams)}
    pairs = list(permutations(teams, 2))
    frames = []
    for knockout in (False, True):
        pf = fixture_frame(ctx, pairs, knockout=knockout)
        pf["i"] = pf["team_a"].map(index)
        pf["j"] = pf["team_b"].map(index)
        frames.append(pf)
    elo = np.array([ctx.elo.get(t, 1500.0) for t in teams])
    return SimulationInputs(ctx.structure, goals_model, frames[0], elo, frames[1])

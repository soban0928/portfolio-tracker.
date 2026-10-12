"""Loading, cleaning and classifying international match results.

Source: Mart Jürisoo, *International football results from 1872 to 2026*
(https://github.com/martj42/international_results), CC0. Scores include extra
time but not penalty shoot-outs; shoot-out winners come from ``shootouts.csv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wcpredict.config import RAW_DIR

#: Competitions involving non-FIFA sides (islands, regions, diaspora teams).
NON_FIFA_COMPETITIONS = (
    "CONIFA",
    "Island Games",
    "Muratti Vase",
    "Viva World Cup",
    "FIFI Wild Cup",
    "ELF Cup",
    "Niamh Challenge Cup",
    "Inter Games",
)

#: Continental/intercontinental finals tournaments (not qualifiers).
MAJOR_FINALS = {
    "UEFA Euro",
    "Copa América",
    "African Cup of Nations",
    "AFC Asian Cup",
    "Gold Cup",
    "CONCACAF Championship",
    "Confederations Cup",
    "Oceania Nations Cup",
}

#: Elo K-factor by competition level (in the spirit of eloratings.net).
IMPORTANCE_K = {
    "world_cup": 60.0,
    "continental_final": 50.0,
    "qualifier": 40.0,
    "nations_league": 40.0,
    "minor_tournament": 30.0,
    "friendly": 20.0,
}

#: Alternative spellings (e.g. FIFA ranking files) -> names used in the results data.
TEAM_ALIASES = {
    "USA": "United States",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "China PR": "China",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo",
    "Republic of Ireland": "Republic of Ireland",
    "Kyrgyz Republic": "Kyrgyzstan",
    "St. Kitts and Nevis": "Saint Kitts and Nevis",
    "St. Lucia": "Saint Lucia",
    "St. Vincent / Grenadines": "Saint Vincent and the Grenadines",
    "St. Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "Swaziland": "Eswatini",
    "FYR Macedonia": "North Macedonia",
    "Brunei Darussalam": "Brunei",
    "Chinese Taipei": "Taiwan",
    "Curacao": "Curaçao",
    "Sao Tome e Principe": "São Tomé and Príncipe",
    "São Tomé e Príncipe": "São Tomé and Príncipe",
    "US Virgin Islands": "United States Virgin Islands",
    "Hong Kong, China": "Hong Kong",
    "Macau": "Macau",
    "Timor-Leste": "East Timor",
    "The Gambia": "Gambia",
}


def standardise_team(name: str) -> str:
    """Map an alternative spelling to the canonical team name."""
    name = name.strip().replace(" (unranked)", "")
    return TEAM_ALIASES.get(name, name)


def classify_competition(tournament: str) -> str:
    """Fine-grained competition level used for Elo weighting.

    Returns one of ``world_cup``, ``continental_final``, ``qualifier``,
    ``nations_league``, ``minor_tournament`` or ``friendly``.
    """
    if tournament in ("Friendly", "FIFA Series"):
        return "friendly"
    if "qualification" in tournament:
        return "qualifier"
    if tournament == "FIFA World Cup":
        return "world_cup"
    if tournament in MAJOR_FINALS:
        return "continental_final"
    if "Nations League" in tournament:
        return "nations_league"
    return "minor_tournament"


def match_type(competition: str) -> str:
    """Coarse match type: ``friendly``, ``competitive`` or ``tournament`` (major finals)."""
    if competition == "friendly":
        return "friendly"
    if competition in ("world_cup", "continental_final"):
        return "tournament"
    return "competitive"


def load_raw(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read ``results.csv`` and ``shootouts.csv`` from ``raw_dir``."""
    results = pd.read_csv(raw_dir / "results.csv", parse_dates=["date"])
    shootouts = pd.read_csv(raw_dir / "shootouts.csv", parse_dates=["date"])
    return results, shootouts


def clean_matches(results: pd.DataFrame, shootouts: pd.DataFrame | None = None) -> pd.DataFrame:
    """Clean raw results into the analysis table.

    Steps: standardise team names, drop non-FIFA competitions and unplayed
    fixtures, add competition level, match type, Elo K, host flags, outcome
    and shoot-out winner. Rows are sorted chronologically with a stable
    ``match_id`` (original order breaks ties within a day).

    Returns:
        DataFrame with one row per match.
    """
    df = results.copy()
    df["_order"] = np.arange(len(df))
    for col in ("home_team", "away_team", "country"):
        df[col] = df[col].astype(str).map(standardise_team)
    df = df[~df["tournament"].str.contains("|".join(NON_FIFA_COMPETITIONS))]
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].astype(bool)

    df["competition"] = df["tournament"].map(classify_competition)
    df["match_type"] = df["competition"].map(match_type)
    df["importance_k"] = df["competition"].map(IMPORTANCE_K)
    df["home_is_host"] = (~df["neutral"]) | (df["home_team"] == df["country"])
    df["away_is_host"] = df["away_team"] == df["country"]
    gd = df["home_score"] - df["away_score"]
    df["result"] = np.select([gd > 0, gd < 0], ["H", "A"], "D")

    if shootouts is not None:
        so = shootouts[["date", "home_team", "away_team", "winner"]].copy()
        for col in ("home_team", "away_team", "winner"):
            so[col] = so[col].astype(str).map(standardise_team)
        so = so.drop_duplicates(["date", "home_team", "away_team"])
        df = df.merge(so.rename(columns={"winner": "shootout_winner"}),
                      on=["date", "home_team", "away_team"], how="left")  # fmt: skip
    else:
        df["shootout_winner"] = pd.NA

    df = df.sort_values(["date", "_order"]).reset_index(drop=True)
    df["match_id"] = np.arange(len(df))
    return df.drop(columns="_order")


def load_matches(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load and clean all matches (see :func:`clean_matches`)."""
    return clean_matches(*load_raw(raw_dir))


def world_cup_matches(matches: pd.DataFrame, year: int) -> pd.DataFrame:
    """Matches of the FIFA World Cup finals in ``year``, with a ``stage`` column.

    Stages are inferred from the fixture count and chronology: the group stage
    is every match before the first knockout date; knockout rounds follow in
    order (round of 32 for 48-team tournaments), then the third-place match
    and the final.

    Raises:
        ValueError: If the tournament is missing or has an unexpected shape.
    """
    wc = matches[(matches["tournament"] == "FIFA World Cup") & (matches["date"].dt.year == year)]
    wc = wc.sort_values(["date", "match_id"]).reset_index(drop=True)
    n_teams = len(set(wc["home_team"]) | set(wc["away_team"]))
    layout = {32: (48, ["R16"] * 8), 48: (72, ["R32"] * 16 + ["R16"] * 8)}
    if n_teams not in layout:
        raise ValueError(f"{year}: unexpected number of teams {n_teams}")
    n_group, early = layout[n_teams]
    stages = ["GROUP"] * n_group + early + ["QF"] * 4 + ["SF"] * 2 + ["THIRD", "FINAL"]
    if len(wc) != len(stages):
        raise ValueError(f"{year}: expected {len(stages)} matches, found {len(wc)}")
    wc["stage"] = stages
    # The third-place match is played the day before the final.
    if not wc.iloc[-2]["date"] < wc.iloc[-1]["date"]:
        raise ValueError(f"{year}: could not separate third-place match from final")
    return wc


def infer_groups(group_matches: pd.DataFrame) -> list[list[str]]:
    """Recover the groups as connected components of the group-stage fixtures.

    Groups are numbered by the order of their first match, which need not match
    FIFA's official letters. Teams within a group are sorted alphabetically.

    Raises:
        ValueError: If any component is not a group of four.
    """
    adjacency: dict[str, set[str]] = {}
    for h, a in zip(group_matches["home_team"], group_matches["away_team"], strict=True):
        adjacency.setdefault(h, set()).add(a)
        adjacency.setdefault(a, set()).add(h)
    groups: list[list[str]] = []
    seen: set[str] = set()
    ordered = pd.unique(group_matches[["home_team", "away_team"]].to_numpy().ravel())
    for team in ordered:
        if team in seen:
            continue
        stack, comp = [team], set()
        while stack:
            t = stack.pop()
            if t not in comp:
                comp.add(t)
                stack.extend(adjacency[t] - comp)
        seen |= comp
        groups.append(sorted(comp))
    if any(len(g) != 4 for g in groups):
        raise ValueError(f"Groups are not all of size 4: {[len(g) for g in groups]}")
    return groups


def knockout_winner(row: pd.Series) -> str:
    """Winner of a knockout match, using the shoot-out winner after a draw.

    Raises:
        ValueError: If the match was drawn and no shoot-out winner is recorded.
    """
    if row["home_score"] > row["away_score"]:
        return str(row["home_team"])
    if row["away_score"] > row["home_score"]:
        return str(row["away_team"])
    winner = row.get("shootout_winner")
    if winner is None or pd.isna(winner):
        raise ValueError(f"Drawn knockout match without shoot-out winner: {dict(row)}")
    return str(winner)


def validate_world_cup(wc: pd.DataFrame) -> dict[str, object]:
    """Integrity checks for one World Cup; returns a small summary.

    Checks: each team plays three group matches, groups are complete,
    knockout winners are determinable, and the knockout tree is consistent
    (every team after the first knockout round won its previous match).
    """
    group = wc[wc["stage"] == "GROUP"]
    counts = pd.concat([group["home_team"], group["away_team"]]).value_counts()
    if not (counts == 3).all():
        raise ValueError(f"Teams without exactly 3 group matches: {counts[counts != 3].to_dict()}")
    groups = infer_groups(group)
    ko = wc[wc["stage"] != "GROUP"].copy()
    ko["winner"] = ko.apply(knockout_winner, axis=1)
    rounds = [s for s in ["R32", "R16", "QF", "SF", "FINAL"] if s in set(ko["stage"])]
    for prev, nxt in zip(rounds[:-1], rounds[1:], strict=True):
        winners = set(ko.loc[ko["stage"] == prev, "winner"])
        players = set(ko.loc[ko["stage"] == nxt, ["home_team", "away_team"]].to_numpy().ravel())
        if players != winners:
            raise ValueError(f"{nxt} participants are not the {prev} winners")
    final = ko[ko["stage"] == "FINAL"].iloc[0]
    return {
        "groups": groups,
        "champion": knockout_winner(final),
        "n_matches": len(wc),
        "n_shootouts": int(ko["shootout_winner"].notna().sum()),
    }

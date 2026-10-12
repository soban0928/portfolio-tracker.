import pandas as pd
import pytest

from tests.conftest import raw_frame
from wcpredict.data import (
    classify_competition,
    clean_matches,
    infer_groups,
    knockout_winner,
    match_type,
    standardise_team,
    validate_world_cup,
    world_cup_matches,
)


@pytest.mark.parametrize(
    ("tournament", "competition", "mtype"),
    [
        ("Friendly", "friendly", "friendly"),
        ("FIFA Series", "friendly", "friendly"),
        ("FIFA World Cup qualification", "qualifier", "competitive"),
        ("FIFA World Cup", "world_cup", "tournament"),
        ("UEFA Euro", "continental_final", "tournament"),
        ("UEFA Nations League", "nations_league", "competitive"),
        ("Gulf Cup", "minor_tournament", "competitive"),
    ],
)
def test_competition_classification(tournament, competition, mtype):
    assert classify_competition(tournament) == competition
    assert match_type(competition) == mtype


def test_standardise_team():
    assert standardise_team("Korea Republic") == "South Korea"
    assert standardise_team("Eritrea (unranked)") == "Eritrea"
    assert standardise_team("France") == "France"


def test_clean_matches_flags(toy_matches):
    m = toy_matches
    assert list(m["result"]) == ["H", "D", "A", "H", "D", "A"]
    assert list(m["match_id"]) == list(range(6))
    # neutral venue in a third country: nobody is host
    euro = m[m["tournament"] == "UEFA Euro"].iloc[0]
    assert not euro["home_is_host"] and not euro["away_is_host"]
    # non-neutral: home team is host
    assert m.iloc[0]["home_is_host"]
    assert m.loc[m["tournament"] == "FIFA World Cup", "importance_k"].iloc[0] == 60
    assert m["shootout_winner"].isna().all()


def test_clean_matches_drops_non_fifa_and_unplayed():
    raw = raw_frame(
        [
            ("2020-01-01", "Jersey", "Guernsey", 1, 0, "Muratti Vase", "Jersey", False),
            ("2020-01-02", "A", "B", None, None, "Friendly", "A", False),
            ("2020-01-03", "A", "B", 1, 0, "Friendly", "A", False),
        ]
    )
    assert len(clean_matches(raw)) == 1


def test_host_flag_for_away_side_at_home():
    raw = raw_frame([("2020-01-01", "A", "Qatar", 0, 1, "FIFA World Cup", "Qatar", True)])
    row = clean_matches(raw).iloc[0]
    assert row["away_is_host"] and not row["home_is_host"]


def test_shootout_merge():
    raw = raw_frame([("2020-01-01", "A", "B", 1, 1, "FIFA World Cup", "C", True)])
    so = pd.DataFrame(
        {"date": pd.to_datetime(["2020-01-01"]), "home_team": ["A"], "away_team": ["B"],
         "winner": ["B"], "first_shooter": ["A"]}
    )  # fmt: skip
    row = clean_matches(raw, so).iloc[0]
    assert knockout_winner(row) == "B"


def test_knockout_winner_requires_shootout():
    row = pd.Series({"home_team": "A", "away_team": "B", "home_score": 0, "away_score": 0,
                     "shootout_winner": pd.NA})  # fmt: skip
    with pytest.raises(ValueError):
        knockout_winner(row)
    home_win, away_win = row.copy(), row.copy()
    home_win["home_score"] = 2
    away_win["away_score"] = 1
    assert knockout_winner(home_win) == "A"
    assert knockout_winner(away_win) == "B"


def test_infer_groups_rejects_bad_shapes():
    df = pd.DataFrame({"home_team": ["A", "B"], "away_team": ["B", "C"]})
    with pytest.raises(ValueError):
        infer_groups(df)


@pytest.mark.parametrize(
    ("year", "champion", "n_matches"),
    [(2018, "France", 64), (2022, "Argentina", 64), (2026, "Spain", 104)],
)
def test_real_world_cups_validate(matches, year, champion, n_matches):
    wc = world_cup_matches(matches, year)
    summary = validate_world_cup(wc)
    assert summary["champion"] == champion
    assert summary["n_matches"] == n_matches
    assert len(summary["groups"]) == (12 if year == 2026 else 8)


def test_world_cup_matches_missing_year(matches):
    with pytest.raises(ValueError):
        world_cup_matches(matches, 2019)

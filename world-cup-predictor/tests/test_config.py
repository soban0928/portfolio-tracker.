from wcpredict.config import TOURNAMENTS


def test_tournament_specs():
    assert set(TOURNAMENTS) == {2018, 2022, 2026}
    for spec in TOURNAMENTS.values():
        assert spec.start < spec.end
    assert TOURNAMENTS[2026].n_teams == 48

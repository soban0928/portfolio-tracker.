import numpy as np
import pandas as pd
import pytest

from wcpredict.elo import compute_elo
from wcpredict.evaluation import training_rows
from wcpredict.goals import GoalsModel
from wcpredict.pipeline import build_context, fixture_frame, simulation_inputs, venue_for


@pytest.fixture(scope="module")
def ctx2022(matches):
    return build_context(matches, compute_elo(matches), 2022)


def test_context_uses_only_pre_tournament_data(ctx2022):
    assert ctx2022.hosts == {"Qatar"} and ctx2022.ko_hosts == {"Qatar"}
    assert len(ctx2022.states) == 32
    assert all(s.last_match < ctx2022.start for s in ctx2022.states.values())
    # Argentina's rating is frozen before the tournament (no World Cup matches counted)
    assert ctx2022.states["Argentina"].elo == pytest.approx(ctx2022.elo["Argentina"])


def test_venue_for():
    hosts = {"Qatar"}
    assert venue_for("Qatar", "Ecuador", hosts) == 1
    assert venue_for("Ecuador", "Qatar", hosts) == -1
    assert venue_for("Spain", "Ecuador", hosts) == 0


def test_fixture_frame_and_inputs(ctx2022, features):
    ff = fixture_frame(ctx2022, [("Qatar", "Ecuador"), ("Ecuador", "Qatar")])
    assert list(ff["venue"]) == [1.0, -1.0]
    assert ff.loc[0, "elo_diff"] == pytest.approx(-ff.loc[1, "elo_diff"])
    assert (ff["is_tournament"] == 1).all() and (ff["rest_home"] == 4).all()
    model = GoalsModel().fit(training_rows(features, ctx2022.start))
    inp = simulation_inputs(ctx2022, model)
    assert len(inp.pair_features) == 32 * 31
    lam = inp.rate_matrix()
    i, j = ctx2022.structure.team_index["Brazil"], ctx2022.structure.team_index["Qatar"]
    assert lam[i, j] > lam[j, i]
    assert np.isfinite(lam).all()
    assert isinstance(inp.ko_pair_features, pd.DataFrame)

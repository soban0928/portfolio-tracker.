"""Evaluate pre-tournament predictions against what actually happened.

For each World Cup every model is trained only on matches before kick-off and
every team's state is frozen at kick-off (:mod:`wcpredict.pipeline`). Then:

* **Group matches** are scored on W/D/L (90-minute results).
* **Knockout matches** are scored on *who advanced*: the dataset's knockout
  scores include extra time, so W/D/L at 90 minutes is not observable there.
  Classifiers give P(advance) = P(win) + 0.5 * P(draw); the goals model uses
  its extra-time/penalties win probability.
* **Tournament outcomes** compare simulated stage probabilities with the stage
  each team actually reached.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wcpredict.data import knockout_winner
from wcpredict.elo import venue_indicator
from wcpredict.evaluation import score
from wcpredict.features import fixture_features
from wcpredict.goals import GoalsModel
from wcpredict.models import MatchModel, encode
from wcpredict.pipeline import TOURNAMENT_REST_DAYS, TournamentContext
from wcpredict.simulation import knockout_win_matrix
from wcpredict.tournament import STAGE_ORDER

EPS = 1e-12


def real_fixture_frame(ctx: TournamentContext) -> pd.DataFrame:
    """Features for every real tournament match using the frozen pre-tournament states.

    The venue comes from the real match (host flags); rest days are the
    tournament default. The real columns (teams, scores, stage, result,
    shoot-out winner) are kept for scoring.
    """
    wc = ctx.matches
    rows = [
        fixture_features(
            ctx.states[r.home_team],
            ctx.states[r.away_team],
            ctx.start,
            venue=venue_indicator(bool(r.home_is_host), bool(r.away_is_host)),
            competition="world_cup",
            rest_home=TOURNAMENT_REST_DAYS,
            rest_away=TOURNAMENT_REST_DAYS,
        )
        for r in wc.itertuples()
    ]
    keep = ["date", "home_team", "away_team", "home_score", "away_score", "stage", "result",
            "shootout_winner"]  # fmt: skip
    return pd.concat([wc[keep].reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def advance_prob_goals(model: GoalsModel, X: pd.DataFrame) -> np.ndarray:
    """P(first team advances) from the goals model incl. extra time and penalties."""
    lam, mu = model.rates(X)
    out = np.empty(len(X))
    for k, (a, b) in enumerate(zip(lam, mu, strict=True)):
        beat, _ = knockout_win_matrix(np.array([[1.0, a], [b, 1.0]]), model.rho_)
        out[k] = beat[0, 1]
    return out


def match_predictions(
    models: list[MatchModel], goals_model: GoalsModel, fixtures: pd.DataFrame
) -> pd.DataFrame:
    """Long table of every model's prediction for every real match.

    Columns: match index, model, p_H, p_D, p_A, p_advance (knockouts), plus the
    real match columns.
    """
    frames = []
    is_ko = (fixtures["stage"] != "GROUP").to_numpy()
    for m in [*models, goals_model]:
        p = m.predict_proba(fixtures)
        if m is goals_model:
            adv = advance_prob_goals(goals_model, fixtures)
        else:
            adv = p[:, 0] + 0.5 * p[:, 1]
        f = fixtures[["date", "home_team", "away_team", "home_score", "away_score", "stage",
                      "result"]].copy()  # fmt: skip
        f["model"] = m.name
        f[["p_H", "p_D", "p_A"]] = p
        f["p_advance"] = np.where(is_ko, adv, np.nan)
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    ko = out["stage"] != "GROUP"
    winners = fixtures.loc[fixtures["stage"] != "GROUP"].apply(knockout_winner, axis=1)
    first_adv = (fixtures.loc[winners.index, "home_team"] == winners).astype(float)
    out.loc[ko, "home_advanced"] = np.tile(first_adv.to_numpy(), len(models) + 1)
    return out


def score_match_predictions(preds: pd.DataFrame) -> pd.DataFrame:
    """Per-model metrics: group-stage W/D/L and knockout advancement."""
    rows = []
    for model, g in preds.groupby("model", sort=False):
        grp = g[g["stage"] == "GROUP"]
        s = score(encode(grp["result"]), grp[["p_H", "p_D", "p_A"]].to_numpy())
        ko = g[g["stage"] != "GROUP"]
        y, p = ko["home_advanced"].to_numpy(), ko["p_advance"].to_numpy()
        p = np.clip(p, EPS, 1 - EPS)
        adv_ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        rows.append(
            {
                "model": model,
                "group_accuracy": s["accuracy"],
                "group_log_loss": s["log_loss"],
                "group_brier": s["brier"],
                "group_rps": s["rps"],
                "n_group": int(s["n"]),
                "ko_log_loss": adv_ll,
                "ko_accuracy": float(((p > 0.5) == (y == 1)).mean()),
                "n_ko": len(ko),
            }
        )
    return pd.DataFrame(rows).set_index("model")


def exit_stage_probs(stages: pd.DataFrame) -> pd.DataFrame:
    """P(team's furthest stage == s) from cumulative reach probabilities."""
    cols = [c for c in stages.columns if c.startswith("p_")]
    names = [c[2:] for c in cols]
    reach = stages[cols].to_numpy()
    out = pd.DataFrame(index=stages.index)
    out["GROUP"] = 1 - reach[:, 0]
    for k, name in enumerate(names):
        nxt = reach[:, k + 1] if k + 1 < len(names) else 0.0
        out[name] = reach[:, k] - nxt
    return out


def tournament_scores(stages: pd.DataFrame, progress: pd.Series) -> dict[str, object]:
    """Compare simulated stage probabilities with the real outcome.

    Returns champion/finalist/semi-finalist ranks and probabilities, per-stage
    Brier scores (with a uniform baseline), and the mean log-probability of each
    team's actual exit stage (with a uniform baseline).
    """
    stages = stages.reindex(progress.index)
    n = len(stages)
    stage_cols = [c for c in stages.columns if c.startswith("p_")]
    rank = {s: i for i, s in enumerate(STAGE_ORDER)}
    out: dict[str, object] = {}

    def ranks(col: str, teams: list[str]) -> dict[str, int]:
        order = stages[col].rank(ascending=False, method="min")
        return {t: int(order[t]) for t in teams}

    champion = progress.index[progress == "WINNER"][0]
    finalists = list(progress.index[progress.isin(["FINAL", "WINNER"])])
    semis = list(progress.index[progress.map(rank) >= rank["SF"]])
    out["champion"] = champion
    out["champion_p_win"] = float(stages.loc[champion, "p_WINNER"])
    out["champion_rank"] = ranks("p_WINNER", [champion])[champion]
    out["favourite"] = str(stages["p_WINNER"].idxmax())
    out["favourite_p_win"] = float(stages["p_WINNER"].max())
    out["finalist_ranks"] = ranks("p_FINAL", finalists)
    out["semifinalist_ranks"] = ranks("p_SF", semis)

    brier = {}
    for col in stage_cols:
        name = col[2:]
        reached = (progress.map(rank) >= rank[name]).astype(float)
        p = stages[col].to_numpy()
        base = reached.mean()
        brier[name] = {
            "model": float(((p - reached) ** 2).mean()),
            "uniform": float(((base - reached) ** 2).mean()),
        }
    out["stage_brier"] = brier

    exits = exit_stage_probs(stages)
    p_actual = np.array([exits.loc[t, s] for t, s in progress.items()])
    uniform = progress.value_counts() / n
    out["exit_log_score"] = float(np.log(np.clip(p_actual, EPS, 1)).mean())
    out["exit_log_score_uniform"] = float(np.log(progress.map(uniform)).mean())
    return out

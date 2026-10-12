"""End-to-end, reproducible pipeline: train, evaluate, simulate, explain.

Writes everything the README and the dashboard use into ``artefacts/``::

    python scripts/run_pipeline.py --sims 100000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from wcpredict.backtest import (
    match_predictions,
    real_fixture_frame,
    score_match_predictions,
    tournament_scores,
)
from wcpredict.calibration import CalibratedModel
from wcpredict.config import ARTEFACT_DIR, SEED, TOURNAMENTS
from wcpredict.data import load_matches
from wcpredict.elo import compute_elo
from wcpredict.evaluation import (
    calibration_table,
    encode,
    oof_predictions,
    summarise_predictions,
    training_rows,
)
from wcpredict.explain import shap_summary
from wcpredict.features import BASE_FEATURES, build_features, state_before, team_states
from wcpredict.goals import GoalsModel, elo_only_goals_model
from wcpredict.models import (
    BaseRateModel,
    HigherRatedModel,
    elo_only,
    logistic,
    random_forest,
    xgboost,
)
from wcpredict.pipeline import build_context, simulation_inputs
from wcpredict.simulation import SimulationResult, simulate
from wcpredict.tournament import actual_progress

CV_YEARS = list(range(2018, 2027))
ML_FACTORIES = {
    "Logistic regression": lambda: logistic(BASE_FEATURES),
    "Random Forest": lambda: random_forest(BASE_FEATURES),
    "XGBoost": lambda: xgboost(BASE_FEATURES),
}


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def to_json(obj: object) -> object:
    """Make numpy scalars JSON-serialisable."""
    if isinstance(obj, dict):
        return {str(k): to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _top(counts: np.ndarray, teams: list[str], total: float, n: int = 3) -> list[dict]:
    """The ``n`` most frequent teams in a bracket position with their probabilities."""
    idx = np.argsort(counts)[::-1][:n]
    return [{"team": teams[i], "p": counts[i] / total} for i in idx if counts[i] > 0]


def bracket_summary(res: SimulationResult, structure, wc: pd.DataFrame) -> list[dict]:
    """Top occupants and winners per knockout match, alongside the real teams."""
    names = structure.round_names
    real = wc[wc["stage"].isin(names)]
    out = []
    for r, name in enumerate(names):
        real_round = real[real["stage"] == name].reset_index(drop=True)
        for k in range(res.slot_counts[r].shape[0]):
            total = res.slot_counts[r][k, 0].sum()
            entry = {
                "round": name,
                "match": k,
                "side_a": _top(res.slot_counts[r][k, 0], structure.teams, total),
                "side_b": _top(res.slot_counts[r][k, 1], structure.teams, total),
                "winner": _top(res.winner_counts[r][k], structure.teams, total),
            }
            if r == 0:
                entry["slot_a"], entry["slot_b"] = (list(x) for x in structure.first_round[k])
            else:
                entry["from"] = list(structure.later_rounds[r - 1][k])
            if k < len(real_round):
                rr = real_round.iloc[k]
                entry["real"] = [rr["home_team"], rr["away_team"]]
            out.append(entry)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sims", type=int, default=100_000)
    parser.add_argument("--out", type=Path, default=ARTEFACT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    out: Path = args.out
    (out / "models").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    matches = load_matches()
    elo_matches = compute_elo(matches)
    features = build_features(elo_matches)
    log(f"features for {len(features):,} matches", t0)

    # ---------------------------------------------------- rolling CV (2018-2026)
    factories = {
        "Base rates": BaseRateModel,
        "Higher-rated team": HigherRatedModel,
        "Elo-only (logistic)": elo_only,
        **ML_FACTORIES,
        "Dixon-Coles (goals)": GoalsModel,
        "Dixon-Coles Elo-only (goals)": elo_only_goals_model,
    }
    oof = pd.concat([oof_predictions(f, features, CV_YEARS) for f in factories.values()])
    summary = summarise_predictions(oof)
    ml = summary.loc[list(ML_FACTORIES)]
    best_name = str(ml["log_loss"].idxmin())
    best_factory = ML_FACTORIES[best_name]
    log(f"rolling CV done; best ML model by log loss: {best_name}", t0)

    calibrated = {
        f"{best_name} + platt": lambda: CalibratedModel(best_factory, "platt"),
        f"{best_name} + isotonic": lambda: CalibratedModel(best_factory, "isotonic"),
    }
    oof_cal = pd.concat([oof_predictions(f, features, CV_YEARS) for f in calibrated.values()])
    oof = pd.concat([oof, oof_cal], ignore_index=True)
    summary = summarise_predictions(oof)
    competitive = summarise_predictions(oof[oof["match_type"] != "friendly"])
    summary.to_csv(out / "cv_summary.csv")
    competitive.to_csv(out / "cv_summary_competitive.csv")
    per_year = (
        pd.concat([summarise_predictions(g).assign(year=yr) for yr, g in oof.groupby("year")])
        .rename_axis("model")
        .reset_index()
    )
    per_year.to_csv(out / "cv_by_year.csv", index=False)
    cal_frames = []
    for model in [best_name, *calibrated, "Dixon-Coles (goals)", "Elo-only (logistic)"]:
        g = oof[oof["model"] == model]
        t = calibration_table(encode(g["result"]), g[["p_H", "p_D", "p_A"]].to_numpy())
        t["model"] = model
        cal_frames.append(t)
    pd.concat(cal_frames).to_csv(out / "calibration_curves.csv", index=False)
    log("calibration done", t0)

    # ---------------------------------------------------- SHAP (best model, pre-2026 data)
    shap_model = best_factory().fit(
        training_rows(features, TOURNAMENTS[2026].start),
        encode(training_rows(features, TOURNAMENTS[2026].start)["result"]),
    )
    recent = features[
        (features["date"] >= "2018-01-01") & (features["date"] < TOURNAMENTS[2026].start)
    ]
    importance, shap_sample = shap_summary(shap_model, recent, n=1000, seed=args.seed)
    importance.to_csv(out / "shap_importance.csv")
    shap_sample.to_csv(out / "shap_sample.csv", index=False)
    log("SHAP done", t0)

    # ---------------------------------------------------- tournaments
    results: dict[str, object] = {
        "best_model": best_name,
        "cv_years": [CV_YEARS[0], CV_YEARS[-1]],
        "n_sims": args.sims,
        "seed": args.seed,
        "tournaments": {},
    }
    for year in TOURNAMENTS:
        ydir = out / str(year)
        ydir.mkdir(exist_ok=True)
        ctx = build_context(matches, elo_matches, year)
        train = training_rows(features, ctx.start)
        y = encode(train["result"])
        models = [
            BaseRateModel().fit(train, y),
            elo_only().fit(train, y),
            *[f().fit(train, y) for f in ML_FACTORIES.values()],
            *[f().fit(train, y) for f in calibrated.values()],
            elo_only_goals_model().fit(train),
        ]
        goals = GoalsModel().fit(train)
        fixtures = real_fixture_frame(ctx)
        preds = match_predictions(models, goals, fixtures)
        preds.to_csv(ydir / "match_predictions.csv", index=False)
        match_scores = score_match_predictions(preds)
        match_scores.to_csv(ydir / "match_scores.csv")

        progress = actual_progress(ctx.matches)
        sims = {}
        for label, gm in [("full", goals), ("elo_only", models[-1])]:
            res = simulate(simulation_inputs(ctx, gm), n_sims=args.sims, seed=args.seed)
            res.stages.to_csv(ydir / f"stages_{label}.csv", index_label="team")
            sims[label] = res
        res = sims["full"]
        res.positions.to_csv(ydir / "positions.csv", index_label="team")
        progress.to_csv(ydir / "actual_progress.csv", index_label="team")
        (ydir / "bracket.json").write_text(
            json.dumps(to_json(bracket_summary(res, ctx.structure, ctx.matches)), indent=1)
        )
        groups = [[ctx.structure.teams[i] for i in g] for g in ctx.structure.groups]
        t_scores = {k: tournament_scores(v.stages, progress) for k, v in sims.items()}
        goals.coefficients().to_csv(ydir / "goals_coefficients.csv")
        joblib.dump(goals, out / "models" / f"{year}_goals.joblib", compress=3)
        joblib.dump(models[2 + list(ML_FACTORIES).index(best_name)],
                    out / "models" / f"{year}_{best_name.replace(' ', '_').lower()}.joblib",
                    compress=3)  # fmt: skip
        results["tournaments"][str(year)] = {
            "start": str(ctx.start.date()),
            "n_train_matches": len(train),
            "hosts": sorted(ctx.hosts),
            "groups": groups,
            "match_scores": match_scores.reset_index().to_dict(orient="records"),
            "tournament": t_scores,
            "sim_meta": res.meta,
            "rho": goals.rho_,
        }
        log(f"{year}: champion {t_scores['full']['champion']} ranked "
            f"{t_scores['full']['champion_rank']} (p={t_scores['full']['champion_p_win']:.3f})",
            t0)  # fmt: skip

    # ---------------------------------------------------- current models for the app
    cur = out / "current"
    cur.mkdir(exist_ok=True)
    end = features["date"].max() + pd.Timedelta(days=1)
    all_rows = training_rows(features, end)
    best_now = best_factory().fit(all_rows, encode(all_rows["result"]))
    goals_now = GoalsModel().fit(all_rows)
    joblib.dump(best_now, cur / "classifier.joblib", compress=3)
    joblib.dump(goals_now, cur / "goals.joblib", compress=3)
    st = team_states(elo_matches)
    active = st.loc[st["date"] >= "2018-01-01", "team"].unique()
    states = pd.DataFrame([vars(state_before(st, t, end)) for t in sorted(active)])
    states.to_csv(cur / "team_states.csv", index=False)
    (cur / "meta.json").write_text(
        json.dumps(
            {"as_of": str(end.date()), "classifier": best_name, "n_train": len(all_rows)}, indent=1
        )
    )
    results["current_as_of"] = str(end.date())
    (out / "summary.json").write_text(json.dumps(to_json(results), indent=1))
    log("done", t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

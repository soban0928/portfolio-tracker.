"""Rolling out-of-sample metrics vs baselines, calibration and SHAP."""

from __future__ import annotations

import streamlit as st
from charts import metric_lines, reliability, shap_bars, shap_strip
from ui import FEATURE_LABELS, csv, page, summary

page("Model Evaluation", "🧪")
s = summary()
best = s["best_model"]
y0, y1 = s["cv_years"]
st.markdown(
    f"Expanding-window cross-validation: for every year {y0}–{y1}, each model is trained on "
    "all matches from 1994 up to 31 December of the previous year and tested on that year's "
    "matches (friendlies included). Lower is better for log loss, Brier and RPS."
)

scope = st.radio("Matches", ["All", "Competitive only (no friendlies)"], horizontal=True)
summ = csv("cv_summary.csv" if scope == "All" else "cv_summary_competitive.csv", index_col=0)
table = summ[["accuracy", "log_loss", "brier", "rps", "ece", "n"]].sort_values("log_loss")
table.columns = ["Accuracy", "Log loss", "Brier", "RPS", "Calib. error", "Matches"]
st.dataframe(
    table.style.format(
        {
            "Accuracy": "{:.1%}",
            "Log loss": "{:.4f}",
            "Brier": "{:.4f}",
            "RPS": "{:.4f}",
            "Calib. error": "{:.4f}",
            "Matches": "{:.0f}",
        }
    ).highlight_min(subset=["Log loss", "Brier", "RPS"], color="#cde2fb"),
)
elo = summ.loc["Elo-only (logistic)", "log_loss"]
gain = elo - summ.loc[best, "log_loss"]
st.info(
    f"**Honest read:** {best} beats the Elo-only baseline by just {gain:.4f} in log loss, and "
    "always backing the higher-rated team is about as accurate as any model. Most of the "
    "signal is in the Elo rating; the extra features add a little calibration, not a "
    "step change. 'Higher-rated team' never predicts draws, so its log loss is infinite."
)

by_year = csv("cv_by_year.csv")
models = [best, "Elo-only (logistic)", "Dixon-Coles (goals)", "Logistic regression"]
st.plotly_chart(metric_lines(by_year, "log_loss", models, "Log loss by test year"))

st.subheader("Calibration: before and after")
cal = csv("calibration_curves.csv")
outcome = (
    st.segmented_control(
        "Outcome",
        ["H", "D", "A"],
        default="H",
        format_func={"H": "First team wins", "D": "Draw", "A": "Second team wins"}.get,
    )
    or "H"
)
cal_models = [best, f"{best} + platt", f"{best} + isotonic"]
left, right = st.columns(2)
with left:
    st.plotly_chart(reliability(cal, cal_models, outcome, "Raw vs calibrated"))
with right:
    st.plotly_chart(
        reliability(
            cal,
            ["Dixon-Coles (goals)", "Elo-only (logistic)"],
            outcome,
            "Goals model and Elo baseline",
        )
    )
ece = summ.loc[cal_models, ["log_loss", "ece"]]
st.caption(
    "Calibration error (count-weighted |predicted − observed|): "
    + ", ".join(f"{m}: {r.ece:.4f} (log loss {r.log_loss:.4f})" for m, r in ece.iterrows())
    + ". The raw model is already well calibrated; isotonic over-fits the calibration window."
)

st.subheader(f"Explainability: SHAP for {best}")
imp = csv("shap_importance.csv", index_col=0)
sample = csv("shap_sample.csv")
left, right = st.columns(2)
with left:
    st.plotly_chart(shap_bars(imp, FEATURE_LABELS))
with right:
    st.plotly_chart(shap_strip(sample, list(imp.index[:6]), FEATURE_LABELS))
st.caption(
    "SHAP values computed on 1,000 matches from 2018 up to the 2026 World Cup, for the "
    "model trained on all earlier data."
)

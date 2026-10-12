"""Most likely occupant of every knockout slot, next to what really happened."""

from __future__ import annotations

import streamlit as st
from ui import STAGE_LABELS, bracket, page, pct, year_picker

page("Simulated Bracket", "🏆")
year = year_picker()
data = bracket(year)
rounds = list(dict.fromkeys(m["round"] for m in data))
st.caption(
    "Each card shows the team that most often filled each side of the match across the "
    "simulations (with its share) and the most frequent winner. ✓ marks a slot where the "
    "most likely team is the one that really played there."
)

# Order each round's matches by bracket position (top to bottom), walking back from the final.
by_round = {r: [m for m in data if m["round"] == r] for r in rounds}
order = {rounds[-1]: [0]}
for later, earlier in zip(rounds[::-1][:-1], rounds[::-1][1:], strict=True):
    order[earlier] = [src for k in order[later] for src in by_round[later][k]["from"]]

cols = st.columns(len(rounds))
for col, rnd in zip(cols, rounds, strict=True):
    with col:
        st.markdown(f"#### {STAGE_LABELS[rnd]}")
        for m in [by_round[rnd][k] for k in order[rnd]]:
            real = m.get("real", [])
            with st.container(border=True):
                for side in ("side_a", "side_b"):
                    top = m[side][0]
                    mark = " ✓" if top["team"] in real else ""
                    st.markdown(f"{top['team']} · {pct(top['p'], 0)}{mark}")
                w = m["winner"][0]
                st.caption(f"Winner: {w['team']} ({pct(w['p'], 0)})")
                if real:
                    st.caption(f"Real: {real[0]} v {real[1]}")

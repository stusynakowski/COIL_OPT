#!/usr/bin/env python
"""
Streamlit dashboard for visualising sweep results.

Run from the sweep_output directory:
    streamlit run sweep_results_app.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Sweep Results", layout="wide")
st.title("📊 Coil Optimisation Sweep Results")

SWEEP_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
summary_path = SWEEP_DIR / "sweep_summary.json"
if not summary_path.exists():
    st.error(f"`sweep_summary.json` not found in `{SWEEP_DIR}`. Run `run_sweep.py` first.")
    st.stop()

with open(summary_path) as f:
    raw = json.load(f)

df = pd.DataFrame(raw)
df["label"] = df.apply(lambda r: f"nc={int(r.ncoils)}, nq={int(r.n_unique_shapes)}", axis=1)
df["Bn_reduction"] = df["init_max_Bn"] / df["final_max_Bn"]


@st.cache_data
def load_loss_history(ncoils: int, n_unique: int) -> np.ndarray | None:
    p = SWEEP_DIR / f"ncoils{ncoils}_nuniq{n_unique}" / "loss_history.txt"
    if p.exists():
        return np.loadtxt(p)
    return None


# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Filters")
    all_nc = sorted(df["ncoils"].unique())
    sel_nc = st.multiselect("ncoils", all_nc, default=all_nc)

    all_nq = sorted(df["n_unique_shapes"].unique())
    sel_nq = st.multiselect("n_unique_shapes", all_nq, default=all_nq)

mask = df["ncoils"].isin(sel_nc) & df["n_unique_shapes"].isin(sel_nq)
filtered = df[mask].copy()

if filtered.empty:
    st.warning("No runs match the current filters.")
    st.stop()

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
st.subheader("Summary Table")
display_cols = [
    "ncoils", "n_unique_shapes", "n_dofs",
    "init_max_Bn", "final_max_Bn", "Bn_reduction",
    "final_loss", "total_base_curve_length",
    "n_iters", "elapsed_s",
]
st.dataframe(
    filtered[display_cols].style.format({
        "init_max_Bn": "{:.4e}",
        "final_max_Bn": "{:.4e}",
        "Bn_reduction": "{:.1f}×",
        "final_loss": "{:.4e}",
        "total_base_curve_length": "{:.3f}",
        "elapsed_s": "{:.2f}",
    }),
    use_container_width=True,
    hide_index=True,
)

# ---------------------------------------------------------------------------
# Bar / scatter plots
# ---------------------------------------------------------------------------
st.subheader("Final max|B·n| by Configuration")

fig_bn = px.bar(
    filtered.sort_values(["ncoils", "n_unique_shapes"]),
    x="label", y="final_max_Bn",
    color="ncoils",
    log_y=True,
    labels={"final_max_Bn": "Final max|B·n|", "label": "Configuration"},
    text_auto=".2e",
)
fig_bn.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig_bn, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Final Loss")
    fig_loss = px.bar(
        filtered.sort_values(["ncoils", "n_unique_shapes"]),
        x="label", y="final_loss",
        color="ncoils",
        log_y=True,
        labels={"final_loss": "Final Loss", "label": "Configuration"},
        text_auto=".2e",
    )
    fig_loss.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_loss, use_container_width=True)

with col2:
    st.subheader("B·n Reduction Factor (init / final)")
    fig_red = px.bar(
        filtered.sort_values(["ncoils", "n_unique_shapes"]),
        x="label", y="Bn_reduction",
        color="ncoils",
        labels={"Bn_reduction": "Reduction ×", "label": "Configuration"},
        text_auto=".0f",
    )
    fig_red.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_red, use_container_width=True)

# ---------------------------------------------------------------------------
# DOFs vs final_max_Bn scatter
# ---------------------------------------------------------------------------
st.subheader("DOFs vs. Final max|B·n|")
fig_scatter = px.scatter(
    filtered,
    x="n_dofs", y="final_max_Bn",
    color="ncoils",
    symbol="n_unique_shapes",
    size="elapsed_s",
    hover_data=["ncoils", "n_unique_shapes", "n_iters", "final_loss"],
    log_y=True,
    labels={"n_dofs": "Number of DOFs", "final_max_Bn": "Final max|B·n|"},
)
st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------------------------------------------------------------------
# Heatmap: ncoils × n_unique_shapes
# ---------------------------------------------------------------------------
st.subheader("Heatmap: Final max|B·n|")

with st.sidebar:
    st.divider()
    heatmap_metric = st.selectbox(
        "Heatmap metric",
        ["final_max_Bn", "final_loss", "Bn_reduction", "n_dofs", "elapsed_s", "n_iters"],
        index=0,
    )

pivot = df.pivot_table(index="ncoils", columns="n_unique_shapes", values=heatmap_metric)

fig_heat = px.imshow(
    pivot,
    text_auto=".3g",
    labels=dict(x="n_unique_shapes", y="ncoils", color=heatmap_metric),
    aspect="auto",
    color_continuous_scale="Viridis_r" if "Bn" in heatmap_metric or "loss" in heatmap_metric else "Viridis",
)
fig_heat.update_xaxes(type="category")
fig_heat.update_yaxes(type="category")
st.plotly_chart(fig_heat, use_container_width=True)

# ---------------------------------------------------------------------------
# Convergence curves
# ---------------------------------------------------------------------------
st.subheader("Convergence Histories")

with st.sidebar:
    st.divider()
    st.header("Convergence")
    conv_runs = st.multiselect(
        "Runs to plot",
        filtered["label"].tolist(),
        default=filtered["label"].tolist(),
    )

fig_conv = go.Figure()
for _, row in filtered[filtered["label"].isin(conv_runs)].iterrows():
    hist = load_loss_history(int(row.ncoils), int(row.n_unique_shapes))
    if hist is not None:
        fig_conv.add_trace(go.Scatter(
            y=hist,
            mode="lines",
            name=row.label,
        ))

fig_conv.update_layout(
    yaxis_type="log",
    xaxis_title="Function Evaluation",
    yaxis_title="Loss",
    legend_title="Configuration",
    height=500,
)
st.plotly_chart(fig_conv, use_container_width=True)

# ---------------------------------------------------------------------------
# Per-ncoils grouped view
# ---------------------------------------------------------------------------
st.subheader("Effect of n_unique_shapes (grouped by ncoils)")

for nc in sorted(filtered["ncoils"].unique()):
    sub = filtered[filtered["ncoils"] == nc].sort_values("n_unique_shapes")
    if len(sub) < 2:
        continue

    with st.expander(f"ncoils = {nc}  ({len(sub)} runs)", expanded=False):
        col_a, col_b = st.columns(2)

        with col_a:
            fig_a = px.line(
                sub, x="n_unique_shapes", y="final_max_Bn",
                markers=True, log_y=True,
                title=f"ncoils={nc}: max|B·n| vs n_unique_shapes",
                labels={"n_unique_shapes": "n_unique_shapes", "final_max_Bn": "Final max|B·n|"},
            )
            st.plotly_chart(fig_a, use_container_width=True)

        with col_b:
            fig_b = go.Figure()
            for _, row in sub.iterrows():
                hist = load_loss_history(int(row.ncoils), int(row.n_unique_shapes))
                if hist is not None:
                    fig_b.add_trace(go.Scatter(
                        y=hist, mode="lines",
                        name=f"nq={int(row.n_unique_shapes)}",
                    ))
            fig_b.update_layout(
                yaxis_type="log",
                title=f"ncoils={nc}: Convergence",
                xaxis_title="Evaluation", yaxis_title="Loss",
                height=400,
            )
            st.plotly_chart(fig_b, use_container_width=True)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from src import evaluation, state, ui

ui.hero("Analytics Dashboard", "Explore the scored applicants. Charts are interactive: hover, zoom and filter.")
ui.need_results()
res = st.session_state["results"].copy()
model = state.load_model()
RC = config.RISK_COLOURS

# ---------- filters
with st.expander("Filters", expanded=False):
    f1, f2, f3 = st.columns(3)
    risk_f = f1.multiselect("Risk band", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    amin, amax = int(res[config.AGE].min()), int(res[config.AGE].max())
    age_f = f2.slider("Age", amin, amax, (amin, amax)) if amin < amax else (amin, amax)
    inc_f = f3.multiselect("Income band", sorted(res["Income band"].dropna().unique()), default=sorted(res["Income band"].dropna().unique()))
res = res[res["Risk"].isin(risk_f) & res[config.AGE].between(*age_f) & res["Income band"].isin(inc_f)]
if res.empty:
    st.warning("No applicants match these filters.")
    st.stop()

k1, k2, k3, k4 = st.columns(4)
with k1: ui.tile("Number of High risk", f"{(res['Risk'] == 'High').sum():,}", f"{(res['Risk'] == 'High').mean():.1%} of the filtered applicants", RC["High"])
with k2: ui.tile("Average risk", f"{res['Fraud Score'].mean():.1%}", "mean fraud probability", ui.P["lavender"])
with k3: ui.tile("Medium risk", f"{(res['Risk'] == 'Medium').sum():,}", "need extra verification", RC["Medium"])
with k4: ui.tile("Applicants shown", f"{len(res):,}", "after filters", ui.P["blue"])
st.write("")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Breakdowns", "Age x Income", "Monitoring"])

with tab1:
    a, b = st.columns(2)
    counts = res["Risk"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
    fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.55, sort=False,
                           marker=dict(colors=[RC[k] for k in counts.index], line=dict(color="white", width=3)), textinfo="label+percent"))
    fig.update_layout(title="Applicants by risk band")
    a.plotly_chart(ui.style_fig(fig, 340), use_container_width=True)
    top_x = float(max(res["Fraud Score"].quantile(0.995), model.thresholds["high"] * 2))
    fig = px.histogram(res[res["Fraud Score"] <= top_x], x="Fraud Score", nbins=40, color_discrete_sequence=[ui.P["blue"]])
    fig.add_vline(x=model.thresholds["medium"], line_dash="dash", line_color="#E0B84F")
    fig.add_vline(x=model.thresholds["high"], line_dash="dash", line_color="#E27A98")
    fig.update_traces(marker_line_color="white", marker_line_width=1)
    fig.update_layout(title="Distribution of fraud scores (dashed lines: Medium and High thresholds)")
    b.plotly_chart(ui.style_fig(fig, 340, legend=False), use_container_width=True)
    if "month" in res.columns:
        m = res.groupby("month").agg(avg=("Fraud Score", "mean"), high=("Risk", lambda s: (s == "High").mean())).reset_index()
        fig = go.Figure()
        fig.add_scatter(x=m["month"], y=m["avg"], name="Average score", mode="lines+markers", line=dict(color="#B79CD6", width=3))
        fig.add_scatter(x=m["month"], y=m["high"], name="Share High risk", mode="lines+markers", line=dict(color="#F2A7BE", width=3))
        fig.update_layout(title="Risk by month"); fig.update_yaxes(tickformat=".1%")
        st.plotly_chart(ui.style_fig(fig, 320), use_container_width=True)

with tab2:
    dims = {"Age band": "Age band", "Income band": "Income band", "Employment status": "employment_status",
            "Housing status": "housing_status", "Device operating system": "device_os", "Payment type": "payment_type", "Source": "source"}
    if "Province" in res.columns:
        dims = {"Province": "Province", **dims}
    c1, c2 = st.columns([2, 2])
    dim = c1.selectbox("Break down by", list(dims))
    measure = c2.radio("Measure", ["Average risk", "Number of High risk", "Share High risk"], horizontal=True)
    col = dims[dim]
    g = res.groupby(col).agg(n=("Fraud Score", "size"), avg=("Fraud Score", "mean"),
                             high=("Risk", lambda s: (s == "High").sum())).reset_index()
    g["share"] = g["high"] / g["n"]
    y = {"Average risk": "avg", "Number of High risk": "high", "Share High risk": "share"}[measure]
    if col == "Age band":
        g[col] = pd.Categorical(g[col], ["Under 30", "30-39", "40-49", "50-59", "60 and over"], ordered=True)
        g = g.sort_values(col)
    fig = go.Figure(go.Bar(x=g[col].astype(str), y=g[y], marker=dict(color=ui.P["lavender"], line=dict(color="#9a90b8", width=1)),
                           customdata=g[["n", "high"]], hovertemplate="%{x}<br>%{y:.3f}<br>applicants: %{customdata[0]:,}<br>High risk: %{customdata[1]:,}<extra></extra>"))
    fig.update_layout(title=f"{measure} by {dim.lower()}")
    if y != "high":
        fig.update_yaxes(tickformat=".1%")
    st.plotly_chart(ui.style_fig(fig, 380, legend=False), use_container_width=True)
    if "Province" not in res.columns:
        st.caption("No Province column in this file. If your data has one (for example SASSA data later), map it on the validation page and it will appear here.")

with tab3:
    order_a = ["Under 30", "30-39", "40-49", "50-59", "60 and over"]
    order_i = ["Lower (up to 0.3)", "Middle (0.3-0.6)", "Higher (above 0.6)"]
    pv = res.pivot_table(index="Age band", columns="Income band", values="Fraud Score", aggfunc="mean")
    pv = pv.reindex(index=[x for x in order_a if x in pv.index], columns=[x for x in order_i if x in pv.columns])
    fig = go.Figure(go.Heatmap(z=pv.values, x=pv.columns, y=pv.index, colorscale=[[0, "#E4F1FF"], [0.5, "#EBDDF8"], [1, "#F7A8BE"]],
                               hovertemplate="Age %{y}<br>Income %{x}<br>average risk %{z:.1%}<extra></extra>", colorbar=dict(tickformat=".0%")))
    fig.update_layout(title="Average fraud risk: age band and income band")
    st.plotly_chart(ui.style_fig(fig, 380, legend=False), use_container_width=True)
    sc = px.scatter(res.sample(min(4000, len(res)), random_state=1), x=config.AGE, y="income", color="Risk", color_discrete_map=RC,
                    opacity=0.75, hover_data=["Applicant ID", "Fraud Score"])
    sc.update_traces(marker=dict(size=8, line=dict(width=0.5, color="#9a90b8")))
    sc.update_layout(title="Applicants by age and income (colour = risk band)")
    st.plotly_chart(ui.style_fig(sc, 380), use_container_width=True)

with tab4:
    # PSI = how far the score distribution moved away from the validation months (FR7)
    edges = np.array(model.info["psi_edges"])
    ref = np.array(model.info["psi_reference_share"])
    now = np.histogram(res["Fraud Score"], bins=edges)[0] / len(res)
    eps = 1e-4
    psi = float(np.sum((now - ref) * np.log((now + eps) / (ref + eps))))
    state_txt, colour = (("Stable", ui.P["mint"]) if psi < 0.1 else (("Moderate shift", RC["Medium"]) if psi < 0.25 else ("Large shift, consider retraining", RC["High"])))
    a, b = st.columns([1, 2])
    with a:
        ui.tile("Score drift (PSI)", f"{psi:.3f}", state_txt, colour)
        st.caption("PSI below 0.10 is stable, 0.10 to 0.25 a moderate shift, above 0.25 a large shift.")
    with b:
        fig = go.Figure()
        fig.add_bar(x=list(range(1, len(ref) + 1)), y=ref, name="Validation months", marker_color=ui.P["lavender"])
        fig.add_bar(x=list(range(1, len(ref) + 1)), y=now, name="This file", marker_color=ui.P["pink"])
        fig.update_layout(barmode="group", title="Score distribution: validation months vs this file (deciles)")
        fig.update_yaxes(tickformat=".0%"); fig.update_xaxes(title="score bin (low to high)")
        st.plotly_chart(ui.style_fig(fig, 320), use_container_width=True)
    if "fraud_bool" in res.columns:
        st.markdown("#### Model check (this file has known fraud labels)")
        y = res["fraud_bool"].values
        r = evaluation.rates_at_threshold(y, res["Fraud Score"].values, model.thresholds["high"] - 1e-9)
        m1, m2, m3 = st.columns(3)
        with m1: ui.tile("Fraud caught (High band)", f"{r['TPR']:.1%}", f"{r['tp']} of {r['tp'] + r['fn']} fraud cases", ui.P["pink"])
        with m2: ui.tile("Genuine wrongly flagged", f"{r['FPR']:.1%}", f"{r['fp']:,} applicants", ui.P["peach"])
        with m3: ui.tile("Precision", f"{r['precision']:.1%}", "share of High risk that is fraud", ui.P["blue"])

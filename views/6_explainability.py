import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap

import config
from src import explain, state, ui

ui.hero("Explainability", "Which features push a score up or down? (SHAP)")
ui.need_results()
res = st.session_state["results"]
model = state.load_model()

# SHAP is slow on big files, so I only explain a random sample of 1,500 applicants (and keep it for the session)
if "shap_cache" not in st.session_state:
    with st.spinner("Calculating SHAP values..."):
        sample = res.sample(min(1500, len(res)), random_state=1)
        X, values, base = explain.shap_for(model, sample[config.REQUIRED_COLUMNS], max_rows=1500)
        st.session_state["shap_cache"] = (X, values, base, sample["Applicant ID"].tolist())
X, values, base, sample_ids = st.session_state["shap_cache"]
labels = [explain.nice(c) for c in X.columns]
st.caption(f"Explained on a random sample of {len(X):,} applicants. SHAP values are on the log-odds scale (before calibration).")

tab1, tab2, tab3 = st.tabs(["Feature importance", "SHAP summary", "Waterfall plot"])

with tab1:
    imp = pd.Series(np.abs(values).mean(axis=0), index=labels).sort_values().tail(15)
    fig = go.Figure(go.Bar(y=imp.index, x=imp.values, orientation="h",
                           marker=dict(color=imp.values, colorscale=[[0, "#CFE6FF"], [0.5, "#E3D4F5"], [1, "#F7B6CB"]], line=dict(color="#9a90b8", width=0.8)),
                           hovertemplate="%{y}<br>mean |SHAP| = %{x:.3f}<extra></extra>"))
    fig.update_layout(title="Top 15 features: average impact on the score")
    fig.update_xaxes(title="mean |SHAP value|")
    st.plotly_chart(ui.style_fig(fig, 480, legend=False), use_container_width=True)
    train_imp = state.load_table("feature_importance_shap.csv")
    if train_imp is not None:
        with st.expander("Importance on the BAF test months (from training)"):
            st.dataframe(train_imp.head(15)[["label", "mean_abs_shap"]].rename(columns={"label": "Feature", "mean_abs_shap": "Mean |SHAP|"}),
                         hide_index=True, use_container_width=True)
    st.markdown("**How to read this:** the longer the bar, the more that feature moves scores, up or down. "
                "It says what the model uses, not what causes fraud.")

with tab2:
    cmap = LinearSegmentedColormap.from_list("pastel", ["#A2D2FF", "#E8DAF6", "#F7A8BE"])
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(values, X.values, feature_names=labels, max_display=12, cmap=cmap, show=False, plot_size=None)
    fig = plt.gcf()
    fig.patch.set_facecolor("white")
    st.pyplot(fig, clear_figure=True)
    st.markdown("**How to read this:** each dot is one applicant. Pink dots have a high value of the feature, blue dots a low value. "
                "Dots to the right push the fraud score up.")

with tab3:
    pick = st.selectbox("Applicant", sample_ids, index=0)
    i = sample_ids.index(pick)
    sv = values[i]
    order = np.argsort(-np.abs(sv))[:10]
    other = sv.sum() - sv[order].sum()
    names = [f"{labels[j]} = {X.iloc[i, j]:.2f}" if abs(X.iloc[i, j]) < 1e5 else labels[j] for j in order] + ["All other features"]
    vals = [sv[j] for j in order] + [other]
    fig = go.Figure(go.Waterfall(
        orientation="h", measure=["relative"] * len(vals) + ["total"], y=names[::-1] + ["Final score"], x=vals[::-1] + [sum(vals)],
        base=base, connector=dict(line=dict(color="#CFC6E3")),
        increasing=dict(marker=dict(color="#F7B6CB", line=dict(color="#c98aa2", width=1))),
        decreasing=dict(marker=dict(color="#B5EAD7", line=dict(color="#86c3ad", width=1))),
        totals=dict(marker=dict(color="#CDB4DB", line=dict(color="#9a82b3", width=1)))))
    fig.update_layout(title=f"Waterfall for applicant {pick}: starts at the average score and adds each feature")
    fig.update_xaxes(title="log-odds of fraud")
    st.plotly_chart(ui.style_fig(fig, 480, legend=False), use_container_width=True)
    prob = res.loc[res["Applicant ID"] == pick, "Fraud Score"].iloc[0]
    st.markdown(f"Calibrated fraud probability for this applicant: **{prob:.1%}**. Pink bars raise the score, green bars lower it.")

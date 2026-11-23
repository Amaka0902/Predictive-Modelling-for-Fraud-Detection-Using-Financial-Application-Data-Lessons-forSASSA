import plotly.graph_objects as go
import streamlit as st

import config
from src import state, ui

P = config.PASTEL
ui.hero("Fraud Detection Dashboard",
        "Predictive modelling for fraud detection using financial application data: lessons for the South African Social Security Agency "
        "(SASSA)")

col1, col2 = st.columns([3, 2], gap="large")
with col1:
    ui.card("Project description",
            "<p>Social grant systems such as SASSA pay out billions of Rands, and fraud is often found only after the money has "
            "been paid. This project builds a machine learning model that gives every new application a fraud risk score "
            "before a decision is made, so investigators can focus on the riskiest cases.</p>"
            "<p>Access to SASSA records was not possible in the project time frame, so the model is developed on the public "
            "Bank Account Fraud (BAF) dataset (Jesus et al., 2022), which has the same problems as grant fraud: fraud is rare, "
            "it changes over time, and wrongly flagging genuine people does harm. The dashboard shows how the finished model would be used.</p>")
with col2:
    ui.card("How to use this dashboard",
            "<ol style='margin:0;padding-left:18px'><li>Upload a file of applications</li><li>Check and map the columns</li>"
            "<li>Run the predictions</li><li>Look at results, explanations and analytics</li><li>Download the report</li></ol>")

st.markdown("#### Research objectives and questions")
c1, c2 = st.columns(2, gap="large")
with c1:
    ui.card("Objectives",
            "<p><b>O1.</b> Identify the key variables and data patterns associated with fraudulent applications.</p>"
            "<p><b>O2.</b> Develop, evaluate and deploy a machine learning model that improves fraud detection while keeping "
            "processing fast and fair for legitimate applicants, and draw lessons for SASSA.</p>")
with c2:
    ui.card("Research questions",
            "<p><b>RQ1.</b> Which demographic and behavioural indicators are high probability predictors of fraud?</p>"
            "<p><b>RQ2.</b> How can machine learning improve fraud detection while keeping services fast and fair, "
            "and how can the model be deployed for SASSA?</p>")

metrics = state.load_json("metrics.json")
if not metrics:
    st.warning("The model has not been trained yet. Run `python train_model.py` first (see the README).")
    st.stop()

st.markdown("#### The model at a glance")
comp = state.load_table("model_comparison.csv")
if comp is None or comp.empty:
    st.warning("Missing `outputs/model_comparison.csv`. Commit `outputs/model_comparison.csv` to the repository and reboot the app.")
    st.stop()
chosen = comp[comp["model"] == metrics["chosen_model"]].iloc[0]
k1, k2, k3, k4, k5 = st.columns(5)
with k1: ui.tile("Chosen model", metrics["chosen_model_name"].split(" (")[0], "picked on validation months", P["lavender"])
with k2: ui.tile("Fraud caught at 5% FPR", f"{chosen['recall_at_5pct_fpr']:.1%}", f"95% CI {chosen['recall_ci_low']:.1%} to {chosen['recall_ci_high']:.1%}", P["pink"])
with k3: ui.tile("PR-AUC", f"{chosen['pr_auc']:.3f}", f"fraud rate is {metrics['fraud_rate']:.1%}", P["blue"])
with k4: ui.tile("Age fairness ratio", f"{metrics['fairness']['ratio']:.2f}", "1.0 = equal false positive rates", P["mint"])
with k5: ui.tile("Scoring time", f"{metrics['latency']['median_ms']:.0f} ms", "median, one applicant", P["peach"])
st.caption("Results are from the held-out test months (6 and 7) of the synthetic BAF Base dataset. They describe the method, not SASSA data.")

tab1, tab_perf, tab2, tab3, tab4 = st.tabs(["Model comparison", "Model performance", "Fairness by age", "Fraud over time", "Other BAF variants"])

with tab1:
    d = comp.sort_values("recall_at_5pct_fpr")
    fig = go.Figure(go.Bar(
        y=d["model_name"], x=d["recall_at_5pct_fpr"], orientation="h",
        marker=dict(color=[ui.P["lavender"] if m != metrics["chosen_model"] else ui.P["pink"] for m in d["model"]],
                    line=dict(color="#9a90b8", width=1)),
        error_x=dict(type="data", symmetric=False, array=d["recall_ci_high"] - d["recall_at_5pct_fpr"],
                     arrayminus=d["recall_at_5pct_fpr"] - d["recall_ci_low"], color="#8c86a8"),
        hovertemplate="%{y}<br>recall at 5% FPR: %{x:.1%}<extra></extra>"))
    fig.update_layout(title="Fraud caught when only 5% of genuine applicants may be flagged (test months)")
    fig.update_xaxes(tickformat=".0%", range=[0, max(0.7, d["recall_ci_high"].max() + 0.05)])
    st.plotly_chart(ui.style_fig(fig, 340, legend=False), use_container_width=True)

    # H2: best non-baseline model against the logistic regression baseline (paired bootstrap on the test months)
    h2 = metrics["h2"]
    diff, lo, hi = h2["difference"] * 100, h2["ci_low"] * 100, h2["ci_high"] * 100
    challenger = h2.get("challenger_name", metrics["chosen_model_name"])
    n_boot = h2.get("n_boot", 200)
    st.markdown(
        f"**Hypothesis H2.** Recall at 5% FPR for **{challenger}** minus the logistic regression baseline: "
        f"**{diff:+.1f} percentage points** (95% CI {lo:+.1f} to {hi:+.1f} points, {n_boot:,} paired bootstrap resamples). "
        + ("The interval is entirely above zero, so the null hypothesis is rejected."
           if h2["h0_rejected"] else
           "The interval includes zero, so no difference in recall could be detected and the null hypothesis is not rejected. "
           "This does not show that the models are equivalent."))
    show = comp[["model_name", "recall_at_5pct_fpr", "pr_auc", "roc_auc", "validation_recall_at_5pct_fpr"]].rename(columns={
        "model_name": "Model", "recall_at_5pct_fpr": "Recall at 5% FPR (test)", "pr_auc": "PR-AUC", "roc_auc": "ROC-AUC",
        "validation_recall_at_5pct_fpr": "Recall at 5% FPR (validation)"})
    st.dataframe(show.style.format({c: "{:.3f}" for c in show.columns[1:]}), hide_index=True, use_container_width=True)

with tab_perf:
    # These charts use the held-out test months. The curves and counts are saved by train_model.py.
    pr = state.load_table("curve_pr.csv")
    roc = state.load_table("curve_roc.csv")
    cm = state.load_table("confusion_matrix.csv")
    cal = state.load_table("calibration.csv")
    topk = state.load_table("topk_capacity.csv")
    if pr is None or cm is None:
        st.info("Run `python train_model.py` again to create the performance curves.")
    else:
        chosen_name = metrics["chosen_model_name"]
        # deeper pastel shades so thin lines can still be seen on a white background
        line_colours = ["#9B7FC7", "#F08CAE", "#6FB1E8", "#5CC9A7", "#F2A97C", "#E3C65A"]
        colour_for = {n: c for n, c in zip(comp["model_name"], line_colours)}
        legend_below = dict(orientation="h", y=-0.28, x=0)
        line_edge = "#8c86a8"

        left, right = st.columns(2, gap="large")
        with left:
            fig = go.Figure()
            for name, part in pr.groupby("model"):
                fig.add_scatter(x=part["recall"], y=part["precision"], mode="lines", name=name,
                                line=dict(color=colour_for.get(name, "#CDB4DB"), width=5 if name == chosen_name else 2.5))
            fig.add_hline(y=metrics["fraud_rate"], line_dash="dot", line_color=line_edge,
                          annotation_text="no skill (fraud rate)", annotation_position="top right")
            fig.update_layout(title="Precision-recall curves (best for rare fraud)")
            fig.update_xaxes(title="Recall (share of fraud caught)", tickformat=".0%")
            fig.update_yaxes(title="Precision", tickformat=".0%")
            fig.update_layout(legend=legend_below)
            st.plotly_chart(ui.style_fig(fig, 420), use_container_width=True)
        with right:
            fig = go.Figure()
            for name, part in roc.groupby("model"):
                fig.add_scatter(x=part["fpr"], y=part["tpr"], mode="lines", name=name,
                                line=dict(color=colour_for.get(name, "#CDB4DB"), width=5 if name == chosen_name else 2.5))
            fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="random guess",
                            line=dict(color=line_edge, dash="dot", width=1.5))
            fig.add_vline(x=config.FPR_HIGH, line_dash="dash", line_color=line_edge,
                          annotation_text="5% FPR", annotation_position="bottom right")
            fig.update_layout(title="ROC curves")
            fig.update_xaxes(title="False positive rate", tickformat=".0%")
            fig.update_yaxes(title="True positive rate (recall)", tickformat=".0%")
            fig.update_layout(legend=legend_below)
            st.plotly_chart(ui.style_fig(fig, 420), use_container_width=True)
        st.caption("The thick line is the chosen model. On rare fraud the ROC curve looks flattering, so the precision-recall curve is the more honest picture.")

        left, right = st.columns(2, gap="large")
        with left:
            level = st.radio("Risk level for the confusion matrix", ["High", "Medium"], horizontal=True)
            row = cm[cm["level"] == level].iloc[0]
            z = [[row["true_negative"], row["false_positive"]], [row["false_negative"], row["true_positive"]]]
            labels = [["Genuine, passed", "Genuine, flagged"], ["Fraud, missed", "Fraud, caught"]]
            text = [[f"{labels[i][j]}<br><b>{int(z[i][j]):,}</b>" for j in range(2)] for i in range(2)]
            # mint = correct decision, pink = mistake (colour is by cell type, not by size)
            kind = [[0, 1], [1, 0]]
            fig = go.Figure(go.Heatmap(z=kind, x=["Not flagged", "Flagged"], y=["Genuine", "Fraud"], text=text,
                                       texttemplate="%{text}", showscale=False, xgap=4, ygap=4,
                                       colorscale=[[0, ui.P["mint"]], [1, ui.P["pink"]]]))
            fig.update_layout(title=f"Confusion matrix at the {level} threshold")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(ui.style_fig(fig, 360, legend=False), use_container_width=True)
            st.caption(f"Recall {row['recall']:.1%}, precision {row['precision']:.1%}. "
                       f"{row['flagged_share']:.1%} of all applications go to a person.")
        with right:
            fig = go.Figure()
            top = max(cal["predicted"].max(), cal["observed"].max()) * 1.1
            fig.add_scatter(x=[0, top], y=[0, top], mode="lines", name="perfect calibration",
                            line=dict(color=line_edge, dash="dot"))
            fig.add_scatter(x=cal["predicted"], y=cal["observed"], mode="lines+markers", name="model (deciles)",
                            line=dict(color="#B79CD6", width=3),
                            marker=dict(size=10, color=ui.P["pink"], line=dict(color=line_edge, width=1)))
            fig.update_layout(title=f"Calibration (Brier score {metrics['brier_test']:.4f})")
            fig.update_xaxes(title="Predicted fraud probability", tickformat=".1%")
            fig.update_yaxes(title="Observed fraud rate", tickformat=".1%")
            st.plotly_chart(ui.style_fig(fig, 360), use_container_width=True)

        fig = go.Figure(go.Bar(
            x=[f"Top {k:.0%}" for k in topk["top_share"]], y=topk["recall"],
            marker=dict(color=ui.P["mint"], line=dict(color=line_edge, width=1)),
            text=[f"{r:.0%} caught<br>lift {l:.0f}x" for r, l in zip(topk["recall"], topk["lift"])], textposition="outside",
            hovertemplate="%{x}<br>fraud caught: %{y:.1%}<extra></extra>"))
        fig.update_layout(title="If investigators can only check the highest-scored applications")
        fig.update_yaxes(title="Share of all fraud found", tickformat=".0%", range=[0, topk["recall"].max() * 1.25])
        st.plotly_chart(ui.style_fig(fig, 340, legend=False), use_container_width=True)
        st.caption("Lift is how many times better than checking applications at random.")

with tab2:
    fa = state.load_table("fairness_age.csv")
    fm = state.load_table("fairness_age_mitigated.csv")
    fig = go.Figure()
    fig.add_bar(x=fa["group"], y=fa["FPR"], name="One threshold", marker=dict(color=ui.P["pink"], line=dict(color="#9a90b8", width=1)))
    fig.add_bar(x=fm["group"], y=fm["FPR"], name="Separate thresholds (mitigation)", marker=dict(color=ui.P["mint"], line=dict(color="#9a90b8", width=1)))
    fig.update_layout(barmode="group", title="False positive rate by age group (genuine applicants wrongly flagged)")
    fig.update_yaxes(tickformat=".1%")
    st.plotly_chart(ui.style_fig(fig, 340), use_container_width=True)

    f = metrics["fairness"]
    text = (f"Predictive equality ratio with one threshold: **{f['ratio']:.2f}** "
            f"(95% CI {f['ratio_ci'][0]:.2f} to {f['ratio_ci'][1]:.2f}). ")
    if "ratio_mitigated_ci" in f:
        text += (f"With separate thresholds it becomes **{f['ratio_mitigated']:.2f}** "
                 f"(95% CI {f['ratio_mitigated_ci'][0]:.2f} to {f['ratio_mitigated_ci'][1]:.2f}). "
                 f"The share of fraud caught changes from {f['recall_global_threshold']:.1%} to {f['recall_group_thresholds']:.1%} "
                 f"({f['recall_change'] * 100:+.1f} points, 95% CI {f['recall_change_ci'][0] * 100:+.1f} to "
                 f"{f['recall_change_ci'][1] * 100:+.1f}). ")
    else:
        text += (f"With separate thresholds it becomes **{f['ratio_mitigated']:.2f}**, and the share of fraud caught changes from "
                 f"{f['recall_global_threshold']:.1%} to {f['recall_group_thresholds']:.1%}. ")
    text += (f"Leaving customer age out of the model gives a ratio of {metrics['without_age']['predictive_equality_ratio']:.2f}.")
    st.markdown(text)
    st.caption("The separate-threshold result is exploratory. Its interval is wide, so it shows that the gap can be narrowed "
               "but not exactly how much recall changes. Using age to set thresholds also needs a legal and policy review before any real use.")

with tab3:
    bm = state.load_table("fraud_by_month.csv")
    fig = go.Figure(go.Scatter(x=bm["month"], y=bm["fraud_rate"], mode="lines+markers",
                               line=dict(color="#B79CD6", width=3), marker=dict(size=10, color=ui.P["pink"], line=dict(color="#8c86a8", width=1)),
                               hovertemplate="Month %{x}<br>fraud rate %{y:.2%}<extra></extra>"))
    fig.update_layout(title="Fraud rate by month (it changes over time, which is why I use a time-based split)")
    fig.update_yaxes(tickformat=".1%")
    fig.update_xaxes(title="Month (0 to 7)")
    st.plotly_chart(ui.style_fig(fig, 320, legend=False), use_container_width=True)

with tab4:
    var = state.load_table("variants_summary.csv")
    if var is None:
        st.info("Run `python run_variants.py` to compare the Base file with Variants I to V.")
    else:
        st.dataframe(var.style.format({c: "{:.3f}" for c in var.columns[1:] if var[c].dtype.kind == "f"}), hide_index=True, use_container_width=True)
        st.caption("Same method on each BAF file. Variants I-V add different kinds of bias or data shift (see Chapter 3).")

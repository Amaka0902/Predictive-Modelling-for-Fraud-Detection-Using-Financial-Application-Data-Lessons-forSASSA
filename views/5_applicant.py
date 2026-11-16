from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from src import explain, state, ui

ui.hero("Applicant Details", "One applicant: the prediction, the risk band, the probability and the reasons.")
ui.need_results()
res = st.session_state["results"]
model = state.load_model()

ids = res["Applicant ID"].tolist()
default = st.session_state.get("selected_id", ids[0])
idx = ids.index(default) if default in ids else 0
chosen = st.selectbox("Applicant ID", ids, index=idx)
st.session_state["selected_id"] = chosen
row = res[res["Applicant ID"] == chosen].iloc[0]
risk = row["Risk"]
prob = float(row["Fraud Score"])

left, right = st.columns([2, 3], gap="large")
with left:
    # most scores are small (fraud is about 1%), so the dial only goes up to 40% (or higher for a very risky applicant)
    top = max(0.4, min(1.0, prob * 1.2))
    t_med, t_high = model.thresholds["medium"], model.thresholds["high"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100, number=dict(suffix="%", font=dict(color=ui.P["text"])),
        gauge=dict(axis=dict(range=[0, top * 100], tickcolor="#b8b0d0"), bar=dict(color="#6F5AA8", thickness=0.22),
                   bgcolor="white", borderwidth=0,
                   steps=[dict(range=[0, t_med * 100], color=config.RISK_COLOURS["Low"]),
                          dict(range=[t_med * 100, t_high * 100], color=config.RISK_COLOURS["Medium"]),
                          dict(range=[t_high * 100, top * 100], color=config.RISK_COLOURS["High"])])))
    st.plotly_chart(ui.style_fig(fig, 280, legend=False), use_container_width=True)
    st.markdown(f"<div style='text-align:center'>{ui.pill(risk + ' risk', config.RISK_COLOURS[risk])}</div>", unsafe_allow_html=True)

with right:
    pct = (res["Fraud Score"] < prob).mean()
    c1, c2, c3 = st.columns(3)
    with c1: ui.tile("Prediction", "Likely fraud" if risk == "High" else ("Needs checking" if risk == "Medium" else "Likely genuine"), "", config.RISK_COLOURS[risk])
    with c2: ui.tile("Risk", risk, "band", config.RISK_COLOURS[risk])
    with c3: ui.tile("Probability", f"{prob:.1%}", f"riskier than {pct:.0%} of this file", ui.P["blue"])
    actions = {"High": "Refer to a human reviewer for manual verification. Do **not** reject automatically.",
               "Medium": "Ask for additional verification, then decide.",
               "Low": "Continue with standard processing."}
    st.markdown(f"**Recommended action:** {actions[risk]}")
    if "fraud_bool" in res.columns:
        st.markdown(f"**Recorded outcome in the file:** {'fraud' if row['fraud_bool'] == 1 else 'not fraud'}")

st.markdown("#### Factors pushing this score up the most")
try:
    X, values, base = explain.shap_for(model, res[res["Applicant ID"] == chosen][config.REQUIRED_COLUMNS], max_rows=1)
    reasons = explain.top_reasons(values[0], list(X.columns), 3)
    if reasons:
        for name, v in reasons:
            st.markdown(f"- {name}  <span class='small'>(pushes the score up)</span>", unsafe_allow_html=True)
    else:
        st.write("No feature pushed this applicant's score up noticeably.")
    st.caption("Open the Explainability page for the full waterfall plot.")
except Exception as e:
    st.caption(f"Reasons are not available: {e}")

st.markdown("#### Application details")
show = row[config.REQUIRED_COLUMNS].to_frame("Value").reset_index().rename(columns={"index": "Feature"})
show["Value"] = show["Value"].astype(str)
a, b = st.columns(2)
a.dataframe(show.iloc[:15], hide_index=True, use_container_width=True)
b.dataframe(show.iloc[15:], hide_index=True, use_container_width=True)

st.markdown("#### Reviewer outcome")
with st.form("review"):
    outcome = st.selectbox("Outcome after manual review", ["Not reviewed yet", "Confirmed fraud", "Not fraud", "Needs more information"])
    note = st.text_area("Notes (do not type names or ID numbers)")
    if st.form_submit_button("Save outcome"):
        path = config.OUTPUT_DIR / "reviews.csv"
        new = pd.DataFrame([{"time": datetime.now().isoformat(timespec="seconds"), "applicant_id": chosen, "score": prob,
                             "risk": risk, "outcome": outcome, "note": note, "model_version": model.info["version"]}])
        new.to_csv(path, mode="a", header=not path.exists(), index=False)
        st.success("Saved to outputs/reviews.csv. These outcomes can be used later to check and retrain the model.")

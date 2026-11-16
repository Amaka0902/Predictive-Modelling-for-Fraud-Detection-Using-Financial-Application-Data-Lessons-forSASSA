import streamlit as st

import config
from src import ui

ui.hero("Prediction Results", "Every applicant with a fraud score and a risk band. Select a row to open the applicant.")
ui.need_results()
res = st.session_state["results"]
model_thr = None

counts = res["Risk"].value_counts()
c1, c2, c3, c4 = st.columns(4)
with c1: ui.tile("Applicants", f"{len(res):,}", "scored", ui.P["lavender"])
with c2: ui.tile("High risk", f"{counts.get('High', 0):,}", f"{counts.get('High', 0) / len(res):.1%} of applicants", config.RISK_COLOURS["High"])
with c3: ui.tile("Medium risk", f"{counts.get('Medium', 0):,}", f"{counts.get('Medium', 0) / len(res):.1%} of applicants", config.RISK_COLOURS["Medium"])
with c4: ui.tile("Low risk", f"{counts.get('Low', 0):,}", f"{counts.get('Low', 0) / len(res):.1%} of applicants", config.RISK_COLOURS["Low"])

st.write("")
f1, f2, f3 = st.columns([2, 2, 2])
risk_pick = f1.multiselect("Risk band", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
low, high = f2.slider("Fraud score range", 0.0, 1.0, (0.0, 1.0), 0.01)
search = f3.text_input("Search Applicant ID")

view = res[res["Risk"].isin(risk_pick) & res["Fraud Score"].between(low, high)]
if search:
    view = view[view["Applicant ID"].str.contains(search.strip(), na=False)]
view = view.sort_values("Fraud Score", ascending=False)

show = view[["Applicant ID", "Fraud Score", "Risk", config.AGE, "income", "employment_status", "housing_status"]].rename(
    columns={config.AGE: "Age", "income": "Income (percentile)", "employment_status": "Employment", "housing_status": "Housing"})
colour = lambda v: f"background-color: {config.RISK_COLOURS.get(v, 'white')}; font-weight: 600"
styled = show.head(2000).style.map(colour, subset=["Risk"]).format({"Fraud Score": "{:.2f}", "Income (percentile)": "{:.1f}"})
st.caption(f"{len(view):,} applicants match. Showing the first 2,000.")
event = st.dataframe(styled, hide_index=True, use_container_width=True, height=430, on_select="rerun", selection_mode="single-row",
                     column_config={"Fraud Score": st.column_config.ProgressColumn("Fraud Score", min_value=0, max_value=1, format="%.2f")})

rows = event.selection.rows if event and event.selection else []
if rows:
    chosen = show.iloc[rows[0]]["Applicant ID"]
    st.session_state["selected_id"] = chosen
    st.success(f"Selected applicant **{chosen}**")
    if st.button("Open applicant details"):
        st.switch_page("views/5_applicant.py")

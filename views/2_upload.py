import pandas as pd
import streamlit as st

import config
from src import ui

ui.hero("Upload Dataset", "Upload a CSV or Excel file with one row per application.")

model_ok = (config.MODEL_DIR / "fraud_model.joblib").exists()
if not model_ok:
    st.warning("The model has not been trained yet, so predictions will not work. Run `python train_model.py` first.")

left, right = st.columns([3, 2], gap="large")
with left:
    ui.card("Choose a file", "<p>Accepted: <b>.csv</b>, <b>.xlsx</b>, <b>.xls</b>. The file should have the BAF columns "
            "(you can map different column names on the next page). Do not upload real personal data to a shared copy of this app.</p>")
    uploaded = st.file_uploader("Drag and drop a file here", type=["csv", "xlsx", "xls"], label_visibility="collapsed")
    sheet = None
    if uploaded is not None and uploaded.name.lower().endswith(("xlsx", "xls")):
        names = pd.ExcelFile(uploaded).sheet_names
        sheet = st.selectbox("Excel sheet", names) if len(names) > 1 else names[0]

with right:
    ui.card("No file handy?", "<p>Use the small sample (3,000 applications from the BAF test months) to try the dashboard.</p>")
    use_sample = st.button("Load the sample file")

df = None
name = None
if use_sample:
    df = pd.read_csv(config.ROOT / "sample_data" / "sample_applications.csv")
    name = "sample_applications.csv"
elif uploaded is not None:
    try:
        df = pd.read_csv(uploaded) if uploaded.name.lower().endswith("csv") else pd.read_excel(uploaded, sheet_name=sheet)
        name = uploaded.name
    except Exception as e:
        st.error(f"I could not read that file: {e}")

if df is not None:
    for key in ("results", "mapping_done", "shap_cache"):
        st.session_state.pop(key, None)
    st.session_state["raw_df"] = df
    st.session_state["file_name"] = name

if "raw_df" in st.session_state:
    d = st.session_state["raw_df"]
    st.success(f"Loaded **{st.session_state['file_name']}**: {len(d):,} rows and {d.shape[1]} columns. Go to **Dataset Validation** next.")
    st.markdown("#### Preview")
    st.dataframe(d.head(20), use_container_width=True)

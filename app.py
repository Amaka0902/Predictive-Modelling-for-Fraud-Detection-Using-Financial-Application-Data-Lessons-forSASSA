"""Streamlit dashboard. Start it with:  streamlit run app.py"""
import streamlit as st

from src import ui

st.set_page_config(page_title="Fraud Detection Dashboard", page_icon=":material/shield:", layout="wide")
ui.apply_theme()

pages = [
    st.Page("views/1_overview.py", title="Overview", icon=":material/home:", default=True),
    st.Page("views/2_upload.py", title="Upload Dataset", icon=":material/upload_file:"),
    st.Page("views/3_validation.py", title="Dataset Validation", icon=":material/fact_check:"),
    st.Page("views/4_results.py", title="Prediction Results", icon=":material/table_chart:"),
    st.Page("views/5_applicant.py", title="Applicant Details", icon=":material/person_search:"),
    st.Page("views/6_explainability.py", title="Explainability", icon=":material/psychology:"),
    st.Page("views/7_analytics.py", title="Analytics Dashboard", icon=":material/insights:"),
    st.Page("views/8_download.py", title="Download", icon=":material/download:"),
]
nav = st.navigation(pages)

with st.sidebar:
    st.markdown("### Fraud screening prototype")
    st.caption("Decision support only: no application is rejected automatically.")
    if "results" in st.session_state:
        st.success(f"{len(st.session_state['results']):,} applicants scored")

nav.run()

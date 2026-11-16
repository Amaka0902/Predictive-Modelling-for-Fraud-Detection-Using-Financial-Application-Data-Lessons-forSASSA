import pandas as pd
import streamlit as st

import config
from src import explain, reports, state, ui

ui.hero("Download", "Generate the Fraud Investigation Report (PDF), an Excel workbook, or the raw CSV.")
ui.need_results()
res = st.session_state["results"]
model = state.load_model()

counts = res["Risk"].value_counts()
summary = {
    "Applicants scored": f"{len(res):,}",
    "High risk": f"{counts.get('High', 0):,} ({counts.get('High', 0) / len(res):.1%})",
    "Medium risk": f"{counts.get('Medium', 0):,} ({counts.get('Medium', 0) / len(res):.1%})",
    "Low risk": f"{counts.get('Low', 0):,} ({counts.get('Low', 0) / len(res):.1%})",
    "Average fraud score": f"{res['Fraud Score'].mean():.3f}",
    "Source file": st.session_state.get("file_name", "-"),
}
model_info = {"Model": model.info["nice_name"], "Version": model.info["version"],
              "High threshold": round(model.thresholds["high"], 4), "Medium threshold": round(model.thresholds["medium"], 4),
              "Training months": str(model.info["train_months"]), "Validation months": str(model.info["validation_months"])}

c1, c2 = st.columns([3, 2], gap="large")
with c1:
    ui.card("Report options", "<p>Choose how many of the highest risk applicants appear in the PDF and Excel high-risk sheet.</p>")
    top_n = st.slider("Number of high-risk applicants to list", 5, 50, 15)
    case_ref = st.text_input("Case reference (optional)")
    investigator = st.text_input("Prepared by (optional)")
with c2:
    ui.card("What you get", "<ul style='margin:0;padding-left:18px'><li><b>PDF</b>: summary, charts, top applicants with reasons, actions</li>"
            "<li><b>Excel</b>: 4 sheets (summary, all applicants, high risk, model info)</li><li><b>CSV</b>: every applicant with score and risk</li></ul>")


@st.cache_data(show_spinner=False)
def build_top(ids_key, n):
    top = res.sort_values("Fraud Score", ascending=False).head(n).copy()
    X, values, _ = explain.shap_for(model, top[config.REQUIRED_COLUMNS], max_rows=n)
    top["Main reasons"] = [", ".join(name for name, _ in explain.top_reasons(values[i], list(X.columns), 3)) or "no strong single reason"
                           for i in range(len(top))]
    return top[["Applicant ID", "Fraud Score", "Risk", "Main reasons"]].reset_index(drop=True)


top = build_top(st.session_state.get("file_name", "") + str(len(res)), top_n)
st.markdown("#### Preview of the high-risk list")
st.dataframe(top, hide_index=True, use_container_width=True)

st.markdown("#### Download")
d1, d2, d3 = st.columns(3)
stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
with d1:
    st.markdown("**PDF report**")
    if st.button("Generate PDF"):
        with st.spinner("Building the report..."):
            st.session_state["pdf"] = reports.to_pdf_bytes(res, summary, top, model.thresholds, model.info, case_ref, investigator)
    if "pdf" in st.session_state:
        st.download_button("Download PDF", st.session_state["pdf"], f"fraud_investigation_report_{stamp}.pdf", "application/pdf")
with d2:
    st.markdown("**Excel workbook**")
    if st.button("Generate Excel"):
        with st.spinner("Building the workbook..."):
            show = res.drop(columns=[c for c in res.columns if c.endswith("_missing")], errors="ignore")
            st.session_state["xlsx"] = reports.to_excel_bytes(show, summary, top, model_info)
    if "xlsx" in st.session_state:
        st.download_button("Download Excel", st.session_state["xlsx"], f"fraud_results_{stamp}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
with d3:
    st.markdown("**CSV file**")
    st.download_button("Download CSV", reports.to_csv_bytes(res), f"fraud_results_{stamp}.csv", "text/csv")

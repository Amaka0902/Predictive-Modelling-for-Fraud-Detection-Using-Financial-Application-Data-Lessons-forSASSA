import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from src import data, state, ui

ui.hero("Dataset Validation", "Check the file and tell the dashboard which column is which feature.")

if "raw_df" not in st.session_state:
    st.info("Upload a dataset first (page 2).")
    st.stop()

raw = st.session_state["raw_df"]
cols = list(raw.columns)

# ---------- feature mapping first, because the checks depend on it
auto = state.guess_mapping(cols)
NONE = "- not mapped -"
with st.expander("Feature mapping (open to change it)", expanded=bool(sum(v is None for v in auto.values()))):
    st.caption("The model needs 30 features. I matched them by name. If your file uses other column names, pick them here.")
    mapping = {}
    grid = st.columns(3)
    for i, feature in enumerate(config.REQUIRED_COLUMNS):
        default = auto[feature] if auto[feature] in cols else NONE
        choice = grid[i % 3].selectbox(feature, [NONE] + cols, index=([NONE] + cols).index(default), key=f"map_{feature}")
        mapping[feature] = None if choice == NONE else choice

    st.markdown("**Optional columns**")
    o1, o2, o3, o4 = st.columns(4)
    id_col = o1.selectbox("Applicant ID", [NONE] + cols, index=([NONE] + cols).index(next((c for c in cols if c.lower().replace(" ", "_") in ("applicant_id", "applicantid", "id")), NONE)))
    prov_col = o2.selectbox("Province", [NONE] + cols, index=([NONE] + cols).index(next((c for c in cols if c.lower() == "province"), NONE)))
    month_col = o3.selectbox("Month", [NONE] + cols, index=([NONE] + cols).index("month" if "month" in cols else NONE))
    label_col = o4.selectbox("Known fraud label", [NONE] + cols, index=([NONE] + cols).index("fraud_bool" if "fraud_bool" in cols else NONE))
    extras = {k: (None if v == NONE else v) for k, v in
              dict(id=id_col, province=prov_col, month=month_col, label=label_col).items()}

# validate on a renamed copy so the checks use the mapped names
renamed = raw.rename(columns={v: k for k, v in mapping.items() if v})
info = data.validate_dataset(renamed)
missing_features = [f for f, c in mapping.items() if c is None]

st.markdown("#### Summary of the file")
c1, c2, c3, c4 = st.columns(4)
with c1: ui.tile("Rows", f"{info['rows']:,}", "applications", ui.P["lavender"])
with c2: ui.tile("Columns", f"{info['columns']}", f"{30 - len(missing_features)} of 30 features found", ui.P["blue"])
with c3: ui.tile("Missing values", f"{info['missing_cells']:,}", "empty cells in the feature columns", ui.P["pink"])
with c4: ui.tile("Duplicate rows", f"{info['duplicate_rows']:,}", "exact copies", ui.P["mint"])

left, right = st.columns([3, 2], gap="large")
with left:
    st.markdown("#### Missing values by column")
    miss = renamed[[c for c in config.REQUIRED_COLUMNS if c in renamed.columns]].isna().sum()
    for col, n in info["minus_one_counts"].items():
        miss[col] = miss.get(col, 0) + n          # -1 means "not available" in BAF so I count it too
    miss = miss[miss > 0].sort_values()
    if miss.empty:
        st.success("No missing values found.")
    else:
        fig = go.Figure(go.Bar(y=miss.index, x=miss.values / len(renamed), orientation="h",
                               marker=dict(color=ui.P["peach"], line=dict(color="#b9a58f", width=1)),
                               hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
        fig.update_xaxes(tickformat=".0%", title="share of rows (empty or -1)")
        st.plotly_chart(ui.style_fig(fig, max(220, 26 * len(miss)), legend=False), use_container_width=True)
with right:
    st.markdown("#### Checks")
    if missing_features:
        st.error(f"{len(missing_features)} features are not mapped: {', '.join(missing_features[:6])}{'...' if len(missing_features) > 6 else ''}")
    else:
        st.success("All 30 model features are mapped.")
    for p in info["problems"]:
        st.warning(p)
    if not info["problems"] and not missing_features:
        st.success("No range or category problems found.")
    if extras["province"] is None:
        st.caption("No Province column: BAF does not have one, so the analytics page uses age, income, employment and housing instead.")

st.markdown("#### Feature mapping table")
status = pd.DataFrame({
    "Feature": config.REQUIRED_COLUMNS,
    "Your column": [mapping[f] or "-" for f in config.REQUIRED_COLUMNS],
    "Status": ["Matched" if mapping[f] == f else ("Mapped" if mapping[f] else "Missing") for f in config.REQUIRED_COLUMNS],
})
st.dataframe(status, hide_index=True, use_container_width=True, height=280)

fill = False
if missing_features:
    fill = st.checkbox("Fill the missing features with blanks anyway (the scores will be less reliable)")
ready = (not missing_features) or fill

if st.button("Run predictions", disabled=not ready or state.load_model() is None):
    with st.spinner("Scoring applications..."):
        state.run_predictions(raw, mapping, extras, fill)
    st.success("Done. Open **Prediction Results** in the sidebar.")
    st.balloons()
if state.load_model() is None:
    st.warning("Model file not found. Train the model first.")

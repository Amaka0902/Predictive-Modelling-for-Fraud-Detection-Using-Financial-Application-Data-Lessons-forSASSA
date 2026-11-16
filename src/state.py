"""Small helpers shared by the pages: loading the model, running predictions, writing the audit log."""
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import config
from src import data

REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve_artifact(base_dir: Path, subdir: str, name: str) -> Path:
    """Resolve artifact path across config directory and repo-relative fallbacks."""
    candidates = [
        Path(base_dir) / name,
        REPO_ROOT / subdir / name,
        REPO_ROOT / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


@st.cache_resource(show_spinner="Loading the fraud model...")
def load_model():
    path = _resolve_artifact(config.MODEL_DIR, "models", "fraud_model.joblib")
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_data
def load_json(name):
    path = _resolve_artifact(config.OUTPUT_DIR, "outputs", name)
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data
def load_table(name):
    path = _resolve_artifact(config.OUTPUT_DIR, "outputs", name)
    return pd.read_csv(path) if path.exists() else None


def guess_mapping(columns):
    """Match required features to uploaded columns: exact name first, then ignoring case/spaces."""
    clean = {c.lower().replace(" ", "_"): c for c in columns}
    mapping = {}
    for feature in config.REQUIRED_COLUMNS:
        mapping[feature] = feature if feature in columns else clean.get(feature.lower())
    return mapping


def band_age(age):
    return pd.cut(pd.to_numeric(age, errors="coerce"), [0, 29, 39, 49, 59, 200],
                  labels=["Under 30", "30-39", "40-49", "50-59", "60 and over"])


def band_income(income):
    return pd.cut(pd.to_numeric(income, errors="coerce"), [-0.01, 0.3, 0.6, 1.01],
                  labels=["Lower (up to 0.3)", "Middle (0.3-0.6)", "Higher (above 0.6)"])


def run_predictions(raw, mapping, extras, fill_missing):
    """Build the model input from the mapping, score it, and keep everything in session_state."""
    model = load_model()
    if model is None:
        raise FileNotFoundError("fraud_model.joblib was not found in models/ or config.MODEL_DIR.")

    frame = pd.DataFrame(index=raw.index)
    for feature in config.REQUIRED_COLUMNS:
        source = mapping.get(feature)
        if source is not None and source in raw.columns:
            frame[feature] = raw[source]
        elif feature in config.CATEGORICAL:
            frame[feature] = "other" if feature == "device_os" else "unknown"
        else:
            frame[feature] = np.nan

    probs = model.predict(frame)
    risk = model.risk_band(probs)

    id_col = extras.get("id")
    ids = raw[id_col].astype(str).values if id_col else np.arange(10001, 10001 + len(raw)).astype(str)
    results = pd.DataFrame({"Applicant ID": ids, "Fraud Score": np.round(probs, 4), "Risk": risk}, index=raw.index)
    out = pd.concat([results, frame], axis=1)
    for key, col in (("province", "Province"), ("month", "month"), ("label", "fraud_bool")):
        if extras.get(key):
            out[col] = raw[extras[key]].values
    out["Age band"] = band_age(out[config.AGE]).astype(str)
    out["Income band"] = band_income(out["income"]).astype(str)

    st.session_state["results"] = out.reset_index(drop=True)
    st.session_state.pop("shap_cache", None)
    write_audit(len(out), (risk == "High").sum(), model.info["version"])
    return out


def write_audit(n, high, version):
    """FR4: every scoring run is written to an audit log."""
    out_dir = _resolve_artifact(config.OUTPUT_DIR, "outputs", "").parent
    out_dir.mkdir(exist_ok=True)
    path = out_dir / "audit_log.csv"
    row = pd.DataFrame([{"time": datetime.now().isoformat(timespec="seconds"), "model_version": version,
                         "applications_scored": n, "high_risk": int(high)}])
    row.to_csv(path, mode="a", header=not path.exists(), index=False)

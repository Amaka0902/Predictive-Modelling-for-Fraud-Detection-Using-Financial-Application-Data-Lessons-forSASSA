"""Tests for the parts that must not break. Run with:  pytest -q"""
import io
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src import data, evaluation, fairness, reports, state

SAMPLE = pd.read_csv(config.ROOT / "sample_data" / "sample_applications.csv")


def test_prepare_has_no_target_or_month():
    X = data.prepare(SAMPLE)
    assert config.TARGET not in X.columns and config.MONTH not in X.columns
    assert len(X) == len(SAMPLE)


def test_prepare_columns_same_for_any_data():
    # even if a category is missing in the new data, the columns must be the same as in training
    a = data.prepare(SAMPLE)
    b = data.prepare(SAMPLE.head(5))
    assert list(a.columns) == list(b.columns)


def test_minus_one_becomes_missing():
    df = SAMPLE.head(3).copy()
    df["prev_address_months_count"] = [-1, 5, 7]
    X = data.prepare(df)
    assert np.isnan(X["prev_address_months_count"].iloc[0])
    assert X["prev_address_months_count_missing"].tolist() == [1, 0, 0]


def test_without_age_drops_age_column():
    assert config.AGE not in data.prepare(SAMPLE, use_age=False).columns


def test_time_split_has_no_overlap():
    df = pd.DataFrame({config.MONTH: np.repeat(np.arange(8), 10), "x": 1})
    tr, va, te = data.split_by_month(df)
    assert len(tr) + len(va) + len(te) == len(df)
    assert tr[config.MONTH].max() < va[config.MONTH].min() <= va[config.MONTH].max() < te[config.MONTH].min()


def test_validation_finds_missing_columns():
    info = data.validate_dataset(SAMPLE.drop(columns=["income"]))
    assert "income" in info["missing_columns"]


def test_recall_at_fpr():
    y = np.array([0] * 95 + [1] * 5)
    perfect = y.astype(float)
    assert evaluation.recall_at_fpr(y, perfect, 0.05) == 1.0
    rng = np.random.default_rng(0)
    y2 = (rng.random(20000) < 0.1).astype(int)
    assert abs(evaluation.recall_at_fpr(y2, rng.random(20000), 0.05) - 0.05) < 0.02


def test_group_thresholds_equalise_fpr():
    rng = np.random.default_rng(1)
    g = np.array(["A"] * 5000 + ["B"] * 5000)
    y = (rng.random(10000) < 0.05).astype(int)
    s = rng.random(10000) + 0.3 * (g == "B")          # group B gets higher scores -> more false positives
    one = fairness.group_report(y, s, g, evaluation.threshold_at_fpr(y, s, 0.05))
    fixed = fairness.group_report(y, s, g, fairness.group_thresholds(y, s, g, 0.05))
    assert fairness.predictive_equality_ratio(fixed) > fairness.predictive_equality_ratio(one)
    assert fairness.predictive_equality_ratio(fixed) > 0.8


def test_guess_mapping_ignores_case():
    cols = [c.upper() for c in config.REQUIRED_COLUMNS]
    m = state.guess_mapping(cols)
    assert m["income"] == "INCOME" and all(v is not None for v in m.values())


@pytest.fixture(scope="module")
def model():
    path = config.MODEL_DIR / "fraud_model.joblib"
    if not path.exists():
        pytest.skip("train the model first (python train_model.py)")
    return joblib.load(path)


def test_model_scores_are_probabilities(model):
    p = model.predict(SAMPLE)
    assert p.min() >= 0 and p.max() <= 1 and len(p) == len(SAMPLE)
    bands = model.risk_band(p)
    assert set(bands) <= {"Low", "Medium", "High"}
    assert 0.02 < (bands == "High").mean() < 0.12          # about 5% of genuine applicants are flagged


def test_model_ranks_fraud_higher(model):
    p = model.predict(SAMPLE)
    assert evaluation.recall_at_fpr(SAMPLE[config.TARGET].values, p) > 0.2


def test_thresholds_are_ordered(model):
    assert model.thresholds["medium"] < model.thresholds["high"]


def test_reports_build(model):
    p = model.predict(SAMPLE.head(300))
    res = SAMPLE.head(300).copy()
    res.insert(0, "Applicant ID", [str(i) for i in range(10001, 10301)])
    res.insert(1, "Fraud Score", p)
    res.insert(2, "Risk", model.risk_band(p))
    res["Age band"] = state.band_age(res[config.AGE]).astype(str)
    top = res.sort_values("Fraud Score", ascending=False).head(5)[["Applicant ID", "Fraud Score", "Risk"]].copy()
    top["Main reasons"] = "test reason"
    summary = {"Applicants scored": "300"}
    pdf = reports.to_pdf_bytes(res, summary, top, model.thresholds, model.info)
    assert pdf[:4] == b"%PDF"
    xlsx = reports.to_excel_bytes(res, summary, top, {"Model": "x"})
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(xlsx))
    assert wb.sheetnames == ["Summary", "All applicants", "High risk", "Model info"]
    assert reports.to_csv_bytes(res).startswith(b"Applicant ID")

"""Robustness check: the same LightGBM pipeline on Base and Variants I-V.
Run:  python run_variants.py     (takes about 10 minutes)"""
import json
import time

import joblib
import numpy as np
import pandas as pd

import config
from src import data, evaluation, fairness, modelling

DESCRIPTIONS = {
    "Base": "Sampled to best represent the original dataset",
    "Variant I": "Higher group size disparity than Base",
    "Variant II": "Higher prevalence disparity than Base",
    "Variant III": "Better separability for one of the groups",
    "Variant IV": "Higher prevalence disparity in train",
    "Variant V": "Better separability in train for one of the groups",
}

params = json.loads((config.OUTPUT_DIR / "metrics.json").read_text()) if (config.OUTPUT_DIR / "metrics.json").exists() else {}
best = joblib.load(config.MODEL_DIR / "fraud_model.joblib").info["lightgbm_settings"]

rows = []
for name in DESCRIPTIONS:
    t = time.time()
    df = data.load_baf(config.DATA_DIR / f"{name}.csv")
    train, valid, test = data.split_by_month(df)
    scorer = modelling.fit_model("lightgbm", data.prepare(train), train[config.TARGET].values, best)
    sv, st_ = scorer.score(data.prepare(valid)), scorer.score(data.prepare(test))
    yv, yt = valid[config.TARGET].values, test[config.TARGET].values
    thr = evaluation.threshold_at_fpr(yv, sv, config.FPR_HIGH)
    rep = fairness.group_report(yt, st_, data.age_group(test[config.AGE]), thr)
    m = evaluation.full_metrics(yt, st_, thr)
    rows.append({"Dataset": name.replace(" ", "_"), "What is different": DESCRIPTIONS[name], "fraud_rate": df[config.TARGET].mean(),
                 "share_50_plus": float((df[config.AGE] >= 50).mean()),
                 "recall_at_5pct_fpr": m["recall_at_5pct_fpr"], "pr_auc": m["pr_auc"],
                 "FPR_under_50": float(rep.loc[rep.group == "Under 50", "FPR"].iloc[0]),
                 "FPR_50_plus": float(rep.loc[rep.group == "50 and over", "FPR"].iloc[0]),
                 "predictive_equality_ratio": fairness.predictive_equality_ratio(rep)})
    print(name, f"{time.time() - t:.0f}s", rows[-1], flush=True)
    del df, train, valid, test
pd.DataFrame(rows).to_csv(config.OUTPUT_DIR / "variants_summary.csv", index=False)
print("saved variants_summary.csv")

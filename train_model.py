"""Train, evaluate and save the fraud model.

Run:  python train_model.py            (uses data/Base.csv)
      python train_model.py --quick    (small sample, only to check the code runs)

Steps follow Chapter 3 of my dissertation:
  1. EDA tests (H1)           2. time-based split           3. train all candidate models
  4. pick the best on VALIDATION months only               5. calibrate + risk thresholds on validation
  6. final report on the TEST months (used once)           7. fairness by age (50)
  8. save model, tables and a model card
"""
import argparse
import json
import time
from datetime import date

import joblib
import numpy as np
import pandas as pd
from scipy import stats

import config
from src import data, evaluation, explain, fairness, modelling

t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def eda_tests(df):
    """H1: is there a relationship between applicant characteristics and fraud?"""
    rows = []
    y = df[config.TARGET].values
    for col in config.NUMERIC:
        x = pd.to_numeric(df[col], errors="coerce").values
        ok = ~np.isnan(x)
        u, p = stats.mannwhitneyu(x[ok & (y == 1)], x[ok & (y == 0)])
        auc = u / ((ok & (y == 1)).sum() * (ok & (y == 0)).sum())     # 0.5 = no relationship
        rows.append({"variable": col, "test": "Mann-Whitney U", "statistic": u, "p_value": p,
                     "effect_size": abs(auc - 0.5) * 2, "effect_name": "|2*AUC-1|"})
    for col in config.CATEGORICAL:
        table = pd.crosstab(df[col], df[config.TARGET])
        chi2, p, dof, _ = stats.chi2_contingency(table)
        cramers_v = np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1)))
        rows.append({"variable": col, "test": "Chi-square", "statistic": chi2, "p_value": p,
                     "effect_size": cramers_v, "effect_name": "Cramer's V"})
    out = pd.DataFrame(rows).sort_values("effect_size", ascending=False)
    out["significant_5pct"] = out["p_value"] < 0.05
    return out


def paired_bootstrap(y, s_a, s_b, n=200, seed=1):
    """Same resamples for both models, so the difference in recall@5%FPR is a fair comparison."""
    rng = np.random.default_rng(seed)
    diffs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(y), len(y))
        diffs[i] = evaluation.recall_at_fpr(y[idx], s_a[idx]) - evaluation.recall_at_fpr(y[idx], s_b[idx])
    return diffs


def psi_reference(scores, bins=10):
    """Reference distribution of scores (validation months) for the drift check later."""
    edges = np.unique(np.quantile(scores, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    share = np.histogram(scores, bins=edges)[0] / len(scores)
    return edges.tolist(), share.tolist()


def main(path, quick, out_dir, model_dir):
    out_dir.mkdir(exist_ok=True, parents=True)
    model_dir.mkdir(exist_ok=True, parents=True)

    log(f"loading {path}")
    df = data.load_baf(path)
    if quick:
        df = df.sample(60000, random_state=1)
    log(f"rows: {len(df):,}   fraud rate: {df[config.TARGET].mean():.4f}")
    train, valid, test = data.split_by_month(df)
    log(f"train {len(train):,} | validation {len(valid):,} | test {len(test):,}")

    # ---------- 1. EDA (uses training + validation months only, never the test months)
    eda = eda_tests(pd.concat([train, valid]))
    eda.to_csv(out_dir / "eda_tests.csv", index=False)
    by_month = df.groupby(config.MONTH)[config.TARGET].agg(applications="size", fraud_rate="mean").reset_index()
    by_month.to_csv(out_dir / "fraud_by_month.csv", index=False)
    log("EDA tests saved. Significant variables: " + str(int(eda["significant_5pct"].sum())) + " of " + str(len(eda)))

    # ---------- 2. features
    Xtr, ytr = data.prepare(train), train[config.TARGET].values
    Xva, yva = data.prepare(valid), valid[config.TARGET].values
    Xte, yte = data.prepare(test), test[config.TARGET].values

    # ---------- 3. LightGBM settings: try a few and keep the best on validation
    best_params, best_score = None, -1
    for params in modelling.LIGHTGBM_GRID:
        m = modelling.fit_model("lightgbm", Xtr, ytr, params)
        r = evaluation.recall_at_fpr(yva, m.score(Xva))
        log(f"lightgbm {params} -> validation recall@5%FPR {r:.3f}")
        if r > best_score:
            best_params, best_score = params, r
    log(f"best LightGBM settings: {best_params}")

    # ---------- 4. train every model, score validation + test
    scorers, val_scores, test_scores = {}, {}, {}
    for name in modelling.MODEL_NAMES:
        log(f"training {name}")
        scorers[name] = modelling.fit_model(name, Xtr, ytr, best_params)
        val_scores[name] = scorers[name].score(Xva)
        test_scores[name] = scorers[name].score(Xte)
        log(f"   validation recall@5%FPR = {evaluation.recall_at_fpr(yva, val_scores[name]):.3f}")

    # ---------- 5. choose the model on VALIDATION only
    val_recall = {n: evaluation.recall_at_fpr(yva, s) for n, s in val_scores.items()}
    chosen = max(val_recall, key=val_recall.get)
    log(f"chosen model (validation): {chosen}")

    # ---------- 6. calibration + thresholds for the chosen model (validation months)
    calibrator = modelling.fit_calibrator(val_scores[chosen], yva)
    cal_val = calibrator.predict(val_scores[chosen])
    thresholds = modelling.make_thresholds(cal_val, yva)
    cal_test = calibrator.predict(test_scores[chosen])
    log(f"thresholds: {thresholds}")

    # ---------- 7. test results for all models (the test months are used once)
    rows = []
    for name in modelling.MODEL_NAMES:
        thr_valid = evaluation.threshold_at_fpr(yva, val_scores[name], config.FPR_HIGH)
        m = evaluation.full_metrics(yte, test_scores[name], thr_valid)
        boots = evaluation.bootstrap_recall(yte, test_scores[name], n=100 if quick else 200)
        lo, hi = evaluation.ci(boots)
        m.update({"model": name, "model_name": modelling.NICE_NAMES[name],
                  "recall_ci_low": lo, "recall_ci_high": hi, "validation_recall_at_5pct_fpr": val_recall[name]})
        rows.append(m)
        log(f"{name:20s} test recall@5%FPR {m['recall_at_5pct_fpr']:.3f} [{lo:.3f}, {hi:.3f}]  PR-AUC {m['pr_auc']:.3f}")
    comparison = pd.DataFrame(rows)
    comparison.to_csv(out_dir / "model_comparison.csv", index=False)

    # H2: chosen model against the logistic regression baseline (paired bootstrap)
    diffs = paired_bootstrap(yte, test_scores[chosen], test_scores["logistic_regression"], n=100 if quick else 200)
    h2 = {"chosen": chosen, "difference": float(np.mean(diffs)), "ci_low": evaluation.ci(diffs)[0],
          "ci_high": evaluation.ci(diffs)[1], "h0_rejected": bool(evaluation.ci(diffs)[0] > 0)}
    log(f"H2: recall@5%FPR difference vs baseline = {h2['difference']:.3f}  CI [{h2['ci_low']:.3f}, {h2['ci_high']:.3f}]")

    # calibration + top-k for the chosen model
    calib_table = evaluation.calibration_table(yte, cal_test)
    calib_table.to_csv(out_dir / "calibration.csv", index=False)
    evaluation.topk_table(yte, cal_test).to_csv(out_dir / "topk_capacity.csv", index=False)
    brier_score = evaluation.brier(yte, cal_test)

    # curves for every model (PR and ROC) and the confusion counts at the High / Medium thresholds
    pr_parts, roc_parts = [], []
    for name in modelling.MODEL_NAMES:
        pr_df, roc_df = evaluation.curve_points(yte, test_scores[name], modelling.NICE_NAMES[name])
        pr_parts.append(pr_df)
        roc_parts.append(roc_df)
    pd.concat(pr_parts).to_csv(out_dir / "curve_pr.csv", index=False)
    pd.concat(roc_parts).to_csv(out_dir / "curve_roc.csv", index=False)
    evaluation.confusion_table(yte, cal_test, {"High": thresholds["high"], "Medium": thresholds["medium"]}) \
        .to_csv(out_dir / "confusion_matrix.csv", index=False)
    log("curves and confusion matrix saved")

    # ---------- 8. fairness by age (chosen model, test months)
    g_test = data.age_group(test[config.AGE])
    g_valid = data.age_group(valid[config.AGE])
    rep = fairness.group_report(yte, cal_test, g_test, thresholds["high"])
    ratio = fairness.predictive_equality_ratio(rep)
    rep.to_csv(out_dir / "fairness_age.csv", index=False)
    boot_ratio = []
    rng = np.random.default_rng(3)
    for _ in range(100 if quick else 200):
        idx = rng.integers(0, len(yte), len(yte))
        r = fairness.group_report(yte[idx], cal_test[idx], g_test[idx], thresholds["high"])
        boot_ratio.append(fairness.predictive_equality_ratio(r))
    ratio_ci = evaluation.ci(boot_ratio)
    log(f"predictive equality ratio (age) = {ratio:.3f}  CI [{ratio_ci[0]:.3f}, {ratio_ci[1]:.3f}]")

    # mitigation: group specific thresholds chosen on validation so each group has 5% FPR
    g_thr = fairness.group_thresholds(yva, cal_val, g_valid, config.FPR_HIGH)
    rep_fixed = fairness.group_report(yte, cal_test, g_test, g_thr)
    rep_fixed.to_csv(out_dir / "fairness_age_mitigated.csv", index=False)
    ratio_fixed = fairness.predictive_equality_ratio(rep_fixed)
    flagged_fixed = np.array([cal_test[i] > g_thr[g_test[i]] for i in range(len(cal_test))]) if quick else \
        np.where(g_test == "50 and over", cal_test > g_thr["50 and over"], cal_test > g_thr["Under 50"])
    recall_fixed = float(flagged_fixed[yte == 1].mean())
    recall_global = float((cal_test[yte == 1] > thresholds["high"]).mean())
    log(f"mitigation: ratio {ratio:.3f} -> {ratio_fixed:.3f}, recall {recall_global:.3f} -> {recall_fixed:.3f}")

    # model without the age column (same settings), to see what age adds
    log("training chosen model type without customer_age")
    Xtr_na, Xva_na, Xte_na = (data.prepare(d, use_age=False) for d in (train, valid, test))
    sc_na = modelling.fit_model(chosen, Xtr_na, ytr, best_params)
    na_val, na_test = sc_na.score(Xva_na), sc_na.score(Xte_na)
    cal_na = modelling.fit_calibrator(na_val, yva)
    thr_na = modelling.make_thresholds(cal_na.predict(na_val), yva)
    rep_na = fairness.group_report(yte, cal_na.predict(na_test), g_test, thr_na["high"])
    no_age = {"recall_at_5pct_fpr": evaluation.recall_at_fpr(yte, na_test),
              "predictive_equality_ratio": fairness.predictive_equality_ratio(rep_na),
              "pr_auc": evaluation.full_metrics(yte, na_test)["pr_auc"]}
    log(f"without age: {no_age}")

    # ---------- 9. stability over time (expanding window on the training months)
    folds = [([0, 1, 2], [3]), ([0, 1, 2, 3], [4]), ([0, 1, 2, 3, 4], [5])]
    stab = []
    for name in ("logistic_regression", "lightgbm"):
        for tr_m, va_m in folds:
            a, b = df[df[config.MONTH].isin(tr_m)], df[df[config.MONTH].isin(va_m)]
            m = modelling.fit_model(name, data.prepare(a), a[config.TARGET].values, best_params)
            stab.append({"model": name, "train_months": f"{tr_m[0]}-{tr_m[-1]}", "validate_month": va_m[0],
                         "recall_at_5pct_fpr": evaluation.recall_at_fpr(b[config.TARGET].values, m.score(data.prepare(b)))})
    pd.DataFrame(stab).to_csv(out_dir / "stability_expanding_window.csv", index=False)
    log("expanding-window stability done")

    # ---------- 10. importance (SHAP on a test sample) and the final object
    edges, share = psi_reference(cal_val)
    info = {
        "name": chosen, "nice_name": modelling.NICE_NAMES[chosen], "version": f"1.0-{date.today().isoformat()}",
        "lightgbm_settings": best_params, "train_months": config.TRAIN_MONTHS, "validation_months": config.VALID_MONTHS,
        "test_months": config.TEST_MONTHS, "psi_edges": edges, "psi_reference_share": share,
        "feature_columns": list(Xtr.columns),
    }
    fraud_model = modelling.FraudModel(scorers[chosen], True, calibrator, thresholds, info)

    sample = test.sample(min(3000, len(test)), random_state=5)
    try:
        Xs, sv, _ = explain.shap_for(fraud_model, sample, max_rows=3000)
        imp = pd.DataFrame({"feature": Xs.columns, "label": [explain.nice(c) for c in Xs.columns],
                            "mean_abs_shap": np.abs(sv).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
        imp.to_csv(out_dir / "feature_importance_shap.csv", index=False)
        log("top features: " + ", ".join(imp["feature"].head(5)))
    except Exception as e:           # SHAP should not stop the training
        log(f"SHAP skipped: {e}")

    # speed for one applicant (non-functional requirement NFR1)
    one = test.iloc[[0]]
    fraud_model.predict(one)
    times = []
    for i in range(100):
        s = time.perf_counter()
        fraud_model.predict(test.iloc[[i]])
        times.append((time.perf_counter() - s) * 1000)
    latency = {"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95))}
    log(f"single applicant scoring time: {latency}")

    joblib.dump(fraud_model, model_dir / "fraud_model.joblib", compress=3)

    metrics = {
        "dataset": str(path.name), "rows": int(len(df)), "fraud_rate": float(df[config.TARGET].mean()),
        "train_rows": int(len(train)), "valid_rows": int(len(valid)), "test_rows": int(len(test)),
        "chosen_model": chosen, "chosen_model_name": modelling.NICE_NAMES[chosen],
        "thresholds": thresholds, "brier_test": brier_score, "h2": h2,
        "fairness": {"ratio": ratio, "ratio_ci": ratio_ci, "ratio_mitigated": ratio_fixed,
                     "recall_global_threshold": recall_global, "recall_group_thresholds": recall_fixed},
        "without_age": no_age, "latency": latency,
        "h1_significant": int(eda["significant_5pct"].sum()), "h1_tested": int(len(eda)),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_model_card(model_dir / "model_card.md", metrics, comparison, rep)
    log("all done")


def write_model_card(path, m, comparison, fairness_rep):
    c = comparison[comparison["model"] == m["chosen_model"]].iloc[0]
    text = f"""# Model card: {m['chosen_model_name']}

**Purpose.** Decision support for screening applications for fraud. It gives a risk score and a risk band.
It does not approve or reject anyone: High risk cases go to a human reviewer.

**Data.** Bank Account Fraud (BAF) dataset ({m['dataset']}), synthetic, {m['rows']:,} applications, fraud rate {m['fraud_rate']:.2%}.
Time based split: months 0-3 train, months 4-5 validation (model choice, calibration, thresholds), months 6-7 test.

**Test results (months 6-7).** Recall at 5% false positive rate: {c['recall_at_5pct_fpr']:.3f}
(95% CI {c['recall_ci_low']:.3f} to {c['recall_ci_high']:.3f}). PR-AUC {c['pr_auc']:.3f}. ROC-AUC {c['roc_auc']:.3f}. Brier score {m['brier_test']:.4f}.

**Risk bands.** High = calibrated score at or above {m['thresholds']['high']:.4f}; Medium = at or above {m['thresholds']['medium']:.4f}; otherwise Low.

**Fairness (age 50 and over vs under 50).** Predictive equality ratio {m['fairness']['ratio']:.2f}
(95% CI {m['fairness']['ratio_ci'][0]:.2f} to {m['fairness']['ratio_ci'][1]:.2f}); 1.0 means equal false positive rates.

**Limitations.** Synthetic bank data, not SASSA data. Labels are the dataset's own fraud flag (detected fraud). One protected attribute only.
Results must not be read as how the model would perform on real grant applications.
"""
    path.write_text(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(config.DATA_DIR / "Base.csv"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None, help="output folder (default: outputs/)")
    args = ap.parse_args()
    from pathlib import Path
    out = Path(args.out) if args.out else config.OUTPUT_DIR
    mdl = Path(args.out) / "models" if args.out else config.MODEL_DIR
    main(Path(args.data), args.quick, out, mdl)

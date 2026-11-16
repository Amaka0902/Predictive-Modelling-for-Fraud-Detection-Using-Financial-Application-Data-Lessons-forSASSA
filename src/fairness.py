"""Fairness check: predictive equality = similar false positive rates for under 50 and 50 and over."""
import numpy as np
import pandas as pd

from src import evaluation


def group_report(y, scores, groups, thresholds):
    """thresholds can be one number or a dict {group: threshold}."""
    y = np.asarray(y)
    scores = np.asarray(scores)
    groups = np.asarray(groups)
    rows = []
    for g in np.unique(groups):
        m = groups == g
        thr = thresholds[g] if isinstance(thresholds, dict) else thresholds
        r = evaluation.rates_at_threshold(y[m], scores[m], thr)
        rows.append({"group": g, "applications": int(m.sum()), "fraud_rate": float(y[m].mean()),
                     "TPR": r["TPR"], "FPR": r["FPR"], "flagged_share": r["flagged_share"],
                     "precision": r["precision"]})
    return pd.DataFrame(rows)


def predictive_equality_ratio(report):
    """smallest group FPR divided by the largest group FPR (1.0 = perfectly equal)."""
    fprs = report["FPR"]
    if len(fprs) < 2 or fprs.max() == 0:
        return float("nan")
    return float(fprs.min() / fprs.max())


def group_thresholds(y, scores, groups, fpr=0.05):
    """Mitigation: one threshold per group so every group has the same false positive rate (set on validation data)."""
    y = np.asarray(y)
    scores = np.asarray(scores)
    groups = np.asarray(groups)
    return {g: evaluation.threshold_at_fpr(y[groups == g], scores[groups == g], fpr) for g in np.unique(groups)}

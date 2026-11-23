"""Metrics. The main one is recall at a fixed 5% false positive rate (Jesus et al., 2022)."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def threshold_at_fpr(y, scores, fpr):
    """Score above which only `fpr` of the genuine applicants are flagged."""
    y = np.asarray(y)
    scores = np.asarray(scores)
    return float(np.quantile(scores[y == 0], 1 - fpr))


def recall_at_fpr(y, scores, fpr=0.05):
    """Best case version: the threshold is set on the same data (this is how the BAF paper reports it)."""
    y = np.asarray(y)
    scores = np.asarray(scores)
    thr = threshold_at_fpr(y, scores, fpr)
    return float((scores[y == 1] > thr).mean())


def rates_at_threshold(y, scores, thr):
    y = np.asarray(y)
    flagged = np.asarray(scores) > thr
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    fn = int((~flagged & (y == 1)).sum())
    tn = int((~flagged & (y == 0)).sum())
    return {
        "TPR": tp / max(tp + fn, 1),
        "FPR": fp / max(fp + tn, 1),
        "precision": tp / max(tp + fp, 1),
        "flagged_share": flagged.mean(),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def bootstrap_recall(y, scores, n=200, fpr=0.05, seed=42):
    """Bootstrap resamples of recall@fpr, returned as an array (used for the confidence interval)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    scores = np.asarray(scores)
    out = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(y), len(y))
        out[i] = recall_at_fpr(y[idx], scores[idx], fpr)
    return out


def ci(values):
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def full_metrics(y, scores, valid_threshold=None, fpr=0.05):
    """All headline measures for one model on one dataset."""
    out = {
        "recall_at_5pct_fpr": recall_at_fpr(y, scores, fpr),
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
    }
    if valid_threshold is not None:
        r = rates_at_threshold(y, scores, valid_threshold)
        out["TPR_at_valid_threshold"] = r["TPR"]
        out["FPR_at_valid_threshold"] = r["FPR"]
        out["precision_at_valid_threshold"] = r["precision"]
    return out


def brier(y, probs):
    return float(brier_score_loss(y, probs))


def calibration_table(y, probs, bins=10):
    """Predicted vs observed fraud rate in score deciles."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(probs)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    return df.groupby("bin").agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size")).reset_index()


def topk_table(y, scores, ks=(0.01, 0.02, 0.05, 0.10)):
    """If investigators can only check the top k% of applications, how many frauds do they find?"""
    y = np.asarray(y)
    order = np.argsort(-np.asarray(scores))
    base_rate = y.mean()
    rows = []
    for k in ks:
        top = y[order[: int(len(y) * k)]]
        rows.append({"top_share": k, "precision": top.mean(), "recall": top.sum() / y.sum(),
                     "lift": top.mean() / base_rate})
    return pd.DataFrame(rows)


def curve_points(y, scores, model, max_points=400):
    """Points for the precision-recall and ROC curves of one model.
    A test set of 200,000 rows gives a huge number of points, so we keep an even sample of them."""
    from sklearn.metrics import precision_recall_curve, roc_curve
    y = np.asarray(y)
    scores = np.asarray(scores)

    precision, recall, _ = precision_recall_curve(y, scores)
    fpr, tpr, _ = roc_curve(y, scores)

    def thin(*arrays):
        idx = np.unique(np.linspace(0, len(arrays[0]) - 1, max_points).astype(int))
        return [a[idx] for a in arrays]

    p, r = thin(precision, recall)
    f, t = thin(fpr, tpr)
    pr = pd.DataFrame({"model": model, "recall": r, "precision": p})
    roc = pd.DataFrame({"model": model, "fpr": f, "tpr": t})
    return pr, roc


def confusion_table(y, scores, thresholds):
    """Counts of caught frauds, missed frauds, false alarms and correct passes at each risk threshold."""
    rows = []
    for label, thr in thresholds.items():
        r = rates_at_threshold(y, scores, thr)
        rows.append({"level": label, "threshold": thr, "true_positive": r["tp"], "false_positive": r["fp"],
                     "false_negative": r["fn"], "true_negative": r["tn"],
                     "recall": r["TPR"], "precision": r["precision"], "flagged_share": r["flagged_share"]})
    return pd.DataFrame(rows)

def holm_adjust(p_values):
    """Holm (1979) step-down adjusted p-values. Controls the family-wise error rate.
    NaN p-values (a variable that cannot be tested) are left as NaN and not counted."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    order = np.argsort(p[ok])
    running = 0.0
    adj_ok = np.empty(m)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[ok][i])   # smallest p is multiplied by m, next by m-1, ...
        adj_ok[i] = min(1.0, running)
    adjusted[ok] = adj_ok
    return adjusted

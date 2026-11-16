"""The models I compare, and the FraudModel object that the dashboard loads."""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import config
from src import data

MODEL_NAMES = ["logistic_regression", "random_forest", "lightgbm", "xgboost", "isolation_forest", "hybrid"]
NICE_NAMES = {
    "logistic_regression": "Logistic regression (baseline)",
    "random_forest": "Random forest",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "isolation_forest": "Isolation Forest (unsupervised)",
    "hybrid": "Hybrid (anomaly score + LightGBM)",
}

#a few LightGBM settings to try; I pick the best one on the VALIDATION months
LIGHTGBM_GRID = [
    dict(num_leaves=15, learning_rate=0.05, n_estimators=300, scale_pos_weight=1),
    dict(num_leaves=31, learning_rate=0.05, n_estimators=300, scale_pos_weight=1),
    dict(num_leaves=31, learning_rate=0.05, n_estimators=300, scale_pos_weight=10),
    dict(num_leaves=63, learning_rate=0.03, n_estimators=400, scale_pos_weight=10),
]


def make_lightgbm(params=None):
    p = dict(num_leaves=31, learning_rate=0.05, n_estimators=300, scale_pos_weight=10)
    p.update(params or {})
    return LGBMClassifier(min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                          random_state=config.SEED, n_jobs=2, verbose=-1, **p)


def make_isolation_forest():
    return make_pipeline(SimpleImputer(strategy="median"),
                         IsolationForest(n_estimators=200, max_samples=1024, random_state=config.SEED, n_jobs=2))


class Scorer:
    """One trained model with a .score(X) method that gives a higher number for riskier applications."""

    def __init__(self, name, clf, iso=None):
        self.name = name
        self.clf = clf
        self.iso = iso          #only used by the hybrid model

    def _anomaly(self, X):
        return -self.iso.score_samples(X)

    def model_input(self, X):
        """The matrix the classifier really sees (hybrid = features + anomaly score)."""
        if self.name == "hybrid":
            X = X.copy()
            X["anomaly_score"] = self._anomaly(X).astype("float32")
        return X

    def score(self, X):
        if self.name == "isolation_forest":
            return self._anomaly(X)
        return self.clf.predict_proba(self.model_input(X))[:, 1]


def fit_model(name, X, y, lgbm_params=None):
    """Train one model. Class imbalance is handled with class weights (not SMOTE)."""
    if name == "logistic_regression":
        clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                            LogisticRegression(C=0.1, class_weight="balanced", max_iter=500))
        return Scorer(name, clf.fit(X, y))
    if name == "random_forest":
        clf = RandomForestClassifier(n_estimators=120, max_depth=14, min_samples_leaf=30, max_features="sqrt",
                                     class_weight="balanced_subsample", n_jobs=2, random_state=config.SEED)
        return Scorer(name, clf.fit(X, y))
    if name == "lightgbm":
        return Scorer(name, make_lightgbm(lgbm_params).fit(X, y))
    if name == "xgboost":
        clf = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                            scale_pos_weight=10, tree_method="hist", n_jobs=2, random_state=config.SEED)
        return Scorer(name, clf.fit(X, y))
    if name == "isolation_forest":
        #unsupervised: it never sees the labels
        return _fit_iso_only(X)
    if name == "hybrid":
        iso = make_isolation_forest().fit(X)
        scorer = Scorer(name, None, iso=iso)
        scorer.clf = make_lightgbm(lgbm_params).fit(scorer.model_input(X), y)
        return scorer
    raise ValueError(name)


def _fit_iso_only(X):
    iso = make_isolation_forest().fit(X)
    return Scorer("isolation_forest", None, iso=iso)


class FraudModel:
    """Everything the dashboard needs in one object: the model, the calibration and the risk thresholds."""

    def __init__(self, scorer, use_age, calibrator, thresholds, info):
        self.scorer = scorer
        self.use_age = use_age
        self.calibrator = calibrator       #turns the raw score into a probability
        self.thresholds = thresholds       #{"medium": ..., "high": ...} on the calibrated score
        self.info = info                   #name, version, training months, etc

    def features(self, raw_df):
        return data.prepare(raw_df, use_age=self.use_age)

    def predict(self, raw_df):
        X = self.features(raw_df)
        raw = self.scorer.score(X)
        return self.calibrator.predict(raw)

    def risk_band(self, probs):
        probs = np.asarray(probs)
        band = np.where(probs >= self.thresholds["high"], "High",
                        np.where(probs >= self.thresholds["medium"], "Medium", "Low"))
        return band


def fit_calibrator(raw_scores, y):
    """Isotonic regression: maps a score to the observed fraud rate in the validation months."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    return iso.fit(raw_scores, y)


def make_thresholds(cal_scores, y):
    """High = flags 5% of genuine applicants, Medium = 15% (measured on the validation months)."""
    genuine = np.asarray(cal_scores)[np.asarray(y) == 0]
    return {"high": float(np.quantile(genuine, 1 - config.FPR_HIGH)),
            "medium": float(np.quantile(genuine, 1 - config.FPR_MEDIUM))}

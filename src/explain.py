"""SHAP explanations and plain-language reasons."""
import numpy as np
import pandas as pd
import shap
from sklearn.impute import SimpleImputer

LABELS = {
    "income": "Applicant income (percentile)",
    "name_email_similarity": "Name and email similarity",
    "prev_address_months_count": "Months at previous address",
    "current_address_months_count": "Months at current address",
    "customer_age": "Customer age",
    "days_since_request": "Days since the request",
    "intended_balcon_amount": "Intended balance transfer amount",
    "zip_count_4w": "Applications from the same zip (4 weeks)",
    "velocity_6h": "Application velocity (6 hours)",
    "velocity_24h": "Application velocity (24 hours)",
    "velocity_4w": "Application velocity (4 weeks)",
    "bank_branch_count_8w": "Applications at the branch (8 weeks)",
    "date_of_birth_distinct_emails_4w": "Emails sharing the date of birth (4 weeks)",
    "credit_risk_score": "Credit risk score",
    "email_is_free": "Free email domain",
    "phone_home_valid": "Home phone valid",
    "phone_mobile_valid": "Mobile phone valid",
    "bank_months_count": "Months with previous bank account",
    "has_other_cards": "Has other cards",
    "proposed_credit_limit": "Proposed credit limit",
    "foreign_request": "Request from a foreign country",
    "session_length_in_minutes": "Session length (minutes)",
    "keep_alive_session": "Keep-alive session",
    "device_distinct_emails_8w": "Distinct emails on the device (8 weeks)",
    "device_fraud_count": "Earlier fraud on the device",
    "anomaly_score": "Anomaly score (Isolation Forest)",
    "prev_address_months_count_missing": "No previous address recorded",
    "bank_months_count_missing": "No previous bank account recorded",
    "intended_balcon_negative": "Intended balance amount is negative",
}


def nice(name):
    if name in LABELS:
        return LABELS[name]
    for prefix in ("payment_type", "employment_status", "housing_status", "source", "device_os"):
        if name.startswith(prefix + "_"):
            return f"{prefix.replace('_', ' ').capitalize()} = {name[len(prefix) + 1:]}"
    return name


def shap_for(fraud_model, raw_df, max_rows=2000, seed=42):
    """Returns (matrix shown to the model, shap values, base value). Values are in log-odds."""
    if len(raw_df) > max_rows:
        raw_df = raw_df.sample(max_rows, random_state=seed)
    X = fraud_model.features(raw_df)
    scorer = fraud_model.scorer
    Xm = scorer.model_input(X)
    clf = scorer.clf
    if scorer.name == "logistic_regression":
        imputer, scaler, lr = clf.steps[0][1], clf.steps[1][1], clf.steps[2][1]
        Xs = pd.DataFrame(scaler.transform(imputer.transform(Xm)), columns=Xm.columns, index=Xm.index)
        explainer = shap.LinearExplainer(lr, Xs)
        values = explainer.shap_values(Xs)
        return Xm, np.asarray(values), float(np.ravel(explainer.expected_value)[0])
    explainer = shap.TreeExplainer(clf)
    values = explainer.shap_values(Xm)
    if isinstance(values, list):          # older shap versions give one array per class
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]
    base = float(np.ravel(explainer.expected_value)[-1])
    return Xm, values, base


def top_reasons(shap_row, columns, n=3):
    """The features that pushed this applicant's score UP the most (these are the 'reason codes')."""
    order = np.argsort(-shap_row)[:n]
    return [(nice(columns[i]), float(shap_row[i])) for i in order if shap_row[i] > 0]

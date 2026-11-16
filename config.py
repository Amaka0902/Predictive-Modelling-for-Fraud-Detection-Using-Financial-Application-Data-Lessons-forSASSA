"""Settings for the project. Everything I might want to change is in this one file."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("BAF_DATA_DIR", ROOT))
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
SEED = 42

TARGET = "fraud_bool"
MONTH = "month"
AGE = "customer_age"
AGE_CUTOFF = 50

TRAIN_MONTHS = [0, 1, 2, 3]
VALID_MONTHS = [4, 5]
TEST_MONTHS = [6, 7]

FPR_HIGH = 0.05
FPR_MEDIUM = 0.15

NUMERIC = [
    "income",
    "name_email_similarity",
    "prev_address_months_count",
    "current_address_months_count",
    "customer_age",
    "days_since_request",
    "intended_balcon_amount",
    "zip_count_4w",
    "velocity_6h",
    "velocity_24h",
    "velocity_4w",
    "bank_branch_count_8w",
    "date_of_birth_distinct_emails_4w",
    "credit_risk_score",
    "email_is_free",
    "phone_home_valid",
    "phone_mobile_valid",
    "bank_months_count",
    "has_other_cards",
    "proposed_credit_limit",
    "foreign_request",
    "session_length_in_minutes",
    "keep_alive_session",
    "device_distinct_emails_8w",
    "device_fraud_count",
]

CATEGORICAL = {
    "payment_type": ["AA", "AB", "AC", "AD", "AE"],
    "employment_status": ["CA", "CB", "CC", "CD", "CE", "CF", "CG"],
    "housing_status": ["BA", "BB", "BC", "BD", "BE", "BF", "BG"],
    "source": ["INTERNET", "TELEAPP"],
    "device_os": ["linux", "macintosh", "other", "windows", "x11"],
}

REQUIRED_COLUMNS = NUMERIC + list(CATEGORICAL)

MINUS_ONE_MEANS_MISSING = [
    "prev_address_months_count",
    "current_address_months_count",
    "bank_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
]

PASTEL = {
    "lavender": "#CDB4DB",
    "pink": "#FFC8DD",
    "blue": "#A2D2FF",
    "mint": "#B5EAD7",
    "peach": "#FFDAC1",
    "butter": "#FFF1B6",
    "text": "#3D3A50",
    "muted": "#7A7691",
    "bg": "#FBF8FF",
}

RISK_COLOURS = {
    "Low": "#B5EAD7",
    "Medium": "#FFE5A8",
    "High": "#F7A8BE",
}

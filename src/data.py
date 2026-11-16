"""Loading and cleaning the data. The same prepare() is used for training and in the dashboard."""
import numpy as np
import pandas as pd

import config


def load_baf(path, columns=None):
    """Read one BAF csv. float32 keeps the memory down (the files have 1 million rows)."""
    df = pd.read_csv(path, usecols=columns)
    for col in df.select_dtypes("float64").columns:
        df[col] = df[col].astype("float32")
    return df


def split_by_month(df):
    """Time based split: train / validation / test (never random, because fraud changes over time)."""
    train = df[df[config.MONTH].isin(config.TRAIN_MONTHS)]
    valid = df[df[config.MONTH].isin(config.VALID_MONTHS)]
    test = df[df[config.MONTH].isin(config.TEST_MONTHS)]
    return train, valid, test


def prepare(df, use_age=True):
    """Turn raw application data into the model matrix.

    - -1 values that mean "not available" become NaN (and get a missing flag for the big ones)
    - categories are one-hot encoded with a FIXED list so train and new data have the same columns
    - month and the target are NOT used as features
    """
    out = pd.DataFrame(index=df.index)

    for col in config.NUMERIC:
        values = pd.to_numeric(df[col], errors="coerce").astype("float32")
        if col in config.MINUS_ONE_MEANS_MISSING:
            is_missing = values == -1
            values = values.mask(is_missing)
            if col in ("prev_address_months_count", "bank_months_count"):
                out[col + "_missing"] = is_missing.astype("float32")
        out[col] = values

    out["intended_balcon_negative"] = (out["intended_balcon_amount"] < 0).astype("float32")

    for col, levels in config.CATEGORICAL.items():
        for level in levels:
            out[f"{col}_{level}"] = (df[col].astype(str) == level).astype("float32")

    if not use_age:
        out = out.drop(columns=[config.AGE])
    return out


def age_group(age):
    """Protected group used for the fairness check."""
    return np.where(pd.to_numeric(age, errors="coerce") >= config.AGE_CUTOFF, "50 and over", "Under 50")


def validate_dataset(df):
    """Simple checks that are shown on the validation page. Returns a dict."""
    missing_cols = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    present = [c for c in config.REQUIRED_COLUMNS if c in df.columns]
    info = {
        "rows": len(df),
        "columns": df.shape[1],
        "missing_columns": missing_cols,
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df[present].isna().sum().sum()),
    }
    #"-1" placeholders are not real missing values but I still want to show them
    minus_one = {}
    for col in config.MINUS_ONE_MEANS_MISSING:
        if col in df.columns:
            minus_one[col] = int((pd.to_numeric(df[col], errors="coerce") == -1).sum())
    info["minus_one_counts"] = minus_one

    problems = []
    if config.AGE in df.columns:
        ages = pd.to_numeric(df[config.AGE], errors="coerce")
        #in BAF the age is stored in 10-year bins (10, 20, ... 90), so i only flag impossible values
        if ((ages < 10) | (ages > 110)).any():
            problems.append(f"{int(((ages < 10) | (ages > 110)).sum())} rows have a customer_age that looks impossible")
    if "income" in df.columns:
        income = pd.to_numeric(df["income"], errors="coerce")
        if ((income < 0) | (income > 1)).any():
            problems.append("income should be between 0 and 1 (BAF stores it as a percentile)")
    for col, levels in config.CATEGORICAL.items():
        if col in df.columns:
            unknown = set(df[col].astype(str).unique()) - set(levels)
            if unknown:
                problems.append(f"{col} has unknown categories: {sorted(unknown)[:5]}")
    info["problems"] = problems
    return info

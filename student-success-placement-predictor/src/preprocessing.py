"""Data cleaning and the scikit-learn preprocessing pipeline.

Two stages, split deliberately to prevent data leakage:

1. ``clean_dataset`` runs on the full raw file *before* the train/test split.
   It only applies rules that need no statistics from the data:
   removing exact duplicates, dropping rows without targets and replacing
   physically impossible values (e.g. CGPA 12.4) with NaN.

2. ``build_preprocessor`` returns an *unfitted* ColumnTransformer. Anything
   that learns from data (median imputation, scaling, one-hot categories,
   variance filtering) is fitted inside a Pipeline on the training fold only,
   so cross-validation and the test set never leak into it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import (BINARY_FEATURES, DATASET_PATH, FEATURE_LABELS,
                       NUMERIC_FEATURES, TARGET_ACADEMIC, TARGET_PROBABILITY,
                       TARGET_READINESS, VALID_RANGES)


def load_raw(path=DATASET_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def detect_outliers_iqr(df: pd.DataFrame, cols=NUMERIC_FEATURES, k: float = 1.5) -> pd.DataFrame:
    """Report IQR-based statistical outliers per column (for documentation).

    Statistical outliers inside the valid domain (e.g. a student with 9
    projects) are genuine, so they are reported but kept.
    """
    rows = []
    for col in cols:
        s = df[col].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        n_out = int(((s < lo) | (s > hi)).sum())
        rows.append({"feature": col, "q1": q1, "q3": q3, "lower_fence": lo,
                     "upper_fence": hi, "n_outliers": n_out,
                     "pct_outliers": round(100 * n_out / max(len(s), 1), 2)})
    return pd.DataFrame(rows)


def clean_dataset(df: pd.DataFrame, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Rule-based cleaning that is safe to run before splitting."""
    report = {"rows_in": int(len(df))}
    df = df.copy()

    n_before = len(df)
    df = df.drop_duplicates()
    report["duplicates_removed"] = int(n_before - len(df))

    n_before = len(df)
    df = df.dropna(subset=[TARGET_ACADEMIC, TARGET_READINESS, TARGET_PROBABILITY])
    report["rows_missing_target_removed"] = int(n_before - len(df))

    invalid = {}
    for col, (lo, hi, _) in VALID_RANGES.items():
        if col not in df.columns:
            continue
        mask = (df[col] < lo) | (df[col] > hi)
        if mask.any():
            invalid[col] = int(mask.sum())
            df.loc[mask, col] = np.nan  # imputed later, inside the pipeline
    report["impossible_values_set_to_nan"] = invalid
    report["missing_values_after_cleaning"] = {
        k: int(v) for k, v in df.isna().sum().items() if v > 0}
    report["rows_out"] = int(len(df))

    if verbose:
        for k, v in report.items():
            print(f"{k:32s}: {v}")
    return df.reset_index(drop=True), report


def build_preprocessor() -> Pipeline:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    binary = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(drop="if_binary", categories=[["No", "Yes"]] * len(BINARY_FEATURES),
                                 handle_unknown="error")),
    ])
    columns = ColumnTransformer(
        [("num", numeric, NUMERIC_FEATURES), ("bin", binary, BINARY_FEATURES)],
        remainder="drop",            # drops student_id, age, gender and targets
        verbose_feature_names_out=False,
    )
    # Feature selection step: removes any constant / near-constant feature.
    return Pipeline([("columns", columns),
                     ("select", VarianceThreshold(threshold=1e-4))])


def transformed_feature_names(fitted_preprocessor: Pipeline) -> list[str]:
    """Names of the columns produced by a *fitted* preprocessor."""
    names = fitted_preprocessor.named_steps["columns"].get_feature_names_out()
    keep = fitted_preprocessor.named_steps["select"].get_support()
    return [n for n, k in zip(names, keep) if k]


def readable_name(transformed_name: str) -> str:
    """Map e.g. 'internship_Yes' -> 'Internship'."""
    base = transformed_name.rsplit("_Yes", 1)[0] if transformed_name.endswith("_Yes") else transformed_name
    return FEATURE_LABELS.get(base, base)


def base_feature(transformed_name: str) -> str:
    return transformed_name[:-4] if transformed_name.endswith("_Yes") else transformed_name

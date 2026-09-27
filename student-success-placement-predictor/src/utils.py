"""Shared constants, paths and helpers used across the project."""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
FIG_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
REPORTS_DIR = OUTPUTS_DIR / "reports"
DATASET_PATH = DATA_DIR / "student_dataset.csv"

RANDOM_SEED = 42
TEST_SIZE = 0.20

# --------------------------------------------------------------------------
# Columns
# --------------------------------------------------------------------------
ID_COL = "student_id"

# Demographic columns are collected for EDA and fairness auditing only.
# They are deliberately NOT used as model inputs (see README > Ethics).
DEMOGRAPHIC_COLS = ["age", "gender"]

NUMERIC_FEATURES = [
    "cgpa",
    "attendance_pct",
    "backlogs",
    "coding_score",
    "aptitude_score",
    "communication_score",
    "projects",
    "certifications",
    "extracurricular_score",
    "technical_skills_score",
    "programming_languages",
    "study_hours_per_week",
]
BINARY_FEATURES = ["internship", "hackathon"]  # values: "Yes" / "No"
MODEL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES

TARGET_ACADEMIC = "academic_performance"
TARGET_READINESS = "placement_readiness"
TARGET_PROBABILITY = "placement_probability"

ACADEMIC_LABELS = ["Poor", "Average", "Good", "Excellent"]
READINESS_LABELS = ["Not Ready", "Developing", "Ready", "Highly Ready"]

GENDERS = ["Male", "Female", "Other"]

# Human-readable names for every model feature (used in the UI and charts)
FEATURE_LABELS = {
    "cgpa": "CGPA",
    "attendance_pct": "Attendance %",
    "backlogs": "Backlogs",
    "coding_score": "Coding / DSA score",
    "aptitude_score": "Aptitude score",
    "communication_score": "Communication score",
    "projects": "Projects",
    "certifications": "Certifications",
    "extracurricular_score": "Extracurricular score",
    "technical_skills_score": "Technical skills score",
    "programming_languages": "Programming languages",
    "study_hours_per_week": "Study hours / week",
    "internship": "Internship",
    "hackathon": "Hackathon participation",
    "age": "Age",
    "gender": "Gender",
}

# Valid domain of every input: (min, max, is_integer)
VALID_RANGES = {
    "age": (16, 35, True),
    "cgpa": (0.0, 10.0, False),
    "attendance_pct": (0.0, 100.0, False),
    "backlogs": (0, 30, True),
    "coding_score": (0.0, 100.0, False),
    "aptitude_score": (0.0, 100.0, False),
    "communication_score": (0.0, 100.0, False),
    "projects": (0, 50, True),
    "certifications": (0, 50, True),
    "extracurricular_score": (0.0, 100.0, False),
    "technical_skills_score": (0.0, 100.0, False),
    "programming_languages": (0, 20, True),
    "study_hours_per_week": (0.0, 100.0, False),
}

# Colours shared by the app and the static figures
PALETTE = {
    "ink": "#1D2B36",
    "teal": "#1F5C63",
    "sage": "#7FA99B",
    "mustard": "#D9A441",
    "brick": "#B3524B",
    "paper": "#F3F5F2",
}
CATEGORY_COLORS = {
    "Poor": "#B3524B", "Not Ready": "#B3524B",
    "Average": "#D9A441", "Developing": "#D9A441",
    "Good": "#7FA99B", "Ready": "#7FA99B",
    "Excellent": "#1F5C63", "Highly Ready": "#1F5C63",
}


def validate_input(record: dict) -> list[str]:
    """Return a list of human-readable validation errors (empty list = valid)."""
    errors: list[str] = []
    for col, (lo, hi, is_int) in VALID_RANGES.items():
        label = FEATURE_LABELS.get(col, col)
        if col not in record or record[col] is None or record[col] == "":
            errors.append(f"{label} is required.")
            continue
        try:
            value = float(record[col])
        except (TypeError, ValueError):
            errors.append(f"{label} must be a number.")
            continue
        if value != value:  # NaN
            errors.append(f"{label} is required.")
            continue
        if value < lo:
            if lo == 0:
                errors.append(f"{label} cannot be negative (you entered {value:g}).")
            else:
                errors.append(f"{label} must be at least {lo:g} (you entered {value:g}).")
        elif value > hi:
            errors.append(f"{label} must be between {lo:g} and {hi:g} (you entered {value:g}).")
        elif is_int and not float(value).is_integer():
            errors.append(f"{label} must be a whole number (you entered {value:g}).")
    for col in BINARY_FEATURES:
        if record.get(col) not in ("Yes", "No"):
            errors.append(f"{FEATURE_LABELS[col]} must be 'Yes' or 'No'.")
    if record.get("gender") not in GENDERS:
        errors.append(f"Gender must be one of: {', '.join(GENDERS)}.")
    return errors


def impact_label(share: float) -> str:
    """Convert a feature's share of total importance into a readable tier."""
    if share >= 0.15:
        return "High"
    if share >= 0.08:
        return "Medium-high"
    if share >= 0.04:
        return "Medium"
    return "Low"

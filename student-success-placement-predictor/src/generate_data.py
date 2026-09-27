"""Synthetic student dataset generator.

IMPORTANT: every record produced here is SYNTHETIC. The relationships between
features and targets are hand-designed assumptions (e.g. "higher coding scores
tend to improve placement readiness") plus random noise. The data does not
describe real students and results obtained on it must not be read as
real-world placement evidence.

Generative story
----------------
Each simulated student has two hidden traits, ``ability`` and ``diligence``.
Observable features are noisy functions of those traits, so features are
correlated the way one would expect (e.g. CGPA with attendance). The three
targets are computed from the observable features plus noise, which means a
model can learn them but never perfectly.

To exercise the preprocessing pipeline the generator also injects realistic
data-quality problems: ~2% missing values in several columns, a handful of
exact duplicate rows and a few impossible entries (e.g. CGPA 12.4).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import (ACADEMIC_LABELS, DATASET_PATH, READINESS_LABELS,
                       RANDOM_SEED)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate_students(n: int = 1200, seed: int = RANDOM_SEED,
                      inject_issues: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ability = rng.normal(0, 1, n)
    diligence = rng.normal(0, 1, n)

    age = rng.choice([19, 20, 21, 22, 23, 24], n, p=[.08, .30, .34, .18, .07, .03])
    gender = rng.choice(["Male", "Female", "Other"], n, p=[.56, .42, .02])

    study_hours = np.clip(14 + 5.5 * diligence + rng.normal(0, 4, n), 1, 45)
    attendance = np.clip(79 + 7 * diligence + 2 * ability + rng.normal(0, 6, n), 35, 100)
    cgpa = np.clip(7.0 + 0.75 * ability + 0.45 * diligence + rng.normal(0, 0.45, n), 4.0, 10.0)
    backlog_rate = np.exp(-0.2 - 1.35 * (cgpa - 6.3))
    backlogs = np.clip(rng.poisson(backlog_rate), 0, 10)

    coding = np.clip(58 + 13 * ability + 5 * diligence + rng.normal(0, 9, n), 5, 100)
    aptitude = np.clip(60 + 11 * ability + 3 * diligence + rng.normal(0, 9, n), 10, 100)
    communication = np.clip(64 + 3 * diligence + 2 * ability + rng.normal(0, 12, n), 15, 100)
    projects = np.clip(rng.poisson(np.clip(1.8 + 0.9 * ability + 0.6 * diligence, 0.2, None)), 0, 10)
    certifications = np.clip(rng.poisson(np.clip(1.6 + 0.8 * diligence + 0.3 * ability, 0.2, None)), 0, 12)
    languages = np.clip(1 + rng.poisson(np.clip(1.4 + 0.6 * ability + 0.2 * projects, 0.2, None)), 1, 8)
    internship = rng.random(n) < _sigmoid(-1.0 + 0.9 * ability + 0.35 * (projects - 2) + 0.3 * diligence)
    hackathon = rng.random(n) < _sigmoid(-0.9 + 0.8 * ability + 0.25 * (projects - 2))
    extracurricular = np.clip(55 + 4 * diligence + 8 * hackathon + rng.normal(0, 14, n), 5, 100)
    technical = np.clip(0.55 * coding + 3.2 * projects + 2.4 * languages + 1.5 * certifications
                        + rng.normal(0, 6, n), 5, 100)

    # ---------------- Targets ----------------
    # Academic performance index: dominated by CGPA, attendance and backlogs
    academic_index = (
        10 * (cgpa - 7.0)
        + 0.25 * (attendance - 80)
        - 2.2 * backlogs
        + 0.18 * (study_hours - 14)
        + rng.normal(0, 2.2, n)
    )
    academic = pd.cut(academic_index, bins=[-np.inf, -9, 1.5, 11, np.inf],
                      labels=ACADEMIC_LABELS).astype(str)

    # Placement score (0-100 scale): skills + experience + academics
    placement_score = (
        0.30 * coding
        + 0.16 * aptitude
        + 0.14 * communication
        + 0.10 * technical
        + 4.5 * (cgpa - 7.0)
        + 2.4 * projects
        + 1.2 * certifications
        + 9.0 * internship
        + 3.5 * hackathon
        + 0.04 * extracurricular
        + 0.06 * (attendance - 80)
        - 2.8 * backlogs
        + rng.normal(0, 3.5, n)
    )
    readiness = pd.cut(placement_score, bins=[-np.inf, 43, 52, 61, np.inf],
                       labels=READINESS_LABELS).astype(str)
    probability = 100 * _sigmoid((placement_score - 52) / 6.5)
    probability = np.clip(probability + rng.normal(0, 3, n), 0.5, 99.5)

    df = pd.DataFrame({
        "student_id": [f"STU{i:05d}" for i in range(1, n + 1)],
        "age": age,
        "gender": gender,
        "cgpa": cgpa.round(2),
        "attendance_pct": attendance.round(1),
        "backlogs": backlogs.astype(int),
        "coding_score": coding.round(0).astype(int),
        "aptitude_score": aptitude.round(0).astype(int),
        "communication_score": communication.round(0).astype(int),
        "projects": projects.astype(int),
        "certifications": certifications.astype(int),
        "internship": np.where(internship, "Yes", "No"),
        "hackathon": np.where(hackathon, "Yes", "No"),
        "extracurricular_score": extracurricular.round(0).astype(int),
        "technical_skills_score": technical.round(0).astype(int),
        "programming_languages": languages.astype(int),
        "study_hours_per_week": study_hours.round(1),
        "academic_performance": academic,
        "placement_readiness": readiness,
        "placement_probability": probability.round(1),
    })

    if inject_issues:
        df = _inject_data_quality_issues(df, rng)
    return df


def _inject_data_quality_issues(df: pd.DataFrame, rng) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    # ~2% missing values in selected columns (stored as float / object)
    for col in ["attendance_pct", "communication_score", "certifications",
                "study_hours_per_week", "extracurricular_score", "internship"]:
        idx = rng.choice(n, size=int(0.02 * n), replace=False)
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(float)
        df.loc[idx, col] = np.nan
    # A few impossible values (data-entry errors) that cleaning must catch
    bad = rng.choice(n, size=6, replace=False)
    df.loc[bad[0], "cgpa"] = 12.4
    df.loc[bad[1], "attendance_pct"] = 134.0
    df.loc[bad[2], "study_hours_per_week"] = 160.0
    df["coding_score"] = df["coding_score"].astype(float)
    df.loc[bad[3], "coding_score"] = -8.0
    df.loc[bad[4], "cgpa"] = -1.0
    df.loc[bad[5], "attendance_pct"] = 250.0
    # Exact duplicate rows (e.g. a record uploaded twice)
    dup_idx = rng.choice(n, size=15, replace=False)
    df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)
    return df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)


def main():
    df = generate_students()
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET_PATH, index=False)
    print(f"Saved {len(df)} rows x {df.shape[1]} columns to {DATASET_PATH}")
    print(df["academic_performance"].value_counts().to_string())
    print(df["placement_readiness"].value_counts().to_string())


if __name__ == "__main__":
    main()

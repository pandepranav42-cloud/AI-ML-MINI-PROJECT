"""End-to-end test cases for the trained predictor.

Run with pytest:              pytest -v tests/
Or generate the Markdown report:  python -m tests.test_cases
   -> outputs/reports/test_cases_report.md
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from src.predict import InvalidInputError, StudentPredictor  # noqa: E402
from src.utils import REPORTS_DIR  # noqa: E402

AVERAGE = dict(age=20, gender="Male", cgpa=7.1, attendance_pct=79, backlogs=0, coding_score=58,
               aptitude_score=60, communication_score=63, projects=2, certifications=1,
               internship="No", hackathon="No", extracurricular_score=55, technical_skills_score=47,
               programming_languages=2, study_hours_per_week=13)
HIGH = dict(AVERAGE, gender="Female", cgpa=9.1, attendance_pct=93, coding_score=86, aptitude_score=81,
            communication_score=76, projects=4, certifications=3, internship="Yes", hackathon="Yes",
            extracurricular_score=72, technical_skills_score=78, programming_languages=5,
            study_hours_per_week=22)
LOW = dict(AVERAGE, age=22, cgpa=5.4, attendance_pct=58, backlogs=4, coding_score=34, aptitude_score=41,
           communication_score=52, projects=0, certifications=0, extracurricular_score=40,
           technical_skills_score=24, programming_languages=1, study_hours_per_week=5)

_predictor = None


def predictor() -> StudentPredictor:
    global _predictor
    if _predictor is None:
        _predictor = StudentPredictor()
    return _predictor


def factor(res, feature):
    return next(f for f in res.factors if f.feature == feature)


# Each case: (id, title, input, expected behaviour text, check(result_or_error) -> bool)
def case_list():
    p = predictor()
    avg = p.predict(AVERAGE)
    return [
        ("TC01", "High-performing student", HIGH,
         "Academic Good/Excellent; readiness Ready/Highly Ready; probability ≥ 70%",
         lambda r: r.academic_performance in ("Good", "Excellent")
         and r.placement_readiness in ("Ready", "Highly Ready") and r.placement_probability >= 70),
        ("TC02", "Average-performing student", AVERAGE,
         "Academic Average/Good; readiness Developing/Ready; probability 20–70%",
         lambda r: r.academic_performance in ("Average", "Good")
         and r.placement_readiness in ("Developing", "Ready") and 20 <= r.placement_probability <= 70),
        ("TC03", "Low-performing student", LOW,
         "Academic Poor; readiness Not Ready; probability < 20%",
         lambda r: r.academic_performance == "Poor" and r.placement_readiness == "Not Ready"
         and r.placement_probability < 20),
        ("TC04", "Student with multiple backlogs (average profile, 5 backlogs)", dict(AVERAGE, backlogs=5),
         "Academic category drops to Poor/Average; probability lower than the same student with 0 backlogs; "
         "backlogs is a negative factor",
         lambda r: r.academic_performance in ("Poor", "Average")
         and r.placement_probability < avg.placement_probability and factor(r, "backlogs").contribution < 0),
        ("TC05", "Excellent coding score (average profile, coding 95)", dict(AVERAGE, coding_score=95),
         "Probability higher than the average student; coding is among the top-3 positive factors",
         lambda r: r.placement_probability > avg.placement_probability
         and "coding_score" in [f.feature for f in sorted(r.factors, key=lambda f: -f.contribution)[:3]]),
        ("TC06", "Poor attendance (average profile, 45% attendance)", dict(AVERAGE, attendance_pct=45),
         "Academic category Poor/Average; attendance contributes negatively",
         lambda r: r.academic_performance in ("Poor", "Average")
         and factor(r, "attendance_pct").contribution < 0),
        ("TC07", "Student with internship (average profile)", dict(AVERAGE, internship="Yes"),
         "Probability higher than the identical student without internship; internship is a positive factor",
         lambda r: r.placement_probability > avg.placement_probability
         and factor(r, "internship").contribution > 0),
        ("TC08", "Student without internship (average profile)", AVERAGE,
         "Valid prediction returned; internship contributes negatively (below-average experience)",
         lambda r: factor(r, "internship").contribution < 0),
        ("TC09", "Invalid CGPA (11.0)", dict(AVERAGE, cgpa=11.0),
         "Rejected with message 'CGPA must be between 0 and 10'",
         lambda e: isinstance(e, InvalidInputError) and any("CGPA must be between 0 and 10" in m for m in e.errors)),
        ("TC10", "Invalid attendance (105%)", dict(AVERAGE, attendance_pct=105),
         "Rejected with message 'Attendance % must be between 0 and 100'",
         lambda e: isinstance(e, InvalidInputError)
         and any("Attendance % must be between 0 and 100" in m for m in e.errors)),
        ("TC11", "Negative backlogs and projects", dict(AVERAGE, backlogs=-1, projects=-2),
         "Rejected with two 'cannot be negative' messages",
         lambda e: isinstance(e, InvalidInputError) and sum("cannot be negative" in m for m in e.errors) == 2),
        ("TC12", "Gender changed, everything else identical", dict(AVERAGE, gender="Female"),
         "Identical predictions (gender is not a model input)",
         lambda r: r.placement_probability == avg.placement_probability
         and r.placement_readiness == avg.placement_readiness
         and r.academic_performance == avg.academic_performance),
    ]


def run_case(record):
    try:
        return predictor().predict(record)
    except InvalidInputError as exc:
        return exc


def describe(out) -> str:
    if isinstance(out, InvalidInputError):
        return "Rejected: " + " / ".join(out.errors)
    top = ", ".join(f"{f.label} {f.contribution:+.1f}" for f in out.top_factors(3))
    return (f"{out.academic_performance} / {out.placement_readiness} / {out.placement_probability:.0f}% "
            f"(top factors: {top})")


@pytest.mark.parametrize("case", case_list(), ids=lambda c: c[0])
def test_case(case):
    cid, title, record, expected, check = case
    out = run_case(record)
    assert check(out), f"{cid} {title}: expected {expected}; got {describe(out)}"


def write_report():
    rows, passed = [], 0
    for cid, title, record, expected, check in case_list():
        out = run_case(record)
        ok = bool(check(out))
        passed += ok
        rows.append(f"| {cid} | {title} | {expected} | {describe(out)} | {'PASS' if ok else 'FAIL'} |")
    txt = ("# Test cases: expected vs actual behaviour\n\n"
           "Generated by `python -m tests.test_cases` against the trained models in `models/`.\n"
           "Output format: Academic performance / Placement readiness / Estimated probability.\n\n"
           f"**Result: {passed} of {len(rows)} passed.**\n\n"
           "| ID | Scenario | Expected behaviour | Actual behaviour | Result |\n|---|---|---|---|---|\n"
           + "\n".join(rows) + "\n")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "test_cases_report.md").write_text(txt)
    print(txt)
    return passed == len(rows)


if __name__ == "__main__":
    sys.exit(0 if write_report() else 1)

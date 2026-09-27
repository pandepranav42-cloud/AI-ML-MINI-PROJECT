"""Headless smoke test of the Streamlit app using streamlit.testing.AppTest.

Run: pytest -v tests/test_app.py
"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
PAGES = ["Dashboard", "Student Prediction", "Analytics", "About"]


def _app():
    return AppTest.from_file(APP, default_timeout=90).run()


def test_every_page_renders_without_exceptions():
    at = _app()
    for page in PAGES:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, f"{page} raised {at.exception}"


def test_predict_button_produces_report():
    at = _app()
    at.sidebar.radio[0].set_value("Student Prediction").run()
    at.selectbox(key="preset").set_value("Sample: high performer").run()
    at.button[0].click().run()
    assert not at.exception
    assert not at.error
    assert any("Prediction report" in m.value for m in at.markdown)


def test_invalid_cgpa_shows_error_message():
    at = _app()
    at.sidebar.radio[0].set_value("Student Prediction").run()
    at.number_input(key="in_cgpa").set_value(11.5)
    at.number_input(key="in_attendance_pct").set_value(120.0)
    at.button[0].click().run()
    messages = " ".join(e.value for e in at.error)
    assert "CGPA must be between 0 and 10" in messages
    assert "Attendance % must be between 0 and 100" in messages
    assert not any("Prediction report" in m.value for m in at.markdown)

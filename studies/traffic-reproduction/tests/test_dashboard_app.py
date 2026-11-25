from datetime import date
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def app():
    at = AppTest.from_file(str(ROOT / "dashboard/app.py"), default_timeout=30).run()
    assert not at.exception
    return at


def rerun(at):
    # Streamlit 1.51 AppTest's ButtonGroup expects a list, including for the
    # single-selection segmented control. The real widget returns a scalar.
    at.get("button_group")[0].set_value([at.session_state["view"]])
    at.run()
    assert not at.exception
    return at


def navigate(at, name):
    at.get("button_group")[0].set_value([name]).run()
    assert not at.exception
    return at


def test_overview_reproduces_verified_headlines():
    at = app()
    assert [m.value for m in at.metric] == ["+61.2%", "+0.9%", "+28.7%", "89"]


def test_offline_coordinate_view_renders():
    at = app()
    at.toggle(key="offline_map").set_value(True)
    rerun(at)
    assert at.metric[0].value == "+61.2%"


@pytest.mark.parametrize("page", ["Camera explorer", "School context", "Coverage & sources"])
def test_every_view_renders(page):
    navigate(app(), page)


def test_vehicle_and_weekday_filters_recompute_and_reset():
    at = app()
    at.selectbox(key="vehicle").select("Motorcycles")
    rerun(at)
    assert at.metric[0].value != "+61.2%"
    at.selectbox(key="day_type").select("Weekdays only")
    rerun(at)
    assert [m.value for m in at.metric[:3]] == ["—", "—", "—"]
    at.button[0].click()
    rerun(at)
    assert at.selectbox(key="vehicle").value == "All vehicles"
    assert at.metric[0].value == "+61.2%"


def test_date_selection_without_weekend_is_not_a_zero_effect():
    at = app()
    at.date_input(key="date_range").set_value((date(2026, 8, 25), date(2026, 8, 28)))
    rerun(at)
    assert at.metric[0].value == "—" and at.metric[-1].value == "0"


def test_empty_camera_filter_has_a_clear_empty_state():
    at = app()
    at.toggle(key="choose_cameras").set_value(True)
    at.multiselect(key="specific_cameras").set_value([])
    rerun(at)
    assert any("No observations match" in message.value for message in at.info)


def test_camera_and_snapshot_diagnostic_change_the_view():
    at = navigate(app(), "Camera explorer")
    at.selectbox(key="camera_picker").select("239")
    rerun(at)
    assert at.session_state["active_camera"] == "239"
    at.radio(key="detail").set_value("Individual snapshots")
    rerun(at)
    at.toggle(key="raw_diagnostic").set_value(True)
    rerun(at)
    assert not at.exception
    at.selectbox(key="camera_picker").select("1188")
    rerun(at)
    assert any("location labels disagree" in warning.value for warning in at.warning)


def test_unavailable_case_and_shift_control_work():
    at = navigate(app(), "School context")
    at.selectbox(key="school_case").select("sam_m1_1_end_1500")
    rerun(at)
    assert any("Insufficient paired" in x.value for x in at.info)
    at.select_slider(key="clock_shift").set_value(30)
    rerun(at)
    assert not at.exception

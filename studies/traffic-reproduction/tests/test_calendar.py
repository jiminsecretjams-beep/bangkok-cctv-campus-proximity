import copy
import json

import pandas as pd
import pytest

from workflow.calendar import (
    camera_context,
    date_dimension,
    enrich_observations,
    expand_calendars,
    normalize_holidays,
    validate_rules,
)


@pytest.fixture
def catalog():
    return pd.DataFrame(
        [
            dict(institution_id="school", institution_name="School"),
            dict(institution_id="university", institution_name="University"),
        ]
    )


@pytest.fixture
def dates():
    return date_dimension(
        ["2026-08-28", "2026-08-29"],
        pd.DataFrame(columns=["date_local", "name_en"]),
        dict(crosschecked_dates=["2026-08-28", "2026-08-29"], source_ids=["official"], review_id="review"),
    )


def rule(**overrides):
    return dict(
        rule_id="r1",
        institution_id="university",
        scope_id="programme:bba",
        scope_kind="programme",
        scope_name="BBA only",
        kind="term",
        value="in_term",
        label="semester1",
        start_date="2026-08-28",
        end_date="2026-08-29",
        source_id="official",
        evidence_basis="official calendar",
        **overrides,
    )


def test_absence_is_unknown_outside_review():
    holidays = pd.DataFrame([dict(date_local="2026-08-12", name_en="Holiday")])
    d = date_dimension(
        ["2026-08-12", "2026-08-25", "2026-09-01"],
        holidays,
        dict(crosschecked_dates=["2026-08-12", "2026-08-25"], source_ids=["official"], review_id="review"),
    )
    assert d.is_public_holiday.tolist()[:2] == [True, False]
    assert pd.isna(d.is_public_holiday.iloc[2])


def test_programme_never_broadcasts_to_university(catalog, dates):
    rules = [rule()]
    validate_rules(rules, catalog, {"official"})
    d = expand_calendars(catalog, dates, rules)
    assert d[d.scope_kind.eq("institution")].term_status.eq("unknown").all()
    assert d[d.scope_kind.eq("programme")].term_status.eq("in_term").all()
    assert d.closure_status.eq("unknown").all()
    assert d.instruction_day_status.eq("unknown").all()


def test_event_never_implies_closed(catalog, dates):
    r = rule()
    r.update(kind="event", value="off_site_competition", scope_kind="cohort")
    d = expand_calendars(catalog, dates, [r])
    assert d.closure_status.eq("unknown").all()
    assert d.term_status.eq("unknown").all()
    assert json.loads(d[d.scope_kind.eq("cohort")].event_ids_json.iloc[0]) == ["r1"]


def test_conflicting_terms_and_closures_are_explicit(catalog, dates):
    a = rule()
    b = copy.deepcopy(a)
    b.update(rule_id="r2", label="semester2")
    d = expand_calendars(catalog, dates, [a, b])
    assert d[d.scope_kind.eq("programme")].term_status.eq("conflict").all()
    a.update(kind="closure", value="closed")
    b.update(kind="closure", value="open")
    d = expand_calendars(catalog, dates, [a, b])
    assert d[d.scope_kind.eq("programme")].closure_status.eq("conflict").all()


def test_many_institutions_do_not_multiply_images(catalog, dates):
    links = pd.DataFrame(
        [
            dict(
                camera_id="1",
                institution_id=x.institution_id,
                institution_name=x.institution_name,
                distance_m=100,
            )
            for x in catalog.itertuples()
        ]
    )
    scopes = expand_calendars(catalog, dates, [rule()])
    context, bridge = camera_context(["1"], dates, links, scopes)
    assert len(context) == 2 and len(bridge) == 4
    source = pd.DataFrame(
        dict(
            image_id=["a", "b"],
            camera_id=["1", "1"],
            observed_at_utc=pd.to_datetime(["2026-08-27T18:10:00Z", "2026-08-28T18:10:00Z"]),
            date_local=["2026-08-28", "2026-08-29"],
            is_weekend=[False, True],
            vehicle_count=pd.array([0, None], dtype="Int64"),
        )
    )
    result = enrich_observations(source, dates, context)
    assert len(result) == 2
    assert result.vehicle_count.iloc[0] == 0 and pd.isna(result.vehicle_count.iloc[1])
    assert result.nearby_term_known_count.eq(0).all()
    assert "programme:bba" in result.nearby_scoped_calendar_context_json.iloc[0]
    pd.testing.assert_frame_equal(source, result[source.columns])


def test_bangkok_date_mismatch_rejected(catalog, dates):
    source = pd.DataFrame(
        dict(
            image_id=["a"],
            camera_id=["1"],
            observed_at_utc=["2026-08-28T18:10:00Z"],
            date_local=["2026-08-28"],
            is_weekend=[False],
        )
    )
    with pytest.raises(ValueError, match="Bangkok"):
        enrich_observations(source, dates, pd.DataFrame())


def test_duplicate_calendar_join_rejected(dates):
    source = pd.DataFrame(
        dict(
            image_id=["a"],
            camera_id=["1"],
            observed_at_utc=["2026-08-28T01:00:00Z"],
            date_local=["2026-08-28"],
            is_weekend=[False],
        )
    )
    with pytest.raises(ValueError, match="not unique"):
        enrich_observations(
            source, pd.concat([dates, dates]), pd.DataFrame(columns=["camera_id", "date_local"])
        )


def test_source_holiday_era_and_boolean_validation():
    data = dict(
        years=[
            dict(
                year=2026,
                year_buddhist_era=2569,
                holidays=[dict(date="2026-08-12", name_en="Holiday", is_substitution=False)],
            )
        ]
    )
    assert normalize_holidays(data).year.tolist() == [2026]
    data["years"][0]["holidays"][0]["is_substitution"] = "false"
    with pytest.raises(ValueError, match="boolean"):
        normalize_holidays(data)
    data["years"][0]["holidays"][0]["is_substitution"] = False
    data["years"][0]["year_buddhist_era"] = 2568
    with pytest.raises(ValueError, match="era"):
        normalize_holidays(data)


def test_narrow_rule_cannot_use_institution_scope_id(catalog):
    r = rule()
    r["scope_id"] = "university"
    with pytest.raises(ValueError, match="masquerade"):
        validate_rules([r], catalog, {"official"})

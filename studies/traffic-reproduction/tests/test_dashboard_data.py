import io
import json

import numpy as np
import pandas as pd
import pytest

from dashboard_data import (
    add_segments,
    anchor_summary,
    csv_bytes,
    load_bundle,
    paired_periods,
    period_summary,
    select_context,
    timeline,
)


@pytest.mark.parametrize("metric", ["vehicle", "car", "motorcycle", "truck", "bus", "bicycle"])
def test_full_selection_reconciles_all_frozen_class_comparisons(bundle, metric):
    actual = paired_periods(bundle["daily"], bundle["context"].camera_id, bundle["manifest"]["dates"], metric)
    actual = actual.set_index(["camera_id", "period"]).sort_index()
    expected = bundle["frozen_periods"].set_index(["camera_id", "period"]).sort_index()
    assert actual.index.equals(expected.index)
    np.testing.assert_allclose(actual.weekday_mean, expected["weekday_" + metric + "_mean"], rtol=1e-10)
    np.testing.assert_allclose(actual.weekend_mean, expected["weekend_" + metric + "_mean"], rtol=1e-10)
    np.testing.assert_allclose(
        actual.relative_pct, expected[metric + "_relative_pct"], rtol=1e-10, equal_nan=True
    )


def test_missing_weekend_is_unavailable_and_sparse_bicycles_not_a_headline(bundle):
    ids = bundle["context"].camera_id
    q = paired_periods(bundle["daily"], ids, ["2026-08-25", "2026-08-26", "2026-08-28"])
    assert q.empty and period_summary(q).median_pct.isna().all()
    q = paired_periods(bundle["daily"], ids, bundle["manifest"]["dates"], "bicycle")
    assert period_summary(q).median_pct.isna().all()


def test_empty_camera_selection_is_not_reinterpreted_as_all(bundle):
    assert select_context(bundle["context"], "All distances", "All education contexts", []).empty


def test_breaks_nulls_and_overnight_instead_of_drawing_zero():
    q = pd.DataFrame(
        dict(
            time=pd.to_datetime(
                ["2026-08-25 07:00", "2026-08-25 08:00", "2026-08-25 09:00", "2026-08-26 07:00"]
            ),
            value=[0, np.nan, 2, 3],
        )
    )
    actual = add_segments(q)
    assert actual.value.tolist() == [0, 2, 3]
    assert actual.segment.nunique() == 3


def test_camera_line_preserves_real_zero_and_excludes_bad_hours(bundle):
    h = bundle["hourly"].copy()
    camera = h.camera_id.iloc[0]
    h.loc[h.camera_id.eq(camera), "vehicle_mean"] = 0.0
    h.loc[h.camera_id.eq(camera) & h.hour_local.eq(8), "coverage_pass"] = False
    q = timeline(h, camera, bundle["manifest"]["dates"])
    assert q.value.eq(0).all() and not q.hour_local.eq(8).any()


def test_calendar_scopes_never_become_university_wide_or_open(bundle):
    cal = bundle["calendar"]
    chula = cal[cal.institution_id.eq("edu_95d76c138f10")]
    assert chula[chula.scope_kind.eq("institution")].term_status.eq("unknown").all()
    assert chula[chula.scope_kind.eq("programme")].term_status.eq("in_term").all()
    assert cal.closure_status.eq("unknown").all()


def test_insufficient_thursday_case_does_not_return_zero(bundle):
    result = anchor_summary(
        bundle["anchors"],
        bundle["manifest"]["dates"],
        ["150"],
        "sam_m1_1_end_1500",
        60,
        0,
        "vehicle",
        "Both day types",
    )
    assert result.empty


def test_downloads_keep_thai_and_missing_counts():
    d = pd.DataFrame(dict(name=["โรงเรียน"], count=[np.nan]))
    encoded = csv_bytes(d)
    assert encoded.startswith(b"\xef\xbb\xbf")
    loaded = pd.read_csv(io.BytesIO(encoded))
    assert loaded.name.iloc[0] == "โรงเรียน" and pd.isna(loaded["count"].iloc[0])


def test_changed_dataset_rejected_before_display(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps(dict(schema_version=1, files={"bad.parquet": dict(sha256="not-the-hash")}))
    )
    (tmp_path / "bad.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="verification failed"):
        load_bundle(tmp_path)

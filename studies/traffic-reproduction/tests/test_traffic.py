"""Statistical contract checks: safe denominators, coverage, weighting and no leakage."""

import numpy as np
import pandas as pd

from workflow.traffic import (
    METRICS,
    PERIODS,
    balanced_periods,
    camera_effects,
    holdout_peaks,
    hourly_cube,
    location_clusters,
    matched_hours,
)


def source_frames(usable=True):
    t = pd.date_range("2026-08-25T00:01:00Z", periods=6, freq="10min")
    d = pd.DataFrame(
        dict(
            image_id=[str(i) for i in range(6)],
            camera_id="A",
            observed_at_utc=t,
            date_local="2026-08-25",
            inference_status="success",
            eligible_temporal=usable,
            mean_luma=100,
            precipitation_mm=0,
        )
    )
    for m in METRICS:
        d[m + "_count"] = 0 if usable else np.nan
        d["raw_" + m + "_count"] = 0
    return d


def cube():
    rows = []
    for day, count in [
        ("2026-08-25", 10),
        ("2026-08-26", 20),
        ("2026-08-27", 30),
        ("2026-08-29", 5),
        ("2026-08-30", 5),
    ]:
        for hour in range(7, 17):
            r = dict(
                camera_id="A",
                date_local=day,
                hour_local=hour,
                is_weekend=pd.Timestamp(day).weekday() >= 5,
                coverage_pass=True,
            )
            for m in METRICS:
                r[m + "_mean"] = count if m == "vehicle" else count / 5
            rows.append(r)
    return pd.DataFrame(rows)


def test_bad_images_and_missing_hours_do_not_become_zero():
    d = source_frames(False)
    h = hourly_cube(d)
    assert h.vehicle_mean.isna().all()
    h = hourly_cube(source_frames())
    valid = h[h.hour_local.eq(7)].iloc[0]
    assert valid.coverage_pass and valid.vehicle_mean == 0
    assert h[h.hour_local.eq(8)].vehicle_mean.isna().all()


def test_bursty_capture_fails_edge_coverage():
    d = source_frames()
    d["observed_at_utc"] = pd.date_range("2026-08-25T00:01:00Z", periods=6, freq="1min")
    h = hourly_cube(d)
    assert not h.coverage_pass.any()


def test_balanced_periods_equal_weight_days_and_do_not_compare_totals():
    daily, p = balanced_periods(cube())
    assert len(daily) == 5 * len(PERIODS)
    assert len(p) == 3
    assert p.weekday_vehicle_mean.eq(20).all()
    assert p.weekend_vehicle_mean.eq(5).all()
    assert p.vehicle_relative_pct.eq(300).all()
    # A general weekday increase is not a morning-specific excess.
    assert camera_effects(p).morning_specific_excess_pct.eq(0).all()


def test_missing_one_weekend_window_excludes_all_periods_for_that_day():
    h = cube()
    h.loc[h.date_local.eq("2026-08-30") & h.hour_local.eq(7), "coverage_pass"] = False
    daily, p = balanced_periods(h)
    assert not daily.date_local.eq("2026-08-30").any()
    assert p.empty  # Cannot silently replace the missing Sunday with zero.


def test_zero_reference_does_not_create_infinite_percentage():
    h = cube()
    for metric in METRICS:
        h.loc[h.is_weekend, metric + "_mean"] = 0
    _, p = balanced_periods(h)
    assert p.vehicle_relative_pct.isna().all()
    assert matched_hours(h).vehicle_ratio.isna().all()


def test_separate_period_dry_screen_does_not_imply_complete_day_screen():
    h = cube()
    h.loc[h.hour_local.eq(10), "coverage_pass"] = False
    _, all_periods = balanced_periods(h)
    assert all_periods.empty
    _, morning_only = balanced_periods(h, period_hours={"morning_07_09": [7, 8]})
    assert len(morning_only) == 1
    assert morning_only.period.iloc[0] == "morning_07_09"


def test_holdout_cannot_change_peak_selection():
    h = cube()
    h = h[~h.is_weekend].copy()
    h["vehicle_mean"] = np.where(h.hour_local.isin([7, 8]), 20.0, 10.0)
    monday = h[h.date_local.eq("2026-08-25")].copy()
    monday["date_local"] = "2026-08-31"
    monday["vehicle_mean"] = np.where(monday.hour_local.isin([14, 15]), 100.0, 1.0)
    a, _ = holdout_peaks(pd.concat([h, monday], ignore_index=True))
    monday["vehicle_mean"] = np.where(monday.hour_local.isin([7, 8]), 1000.0, 1.0)
    b, _ = holdout_peaks(pd.concat([h, monday], ignore_index=True))
    assert a.start_hour.iloc[0] == b.start_hour.iloc[0] == 7
    assert a.discovery_relative_pct.iloc[0] == b.discovery_relative_pct.iloc[0]
    assert a.holdout_relative_pct.iloc[0] < 0 < b.holdout_relative_pct.iloc[0]


def test_close_views_group_by_geography_without_using_outcomes():
    d = pd.DataFrame(
        dict(camera_id=["A", "B", "C"], latitude=[13.7, 13.7001, 13.71], longitude=[100.5, 100.5, 100.5])
    )
    groups = location_clusters(d)
    assert groups["A"] == groups["B"] != groups["C"]

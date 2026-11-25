"""Independently reconcile exported insights against original per-image counts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def verify(out, calendar):
    manifest = json.loads((out / "manifest.json").read_text())
    for name, expected in manifest["inputs"].items():
        assert digest(ROOT / name) == expected, name
    for name, expected in manifest["code"].items():
        assert digest(ROOT / name) == expected, name
    for name, expected in manifest["files"].items():
        assert digest(out / name) == expected["sha256"], name
    counts = [m + "_count" for m in ["car", "motorcycle", "bus", "truck", "bicycle", "vehicle"]]
    source = pd.read_parquet(
        calendar / "observations_with_calendar.parquet",
        columns=["image_id", "camera_id", "date_local", "observed_at_utc", "eligible_temporal"] + counts,
    )
    assert source.image_id.is_unique and len(source) == 61319
    source["hour_local"] = source.observed_at_utc.dt.tz_convert("Asia/Bangkok").dt.hour
    means = source[source.eligible_temporal].groupby(["camera_id", "date_local", "hour_local"])[counts].mean()
    cube = pd.read_parquet(out / "camera_day_hour.parquet")
    passed = cube[cube.coverage_pass].set_index(["camera_id", "date_local", "hour_local"])
    direct = means.reindex(passed.index)
    for col in counts:
        np.testing.assert_allclose(direct[col], passed[col.replace("_count", "_mean")], rtol=1e-12)
    periods = {"morning_07_09": [7, 8], "midday_10_14": [10, 11, 12, 13], "afternoon_15_17": [15, 16]}
    required = set(sum(periods.values(), []))
    direct_rows = []
    for camera, camera_hours in passed.reset_index().groupby("camera_id"):
        days = {}
        for day, rows in camera_hours.groupby("date_local"):
            if required <= set(rows.hour_local):
                days[day] = rows.set_index("hour_local")
        weekdays = [x for x in days if pd.Timestamp(x).weekday() < 5]
        weekend = [x for x in days if pd.Timestamp(x).weekday() >= 5]
        if len(weekdays) < 3 or len(weekend) < 2:
            continue
        for period, hours in periods.items():
            # Re-derive from per-image means, without importing analysis functions.
            a = np.mean(
                [
                    means.loc[[(camera, day, hour) for hour in hours], "vehicle_count"].mean()
                    for day in weekdays
                ]
            )
            b = np.mean(
                [
                    means.loc[[(camera, day, hour) for hour in hours], "vehicle_count"].mean()
                    for day in weekend
                ]
            )
            direct_rows.append(
                dict(
                    camera_id=camera,
                    period=period,
                    weekday_vehicle_mean=a,
                    weekend_vehicle_mean=b,
                    vehicle_relative_pct=100 * (a / b - 1) if b >= 1 else np.nan,
                )
            )
    expected = pd.DataFrame(direct_rows).set_index(["camera_id", "period"]).sort_index()
    published = pd.read_csv(out / "camera_period_comparisons.csv", dtype={"camera_id": str})
    published = published.set_index(["camera_id", "period"]).sort_index()
    assert expected.index.equals(published.index)
    for col in expected:
        np.testing.assert_allclose(expected[col], published[col], rtol=1e-12)
    headline = pd.read_csv(out / "headline_period_evidence.csv").set_index("period")
    for period, rows in expected.reset_index().groupby("period"):
        assert np.isclose(rows.vehicle_relative_pct.median(), headline.loc[period, "median_relative_pct"])
        assert rows.vehicle_relative_pct.gt(0).sum() == headline.loc[period, "positive_cameras"]
    result = dict(
        result="pass",
        original_source_rows=61319,
        source_hour_cells_checked=len(passed),
        independently_recomputed_camera_period_rows=len(expected),
        checks=[
            "All frozen input, code and listed artifact hashes match",
            "All six accepted hourly class/total means equal direct per-image aggregation",
            "Every primary camera/period mean and percentage independently recomputed",
            "Every headline median and positive-camera count independently recomputed",
        ],
    )
    (out / "independent_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def verify_proximity():
    """Recompute all camera/institution distances without using the saved links."""
    campus = ROOT / "inputs/campus"
    cameras = pd.read_csv(campus / "cctv_campus_flags.csv", dtype={"camera_id": str})
    institutions = pd.read_csv(campus / "institutions_clean.csv")
    links = pd.read_csv(campus / "camera_institution_matches.csv", dtype={"camera_id": str})
    a = np.radians(cameras.latitude.to_numpy())[:, None]
    b = np.radians(institutions.latitude.to_numpy())[None, :]
    x = np.radians(cameras.longitude.to_numpy())[:, None]
    y = np.radians(institutions.longitude.to_numpy())[None, :]
    haversine = np.sin((b - a) / 2) ** 2 + np.cos(a) * np.cos(b) * np.sin((y - x) / 2) ** 2
    distances = 6371008.8 * 2 * np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))
    ci, ii = np.where(distances <= 1500)
    expected = (
        pd.DataFrame(
            dict(
                camera_id=cameras.camera_id.to_numpy()[ci],
                institution_id=institutions.institution_id.to_numpy()[ii],
                distance_m=distances[ci, ii],
            )
        )
        .set_index(["camera_id", "institution_id"])
        .sort_index()
    )
    actual = links.set_index(["camera_id", "institution_id"]).sort_index()
    assert actual.index.is_unique and expected.index.equals(actual.index)
    # Original catalogue exports distances to the nearest millimetre.
    np.testing.assert_allclose(expected.distance_m, actual.distance_m, atol=0.000501, rtol=0)
    assert len(cameras) == 100 and len(institutions) == 1154 and len(actual) == 2306
    print("PASS: all 2,306 proximity links independently recalculated from coordinates.")

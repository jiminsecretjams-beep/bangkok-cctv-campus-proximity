"""Pure dashboard calculations; use frozen accepted-day values, never camera sums."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PERIODS = ["morning_07_09", "midday_10_14", "afternoon_15_17"]
PERIOD_NAMES = {
    "morning_07_09": "Morning · 07–09",
    "midday_10_14": "Midday · 10–14",
    "afternoon_15_17": "Afternoon · 15–17",
}
VEHICLES = {
    "All vehicles": "vehicle",
    "Cars": "car",
    "Motorcycles": "motorcycle",
    "Buses": "bus",
    "Trucks": "truck",
    "Bicycles": "bicycle",
}


def load_bundle(folder: Path) -> dict:
    """Fail clearly on incomplete or changed portable data."""
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported dashboard dataset version")
    for name, item in manifest["files"].items():
        if Path(name).name != name:
            raise ValueError("Invalid data manifest path")
        p = folder / name
        if hashlib.sha256(p.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Dataset verification failed: {name}")
    result = {p.stem: pd.read_parquet(p) for p in folder.glob("*.parquet")}
    for d in result.values():
        if "camera_id" in d:
            d["camera_id"] = d.camera_id.astype(str)
    for name in ["summary", "rules", "sources", "verification"]:
        result[name] = json.loads((folder / (name + ".json")).read_text())
    result["manifest"] = manifest
    if not result["frames"].image_id.is_unique or result["context"].camera_id.duplicated().any():
        raise ValueError("Duplicate dataset keys")
    return result


def select_context(context: pd.DataFrame, band: str, education: str, camera_ids=None) -> pd.DataFrame:
    d = context.copy()
    if band != "All distances":
        d = d[d.nearest_school_band.eq(band)]
    if education == "Also near a university / college":
        d = d[d.near_university_college]
    elif education == "No university / college catalogue match":
        d = d[~d.near_university_college]
    if camera_ids is not None:
        d = d[d.camera_id.isin(camera_ids)]
    return d.sort_values("camera_id", key=lambda s: s.astype(int))


def select_rows(d: pd.DataFrame, camera_ids, dates, day_type="Both day types") -> pd.DataFrame:
    q = d[d.camera_id.isin(camera_ids) & d.date_local.isin(dates)].copy()
    if day_type != "Both day types":
        weekend = pd.to_datetime(q.date_local).dt.dayofweek.ge(5)
        q = q[weekend if day_type == "Weekend only" else ~weekend]
    return q


def paired_periods(daily: pd.DataFrame, camera_ids, dates, metric="vehicle", day_type="Both day types"):
    """Exact same accepted eight-hour days as the verified analysis."""
    q = select_rows(daily, camera_ids, dates, day_type)
    output_columns = [
        "camera_id",
        "period",
        "weekday_mean",
        "weekend_mean",
        "weekday_days",
        "weekend_days",
        "difference",
        "relative_pct",
    ]
    if q.empty or set(q.is_weekend.unique()) != {False, True}:
        return pd.DataFrame(columns=output_columns)
    g = q.groupby(["camera_id", "period", "is_weekend"]).agg(
        value=(metric + "_mean", "mean"), n=("date_local", "nunique")
    )
    p = (
        g.xs(False, level="is_weekend")
        .rename(columns={"value": "weekday_mean", "n": "weekday_days"})
        .join(
            g.xs(True, level="is_weekend").rename(columns={"value": "weekend_mean", "n": "weekend_days"}),
            how="inner",
        )
        .reset_index()
    )
    p = p[p.weekday_days.ge(3) & p.weekend_days.ge(2)].copy()
    p["difference"] = p.weekday_mean - p.weekend_mean
    minimum = 1 if metric == "vehicle" else 0.25
    p["relative_pct"] = 100 * (p.weekday_mean / p.weekend_mean.where(p.weekend_mean.ge(minimum)) - 1)
    return p[output_columns]


def period_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key in PERIODS:
        q = pairs[pairs.period.eq(key)]
        valid = q.relative_pct.dropna()
        # Sparse class comparisons are not promoted to network headline claims.
        enough = len(valid) >= 10
        rows.append(
            dict(
                period=key,
                label=PERIOD_NAMES[key],
                balanced_cameras=len(q),
                ratio_cameras=len(valid),
                median_pct=float(valid.median()) if enough else np.nan,
                q25=float(valid.quantile(0.25)) if enough else np.nan,
                q75=float(valid.quantile(0.75)) if enough else np.nan,
                positive_cameras=int(valid.gt(0).sum()),
                status="estimated" if enough else "insufficient",
            )
        )
    return pd.DataFrame(rows)


def hour_profile(hourly: pd.DataFrame, camera_ids, dates, metric="vehicle", day_type="Both day types"):
    q = select_rows(hourly, camera_ids, dates, day_type)
    q = q[q.coverage_pass & q.hour_local.between(7, 16)]
    if q.empty or set(q.is_weekend.unique()) != {False, True}:
        return pd.DataFrame(columns=["hour_local", "median_pct", "q25", "q75", "cameras"])
    g = q.groupby(["camera_id", "hour_local", "is_weekend"]).agg(
        value=(metric + "_mean", "mean"), n=("date_local", "nunique")
    )
    p = (
        g.xs(False, level="is_weekend")
        .add_prefix("weekday_")
        .join(g.xs(True, level="is_weekend").add_prefix("weekend_"), how="inner")
        .reset_index()
    )
    p = p[p.weekday_n.ge(3) & p.weekend_n.ge(2)].copy()
    minimum = 1 if metric == "vehicle" else 0.25
    p["pct"] = 100 * (p.weekday_value / p.weekend_value.where(p.weekend_value.ge(minimum)) - 1)
    s = (
        p.groupby("hour_local")
        .pct.agg(
            median_pct="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
            cameras="count",
        )
        .reset_index()
    )
    s.loc[s.cameras.lt(10), ["median_pct", "q25", "q75"]] = np.nan
    return s


def add_segments(d: pd.DataFrame, time_col="time", value_col="value", gap_minutes=60) -> pd.DataFrame:
    """Break lines at excluded observations and overnight/acquisition gaps."""
    q = d.sort_values(time_col).copy()
    missing = q[value_col].isna()
    gap = q[time_col].diff().dt.total_seconds().gt(gap_minutes * 60 + 1)
    q["segment"] = (missing | missing.shift(fill_value=True) | gap).cumsum().astype(str)
    return q[~missing]


def timeline(hourly, camera_id, dates, metric="vehicle", day_type="Both day types"):
    q = select_rows(hourly, [camera_id], dates, day_type).copy()
    q["time"] = pd.to_datetime(q.date_local + " " + q.hour_local.astype(str).str.zfill(2) + ":00")
    q["value"] = q[metric + "_mean"]
    q.loc[~q.coverage_pass, "value"] = np.nan
    return add_segments(q)


def anchor_summary(anchors, dates, cameras, schedule_id, width, shift, metric, day_type):
    q = select_rows(anchors, cameras, dates, day_type)
    q = q[
        q.schedule_id.eq(schedule_id)
        & q.width_minutes.eq(width)
        & q.anchor_shift_minutes.eq(shift)
        & q.coverage_pass
    ]
    if q.empty:
        return pd.DataFrame(columns=["baseline", "days", "event_mean", "baseline_mean", "difference"])
    return (
        q.groupby("baseline")
        .agg(
            days=("date_local", "nunique"),
            event_mean=(metric + "_event_mean", "mean"),
            baseline_mean=(metric + "_baseline_mean", "mean"),
            difference=(metric + "_difference", "mean"),
            before_18=("pair_ends_before_18", "all"),
        )
        .reset_index()
    )


def csv_bytes(frame: pd.DataFrame) -> bytes:
    """UTF-8 BOM preserves Thai names in spreadsheet apps; nulls remain blank."""
    return frame.to_csv(index=False).encode("utf-8-sig")

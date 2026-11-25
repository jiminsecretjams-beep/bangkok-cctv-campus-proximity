"""Coverage-balanced, within-camera descriptions of snapshot vehicle presence."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

CLASSES = ["car", "motorcycle", "bus", "truck", "bicycle"]
METRICS = CLASSES + ["vehicle"]
PERIODS = {"morning_07_09": [7, 8], "midday_10_14": [10, 11, 12, 13], "afternoon_15_17": [15, 16]}
TRAIN = {"2026-08-25", "2026-08-26", "2026-08-27", "2026-08-28"}
HOLDOUT = "2026-08-31"


def hourly_cube(source, *, raw=False, dry=False, strict=False):
    """Full 100 x 7 x 15 grid; excluded means remain null, true zeros remain zero."""
    d = source.copy()
    if d.image_id.duplicated().any():
        raise ValueError("Duplicate image rows")
    local = pd.to_datetime(d.observed_at_utc, utc=True).dt.tz_convert("Asia/Bangkok")
    if not local.dt.strftime("%Y-%m-%d").eq(d.date_local).all():
        raise ValueError("Bangkok date mismatch")
    d["hour_local"] = local.dt.hour
    d["minute_in_hour"] = local.dt.minute + local.dt.second / 60
    ordered = d.sort_values(["camera_id", "observed_at_utc"])
    gaps = ordered.groupby("camera_id").observed_at_utc.diff().dt.total_seconds()
    cadence = gaps[gaps.between(60, 1800)].groupby(ordered.camera_id).median().clip(120, 1200)
    d["usable"] = d.inference_status.eq("success") & (True if raw else d.eligible_temporal)
    if dry:
        d["usable"] &= d.precipitation_mm.le(0.1)
    keys = ["camera_id", "date_local", "hour_local"]
    grid = pd.MultiIndex.from_product(
        [sorted(d.camera_id.unique()), sorted(d.date_local.unique()), range(6, 21)], names=keys
    )
    all_hours = d.groupby(keys).size().rename("n_acquired")
    good = d[d.usable & d.hour_local.between(6, 20)].copy()
    aggregate = dict(
        n_usable=("image_id", "size"),
        first_minute=("minute_in_hour", "min"),
        last_minute=("minute_in_hour", "max"),
        mean_luma=("mean_luma", "mean"),
    )
    for metric in METRICS:
        col = ("raw_" if raw else "") + metric + "_count"
        aggregate[metric + "_mean"] = (col, "mean")
        aggregate[metric + "_sum"] = (col, "sum")
    h = good.groupby(keys).agg(**aggregate).reindex(grid).join(all_hours).reset_index()
    h[["n_acquired", "n_usable"]] = h[["n_acquired", "n_usable"]].fillna(0).astype(int)
    h["n_expected"] = 3600 / h.camera_id.map(cadence).fillna(600)
    h["usable_coverage"] = (h.n_usable / h.n_expected).clip(upper=1)
    minimum, coverage, edge = (5, 0.8, 15) if strict else (3, 0.5, 21)
    h["coverage_pass"] = (
        h.n_usable.ge(minimum)
        & h.usable_coverage.ge(coverage)
        & h.first_minute.le(edge)
        & h.last_minute.ge(60 - edge)
    )
    values = [m + suffix for m in METRICS for suffix in ["_mean", "_sum"]]
    h.loc[~h.coverage_pass, values] = np.nan
    h["is_weekend"] = pd.to_datetime(h.date_local).dt.dayofweek.ge(5)
    return h


def matched_hours(hourly, *, minimum_weekdays=3, minimum_weekend=2):
    h = hourly[hourly.coverage_pass & hourly.hour_local.between(7, 16)]
    means = [m + "_mean" for m in METRICS]
    a = h.groupby(["camera_id", "hour_local", "is_weekend"]).agg(
        **{c: (c, "mean") for c in means}, n_days=("date_local", "nunique")
    )
    if not {False, True} <= set(a.index.get_level_values("is_weekend")):
        return pd.DataFrame()
    weekday = a.xs(False, level="is_weekend").add_prefix("weekday_")
    weekend = a.xs(True, level="is_weekend").add_prefix("weekend_")
    p = weekday.join(weekend, how="inner").reset_index()
    p = p[p.weekday_n_days.ge(minimum_weekdays) & p.weekend_n_days.ge(minimum_weekend)].copy()
    p["vehicle_ratio"] = p.weekday_vehicle_mean / p.weekend_vehicle_mean.where(p.weekend_vehicle_mean.ge(1))
    return p


def balanced_periods(hourly, *, minimum_weekdays=3, minimum_weekend=2, period_hours=None):
    """Same eight clock hours and same accepted days for all three periods."""
    period_hours = PERIODS if period_hours is None else period_hours
    needed = set(sum(period_hours.values(), []))
    h = hourly[hourly.coverage_pass & hourly.hour_local.isin(needed)].copy()
    complete = h.groupby(["camera_id", "date_local"]).hour_local.nunique()
    accepted = set(complete[complete.eq(len(needed))].index)
    h = h.loc[[tuple(r) in accepted for r in h[["camera_id", "date_local"]].to_numpy()]]
    period_for_hour = {hour: period for period, hours in period_hours.items() for hour in hours}
    h["period"] = h.hour_local.map(period_for_hour)
    values = [m + "_mean" for m in METRICS]
    daily = h.groupby(["camera_id", "date_local", "is_weekend", "period"])[values].mean().reset_index()
    grouped = daily.groupby(["camera_id", "period", "is_weekend"]).agg(
        **{v: (v, "mean") for v in values}, n_days=("date_local", "nunique")
    )
    if True not in grouped.index.get_level_values("is_weekend"):
        return daily, pd.DataFrame()
    p = (
        grouped.xs(False, level="is_weekend")
        .add_prefix("weekday_")
        .join(grouped.xs(True, level="is_weekend").add_prefix("weekend_"), how="inner")
        .reset_index()
    )
    p = p[p.weekday_n_days.ge(minimum_weekdays) & p.weekend_n_days.ge(minimum_weekend)].copy()
    for metric in METRICS:
        a, b = p["weekday_" + metric + "_mean"], p["weekend_" + metric + "_mean"]
        p[metric + "_difference"] = a - b
        p[metric + "_relative_pct"] = 100 * (a / b.where(b.ge(1 if metric == "vehicle" else 0.25)) - 1)
    return daily, p


def camera_effects(periods):
    selected = ["weekday_vehicle_mean", "weekend_vehicle_mean", "vehicle_relative_pct", "vehicle_difference"]
    p = periods.set_index(["camera_id", "period"])[selected].unstack("period")
    p.columns = [f"{period}_{metric}" for metric, period in p.columns]
    p = p.reset_index()
    a = p["morning_07_09_weekday_vehicle_mean"] / p["midday_10_14_weekday_vehicle_mean"].where(
        p["midday_10_14_weekday_vehicle_mean"].ge(1)
    )
    b = p["morning_07_09_weekend_vehicle_mean"] / p["midday_10_14_weekend_vehicle_mean"].where(
        p["midday_10_14_weekend_vehicle_mean"].ge(1)
    )
    p["morning_specific_ratio_of_ratios"] = a / b.where(b.gt(0))
    p["morning_specific_excess_pct"] = 100 * (p.morning_specific_ratio_of_ratios - 1)
    return p


def location_clusters(cameras, radius_m=50):
    """Connected components of nearby supplied coordinates, ignoring observed counts."""
    c = cameras.sort_values("camera_id").reset_index(drop=True)
    parent = list(range(len(c)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    lat, lon = np.radians(c.latitude.to_numpy()), np.radians(c.longitude.to_numpy())
    for i in range(len(c)):
        for j in range(i):
            distance = (
                6371000
                * 2
                * np.arcsin(
                    np.sqrt(
                        np.sin((lat[i] - lat[j]) / 2) ** 2
                        + np.cos(lat[i]) * np.cos(lat[j]) * np.sin((lon[i] - lon[j]) / 2) ** 2
                    )
                )
            )
            if distance <= radius_m:
                parent[find(i)] = find(j)
    return dict(zip(c.camera_id, ["location_" + c.camera_id.iloc[find(i)] for i in range(len(c))]))


def holdout_peaks(hourly):
    """Select two-hour peaks on Tue-Fri only; report Monday without reselection."""
    h = hourly[hourly.coverage_pass & hourly.hour_local.between(7, 16)]
    candidates, daily = [], []
    for camera, c in h.groupby("camera_id"):
        train = c[c.date_local.isin(TRAIN)]
        for start in range(7, 16):
            peak = {start, start + 1}
            days = {d: set(g.hour_local) for d, g in train.groupby("date_local") if peak <= set(g.hour_local)}
            if len(days) < 3:
                continue
            baseline = set.intersection(*days.values()) - peak
            if len(baseline) < 5:
                continue
            rows = []
            for day in sorted(days) + [HOLDOUT]:
                g = c[c.date_local.eq(day)]
                if not (peak | baseline) <= set(g.hour_local):
                    continue
                pm = float(g[g.hour_local.isin(peak)].vehicle_mean.mean())
                bm = float(g[g.hour_local.isin(baseline)].vehicle_mean.mean())
                rows.append(
                    dict(
                        camera_id=camera,
                        date_local=day,
                        start_hour=start,
                        end_hour=start + 2,
                        split="holdout" if day == HOLDOUT else "discovery",
                        peak_mean=pm,
                        baseline_mean=bm,
                        relative_pct=100 * (pm / bm - 1) if bm >= 1 else np.nan,
                        baseline_hours_json=json.dumps(sorted(baseline)),
                    )
                )
            discovery = [r for r in rows if r["split"] == "discovery"]
            if len(discovery) < 3:
                continue
            held = next((r for r in rows if r["split"] == "holdout"), None)
            delta = float(np.mean([r["relative_pct"] for r in discovery]))
            candidates.append(
                dict(
                    camera_id=camera,
                    start_hour=start,
                    end_hour=start + 2,
                    discovery_days=len(discovery),
                    discovery_relative_pct=delta,
                    positive_discovery_days=sum(r["relative_pct"] > 0 for r in discovery),
                    holdout_available=held is not None,
                    holdout_relative_pct=held["relative_pct"] if held else np.nan,
                    baseline_hours_json=json.dumps(sorted(baseline)),
                )
            )
            daily.extend(rows)
    result = pd.DataFrame(candidates)
    best = (
        result.sort_values(
            ["camera_id", "discovery_relative_pct", "start_hour"], ascending=[True, False, True]
        )
        .drop_duplicates("camera_id")
        .copy()
    )
    best["discovery_screen_pass"] = best.discovery_relative_pct.ge(20) & best.positive_discovery_days.ge(3)
    best["holdout_direction_repeated"] = best.holdout_available & best.holdout_relative_pct.gt(0)
    return best, pd.DataFrame(daily)


def interval_values(part, start, end):
    q = part[part.minute_local.ge(start) & part.minute_local.lt(end)]
    good = q[q.eligible_temporal]
    duration = end - start
    expected = duration / 10
    accepted = (
        len(good) >= max(2, math.ceil(duration / 60 * 3))
        and len(good) >= expected * 0.5
        and good.minute_local.min() <= start + duration * 0.35
        and good.minute_local.max() >= end - duration * 0.35
    )
    return dict(
        n=len(good),
        accepted=accepted,
        **{m: float(good[m + "_count"].mean()) if accepted else np.nan for m in METRICS},
    )


def school_anchor_comparisons(source, schedules):
    """Documented clock references with explicit date applicability and clock sensitivity."""
    d = source.copy()
    t = pd.to_datetime(d.observed_at_utc, utc=True).dt.tz_convert("Asia/Bangkok")
    d["minute_local"] = t.dt.hour * 60 + t.dt.minute + t.dt.second / 60
    rows = []
    for s in schedules[schedules.temporal_status.str.startswith("documented")].to_dict("records"):
        cam = d[d.camera_id.eq(str(s["camera_id"]))]
        for day, part in cam.groupby("date_local"):
            if pd.Timestamp(day).weekday() not in json.loads(s["weekdays_json"]):
                continue
            for offset in [-30, -15, 0, 15, 30]:
                anchor = s["anchor_minute"] + offset
                for width in [30, 60, 90]:
                    a = anchor - width if s["event"] == "arrival" else anchor
                    b = a + width
                    if a < 360 or b > 1260:
                        continue
                    event = interval_values(part, a, b)
                    for baseline, first, last in [("earlier", a - width, a), ("later", b, b + width)]:
                        if first < 360 or last > 1260:
                            continue
                        ref = interval_values(part, first, last)
                        passed = event["accepted"] and ref["accepted"]
                        rows.append(
                            dict(
                                camera_id=s["camera_id"],
                                date_local=day,
                                schedule_id=s["schedule_id"],
                                scope=s["scope"],
                                temporal_status=s["temporal_status"],
                                source_url=s["source_url"],
                                date_applicability=(
                                    "term_context_supported_daily_attendance_unknown"
                                    if s["camera_id"] == "139"
                                    else "class_timetable_reference_august_effective_interval_unverified"
                                ),
                                anchor_shift_minutes=offset,
                                width_minutes=width,
                                baseline=baseline,
                                window_start_minute=a,
                                window_end_minute=b,
                                baseline_start_minute=first,
                                baseline_end_minute=last,
                                event_frames=event["n"],
                                baseline_frames=ref["n"],
                                coverage_pass=passed,
                                pair_ends_before_18=max(b, last) <= 1080,
                                **{m + "_event_mean": event[m] if passed else np.nan for m in METRICS},
                                **{m + "_baseline_mean": ref[m] if passed else np.nan for m in METRICS},
                                **{
                                    m + "_difference": event[m] - ref[m] if passed else np.nan
                                    for m in METRICS
                                },
                            )
                        )
    return pd.DataFrame(rows)

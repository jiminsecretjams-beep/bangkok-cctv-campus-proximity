#!/usr/bin/env python3
"""Join validated whole-image traffic observations to the existing campus dimension.

Run with the frozen worker's Python environment (pandas and pyarrow are already pinned).
This downstream step never runs inference or changes the frozen worker or source tables.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
JOB = "whole-image-f4ba6aa4fcd2ca80"
DEFAULT_TRAFFIC = ROOT / "bangkok-cctv-worker-batch-01/processing/data/combined" / (JOB + "-all-batches-20261111")
BOOL_FIELDS = {"near_any_education", "near_school", "near_university_college", "review_required", "within_radius_of_mapped_university_boundary"}
INT_FIELDS = {"image_count", "institutions_within_radius", "schools_within_radius", "universities_colleges_within_radius"}
FLOAT_FIELDS = {"latitude", "longitude", "radius_m", "nearest_distance_m", "nearest_school_distance_m", "nearest_university_college_distance_m", "nearest_mapped_university_boundary_distance_m"}
CAMERA_NAMES = {"location": "camera_location", "location_detail": "camera_location_detail", "location_en": "camera_location_en", "location_detail_en": "camera_location_detail_en", "latitude": "camera_latitude", "longitude": "camera_longitude", "image_count": "camera_reported_image_count", "coordinate_status": "camera_coordinate_status"}
MONTHS = {name: i for i, name in enumerate("January February March April May June July August September October November December".split(), 1)}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def camera_keys(series):
    require(series.notna().all(), "Camera IDs must not be missing")
    require(series.map(lambda v: isinstance(v, str) and bool(v) and v == v.strip()).all(), "Camera IDs must be nonempty strings without surrounding whitespace")


def nullable_boolean(series):
    require(series.isin(["True", "False", ""]).all(), "Unexpected boolean text; preserve unknown values explicitly")
    return series.map({"True": True, "False": False, "": pd.NA}).astype("boolean")


def read_campus_flags(path):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    camera_keys(frame.camera_id)
    require(not frame.camera_id.duplicated().any(), "Duplicate camera IDs in campus dimension would multiply observations")
    for column in frame:
        if column in BOOL_FIELDS:
            frame[column] = nullable_boolean(frame[column])
        elif column in INT_FIELDS | FLOAT_FIELDS:
            frame[column] = pd.to_numeric(frame[column].replace("", pd.NA), errors="raise").astype("Int64" if column in INT_FIELDS else "Float64")
        elif column != "camera_id":
            frame[column] = frame[column].replace("", pd.NA).astype("string")
    require(frame.radius_m.notna().all() and frame.radius_m.eq(1500).all(), "Campus flags must use the assigned inclusive 1,500 m radius")
    # Camera IP addresses are infrastructure metadata and are not analysis features.
    frame = frame.drop(columns=["ip_address"], errors="ignore")
    return frame.rename(columns={c: CAMERA_NAMES.get(c, "campus_" + c) for c in frame if c != "camera_id"})


def join_observations(traffic, campus):
    camera_keys(traffic.camera_id)
    camera_keys(campus.camera_id)
    require(not campus.camera_id.duplicated().any(), "Duplicate campus camera key")
    require(set(traffic.camera_id) == set(campus.camera_id), "Traffic/campus camera sets differ; investigate before joining")
    require(not (set(traffic) & set(campus) - {"camera_id"}), "Unexpected overlapping column names")
    joined = traffic.merge(campus, on="camera_id", how="left", sort=False, validate="many_to_one", indicator="campus_join_status")
    require(len(joined) == len(traffic) and joined.campus_join_status.eq("both").all(), "Campus join lost or multiplied source observations")
    for column in traffic:
        pd.testing.assert_series_equal(joined[column], traffic[column].reset_index(drop=True))
    joined["campus_join_status"] = "matched"
    local = pd.to_datetime(joined.observed_at_utc, utc=True).dt.tz_convert("Asia/Bangkok")
    if "observed_at_local" in joined:
        require(joined.observed_at_local.equals(local), "Original Bangkok/UTC timestamps disagree")
    if "date_local" in joined:
        require(joined.date_local.astype("string").equals(local.dt.strftime("%Y-%m-%d").astype("string")), "Original local dates disagree with Bangkok time")
    if "hour_local" in joined:
        require(joined.hour_local.astype("Int64").equals(local.dt.hour.astype("Int64")), "Original local hours disagree with Bangkok time")
    if "is_weekend" in joined:
        expected = local.dt.dayofweek.ge(5).astype("boolean").mask(local.isna())
        require(joined.is_weekend.equals(expected), "Original weekend flags disagree with Bangkok time")
    joined["minute_local"] = local.dt.minute.astype("Int8")
    joined["weekday_index_local"] = local.dt.dayofweek.astype("Int8")
    joined["day_type"] = pd.Series("weekday", index=joined.index, dtype="string").mask(local.dt.dayofweek.ge(5), "weekend").mask(local.isna(), "unknown")
    joined["teaching_day"] = pd.Series(pd.NA, index=joined.index, dtype="boolean")
    joined["calendar_status"] = "unknown_no_verified_institution_calendar"
    return joined


def parse_event_dates(text):
    tokens = re.findall(r"(" + "|".join(MONTHS) + r")\s+(\d{1,2})(?:,\s*(\d{4}))?", text)
    require(1 <= len(tokens) <= 2, "Unrecognized calendar date text: " + text)
    years = [int(y) for _, _, y in tokens if y]
    require(bool(years), "Calendar text has no explicit year: " + text)
    values = []
    for month, day, year in tokens:
        values.append(date(int(year) if year else years[-1], MONTHS[month], int(day)))
    if len(values) == 2 and values[0] > values[1] and not tokens[0][2]:
        values[0] = values[0].replace(year=values[0].year - 1)
    require(values[0] <= values[-1], "Reversed calendar range")
    return values[0], values[-1]


def normalize_calendar(path, start, end):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    rows = []
    for event in frame.to_dict("records"):
        try:
            first, last = parse_event_dates(event["date_information"])
            parsed, overlaps = "parsed_from_supplied_text", first <= end and last >= start
        except ValueError:
            first = last = None
            parsed, overlaps = "unparsed_preserved", None
        rows.append(dict(event, event_start_date=first, event_end_date=last, date_parse_status=parsed,
                         event_date_overlaps_capture_period=overlaps,
                         verification_status="supplied_not_independently_verified",
                         applicability="named_programme_only_not_all_nearby_institutions"))
    return pd.DataFrame(rows)


def hourly_summary(frame):
    work = frame.copy(deep=False)
    local = pd.to_datetime(frame.observed_at_utc, utc=True).dt.tz_convert("Asia/Bangkok")
    work = work.assign(date_local=local.dt.strftime("%Y-%m-%d"), hour_local=local.dt.hour,
                       review_image=frame.quality_status.eq("review"), invalid_image=frame.quality_status.eq("invalid"))
    keys = ["camera_id", "date_local", "hour_local"]
    dated = work[local.notna()]
    require(not dated.empty, "No timestamped source observations")
    days = pd.date_range(local.min().normalize(), local.max().normalize(), freq="D").strftime("%Y-%m-%d")
    grid = pd.MultiIndex.from_product([sorted(frame.camera_id.unique()), days, range(24)], names=keys)
    result = dated.groupby(keys).agg(source_images=("image_id", "size"), eligible_temporal_images=("eligible_temporal", "sum"), review_images=("review_image", "sum"), invalid_images=("invalid_image", "sum")).reindex(grid, fill_value=0)
    result["excluded_images"] = result.source_images - result.eligible_temporal_images
    eligible = dated[dated.eligible_temporal]
    means = eligible.groupby(keys).agg(mean_visible_vehicle_count=("vehicle_count", "mean"), median_visible_vehicle_count=("vehicle_count", "median"), mean_visible_motor_vehicle_count=("motor_vehicle_count", "mean"))
    result = result.join(means)
    result["eligible_fraction"] = (result.eligible_temporal_images / result.source_images).where(result.source_images.gt(0))
    result["coverage_status"] = "observed_eligible"
    result.loc[result.eligible_temporal_images.eq(0), "coverage_status"] = "no_eligible_observations"
    result.loc[result.source_images.eq(0), "coverage_status"] = "no_observations"
    result = result.reset_index()
    result["day_type"] = pd.to_datetime(result.date_local).dt.dayofweek.map(lambda n: "weekend" if n >= 5 else "weekday")
    return result


def write_csv(frame, path):
    portable = frame.copy()
    for column in ["quality_flags", "exclusion_reasons"]:
        if column in portable:
            portable[column] = portable[column].map(lambda value: json.dumps(value.tolist() if hasattr(value, "tolist") else value, ensure_ascii=False))
    portable.to_csv(path, index=False, encoding="utf-8-sig", na_rep="")


def build(traffic_folder, flags_path, calendar_path, output):
    paths = {"traffic": traffic_folder / "traffic_full_frame.parquet", "traffic_import_manifest": traffic_folder / "import_manifest.json", "campus_flags": flags_path, "supplied_calendar": calendar_path, "builder": Path(__file__)}
    sources = {name: dict(path=str(path.resolve().relative_to(ROOT)) if path.resolve().is_relative_to(ROOT) else str(path.resolve()), sha256=sha256(path)) for name, path in paths.items()}
    if output.exists():
        previous = json.loads((output / "manifest.json").read_text())
        require(previous["sources"] == sources, "Existing output belongs to different inputs/code; choose a fresh --output path")
        require(all(sha256(output / name) == digest for name, digest in previous["files"].items()), "Existing output checksum mismatch")
        return dict(status="already_verified", output=str(output), **previous["coverage"])
    manifest = json.loads(paths["traffic_import_manifest"].read_text())
    require(manifest["status"] == "completed" and manifest["job_id"] == JOB and not manifest["missing_batches"], "Expected the complete frozen job")
    require(sha256(paths["traffic"]) == manifest["files"]["traffic_full_frame.parquet"], "Combined traffic checksum differs from its validated import manifest")
    traffic = pd.read_parquet(paths["traffic"])
    require(len(traffic) == manifest["images"] == 61319 and traffic.camera_id.nunique() == manifest["cameras"] == 100, "Combined source coverage differs from frozen assignment")
    require(traffic.image_id.is_unique and traffic.inference_status.eq("success").all(), "Unexpected duplicated or unfinished source image")
    campus = read_campus_flags(flags_path)
    joined = join_observations(traffic, campus)
    require(joined.loc[joined.eligible_temporal, ["vehicle_count", "motor_vehicle_count"]].notna().all().all(), "Eligible observations contain unknown counts")
    local = joined.observed_at_utc.dt.tz_convert("Asia/Bangkok")
    calendar = normalize_calendar(calendar_path, local.min().date(), local.max().date())
    hourly = hourly_summary(joined)
    coverage = joined.groupby("camera_id").agg(source_images=("image_id", "size"), eligible_temporal_images=("eligible_temporal", "sum"), first_observed_at_utc=("observed_at_utc", "min"), last_observed_at_utc=("observed_at_utc", "max")).reset_index().merge(campus, on="camera_id", validate="one_to_one")
    coverage["reported_image_count_difference"] = coverage.source_images - coverage.camera_reported_image_count
    checks = dict(many_to_one_join=True, source_rows_and_order_preserved=True, all_original_columns_unchanged=True,
                  source_camera_set_preserved=True, unique_image_ids=bool(joined.image_id.is_unique),
                  radius_m=1500, bangkok_time_fields_reconciled=True,
                  temporal_eligibility_unchanged=bool(joined.eligible_temporal.equals(traffic.eligible_temporal)),
                  teaching_day_unknown_preserved=bool(joined.teaching_day.isna().all()),
                  hourly_source_count_reconciles=int(hourly.source_images.sum()) == int(local.notna().sum()),
                  hourly_eligible_count_reconciles=int(hourly.eligible_temporal_images.sum()) == int(joined.eligible_temporal.sum()),
                  missing_or_excluded_hours_have_null_means=bool(hourly.loc[hourly.eligible_temporal_images.eq(0), "mean_visible_vehicle_count"].isna().all()),
                  camera_source_count_reconciles=int(coverage.source_images.sum()) == len(joined))
    require(all(value is not False for value in checks.values()), "Analysis dataset reconciliation failed")
    stats = dict(source_images=len(joined), cameras=int(joined.camera_id.nunique()), eligible_temporal_images=int(joined.eligible_temporal.sum()),
                 quality_status={str(k): int(v) for k, v in joined.quality_status.value_counts().items()},
                 capture_start_local=local.min().isoformat(), capture_end_local=local.max().isoformat(),
                 near_school_cameras=int(campus.campus_near_school.sum()), near_higher_education_cameras=int(campus.campus_near_university_college.sum()),
                 supplied_calendar_events=len(calendar), parsed_calendar_events=int(calendar.date_parse_status.eq("parsed_from_supplied_text").sum()),
                 verified_teaching_day_rows=0, hourly_grid_rows=len(hourly),
                 unknown_timestamp_rows=int(local.isna().sum()))
    limitations = ["The supplied calendar is programme-specific and has not been independently verified; teaching_day stays unknown for every camera/date.",
                   "Every camera has a nearby recorded school: the school flag supplies no unexposed camera group.",
                   "Higher-education no-matches retain campus-edge, threshold and incomplete-catalog review flags; they are not certified controls.",
                   "Only seven capture dates are present; weekday/weekend differences do not establish a campus commuting effect.",
                   "Counts are provisional visible vehicle instances per full image, including parked vehicles; hourly means are snapshot averages, not vehicles per hour or unique journeys.",
                   "Quality-excluded rows retain raw predictions and original null analysis counts. Use eligible_temporal for descriptive time comparisons."]
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + ".building-", dir=output.parent))
    joined.to_parquet(staging / "traffic_campus.parquet", index=False)
    write_csv(joined, staging / "traffic_campus.csv")
    joined[joined.eligible_temporal].to_parquet(staging / "traffic_campus_temporal.parquet", index=False)
    write_csv(hourly, staging / "camera_hourly_coverage.csv")
    write_csv(coverage, staging / "camera_coverage.csv")
    write_csv(calendar, staging / "calendar_events.csv")
    dictionary = {c: dict(dtype=str(joined[c].dtype), source="frozen traffic export; preserved unchanged" if c in traffic else "campus reference dimension" if c in campus else "derived or explicit unknown") for c in joined}
    dictionary["teaching_day"]["meaning"] = "Nullable boolean; unknown until an applicable institution calendar is independently verified. Weekdays are not teaching-day labels."
    dictionary["campus_join_status"]["meaning"] = "matched: exactly one camera dimension row joined to this observation"
    dictionary["camera_reported_image_count"]["meaning"] = "Original metadata count, retained separately from actual source-image counts"
    dictionary["day_type"]["meaning"] = "Bangkok local weekday/weekend classification; no holiday or teaching-calendar inference"
    dictionary["camera_latitude"]["meaning"] = "Camera coordinate, not a campus coordinate"
    dictionary["camera_longitude"]["meaning"] = "Camera coordinate, not a campus coordinate"
    (staging / "data_dictionary.json").write_text(json.dumps(dictionary, indent=2, ensure_ascii=False) + "\n")
    report = dict(status="passed", checks=checks, coverage=stats, limitations=limitations)
    (staging / "validation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (staging / "README.md").write_text(f"""# Traffic and campus analysis handoff

{stats['source_images']:,} original observations / {stats['cameras']} cameras; {stats['eligible_temporal_images']:,} observations eligible for descriptive temporal analysis. Capture window: {stats['capture_start_local']} to {stats['capture_end_local']} (Asia/Bangkok).

- traffic_campus.parquet / traffic_campus.csv: authoritative one-row-per-source-image table. All original traffic fields are unchanged. Camera metadata has camera_ names; campus reference fields have campus_ names. Campus dimension joins are many-to-one on string camera_id. No inference was rerun.
- traffic_campus_temporal.parquet: exactly the original eligible_temporal subset; calendar eligibility is not implied.
- camera_hourly_coverage.csv: every camera, capture-period date and local hour. source_images=0 means no observation. Vehicle means/medians remain blank when no eligible snapshots exist; observed zero counts remain zero. Statistics are averages of snapshots, not traffic flow. Hours are not weighted to infer missing snapshots.
- camera_coverage.csv: actual image coverage and campus dimension; metadata-reported image counts remain separate.
- calendar_events.csv: supplied programme-specific evidence with parsed event dates, original text and URL. Event-date overlap is not a teaching-day classification. Source pages were not independently rechecked in this offline build.
- data_dictionary.json, validation_report.json and manifest.json: types, lineage, checksums and validation.

## Load

```python
import pandas as pd
df = pd.read_parquet('traffic_campus.parquet')
analysis = df.loc[df['eligible_temporal']].copy()
# teaching_day is unknown. Do not replace it with weekday/weekend.
```

CSV is UTF-8 with BOM. quality_flags and exclusion_reasons are JSON arrays. Blank numeric/boolean values mean unknown; numeric zero means a successful zero count. Counts include parked vehicles.

## Limits

""" + "\n".join("- " + line for line in limitations) + "\n\nCampus reference attribution: © OpenStreetMap contributors (ODbL), plus the web scraping team's source records. See the project README and original reference provenance.\n")
    saved = pd.read_parquet(staging / "traffic_campus.parquet")
    pd.testing.assert_frame_equal(saved, joined)
    require(pd.read_parquet(staging / "traffic_campus_temporal.parquet").shape[0] == stats["eligible_temporal_images"], "Saved temporal subset count mismatch")
    require(all(sha256(paths[name]) == source["sha256"] for name, source in sources.items()), "An input changed during the build")
    run = dict(schema_version=1, algorithm="campus-traffic-join-v1", status="completed", created_at=datetime.now(timezone.utc).isoformat(),
               job_id=JOB, sources=sources, coverage=stats, files={p.name: sha256(p) for p in sorted(staging.iterdir()) if p.is_file()})
    (staging / "manifest.json").write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n")
    staging.rename(output)
    return dict(status="completed", output=str(output.resolve()), **stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traffic-folder", type=Path, default=DEFAULT_TRAFFIC)
    parser.add_argument("--campus-flags", type=Path, default=ROOT / "data/processed/cctv_campus_flags.csv")
    parser.add_argument("--calendar", type=Path, default=ROOT / "data/raw/webscraping/data/academic_calendar.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/analysis/campus_traffic")
    args = parser.parse_args()
    print(json.dumps(build(args.traffic_folder, args.campus_flags, args.calendar, args.output), indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Add source-qualified public-holiday reference fields without changing v1 observations."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from build_analysis_dataset import ROOT, require, sha256, source_metadata, write_csv


def validate_reference(holidays, metadata):
    years = set(metadata["available_years"])
    require(bool(years), "Holiday reference has no declared years")
    dates = pd.to_datetime(holidays.date, format="%Y-%m-%d", errors="raise")
    require(dates.dt.year.isin(years).all(), "Holiday date outside declared year coverage")
    require(dates.dt.year.eq(pd.to_numeric(holidays.year)).all(), "Holiday date/year mismatch")
    require((dates.dt.year + 543).eq(pd.to_numeric(holidays.buddhist_year)).all(), "Holiday Buddhist-year mismatch")
    require(holidays.is_substitution.isin(["True", "False"]).all(), "Unexpected substitution boolean text")
    require(len(holidays) == metadata["rows"], "Holiday rows differ from reviewed import")
    return years


def attach_holidays(frame, dates, holidays, years):
    require(dates.index.equals(frame.index), "Holiday date index must match source row order")
    original = frame.copy(deep=False)
    result = frame.copy()
    normalized = pd.to_datetime(dates, errors="raise").dt.strftime("%Y-%m-%d")
    names = {}
    for value, group in holidays.groupby("date", sort=True):
        names[value] = json.dumps(sorted(set(str(row.name_en or row.name_th) for row in group.itertuples())), ensure_ascii=False)
    covered = pd.to_datetime(normalized).dt.year.isin(years) & normalized.notna()
    listed = normalized.isin(names).astype("boolean").mask(~covered)
    additions = {"public_holiday_listed", "public_holiday_names", "public_holiday_reference_status"}
    require(not additions.intersection(frame.columns), "Holiday fields already exist; preserve the input version")
    result["public_holiday_listed"] = listed
    result["public_holiday_names"] = normalized.map(names).fillna("[]").astype("string").mask(~covered)
    result["public_holiday_reference_status"] = pd.Series("unknown_reference_year_or_date", index=frame.index, dtype="string")
    result.loc[covered, "public_holiday_reference_status"] = "not_listed_in_supplied_reference"
    result.loc[covered & listed.fillna(False), "public_holiday_reference_status"] = "listed_in_supplied_reference"
    pd.testing.assert_frame_equal(result[original.columns], original)
    return result


def build(baseline, reference, reference_manifest, output):
    meta = json.loads(reference_manifest.read_text())
    old = json.loads((baseline / "manifest.json").read_text())
    inputs = {"baseline_manifest": baseline / "manifest.json", "public_holidays": reference,
              "public_holiday_source_manifest": reference_manifest, "holiday_builder": Path(__file__),
              "holiday_snapshot": ROOT / "data/reference/public_holidays/thai_public_holidays.json",
              "provided_holiday_script": ROOT / meta["script_path"]}
    sources = {key: source_metadata(path) for key, path in inputs.items()}
    if output.exists():
        previous = json.loads((output / "manifest.json").read_text())
        require(previous["sources"] == sources, "Different inputs/code; choose a fresh output path")
        require(all(sha256(output / name) == digest for name, digest in previous["files"].items()), "Existing output checksum mismatch")
        return dict(status="already_verified", output=str(output), **previous["coverage"])
    require(old["status"] == "completed" and old["coverage"]["source_images"] == 61319, "Expected the completed baseline analysis handoff")
    require(sha256(inputs["holiday_snapshot"]) == meta["snapshot_sha256"], "Holiday snapshot checksum changed")
    require(sha256(inputs["provided_holiday_script"]) == meta["script_sha256"], "Provided script changed")
    for name, digest in old["files"].items():
        require(sha256(baseline / name) == digest, "Baseline checksum mismatch: " + name)
    holidays = pd.read_csv(reference, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    years = validate_reference(holidays, meta)
    traffic = pd.read_parquet(baseline / "traffic_campus.parquet")
    local_dates = traffic.observed_at_utc.dt.tz_convert("Asia/Bangkok").dt.strftime("%Y-%m-%d")
    joined = attach_holidays(traffic, local_dates, holidays, years)
    hourly = pd.read_csv(baseline / "camera_hourly_coverage.csv", dtype={"camera_id": str})
    hourly_joined = attach_holidays(hourly, hourly.date_local, holidays, years)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + ".building-", dir=output.parent))
    for name in old["files"]:
        shutil.copy2(baseline / name, staging / name)
    joined.to_parquet(staging / "traffic_campus.parquet", index=False)
    write_csv(joined, staging / "traffic_campus.csv")
    joined[joined.eligible_temporal].to_parquet(staging / "traffic_campus_temporal.parquet", index=False)
    write_csv(hourly_joined, staging / "camera_hourly_coverage.csv")
    shutil.copy2(reference, staging / "public_holidays.csv")
    shutil.copy2(reference_manifest, staging / "public_holiday_source_manifest.json")
    report = json.loads((staging / "validation_report.json").read_text())
    coverage = dict(old["coverage"], holiday_reference_rows=len(holidays), holiday_reference_years=sorted(years),
                    listed_public_holiday_observations=int(joined.public_holiday_listed.fillna(False).sum()),
                    unknown_holiday_reference_observations=int(joined.public_holiday_listed.isna().sum()))
    report["coverage"] = coverage
    report["checks"].update(holiday_join_preserves_every_original_column=True, baseline_dataset_unmodified=True,
                            holiday_dates_use_bangkok_time=True, holiday_year_coverage_validated=True,
                            teaching_day_unchanged=bool(joined.teaching_day.equals(traffic.teaching_day)))
    require(all(bool(value) for value in report["checks"].values()), "Holiday handoff validation failed")
    report["limitations"].extend(meta["limitations"][:3])
    (staging / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    dictionary = json.loads((staging / "data_dictionary.json").read_text())
    for name, meaning in {
        "public_holiday_listed": "Nullable boolean: whether the Bangkok date appears in the supplied third-party reference; unknown for uncovered years/dates. False does not establish a teaching day or official calendar completeness.",
        "public_holiday_names": "JSON list of English/Thai names for that date, grouped before the join to avoid multiplying rows; unknown for uncovered dates.",
        "public_holiday_reference_status": "Listed / not listed in supplied reference, or unknown reference year/date. This is not a verified school calendar.",
    }.items():
        dictionary[name] = dict(dtype=str(joined[name].dtype), source="provided holiday script and pinned third-party reference", meaning=meaning)
    (staging / "data_dictionary.json").write_text(json.dumps(dictionary, ensure_ascii=False, indent=2) + "\n")
    with (staging / "README.md").open("a") as stream:
        stream.write("\n## Public holiday reference update (v2)\n\nThe three public_holiday_ columns are source-qualified flags from the friend's preserved script and a commit-pinned JSON snapshot. The source lists 46 events across 2025–2026 (22 in 2026), advertises last_updated 2025-02-20, and has not been independently verified as a complete current official calendar. Names are grouped by date before joining; original observation columns, quality flags and teaching_day values are unchanged.\n\nNo listed holiday overlaps 25–31 August 2026, so the captured data do not support a public-holiday traffic comparison. Weekday/weekend comparisons remain possible. Nonholiday does not imply classes were held. public_holidays.csv and public_holiday_source_manifest.json provide the reference, scope and provenance. The prior v1 dataset remains unchanged.\n")
    pd.testing.assert_frame_equal(pd.read_parquet(staging / "traffic_campus.parquet"), joined)
    require(len(joined) == len(traffic) == 61319 and joined.camera_id.nunique() == 100, "Source coverage changed")
    require(all(sha256(path) == sources[name]["sha256"] for name, path in inputs.items()), "Input changed during holiday build")
    run = dict(schema_version=2, algorithm="campus-traffic-public-holidays-v1", status="completed",
               created_at=datetime.now(timezone.utc).isoformat(), job_id=old["job_id"], sources=sources,
               coverage=coverage, files={p.name: sha256(p) for p in sorted(staging.iterdir()) if p.is_file()})
    (staging / "manifest.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n")
    staging.rename(output)
    return dict(status="completed", output=str(output.resolve()), **coverage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "data/analysis/campus_traffic")
    parser.add_argument("--reference", type=Path, default=ROOT / "data/raw/webscraping/data/public_holidays.csv")
    parser.add_argument("--reference-manifest", type=Path, default=ROOT / "data/reference/public_holidays/source_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/analysis/campus_traffic_holidays_v2")
    args = parser.parse_args()
    print(json.dumps(build(args.baseline, args.reference, args.reference_manifest, args.output), indent=2))

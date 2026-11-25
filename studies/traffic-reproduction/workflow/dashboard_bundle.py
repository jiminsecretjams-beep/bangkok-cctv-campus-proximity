"""Prepare the portable dashboard's minimum verified data; never runs inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CAMPUS = ROOT / "inputs/campus"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(analysis, calendar, output):
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((analysis / "manifest.json").read_text())
    for relative, rec in manifest["files"].items():
        assert digest(analysis / relative) == rec["sha256"], relative
    assert json.loads((analysis / "independent_verification.json").read_text())["result"] == "pass"
    sources = {}

    def parquet(name, source, columns=None):
        sources[str(source.relative_to(ROOT))] = digest(source)
        if source.suffix == ".parquet":
            d = pd.read_parquet(source, columns=columns)
        else:
            d = pd.read_csv(
                source, dtype={"camera_id": str}, keep_default_na=False, float_precision="round_trip"
            )
            if columns:
                d = d[columns]
        if "camera_id" in d:
            d["camera_id"] = d.camera_id.astype(str)
        d.to_parquet(output / (name + ".parquet"), index=False, compression="zstd")
        return d

    # Numeric CSV fields preserve missing values rather than becoming empty text.
    csv_tables = {
        "daily": "balanced_camera_day_period.csv",
        "context": "camera_context.csv",
        "eligibility": "camera_eligibility.csv",
        "frozen_periods": "camera_period_comparisons.csv",
        "anchors": "school_anchor_daily_sensitivity.csv",
        "sensitivity": "sensitivity_checks.csv",
        "peaks": "weekday_peak_holdout.csv",
        "class_summary": "vehicle_class_comparisons.csv",
    }
    for name, file in csv_tables.items():
        source = analysis / file
        sources[str(source.relative_to(ROOT))] = digest(source)
        pd.read_csv(source, dtype={"camera_id": str}, float_precision="round_trip").to_parquet(
            output / (name + ".parquet"), index=False
        )
    hourly = parquet("hourly", analysis / "camera_day_hour.parquet")
    classes = ["car", "motorcycle", "bus", "truck", "bicycle", "vehicle"]
    frame_columns = [
        "image_id",
        "camera_id",
        "observed_at_utc",
        "date_local",
        "hour_local",
        "is_weekend",
        "quality_status",
        "quality_reason",
        "eligible_temporal",
    ]
    frame_columns += [prefix + m + "_count" for prefix in ["", "raw_"] for m in classes]
    frames = parquet("frames", calendar / "observations_with_calendar.parquet", frame_columns)
    parquet("dates", calendar / "date_labels.csv")
    parquet("calendar", calendar / "institution_scope_calendar_daily.csv")
    parquet("camera_calendar", calendar / "camera_calendar_daily.csv")
    parquet(
        "institutions",
        CAMPUS / "institutions_clean.csv",
        [
            "institution_id",
            "institution_name",
            "institution_type",
            "latitude",
            "longitude",
            "point_role",
            "sources",
            "source_urls",
            "official_url",
        ],
    )
    parquet(
        "links",
        CAMPUS / "camera_institution_matches.csv",
        ["camera_id", "institution_id", "institution_name", "institution_type", "distance_m"],
    )
    json_files = {
        "summary.json": analysis / "summary.json",
        "rules.json": calendar / "rules.json",
        "sources.json": calendar / "reviewed_sources.json",
        "verification.json": analysis / "independent_verification.json",
    }
    for name, source in json_files.items():
        sources[str(source.relative_to(ROOT))] = digest(source)
        (output / name).write_bytes(source.read_bytes())
    for name in ["FINDINGS.md", "METHODS.md"]:
        (output / name).write_bytes((ROOT / name).read_bytes())
    assert len(frames) == 61319 and frames.image_id.is_unique and len(hourly) == 10500
    frozen = pd.read_parquet(output / "frozen_periods.parquet")
    assert len(frozen) == 267
    report = dict(
        schema_version=1,
        dataset_id="bangkok-school-zone-20260825-31-v1",
        dates=sorted(frames.date_local.unique()),
        images=len(frames),
        cameras=frames.camera_id.nunique(),
        usable_images=int(frames.eligible_temporal.sum()),
        hour_cells=len(hourly),
        source_hashes=sources,
        files={
            p.name: dict(bytes=p.stat().st_size, sha256=digest(p))
            for p in output.iterdir()
            if p.is_file() and p.name != "manifest.json"
        },
    )
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "files": len(report["files"]),
                "images": len(frames),
                "bytes": sum(x["bytes"] for x in report["files"].values()),
            },
            indent=2,
        )
    )

"""Full-image vehicle counting, resumable inference, and analysis exports without ROIs.

Raw predictions exist even for quality-review/invalid images. Analysis counts are
nullable and restricted to usable, unique acquisitions with successful inference.
SQLite checkpoints hold one output per exact image content and model profile.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import sqlite3
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import digest, file_hash, load_config
from .detector import YOLODetector, verify_model
from .quality import assess_quality
from .storage import SCHEMAS, atomic_json, now, read_json, read_table

CLASSES = ("car", "motorcycle", "bus", "truck", "bicycle")
COUNT_COLUMNS = [c + "_count" for c in CLASSES] + ["vehicle_count"]
THRESHOLD = 0.25


def profile(config):
    return dict(
        algorithm="full-frame-all-decodable-v1",
        detector=config.detector.model_dump(),
        target_classes=CLASSES,
        thresholds=dict.fromkeys(CLASSES, THRESHOLD),
        counting_scope="full_image",
        decoder="PIL-exif-rgb-v1",
        versions={
            n: importlib.metadata.version(n) for n in ["ultralytics", "torch", "torchvision", "Pillow"]
        },
    )


def count_boxes(boxes):
    counts = dict.fromkeys(CLASSES, 0)
    for box in boxes:
        label = box["mapped_class"]
        if label in counts and box["confidence"] >= THRESHOLD:
            counts[label] += 1
    return {**{c + "_count": n for c, n in counts.items()}, "vehicle_count": sum(counts.values())}


def connect(path):
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS outputs (key TEXT PRIMARY KEY, status TEXT, payload TEXT)")
    db.commit()
    return db


def output_key(image, profile_hash):
    return digest([image["content_sha256"], profile_hash])


def infer(config, folder, images, raw_root, profile_hash, limit=None, detector=None, snapshot=None):
    """Do not consult region or quality status; attempt every decodable image."""
    db = connect(folder / "checkpoints.sqlite")
    done = {r[0] for r in db.execute("SELECT key FROM outputs WHERE status='success'")}
    started = time.monotonic()
    predicted = cached = failures = processed = 0
    errors = []
    last_progress = started
    last_snapshot = 0
    snapshot_interval = 500
    try:
        for image in images:
            if image["decode_status"] != "success":
                continue
            key = output_key(image, profile_hash)
            processed += 1
            if key in done:
                cached += 1
            else:
                if limit is not None and predicted >= limit:
                    break
                if detector is None:
                    detector = YOLODetector(config, target_classes=CLASSES)
                value = None
                error = None
                for _ in range(config.detector.max_attempts):
                    try:
                        value = detector.predict([raw_root / image["source_path"]])[0]
                        break
                    except (RuntimeError, OSError, ValueError) as exc:
                        error = exc
                if value is None:
                    failures += 1
                    value = dict(error_type=type(error).__name__, error=str(error)[:500])
                    errors.append(dict(image_id=image["image_id"], **value))
                    status = "failed"
                else:
                    status = "success"
                    done.add(key)
                    predicted += 1
                db.execute("INSERT OR REPLACE INTO outputs VALUES (?,?,?)", (key, status, json.dumps(value)))
                if failures >= 10 and predicted == 0:
                    db.commit()
                    raise RuntimeError(f"Model failed on first ten inputs: {errors[-1]}")
            elapsed = time.monotonic() - started
            if processed % 32 == 0:
                db.commit()
            if snapshot and predicted - last_snapshot >= snapshot_interval:
                db.commit()
                snapshot()
                last_snapshot = predicted
                snapshot_interval = 5000
            if time.monotonic() - last_progress > 20 or processed == len(images):
                db.commit()
                record = dict(
                    status="running",
                    updated_at=now(),
                    accounted_images=processed,
                    total_images=len(images),
                    newly_predicted=predicted,
                    reused_images=cached,
                    failed_images=failures,
                    unique_successful_outputs=len(done),
                    elapsed_seconds=elapsed,
                    estimated_remaining_seconds=(len(images) - processed)
                    * elapsed
                    / max(1, predicted + cached),
                )
                atomic_json(folder / "progress.json", record)
                print(json.dumps(record), flush=True)
                last_progress = time.monotonic()
        db.commit()
    finally:
        db.close()
    return dict(
        newly_predicted=predicted,
        reused_images=cached,
        failed_attempts=failures,
        elapsed_seconds=time.monotonic() - started,
    )


def assemble_frame(images, quality, observations, outputs, profile_hash, run_id):
    qmap = {r["image_id"]: r for r in quality}
    omap = {r["observation_id"]: r for r in observations}
    rows = []
    for image in images:
        q = qmap[image["image_id"]]
        obs = omap[image["observation_id"]]
        key = output_key(image, profile_hash)
        output = outputs.get(key)
        status = (
            "decode_failure" if image["decode_status"] != "success" else output[0] if output else "pending"
        )
        value = output[1] if output else {}
        success = status == "success"
        boxes = value.get("boxes", [])
        retained_count = value.get("retained_detection_count", len(boxes))
        reasons = []
        if not success:
            reasons.append(status)
        if q["quality_status"] != "usable":
            reasons.append("quality_" + q["quality_status"])
        if obs["identity_status"] != "verified" or obs["selected_image_id"] != image["image_id"]:
            reasons.append("non_unique_acquisition")
        if success and (value.get("candidate_limit_reached") or retained_count >= 1000):
            reasons.append("detection_limit_reached")
        eligible = not reasons
        timestamp_ok = image["timestamp_status"] == "verified" and image["observed_at_utc"] is not None
        row = {
            **image,
            "run_id": run_id,
            "counting_scope": "full_image",
            "model": "YOLO26x",
            "inference_key": key,
            "inference_status": status,
            "inference_error": value.get("error"),
            "retained_detection_count": retained_count if success else None,
            "quality_status": q["quality_status"],
            "quality_flags": q["quality_flags"],
            "quality_reason": q["decision_reason"],
            "mean_luma": q.get("mean_luma"),
            "blur_score": q.get("laplacian_variance"),
            "stale_group_id": q.get("stale_group_id"),
            "is_selected_acquisition": obs["selected_image_id"] == image["image_id"],
            "eligible_analysis": eligible,
            "eligible_temporal": eligible and timestamp_ok,
            "exclusion_reasons": reasons + ([] if timestamp_ok else ["unverified_time"]),
            "validation_status": "provisional",
            "count_confidence_threshold": THRESHOLD,
        }
        counts = value.get("counts", count_boxes(boxes)) if success else dict.fromkeys(COUNT_COLUMNS)
        row.update({"raw_" + k: v for k, v in counts.items()})
        row.update({k: v if eligible else None for k, v in counts.items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    for col in COUNT_COLUMNS:
        df[col] = df[col].astype("Int64")
        df["raw_" + col] = df["raw_" + col].astype("Int64")
    local = pd.to_datetime(df.observed_at_utc, utc=True).dt.tz_convert("Asia/Bangkok")
    df["observed_at_local"] = local
    df["date_local"] = local.dt.strftime("%Y-%m-%d")
    df["hour_local"] = local.dt.hour.astype("Int64")
    df["weekday_local"] = local.dt.day_name()
    df["is_weekend"] = local.dt.dayofweek.ge(5)
    return df


def write_frame(df, path):
    temp = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".parquet":
        df.to_parquet(temp, index=False, compression="zstd")
    else:
        serial = df.copy()
        for col in serial.columns:
            if serial[col].map(lambda v: isinstance(v, (list, tuple))).any():
                serial[col] = serial[col].map(
                    lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, tuple)) else v
                )
        serial.to_csv(temp, index=False)
    temp.replace(path)


def comparison(df, baseline):
    old = pd.read_parquet(baseline / "tables/traffic_features.parquet")
    old = old[old.eligible_spatial].copy()
    # One image can have several old road regions; sum those before joining once.
    common = ["car_count", "motorcycle_count", "bus_count", "truck_count", "vehicle_count"]
    old = (
        old.groupby("image_id", as_index=False)[common].sum().rename(columns={c: "roi_" + c for c in common})
    )
    matched = df[df.inference_status.eq("success")].merge(old, on="image_id", validate="one_to_one")
    matched["full_frame_four_class_count"] = sum(matched["raw_" + c] for c in common[:-1])
    matched["four_class_difference"] = matched.full_frame_four_class_count - matched.roi_vehicle_count
    matched["four_class_percent_change"] = (
        100 * matched.four_class_difference / matched.roi_vehicle_count.where(matched.roi_vehicle_count.ne(0))
    ).astype("Float64")
    return matched


def export(folder, config, baseline, images, profile_hash, run_id):
    db = connect(folder / "checkpoints.sqlite")
    outputs = {}
    for key, status, payload in db.execute("SELECT key,status,payload FROM outputs"):
        value = json.loads(payload)
        if status == "success":
            boxes = value.pop("boxes")
            value.update(counts=count_boxes(boxes), retained_detection_count=len(boxes))
        outputs[key] = (status, value)
    frame = assemble_frame(
        images,
        read_table(folder, "quality_results"),
        read_table(folder, "observations"),
        outputs,
        profile_hash,
        run_id,
    )
    cameras = pd.read_parquet(folder / "tables/cameras.parquet").drop(columns=["schema_version", "run_id"])
    frame = frame.merge(cameras, on="camera_id", how="left", validate="many_to_one")
    # Do not export a separate object row per duplicate acquisition: key links the
    # normalized detection table to every relevant image without duplicating boxes.
    detections_path = folder / "detections.parquet"
    temp = detections_path.with_suffix(".parquet.tmp")
    schema = SCHEMAS["detections"].append(pa.field("passes_count_threshold", pa.bool_()))
    with pq.ParquetWriter(temp, schema, compression="zstd") as writer:
        chunk = []
        for key, status, payload in db.execute("SELECT key,status,payload FROM outputs"):
            if status != "success":
                continue
            value = json.loads(payload)
            for box in value["boxes"]:
                chunk.append(
                    {
                        **box,
                        "inference_key": key,
                        "schema_version": 1,
                        "run_id": run_id,
                        "passes_count_threshold": box["confidence"] >= THRESHOLD,
                    }
                )
            if len(chunk) >= 8192:
                writer.write_table(pa.Table.from_pylist(chunk, schema=schema))
                chunk = []
        if chunk:
            writer.write_table(pa.Table.from_pylist(chunk, schema=schema))
    temp.replace(detections_path)
    db.close()
    del outputs
    for suffix in ["parquet", "csv"]:
        write_frame(frame, folder / ("traffic_full_frame." + suffix))
    matched = comparison(frame, baseline)
    write_frame(matched, folder / "comparison_vs_roi.csv")
    comparisons = []
    for c in ["car", "motorcycle", "bus", "truck"]:
        comparisons.append(
            dict(
                vehicle_type=c,
                matched_images=len(matched),
                previous_roi_detections=int(matched["roi_" + c + "_count"].sum()),
                full_frame_detections=int(matched["raw_" + c + "_count"].sum()),
            )
        )
    pd.DataFrame(comparisons).to_csv(folder / "comparison_summary.csv", index=False)
    usable = frame[frame.eligible_temporal].copy()
    daily_hour = (
        usable.groupby(
            ["camera_id", "date_local", "weekday_local", "is_weekend", "hour_local"], observed=True
        )
        .agg(**{c: (c, "mean") for c in COUNT_COLUMNS}, n_images=("image_id", "size"))
        .reset_index()
    )
    daily_hour.to_csv(folder / "camera_day_hour.csv", index=False)
    hourly = (
        daily_hour.groupby(["camera_id", "is_weekend", "hour_local"], observed=True)
        .agg(
            **{c: (c, "mean") for c in COUNT_COLUMNS},
            n_days=("date_local", "nunique"),
            n_images=("n_images", "sum"),
        )
        .reset_index()
    )
    hourly.to_csv(folder / "hourly_profiles.csv", index=False)
    coverage = (
        frame.groupby(["camera_id", "date_local", "hour_local"], observed=True)
        .agg(
            source_images=("image_id", "size"),
            inferred_images=("inference_status", lambda s: s.eq("success").sum()),
            usable_images=("eligible_temporal", "sum"),
        )
        .reset_index()
    )
    coverage.to_csv(folder / "coverage_by_camera_hour.csv", index=False)
    by_camera = (
        frame.groupby("camera_id")
        .agg(
            source_images=("image_id", "size"),
            inferred_images=("inference_status", lambda s: s.eq("success").sum()),
            usable_images=("eligible_temporal", "sum"),
            **{"mean_" + c: (c, "mean") for c in COUNT_COLUMNS},
        )
        .reset_index()
    )
    by_camera.to_csv(folder / "camera_summary.csv", index=False)
    errors = []
    if len(frame) != len(images) or frame.image_id.duplicated().any():
        errors.append("image identity mismatch")
    success = frame.inference_status.eq("success")
    if (
        not frame.loc[success, "raw_vehicle_count"]
        .eq(frame.loc[success, ["raw_" + c + "_count" for c in CLASSES]].sum(axis=1))
        .all()
    ):
        errors.append("class totals mismatch")
    if frame.loc[~frame.eligible_analysis, COUNT_COLUMNS].notna().any().any():
        errors.append("ineligible counts must be null")
    if frame.loc[~success, ["raw_" + c for c in COUNT_COLUMNS]].notna().any().any():
        errors.append("failed/pending inference must be null")
    complete = not frame.inference_status.isin(["pending", "failed"]).any()
    report = dict(
        status="completed" if complete and not errors else "partial",
        updated_at=now(),
        run_id=run_id,
        total_images=len(frame),
        cameras=frame.camera_id.nunique(),
        inference_status=frame.inference_status.value_counts().to_dict(),
        quality_status=frame.quality_status.value_counts().to_dict(),
        eligible_analysis_images=int(frame.eligible_analysis.sum()),
        eligible_temporal_images=int(frame.eligible_temporal.sum()),
        raw_vehicle_detections=int(frame.raw_vehicle_count.sum()),
        raw_class_totals={c: int(frame["raw_" + c + "_count"].sum()) for c in CLASSES},
        analysis_class_totals={c: int(frame[c + "_count"].sum()) for c in CLASSES},
        comparison_matched_images=len(matched),
        comparison_scope="same-image four-class counts; changed counting region, not accuracy",
        validation_errors=errors,
        accuracy_precision_recall=None,
        limitations=[
            "Counts are visible vehicle instances per snapshot, not unique trips or traffic flow.",
            "Full images include all visible roads, parked vehicles, and distant vehicles.",
            "Counts and quality rules have not been validated against full-image human labels.",
            "Raw counts include invalid/review-quality images; filter eligible_temporal for preliminary time patterns.",
            "Camera view changes and overlapping cameras are not resolved; do not sum cameras as unique city traffic.",
        ],
    )
    atomic_json(folder / "summary.json", report)
    atomic_json(folder / "progress.json", report)
    if errors:
        raise ValueError(errors)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/yolo26x.yaml"))
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/full_frame/yolo26x_all_20261109"))
    parser.add_argument(
        "--limit",
        type=int,
        help="Infer at most this many new unique images; export an explicitly partial dataframe",
    )
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config.project_root)
    baseline = args.baseline or root / "data" / read_json(root / "data/current_release.json")["release_path"]
    baseline = baseline.resolve()
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    verify_model(config)
    settings = profile(config)
    profile_hash = digest(settings)
    manifest = folder / "manifest.json"
    run_id = folder.name
    if manifest.exists():
        old = read_json(manifest)
        if old["profile_hash"] != profile_hash or old["baseline_release"] != str(baseline):
            raise ValueError("Cannot change model/settings/source under a resumed run")
        if old["source_manifest_sha256"] != file_hash(baseline / "manifest.json"):
            raise ValueError("Baseline manifest changed")
    else:
        atomic_json(
            manifest,
            dict(
                run_id=run_id,
                created_at=now(),
                profile=settings,
                profile_hash=profile_hash,
                baseline_release=str(baseline),
                source_manifest_sha256=file_hash(baseline / "manifest.json"),
                code_hashes={
                    p.name: file_hash(p) for p in [Path(__file__), Path(__file__).parent / "detector.py"]
                },
            ),
        )
    record = read_json(manifest)
    record.setdefault("execution_sessions", []).append(
        dict(
            started_at=now(),
            new_image_limit=args.limit,
            export_only=args.export_only,
            code_hashes={
                p.name: file_hash(p) for p in [Path(__file__), Path(__file__).parent / "detector.py"]
            },
        )
    )
    atomic_json(manifest, record)
    for name in ["images", "observations", "cameras"]:
        dest = folder / "tables" / (name + ".parquet")
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(baseline / "tables" / dest.name, dest)
    inventory = read_json(baseline / "reports/inventory_summary.json")
    atomic_json(folder / "inventory_summary.json", inventory)
    images = read_table(folder, "images")
    if not (folder / "tables/quality_results.parquet").exists():
        print("Rechecking image quality without any camera regions", flush=True)
        assess_quality(config, folder, run_id, regions=[])
    if not args.export_only:
        # Interleave cameras chronologically so partial exports are not confined
        # to one camera. This changes execution order, never dataset membership.
        inference_images = sorted(
            images, key=lambda im: (str(im["observed_at_utc"] or ""), im["camera_id"], im["image_id"])
        )
        stats = infer(
            config,
            folder,
            inference_images,
            Path(inventory["raw_root"]),
            profile_hash,
            args.limit,
            snapshot=lambda: export(folder, config, baseline, images, profile_hash, run_id),
        )
        atomic_json(folder / "last_session.json", stats)
    print(json.dumps(export(folder, config, baseline, images, profile_hash, run_id), indent=2), flush=True)


if __name__ == "__main__":
    main()

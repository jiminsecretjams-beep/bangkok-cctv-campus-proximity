"""Full-image vehicle counting, resumable inference, and analysis exports without ROIs.

Raw predictions exist even for quality-review/invalid images. Analysis counts are
nullable and restricted to usable, unique acquisitions with successful inference.
SQLite checkpoints hold one output per exact image content and model profile.
"""

from __future__ import annotations

import importlib.metadata
import json
import sqlite3
import time

import pandas as pd

from .config import digest
from .detector import YOLODetector
from .storage import atomic_json, now

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

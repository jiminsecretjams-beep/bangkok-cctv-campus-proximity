"""Portable, disjoint MacBook batches and strictly verified return bundles.

No reviewer or ROI is involved. Tests inject a fixture detector; the worker uses
the same pinned YOLO26x/full-frame measurement as the local full-image pipeline.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import digest, file_hash, load_config
from .full_frame import (
    CLASSES,
    COUNT_COLUMNS,
    THRESHOLD,
    assemble_frame,
    connect,
    count_boxes,
    infer,
    profile,
    write_frame,
)
from .inventory import inventory_sources, load_cameras
from .quality import assess_quality
from .storage import SCHEMAS, atomic_json, now, read_json, read_table

REPOSITORY = "jiminsecretjams-beep/bangkok-cctv-campus-proximity"
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,59}$")
RESULT_FILES = {"traffic_full_frame.parquet", "detections.parquet", "summary.json", "profile.json"}


def safe_id(value: str) -> str:
    if not IDENTIFIER.fullmatch(value):
        raise ValueError("Use a short identifier containing only letters, numbers, hyphens and underscores")
    return value


def code_hash(root: Path) -> str:
    return digest({p.name: file_hash(p) for p in sorted((root / "src/bkk_cctv").glob("*.py"))})


def metadata_semantics(path: Path) -> str:
    rows = load_cameras(path)
    return digest(
        [
            {k: v for k, v in row.items() if k != "metadata_sha256"}
            for row in sorted(rows, key=lambda r: r["camera_id"])
        ]
    )


def make_job(config_path: Path, metadata: Path, output: Path, batches=4) -> dict:
    """Freeze source/model/code and balanced, disjoint camera allocations; no inference."""
    if output.exists():
        raise ValueError("Job already exists; use a new file to create a new job")
    cfg = load_config(config_path)
    root = Path(cfg.project_root)
    cameras = load_cameras(metadata)
    if not 1 <= batches <= len(cameras):
        raise ValueError("Invalid batch count")
    groups = {f"batch-{i + 1:02d}": dict(camera_ids=[], expected_images=0) for i in range(batches)}
    for camera in sorted(cameras, key=lambda r: (-r["declared_image_count"], int(r["camera_id"]))):
        batch = min(groups, key=lambda b: (groups[b]["expected_images"], b))
        groups[batch]["camera_ids"].append(camera["camera_id"])
        groups[batch]["expected_images"] += camera["declared_image_count"]
    for group in groups.values():
        group["camera_ids"].sort(key=int)
    sources = read_json(root / "sources.json")
    definition = dict(
        schema_version=1,
        measurement_scope="full_image",
        classes=list(CLASSES),
        count_threshold=THRESHOLD,
        archive=sources["archive"],
        metadata_source=sources["metadata"],
        metadata_semantic_sha256=metadata_semantics(metadata),
        config_path=str(config_path.relative_to(root))
        if config_path.is_relative_to(root)
        else str(config_path),
        config_sha256=file_hash(config_path),
        source_code_sha256=code_hash(root),
        environment_lock_sha256=file_hash(root / "uv.lock"),
        overrides_sha256=file_hash(cfg.path(cfg.overrides)),
        expected_profile=profile(cfg),
        cameras={r["camera_id"]: r["declared_image_count"] for r in cameras},
        batches=groups,
    )
    job = dict(**definition, job_id="whole-image-" + digest(definition)[:16], created_at=now())
    atomic_json(output, job)
    return job


def load_job(path: Path) -> dict:
    job = read_json(path)
    definition = {k: v for k, v in job.items() if k not in {"job_id", "created_at"}}
    if job["job_id"] != "whole-image-" + digest(definition)[:16]:
        raise ValueError("Frozen job definition was edited")
    all_cameras = [c for b in job["batches"].values() for c in b["camera_ids"]]
    if len(all_cameras) != len(set(all_cameras)) or set(all_cameras) != set(job["cameras"]):
        raise ValueError("Camera allocations overlap or are incomplete")
    return job


def verify_job_environment(job: dict, cfg, root: Path) -> None:
    checks = [
        (file_hash(root / "uv.lock"), job["environment_lock_sha256"], "Dependency lock"),
        (code_hash(root), job["source_code_sha256"], "Processing source"),
        (file_hash(root / job["config_path"]), job["config_sha256"], "Worker config"),
        (file_hash(cfg.path(cfg.overrides)), job["overrides_sha256"], "Quality overrides"),
        (digest(profile(cfg)), digest(job["expected_profile"]), "Installed model/runtime profile"),
    ]
    for actual, expected, name in checks:
        if actual != expected:
            raise ValueError(
                f"{name} differs from the frozen job. Use the job's code and uv sync --frozen --python 3.11; do not retune workers."
            )


def ensure_inputs(job: dict, cfg, root: Path) -> None:
    import gdown

    for source, target in [
        (job["archive"], cfg.path(cfg.source.archive)),
        (job["metadata_source"], cfg.path(cfg.source.metadata)),
    ]:
        if not target.exists():
            if shutil.disk_usage(root).free < 6 * 1024**3:
                raise ValueError(
                    "Allow at least 6 GiB of free space for the archive, selected images and results"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".part")
            result = gdown.download(
                id=source["file_id"], output=str(temporary), quiet=False, use_cookies=False, resume=True
            )
            if result is None:
                raise ValueError(
                    "Drive download failed. Download the course file in your browser, place it at the documented path, and retry."
                )
            if source.get("sha256") and file_hash(temporary) != source["sha256"]:
                raise ValueError("Downloaded source checksum differs; .part preserved")
            temporary.replace(target)
        if source.get("sha256") and file_hash(target) != source["sha256"]:
            raise ValueError("Existing source archive checksum differs")
    if metadata_semantics(cfg.path(cfg.source.metadata)) != job["metadata_semantic_sha256"]:
        raise ValueError(
            "Camera metadata values differ from the job (line-ending differences alone are allowed)"
        )
    model_sources = read_json(root / "model_sources.json")["yolo26x"]["files"]
    for source in model_sources:
        target = root / source["path"]
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".part")
            with (
                urllib.request.urlopen(source["url"], timeout=120) as response,
                temporary.open("wb") as stream,
            ):
                shutil.copyfileobj(response, stream, length=1024 * 1024)
            if file_hash(temporary) != source["sha256"]:
                raise ValueError("Model download checksum mismatch")
            temporary.replace(target)
        if file_hash(target) != source["sha256"]:
            raise ValueError("Existing selected model checksum mismatch")


def validate_frame(df: pd.DataFrame, expected_camera_counts: dict) -> None:
    if df.image_id.duplicated().any() or df.image_id.isna().any():
        raise ValueError("Duplicated or missing image IDs")
    actual = {str(k): int(v) for k, v in df.groupby("camera_id").size().items()}
    if actual != expected_camera_counts:
        raise ValueError(f"Batch camera/image membership mismatch: {actual}")
    if not df.counting_scope.eq("full_image").all():
        raise ValueError("Mixed measurement scopes")
    if df.inference_status.isin(["pending", "failed"]).any():
        raise ValueError("Incomplete predictions cannot be returned as a completed batch; resume first")
    success = df.inference_status.eq("success")
    cols = [c + "_count" for c in CLASSES]
    raw = ["raw_" + c for c in cols]
    if df.loc[success, raw + ["raw_vehicle_count"]].isna().any().any():
        raise ValueError("Successful prediction missing raw counts")
    for group, total in [(cols, "vehicle_count"), (raw, "raw_vehicle_count")]:
        present = df[total].notna()
        values = df.loc[present, group + [total]].astype(float)
        if (values.lt(0) | values.mod(1).ne(0)).any().any():
            raise ValueError("Counts must be nonnegative integers")
        if not df.loc[present, total].eq(df.loc[present, group].sum(axis=1)).all():
            raise ValueError("Vehicle class totals disagree")
    if df.loc[~df.eligible_analysis, COUNT_COLUMNS].notna().any().any():
        raise ValueError("Ineligible analysis counts must be null")
    if df.loc[~success, ["raw_" + c for c in COUNT_COLUMNS]].notna().any().any():
        raise ValueError("Failed/undecodable predictions must not become zero")
    if df.loc[df.eligible_analysis, "quality_status"].ne("usable").any():
        raise ValueError("Quality-gated analysis contract violated")


def export_result(cfg, folder: Path, job: dict, batch: str, worker: str, settings: dict) -> dict:
    profile_hash = digest(settings)
    images = read_table(folder, "images")
    outputs = {}
    db = connect(folder / "checkpoints.sqlite")
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
        folder.name,
    )
    frame["checkpoint_inference_key"] = frame.inference_key
    frame["is_weekend"] = frame["is_weekend"].astype("boolean").mask(frame.observed_at_utc.isna(), pd.NA)
    frame["inference_key"] = frame.inference_key.map(lambda k: digest([job["job_id"], batch, k]))
    frame["job_id"], frame["batch_id"], frame["worker_id"] = job["job_id"], batch, worker
    frame["raw_motor_vehicle_count"] = sum(frame["raw_" + c + "_count"] for c in CLASSES if c != "bicycle")
    frame["motor_vehicle_count"] = sum(frame[c + "_count"] for c in CLASSES if c != "bicycle")
    write_frame(frame, folder / "traffic_full_frame.parquet")
    write_frame(frame, folder / "traffic_full_frame.csv")
    schema = SCHEMAS["detections"].append(pa.field("passes_count_threshold", pa.bool_()))
    temp = folder / "detections.parquet.tmp"
    with pq.ParquetWriter(temp, schema, compression="zstd") as writer:
        rows = []
        for key, status, payload in db.execute("SELECT key,status,payload FROM outputs"):
            if status != "success":
                continue
            for box in json.loads(payload)["boxes"]:
                rows.append(
                    dict(
                        **box,
                        inference_key=digest([job["job_id"], batch, key]),
                        schema_version=1,
                        run_id=folder.name,
                        passes_count_threshold=box["confidence"] >= THRESHOLD,
                    )
                )
            if len(rows) >= 8192:
                writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                rows = []
        if rows:
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    temp.replace(folder / "detections.parquet")
    db.close()
    complete = not frame.inference_status.isin(["pending", "failed"]).any()
    expected = {c: job["cameras"][c] for c in job["batches"][batch]["camera_ids"]}
    if complete:
        validate_frame(frame, expected)
    summary = dict(
        status="completed" if complete else "partial",
        job_id=job["job_id"],
        batch_id=batch,
        worker_id=worker,
        rows=len(frame),
        successful_images=int(frame.inference_status.eq("success").sum()),
        analysis_images=int(frame.eligible_analysis.sum()),
        inference_status=frame.inference_status.value_counts().to_dict(),
        created_at=now(),
    )
    atomic_json(folder / "summary.json", summary)
    atomic_json(folder / "profile.json", settings)
    return summary


def run_batch(
    job_path: Path,
    batch: str,
    worker: str,
    *,
    download=True,
    limit=None,
    detector=None,
    verify_environment=True,
) -> dict:
    safe_id(batch)
    safe_id(worker)
    job = load_job(job_path)
    root = job_path.resolve().parent.parent
    if batch not in job["batches"]:
        raise ValueError("Choose one of the batch IDs in jobs.json")
    cfg = load_config(root / job["config_path"])
    if verify_environment:
        verify_job_environment(job, cfg, root)
    if download:
        ensure_inputs(job, cfg, root)
    elif (
        file_hash(cfg.path(cfg.source.archive)) != job["archive"]["sha256"]
        or metadata_semantics(cfg.path(cfg.source.metadata)) != job["metadata_semantic_sha256"]
    ):
        raise ValueError("Offline source inputs differ from job")
    cfg.source.cameras = job["batches"][batch]["camera_ids"]
    folder = root / "data/distributed" / job["job_id"] / batch
    folder.mkdir(parents=True, exist_ok=True)
    settings = profile(cfg)
    identity = dict(job_id=job["job_id"], batch_id=batch, worker_id=worker, profile_hash=digest(settings))
    lock = folder / "worker_identity.json"
    if lock.exists() and read_json(lock) != identity:
        raise ValueError("Existing batch belongs to a different worker/job/profile; do not overwrite")
    atomic_json(lock, identity)
    # One writer per checkpoint database. OS lock is released even after a crash.
    import fcntl

    with (folder / ".worker.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("This batch is already running; do not start a second writer") from exc
        if not (folder / "inventory_summary.json").exists():
            inventory_sources(cfg, folder, folder.name)
        images = read_table(folder, "images")
        if Counter(r["camera_id"] for r in images) != Counter(
            {c: job["cameras"][c] for c in cfg.source.cameras}
        ):
            raise ValueError("Inventory does not match assigned camera counts")
        if not (folder / "tables/quality_results.parquet").exists():
            assess_quality(cfg, folder, folder.name, regions=[])
        infer(
            cfg,
            folder,
            sorted(images, key=lambda r: (str(r["observed_at_utc"]), r["camera_id"], r["image_id"])),
            Path(read_json(folder / "inventory_summary.json")["raw_root"]),
            digest(settings),
            limit=limit,
            detector=detector,
        )
        result = export_result(cfg, folder, job, batch, worker, settings)
    return dict(**result, folder=str(folder))


def package_batch(job_path: Path, batch: str) -> Path:
    job = load_job(job_path)
    safe_id(batch)
    folder = job_path.resolve().parent.parent / "data/distributed" / job["job_id"] / batch
    summary = read_json(folder / "summary.json")
    if summary["status"] != "completed":
        raise ValueError("Batch is partial; finish/resume processing before returning it")
    expected = {c: job["cameras"][c] for c in job["batches"][batch]["camera_ids"]}
    validate_frame(pd.read_parquet(folder / "traffic_full_frame.parquet"), expected)
    if digest(read_json(folder / "profile.json")) != digest(job["expected_profile"]):
        raise ValueError("Batch model/runtime profile differs from job")
    manifest = dict(
        schema_version=1,
        job_id=job["job_id"],
        batch_id=batch,
        worker_id=summary["worker_id"],
        expected_images=job["batches"][batch]["expected_images"],
        archive_sha256=job["archive"]["sha256"],
        job_definition_sha256=digest(job),
        files={name: file_hash(folder / name) for name in sorted(RESULT_FILES)},
    )
    atomic_json(folder / "result_manifest.json", manifest)
    out = folder.parent / f"{job['job_id']}_{batch}_{safe_id(summary['worker_id'])}.zip"
    temporary = out.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(RESULT_FILES | {"result_manifest.json"}):
            z.write(folder / name, name)
    temporary.replace(out)
    return out


def verify_bundle(path: Path, job: dict, target: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if set(names) != RESULT_FILES | {"result_manifest.json"} or len(names) != len(set(names)):
            raise ValueError("Unexpected or duplicated files in result bundle")
        if sum(i.file_size for i in archive.infolist()) > 4 * 1024**3:
            raise ValueError("Result bundle exceeds the allowed uncompressed size")
        manifest = json.loads(archive.read("result_manifest.json"))
        if manifest.get("job_id") != job["job_id"] or manifest.get("job_definition_sha256") != digest(job):
            raise ValueError("Result belongs to a different frozen job")
        batch = manifest["batch_id"]
        if batch not in job["batches"] or manifest.get("archive_sha256") != job["archive"]["sha256"]:
            raise ValueError("Unknown batch or raw archive mismatch")
        if set(manifest["files"]) != RESULT_FILES:
            raise ValueError("Missing result file checksums")
        import hashlib

        payloads = {name: archive.read(name) for name in RESULT_FILES}
        if any(
            hashlib.sha256(data).hexdigest() != manifest["files"][name] for name, data in payloads.items()
        ):
            raise ValueError("Result file checksum mismatch")
        if digest(json.loads(payloads["profile.json"])) != digest(job["expected_profile"]):
            raise ValueError("Result model/runtime profile differs")
        target.mkdir(parents=True, exist_ok=False)
        for name, data in payloads.items():
            (target / name).write_bytes(data)
        atomic_json(target / "result_manifest.json", manifest)
    frame = pd.read_parquet(target / "traffic_full_frame.parquet")
    validate_frame(frame, {c: job["cameras"][c] for c in job["batches"][batch]["camera_ids"]})
    if not frame.batch_id.eq(batch).all() or not frame.job_id.eq(job["job_id"]).all():
        raise ValueError("Batch provenance columns do not match result manifest")
    summary = read_json(target / "summary.json")
    if summary["status"] != "completed" or summary["rows"] != len(frame):
        raise ValueError("Returned summary is incomplete or inconsistent")
    # Reconcile returned boxes with the raw count columns independently of the worker summary.
    bykey = {key: rows for key, rows in frame[frame.inference_status.eq("success")].groupby("inference_key")}
    counts = {}
    indices = set()
    for chunk in pq.ParquetFile(target / "detections.parquet").iter_batches(batch_size=8192):
        for box in chunk.to_pylist():
            key = box["inference_key"]
            pair = (key, box["detection_index"])
            if key not in bykey or pair in indices or box["mapped_class"] not in CLASSES:
                raise ValueError("Orphan, duplicated or unknown-class detection")
            indices.add(pair)
            image = bykey[key].iloc[0]
            values = [box[c] for c in ["x1", "y1", "x2", "y2", "confidence"]]
            if not all(math.isfinite(v) for v in values) or not (
                0 <= box["x1"] < box["x2"] <= image.width_px + 0.01
                and 0 <= box["y1"] < box["y2"] <= image.height_px + 0.01
                and job["expected_profile"]["detector"]["retain_confidence_min"] <= box["confidence"] <= 1
            ):
                raise ValueError("Invalid detection coordinates/confidence")
            if box["confidence"] >= THRESHOLD:
                counts.setdefault(key, Counter())[box["mapped_class"]] += 1
    for key, rows in bykey.items():
        expected = counts.get(key, Counter())
        for name in CLASSES:
            if not rows["raw_" + name + "_count"].eq(expected[name]).all():
                raise ValueError("Raw counts do not reconcile with returned detections")
    return manifest


def merge_bundles(job_path: Path, bundles: list[Path], output: Path, allow_partial=False) -> dict:
    job = load_job(job_path)
    if output.exists():
        raise ValueError("Import output exists; choose a new folder to preserve previous results")
    staging = output.with_name(output.name + ".importing")
    if staging.exists():
        raise ValueError("A previous incomplete import exists; inspect it before retrying")
    staging.mkdir(parents=True)
    seen = set()
    manifests = []
    frames = []
    detection_files = []
    for index, path in enumerate(bundles):
        target = staging / "batches" / str(index + 1)
        manifest = verify_bundle(path, job, target)
        if manifest["batch_id"] in seen:
            raise ValueError("Duplicate batch returned; choose one final result per batch")
        seen.add(manifest["batch_id"])
        manifests.append(manifest)
        frames.append(pd.read_parquet(target / "traffic_full_frame.parquet"))
        detection_files.append(target / "detections.parquet")
    missing = sorted(set(job["batches"]) - seen)
    if missing and not allow_partial:
        raise ValueError(
            f"Missing batches: {missing}. Use --allow-partial only for an explicitly partial import."
        )
    if not frames:
        raise ValueError("No returned bundles supplied")
    merged = pd.concat(frames, ignore_index=True).sort_values(["camera_id", "observed_at_utc", "image_id"])
    expected = {c: job["cameras"][c] for batch in seen for c in job["batches"][batch]["camera_ids"]}
    validate_frame(merged, expected)
    write_frame(merged, staging / "traffic_full_frame.parquet")
    write_frame(merged, staging / "traffic_full_frame.csv")
    schema = pq.read_schema(detection_files[0])
    with pq.ParquetWriter(staging / "detections.parquet", schema, compression="zstd") as writer:
        for path in detection_files:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=8192):
                writer.write_batch(batch)
    result = dict(
        job_id=job["job_id"],
        status="partial" if missing else "completed",
        batch_ids=sorted(seen),
        missing_batches=missing,
        images=len(merged),
        cameras=merged.camera_id.nunique(),
        eligible_temporal=int(merged.eligible_temporal.sum()),
        imported_at=now(),
        returned_manifests=manifests,
        files={
            name: file_hash(staging / name) for name in ["traffic_full_frame.parquet", "detections.parquet"]
        },
    )
    atomic_json(staging / "import_manifest.json", result)
    staging.rename(output)
    return result


def upload_bundle(job_path: Path, batch: str, repository: str) -> dict:
    """Publish only the explicitly packaged result ZIP; no commit/push of raw data."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Repository must be owner/name")
    bundle = package_batch(job_path, batch)
    job = load_job(job_path)
    tag = f"result-{job['job_id']}-{batch}-{os.urandom(4).hex()}"
    subprocess.run(
        [
            "gh",
            "release",
            "create",
            tag,
            str(bundle),
            "--repo",
            repository,
            "--title",
            f"CCTV {batch} result",
            "--notes",
            f"Completed whole-image batch for frozen job {job['job_id']}. Import with processing/scripts/distributed_worker.py. No raw images or weights included.",
            "--latest=false",
        ],
        check=True,
    )
    return dict(
        repository=repository,
        tag=tag,
        bundle=str(bundle),
        url=f"https://github.com/{repository}/releases/tag/{tag}",
    )

"""Bounded image inventory and safe extraction with per-file resume checkpoints."""

from __future__ import annotations

import csv
import hashlib
import io
import shutil
import stat
import unicodedata
import zipfile
from pathlib import Path, PurePosixPath

from PIL import Image, ImageOps, UnidentifiedImageError

from .config import PipelineConfig, digest, file_hash
from .observations import resolve_observations
from .storage import atomic_json, event, read_json, write_table
from .timestamps import parse_source_identity

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def safe_members(archive: zipfile.ZipFile, max_bytes: int) -> list[tuple[zipfile.ZipInfo, str]]:
    seen = set()
    result = []
    total = 0
    for info in archive.infolist():
        raw = info.filename.replace("\\", "/")
        p = PurePosixPath(raw)
        if p.is_absolute() or ".." in p.parts or (p.parts and ":" in p.parts[0]):
            raise ValueError(f"Unsafe archive path: {info.filename}")
        if stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError(f"Archive symlink rejected: {info.filename}")
        normalized = unicodedata.normalize("NFC", str(p))
        if normalized.casefold() in seen:
            raise ValueError(f"Archive path collision: {normalized}")
        seen.add(normalized.casefold())
        total += info.file_size
        result.append((info, normalized))
    if total > max_bytes:
        raise ValueError(f"Archive expands to {total} bytes, exceeding configured limit")
    return result


def load_cameras(path: Path) -> list[dict]:
    sha = file_hash(path)
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):

            def number(name):
                try:
                    value = float(r[name])
                    return value if value == value and abs(value) != float("inf") else None
                except (ValueError, KeyError):
                    return None

            rows.append(
                dict(
                    camera_id=str(r["camera_id"]),
                    location_th_raw=r.get("location"),
                    detail_th_raw=r.get("location_detail"),
                    location_en_raw=r.get("location_en"),
                    detail_en_raw=r.get("location_detail_en"),
                    latitude_raw=r.get("latitude"),
                    longitude_raw=r.get("longitude"),
                    latitude=number("latitude"),
                    longitude=number("longitude"),
                    location_verified_name=None,
                    location_review_status="unreviewed",
                    declared_image_count=int(r["image_count"]) if r.get("image_count") else None,
                    metadata_sha256=sha,
                )
            )
    if len({r["camera_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate camera IDs in metadata")
    return rows


def inventory_sources(config: PipelineConfig, work: Path, run_id: str) -> dict:
    archive_path = config.path(config.source.archive)
    dataset_id = file_hash(archive_path)
    extraction = config.path(f"data/raw/{dataset_id}/images")
    extraction.mkdir(parents=True, exist_ok=True)
    checkpoint = work / "inventory_checkpoint.json"
    saved = read_json(checkpoint) if checkpoint.exists() else {"dataset_id": dataset_id, "images": {}}
    if saved["dataset_id"] != dataset_id:
        raise ValueError("Source archive changed during resume")
    cameras = load_cameras(config.path(config.source.metadata))
    rows = []
    manifest = []
    with zipfile.ZipFile(archive_path) as archive:
        members = safe_members(archive, int(config.source.max_uncompressed_gb * 1024**3))
        candidates = {}
        for info, name in members:
            if info.is_dir() or PurePosixPath(name).suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            camera = parse_source_identity(name, config.source)["camera_id"]
            if config.source.cameras and camera not in config.source.cameras:
                continue
            candidates.setdefault(camera, []).append(name)
        allowed = set()
        for names in candidates.values():
            names.sort()
            limit = config.source.max_images_per_camera
            if limit and len(names) > limit:
                indices = (
                    [round(i * (len(names) - 1) / (limit - 1)) for i in range(limit)]
                    if limit > 1
                    else [len(names) // 2]
                )
                names = [names[i] for i in indices]
            allowed.update(names)
        need = sum(i.file_size for i, name in members if not i.is_dir() and not (extraction / name).exists())
        if need + 256 * 1024**2 > shutil.disk_usage(extraction).free:
            raise ValueError("Not enough free storage for extraction plus safety margin")
        for info, name in members:
            parsed = parse_source_identity(name, config.source)
            selected = name in allowed
            manifest.append(
                dict(
                    path=name,
                    size=info.file_size,
                    compressed_size=info.compress_size,
                    is_directory=info.is_dir(),
                    selected=selected,
                )
            )
            if not selected:
                continue
            image_id = digest({"dataset_id": dataset_id, "normalized_archive_member_path": name})
            target = extraction / name
            cached = saved["images"].get(image_id)
            if cached and target.exists() and file_hash(target) == cached["content_sha256"]:
                row = {**cached, **parsed}
            else:
                data = archive.read(info)  # ZIP CRC verified by zipfile
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_suffix(target.suffix + ".tmp")
                tmp.write_bytes(data)
                tmp.replace(target)
                row = dict(
                    image_id=image_id,
                    dataset_id=dataset_id,
                    source_path=name,
                    content_sha256=hashlib.sha256(data).hexdigest(),
                    size_bytes=len(data),
                    width_px=None,
                    height_px=None,
                    decode_status="failed",
                    **parsed,
                )
                try:
                    with Image.open(io.BytesIO(data)) as im:
                        im = ImageOps.exif_transpose(im)
                        im.load()
                        row.update(width_px=im.width, height_px=im.height, decode_status="success")
                except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
                    pass
            rows.append(row)
            saved["images"][image_id] = row
            if len(rows) % 500 == 0:
                atomic_json(checkpoint, saved)
                event(work, "inventory", processed=len(rows))
                print(f"Inventory: {len(rows)} images", flush=True)
    # JSON checkpoints serialize timestamps; reparsing above restores datetime types.
    if not rows:
        raise ValueError("No candidate images matched the source selection")
    mapping_version = digest({"parser": config.source.model_dump(), "algorithm": "acquisition-v1"})
    observations = resolve_observations(rows, config.source.acquisition_verified, mapping_version)
    for row in rows:
        saved["images"][row["image_id"]] = row
    atomic_json(checkpoint, saved)
    write_table(work, "cameras", cameras, run_id)
    write_table(work, "images", rows, run_id)
    write_table(work, "observations", observations, run_id)
    atomic_json(work / "archive_manifest.json", manifest)
    actual = {r["camera_id"]: sum(x["camera_id"] == r["camera_id"] for x in rows) for r in cameras}
    times = [r["observed_at_utc"] for r in rows if r["observed_at_utc"]]
    summary = dict(
        dataset_id=dataset_id,
        raw_root=str(extraction),
        image_count=len(rows),
        acquisition_count=len(observations),
        archive_members=len(manifest),
        source_subset=bool(config.source.cameras or config.source.max_images_per_camera),
        decoded=sum(r["decode_status"] == "success" for r in rows),
        first_observed_at=min(times).isoformat() if times else None,
        last_observed_at=max(times).isoformat() if times else None,
        unmatched_camera_ids=sorted({r["camera_id"] for r in rows if r["camera_id"] not in actual}, key=str),
        reconciliation=[
            dict(camera_id=r["camera_id"], declared=r["declared_image_count"], actual=actual[r["camera_id"]])
            for r in cameras
        ],
    )
    atomic_json(work / "inventory_summary.json", summary)
    return summary

"""Atomic local storage, explicit schemas, and immutable release manifests."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

UTC = pa.timestamp("us", tz="UTC")
S, INT, L, F = pa.string(), pa.int32(), pa.int64(), pa.float64()
STRINGS = pa.list_(S)


def schema(*fields: tuple[str, pa.DataType]) -> pa.Schema:
    return pa.schema([("schema_version", INT), ("run_id", S), *fields])


SCHEMAS = {
    "cameras": schema(
        ("camera_id", S),
        ("location_th_raw", S),
        ("detail_th_raw", S),
        ("location_en_raw", S),
        ("detail_en_raw", S),
        ("latitude_raw", S),
        ("longitude_raw", S),
        ("latitude", F),
        ("longitude", F),
        ("location_verified_name", S),
        ("location_review_status", S),
        ("declared_image_count", INT),
        ("metadata_sha256", S),
    ),
    "images": schema(
        ("image_id", S),
        ("dataset_id", S),
        ("source_path", S),
        ("content_sha256", S),
        ("camera_id", S),
        ("size_bytes", L),
        ("width_px", INT),
        ("height_px", INT),
        ("decode_status", S),
        ("raw_timestamp_text", S),
        ("timestamp_source", S),
        ("timestamp_status", S),
        ("acquired_at_utc", UTC),
        ("camera_at_utc", UTC),
        ("observed_at_utc", UTC),
        ("time_basis", S),
        ("parser_version", S),
        ("observation_id", S),
    ),
    "observations": schema(
        ("observation_id", S),
        ("camera_id", S),
        ("acquisition_token", S),
        ("identity_status", S),
        ("selected_image_id", S),
        ("member_count", INT),
        ("observed_at_utc", UTC),
        ("timestamp_status", S),
        ("mapping_version", S),
        ("resolution_reason", S),
    ),
    "quality_results": schema(
        ("image_id", S),
        ("quality_version", S),
        ("quality_status", S),
        ("quality_flags", STRINGS),
        ("mean_luma", F),
        ("luma_std", F),
        ("dark_fraction", F),
        ("bright_fraction", F),
        ("laplacian_variance", F),
        ("test_pattern_score", F),
        ("road_phash", S),
        ("similarity_previous", F),
        ("stale_run_seconds", F),
        ("previous_image_id", S),
        ("stale_group_id", S),
        ("decision_source", S),
        ("decision_reason", S),
        ("override_id", S),
    ),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with tmp.open("w") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, default=str, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_table(work: Path, name: str, rows: list[dict], run_id: str) -> None:
    folder = work / "tables"
    folder.mkdir(parents=True, exist_ok=True)
    data = [{**r, "schema_version": 1, "run_id": run_id} for r in rows]
    table = pa.Table.from_pylist(data, schema=SCHEMAS[name])
    path = folder / f"{name}.parquet"
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd", row_group_size=100_000)
    os.replace(tmp, path)


def read_table(work: Path, name: str) -> list[dict]:
    return pq.read_table(work / "tables" / f"{name}.parquet").to_pylist()


def event(work: Path, stage: str, **values) -> None:
    path = work / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps({"at": now(), "stage": stage, **values}, default=str) + "\n")

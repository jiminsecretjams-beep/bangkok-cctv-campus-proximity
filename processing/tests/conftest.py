from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from bkk_cctv.config import digest, file_hash


def jpeg(array):
    output = io.BytesIO()
    Image.fromarray(array).save(output, format="PNG")
    return output.getvalue()


@pytest.fixture
def sample_project(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "data/raw").mkdir(parents=True)
    (tmp_path / "models").mkdir()
    rng = np.random.default_rng(12)
    road = jpeg(rng.integers(60, 190, size=(240, 320, 3), dtype=np.uint8))
    white = jpeg(np.full((240, 320, 3), 255, np.uint8))
    files = {
        "images/1/1_road_20260825_070000.png": road,
        "images/1/1_road_20260825_073000.png": white,
        "images/1/1_alias_20260825_070000.png": road,
        "images/2/2_road_20260825_070000.png": b"not an image",
        "images/3/3_road_missing.png": road,
    }
    archive = tmp_path / "data/raw/images.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    metadata = tmp_path / "data/raw/cctv_info.csv"
    with metadata.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["camera_id", "latitude", "longitude", "image_count"])
        writer.writeheader()
        for camera, n in [("1", 3), ("2", 1), ("3", 1)]:
            writer.writerow(dict(camera_id=camera, latitude=13.7, longitude=100.5, image_count=n))
    weights = tmp_path / "models/test.pt"
    weights.write_bytes(b"fixture detector weights; never used as real YOLO")
    # Source IDs follow the same public identity contract, with exact archive hash.
    archive_sha = file_hash(archive)
    first = digest(
        dict(dataset_id=archive_sha, normalized_archive_member_path="images/1/1_alias_20260825_070000.png")
    )
    regions = [
        dict(
            region_version_id="region-1",
            camera_id="1",
            camera_view_id="view-1",
            roi_id="road_total",
            reference_image_id=first,
            reference_width_px=320,
            reference_height_px=240,
            polygon_normalized=[dict(x=0, y=0), dict(x=1, y=0), dict(x=1, y=1), dict(x=0, y=1)],
            member_image_ids=[],
            valid_from_utc=None,
            valid_to_utc=None,
            review_status="approved",
        )
    ]
    (tmp_path / "configs/regions.json").write_text(json.dumps(regions))
    (tmp_path / "configs/reviews.jsonl").write_text("")
    config = dict(
        project_root=str(tmp_path),
        source=dict(
            archive=str(archive),
            metadata=str(metadata),
            filename_regex=r"^(?P<camera_id>\d+)_.*_(?P<timestamp>\d{8}_\d{6})\.png$",
            timestamp_verified=True,
            acquisition_verified=True,
        ),
        detector=dict(weights=str(weights), weights_sha256=file_hash(weights), input_size=320, batch_size=2),
        regions="configs/regions.json",
        overrides="configs/reviews.jsonl",
        validation=dict(status="provisional"),
    )
    path = tmp_path / "configs/pipeline.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


class FixtureDetector:
    """Deterministic test double, used only in contract tests, never real releases."""

    def __init__(self, fail=False, empty=False):
        self.fail = fail
        self.empty = empty
        self.calls = 0

    def predict(self, paths: list[Path]):
        self.calls += 1
        if self.fail:
            raise RuntimeError("Forced inference failure")
        boxes = (
            []
            if self.empty
            else [
                dict(
                    detection_index=0,
                    source_class_id=2,
                    source_class_name="car",
                    mapped_class="car",
                    confidence=0.9,
                    x1=10.0,
                    y1=20.0,
                    x2=100.0,
                    y2=100.0,
                )
            ]
        )
        return [dict(boxes=boxes, timing=dict(preprocess=0.0, inference=1.0, postprocess=0.0)) for p in paths]

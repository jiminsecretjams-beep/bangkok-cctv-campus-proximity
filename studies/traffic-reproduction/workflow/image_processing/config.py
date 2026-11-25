"""Strict, hashed configuration. All relative paths resolve against the project root."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str
        ).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)


class Source(StrictModel):
    archive: str = "data/raw/cctv_images.zip"
    metadata: str = "data/raw/cctv_info.csv"
    timezone: str = "Asia/Bangkok"
    filename_regex: str = r"^(?P<camera_id>\d+)_.*(?P<timestamp>\d{8}_\d{6})"
    timestamp_format: str = "%Y%m%d_%H%M%S"
    timestamp_verified: bool = False
    acquisition_verified: bool = False
    time_basis: Literal["acquired", "camera"] = "acquired"
    verification_note: str = "Unverified source naming convention"
    cameras: list[str] = Field(default_factory=list)
    max_images_per_camera: int | None = Field(default=None, ge=1)
    max_uncompressed_gb: float = Field(default=15, gt=0)


class Detector(StrictModel):
    backend: Literal["ultralytics", "dfine"] = "ultralytics"
    weights: str = "models/yolo11n.pt"
    weights_sha256: str | None = None
    auxiliary_sha256: dict[str, str] = Field(default_factory=dict)
    model_revision: str | None = None
    device: str = "cpu"
    mps_memory_fraction: float = Field(default=0.30, gt=0, le=0.5)
    agnostic_nms: bool = False
    input_size: int = Field(default=960, ge=64)
    batch_size: int = Field(default=4, ge=1, le=128)
    retain_confidence_min: float = Field(default=0.10, ge=0, le=1)
    iou: float = Field(default=0.5, gt=0, le=1)
    max_detections: int = Field(default=1000, ge=1)
    max_attempts: int = Field(default=2, ge=1, le=5)


class Quality(StrictModel):
    diagnostic_width: int = Field(default=320, ge=64)
    min_width: int = Field(default=160, ge=1)
    min_height: int = Field(default=120, ge=1)
    blank_fraction: float = Field(default=0.99, gt=0, le=1)
    blank_std: float = Field(default=4, ge=0)
    dark_pixel_max: int = Field(default=20, ge=0, le=255)
    bright_pixel_min: int = Field(default=245, ge=0, le=255)
    dark_review_fraction: float = Field(default=0.8, ge=0, le=1)
    low_luma_review_threshold: float = Field(default=70, ge=0, le=255)
    glare_review_fraction: float = Field(default=0.55, ge=0, le=1)
    blur_review_threshold: float = Field(default=8, ge=0)
    test_pattern_min_score: float = Field(default=0.88, ge=0, le=1)
    frozen_min_frames: int = Field(default=3, ge=2)
    frozen_min_seconds: float = Field(default=600, ge=0)
    frozen_max_gap_seconds: float = Field(default=7200, gt=0)
    similarity_review: float = Field(default=0.995, ge=0, le=1)
    night_validated: bool = False


class ValidationPolicy(StrictModel):
    status: Literal["pilot", "provisional", "validated"] = "provisional"
    labels: str | None = None
    split_manifest: str | None = None
    minimum_evaluation_images: int = Field(default=150, ge=1)
    maximum_mae: float | None = Field(default=None, ge=0)
    maximum_absolute_bias: float | None = Field(default=None, ge=0)
    maximum_false_acceptance: float | None = Field(default=None, ge=0, le=1)
    maximum_false_rejection: float | None = Field(default=None, ge=0, le=1)
    scope: list[str] = Field(default_factory=list)


class PipelineConfig(StrictModel):
    schema_version: Literal[1] = 1
    project_root: str = "."
    source: Source = Field(default_factory=Source)
    detector: Detector = Field(default_factory=Detector)
    quality: Quality = Field(default_factory=Quality)
    validation: ValidationPolicy = Field(default_factory=ValidationPolicy)
    regions: str = "configs/camera_views.json"
    overrides: str = "configs/reviews.jsonl"
    count_thresholds: dict[str, float] = Field(
        default_factory=lambda: dict.fromkeys(["car", "motorcycle", "bus", "truck"], 0.25)
    )
    seed: int = 20261108

    @model_validator(mode="after")
    def thresholds_valid(self):
        if set(self.count_thresholds) != {"car", "motorcycle", "bus", "truck"}:
            raise ValueError("Exactly four target classes required")
        if any(not self.detector.retain_confidence_min <= v <= 1 for v in self.count_thresholds.values()):
            raise ValueError("Counting thresholds must be >= inference retention floor and <=1")
        return self

    def path(self, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else Path(self.project_root) / p


def load_config(path: Path) -> PipelineConfig:
    raw = yaml.safe_load(path.read_text()) or {}
    root = Path(raw.get("project_root", ".."))
    raw["project_root"] = (
        str((path.resolve().parent / root).resolve()) if not root.is_absolute() else str(root)
    )
    return PipelineConfig.model_validate(raw)

"""Pinned YOLO vehicle detector for whole-image counting."""

from __future__ import annotations

import os
from pathlib import Path

import cv2

from .config import PipelineConfig, file_hash
from .imaging import load_rgb

TARGETS = ("car", "motorcycle", "bus", "truck", "bicycle")


def verify_model(config: PipelineConfig) -> Path:
    c = config.detector
    path = config.path(c.weights)
    if not path.is_file() or not c.weights_sha256 or file_hash(path) != c.weights_sha256:
        raise ValueError(
            "Model missing or checksum unpinned/mismatched; obtain the pinned YOLO26x weights described in README.md"
        )
    return path


def configure_device(config: PipelineConfig):
    import torch

    torch.set_num_threads(4)
    if config.detector.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS unavailable; run with Mac accelerator access or select CPU/CUDA explicitly"
            )
        torch.mps.set_per_process_memory_fraction(config.detector.mps_memory_fraction)
    return torch


class YOLODetector:
    def __init__(self, config: PipelineConfig, target_classes=TARGETS):
        path = verify_model(config)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(config.path("data/ultralytics")))
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        os.environ.setdefault("YOLO_OFFLINE", "true")
        os.environ.setdefault("MPLCONFIGDIR", str(config.path("data/matplotlib")))
        from ultralytics import YOLO

        configure_device(config)
        self.model = YOLO(str(path))
        self.config = config.detector
        self.class_ids = [k for k, v in self.model.names.items() if v in target_classes]
        if len(self.class_ids) != len(target_classes):
            raise ValueError("Checkpoint class names do not contain all requested vehicle classes")

    def predict(self, paths: list[Path]) -> list[dict]:
        arrays = [cv2.cvtColor(load_rgb(p), cv2.COLOR_RGB2BGR) for p in paths]
        c = self.config
        results = self.model.predict(
            arrays,
            device=c.device,
            imgsz=c.input_size,
            conf=c.retain_confidence_min,
            iou=c.iou,
            max_det=c.max_detections,
            classes=self.class_ids,
            rect=False,
            verbose=False,
            save=False,
            nms=True,
            agnostic_nms=c.agnostic_nms,
        )
        output = []
        for result in results:
            boxes = []
            if result.boxes is not None:
                for i, (xyxy, confidence, label) in enumerate(
                    zip(
                        result.boxes.xyxy.cpu().tolist(),
                        result.boxes.conf.cpu().tolist(),
                        result.boxes.cls.cpu().tolist(),
                    )
                ):
                    name = result.names[int(label)]
                    boxes.append(
                        dict(
                            detection_index=i,
                            source_class_id=int(label),
                            source_class_name=name,
                            mapped_class=name,
                            confidence=float(confidence),
                            **dict(zip(("x1", "y1", "x2", "y2"), xyxy)),
                        )
                    )
            output.append(dict(boxes=boxes, timing=result.speed))
        if len(output) != len(paths):
            raise RuntimeError("Detector returned an incorrect number of image results")
        return output

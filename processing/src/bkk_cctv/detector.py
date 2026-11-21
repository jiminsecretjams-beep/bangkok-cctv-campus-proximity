"""Pinned local detector adapters and checksummed successful-batch cache."""

from __future__ import annotations

import importlib.metadata
import os
import time
from pathlib import Path

import cv2

from .config import PipelineConfig, digest, file_hash
from .imaging import load_rgb
from .regions import match_regions
from .storage import SCHEMAS, atomic_json, event, read_json, read_table, write_table

TARGETS = ("car", "motorcycle", "bus", "truck")
CLASS_ALIASES = {"motorbike": "motorcycle"}


def verify_model(config: PipelineConfig) -> Path:
    c = config.detector
    path = config.path(c.weights)
    if not path.is_file() or not c.weights_sha256 or file_hash(path) != c.weights_sha256:
        raise ValueError("Model missing or checksum unpinned/mismatched; run pin-model first")
    if c.backend == "dfine" and set(c.auxiliary_sha256) != {"config.json", "preprocessor_config.json"}:
        raise ValueError("D-FINE requires pinned model and image-processor configurations")
    if c.backend == "dfine" and path.name != "model.safetensors":
        raise ValueError("D-FINE weights must be named model.safetensors for local checkpoint loading")
    for name, sha in c.auxiliary_sha256.items():
        if (
            Path(name).name != name
            or not (path.parent / name).is_file()
            or file_hash(path.parent / name) != sha
        ):
            raise ValueError(f"Model auxiliary file missing or checksum mismatch: {name}")
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


def suppress_vehicle_duplicates(boxes: list[dict], iou: float, limit: int) -> list[dict]:
    """Greedy cross-class suppression; preserve adjacent and low-overlap vehicles."""
    selected = []
    for box in sorted(boxes, key=lambda b: (-b["confidence"], b["detection_index"])):
        area = max(0, box["x2"] - box["x1"]) * max(0, box["y2"] - box["y1"])
        if not area:
            continue
        duplicate = False
        for kept in selected:
            intersection = max(0, min(box["x2"], kept["x2"]) - max(box["x1"], kept["x1"])) * max(
                0, min(box["y2"], kept["y2"]) - max(box["y1"], kept["y1"])
            )
            kept_area = (kept["x2"] - kept["x1"]) * (kept["y2"] - kept["y1"])
            if intersection / (area + kept_area - intersection) > iou:
                duplicate = True
                break
        if not duplicate:
            selected.append({**box, "detection_index": len(selected)})
            if len(selected) >= limit:
                break
    return selected


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


class DFineDetector:
    """Objects365→COCO D-FINE, local safetensors only, pure PyTorch on MPS/CUDA/CPU."""

    def __init__(self, config: PipelineConfig):
        path = verify_model(config)
        self.torch = configure_device(config)
        try:
            from transformers import AutoImageProcessor, DFineForObjectDetection
        except ImportError as exc:
            raise RuntimeError("Install the strongest extra: uv sync --extra strongest") from exc
        self.config = config.detector
        self.processor = AutoImageProcessor.from_pretrained(path.parent, local_files_only=True)
        self.model = (
            DFineForObjectDetection.from_pretrained(path.parent, local_files_only=True, use_safetensors=True)
            .eval()
            .to(self.config.device)
        )
        self.names = {int(k): v for k, v in self.model.config.id2label.items()}
        if not set(TARGETS).issubset({CLASS_ALIASES.get(v, v) for v in self.names.values()}):
            raise ValueError("D-FINE checkpoint lacks required vehicle classes")

    def predict(self, paths: list[Path]) -> list[dict]:
        torch, c = self.torch, self.config

        def synchronize():
            if c.device == "mps":
                torch.mps.synchronize()
            elif c.device.startswith("cuda"):
                torch.cuda.synchronize()

        start = time.perf_counter()
        images = [load_rgb(p) for p in paths]
        sizes = [(im.shape[0], im.shape[1]) for im in images]
        inputs = self.processor(
            images=images, size={"height": c.input_size, "width": c.input_size}, return_tensors="pt"
        ).to(c.device)
        synchronize()
        prepared = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model(**inputs)
        synchronize()
        inferred = time.perf_counter()
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=sizes, threshold=c.retain_confidence_min
        )
        output = []
        for result, (height, width) in zip(results, sizes):
            boxes = []
            for xyxy, score, label in zip(
                result["boxes"].cpu().tolist(),
                result["scores"].cpu().tolist(),
                result["labels"].cpu().tolist(),
            ):
                name = self.names[label]
                mapped = CLASS_ALIASES.get(name, name)
                if mapped not in TARGETS:
                    continue
                x1, y1, x2, y2 = xyxy
                x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
                y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append(
                    dict(
                        detection_index=len(boxes),
                        source_class_id=label,
                        source_class_name=name,
                        mapped_class=mapped,
                        confidence=float(score),
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                    )
                )
            if c.agnostic_nms:
                boxes = suppress_vehicle_duplicates(boxes, c.iou, c.max_detections)
            else:
                boxes = boxes[: c.max_detections]
            output.append(
                dict(
                    boxes=boxes,
                    candidate_limit_reached=len(result["scores"]) >= self.model.config.num_queries,
                )
            )
        finished = time.perf_counter()
        timing = dict(
            preprocess=(prepared - start) * 1000 / len(paths),
            inference=(inferred - prepared) * 1000 / len(paths),
            postprocess=(finished - inferred) * 1000 / len(paths),
        )
        for value in output:
            value["timing"] = timing
        return output


def make_detector(config: PipelineConfig):
    return DFineDetector(config) if config.detector.backend == "dfine" else YOLODetector(config)


class InferenceCache:
    def __init__(self, folder: Path):
        self.folder = folder
        self.indexes = {}

    def index(self, prefix: str) -> dict:
        if prefix not in self.indexes:
            path = self.folder / "index" / f"{prefix}.json"
            self.indexes[prefix] = read_json(path) if path.exists() else {}
        return self.indexes[prefix]

    def get(self, key: str) -> dict | None:
        entry = self.index(key[:2]).get(key)
        if not entry:
            return None
        path = self.folder / entry["path"]
        if not path.exists() or file_hash(path) != entry["sha256"]:
            return None
        artifact = read_json(path)
        value = artifact.get(key)
        if not isinstance(value, dict) or not isinstance(value.get("boxes"), list) or "timing" not in value:
            return None
        return value

    def put_batch(self, values: dict[str, dict]) -> None:
        artifact_id = digest(values)
        relative = f"batches/{artifact_id}.json"
        path = self.folder / relative
        if not path.exists():
            atomic_json(path, values)
        sha = file_hash(path)
        prefixes = set()
        for key in values:
            self.index(key[:2])[key] = dict(path=relative, sha256=sha)
            prefixes.add(key[:2])
        for prefix in prefixes:
            atomic_json(self.folder / "index" / f"{prefix}.json", self.index(prefix))


def inference_key(image: dict, config: PipelineConfig) -> str:
    dependencies = ["ultralytics", "torch", "torchvision", "numpy", "Pillow"]
    if config.detector.backend == "dfine":
        dependencies += ["transformers", "safetensors"]
    versions = {name: importlib.metadata.version(name) for name in dependencies}
    return digest(
        dict(
            content=image["content_sha256"],
            decoder="PIL-exif-rgb-v1",
            model=config.detector.weights_sha256,
            settings=config.detector.model_dump(),
            class_map=TARGETS,
            class_aliases=CLASS_ALIASES,
            versions=versions,
            profile="detector-adapters-float32-v3",
        )
    )


def run_inference(
    config: PipelineConfig, work: Path, run_id: str, regions: list[dict], detector=None
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if detector is None:
        verify_model(config)

    images = read_table(work, "images")
    quality = {r["image_id"]: r for r in read_table(work, "quality_results")}
    observations = {r["observation_id"]: r for r in read_table(work, "observations")}
    root = Path(read_json(work / "inventory_summary.json")["raw_root"])
    cache = InferenceCache(config.path("data/cache/inference"))
    outcomes = []
    pending = []
    seen_keys = set()
    output_path = work / "tables/detections.parquet"
    temporary = output_path.with_suffix(".parquet.tmp")
    writer = pq.ParquetWriter(temporary, SCHEMAS["detections"], compression="zstd")
    started = time.perf_counter()

    def success(image, key, value, hit, attempts):
        boxes = value["boxes"]
        timing = value.get("timing", {})
        outcomes.append(
            dict(
                image_id=image["image_id"],
                inference_key=key,
                status="success",
                attempt_count=attempts,
                device=config.detector.device,
                model_weights_sha256=config.detector.weights_sha256,
                preprocess_ms=timing.get("preprocess"),
                inference_ms=timing.get("inference"),
                postprocess_ms=timing.get("postprocess"),
                retained_detection_count=len(boxes),
                max_det_reached=value.get("candidate_limit_reached", False)
                or len(boxes) >= config.detector.max_detections,
                cache_hit=hit,
            )
        )
        if key not in seen_keys:
            seen_keys.add(key)
            if boxes:
                writer.write_table(
                    pa.Table.from_pylist(
                        [{**b, "inference_key": key, "schema_version": 1, "run_id": run_id} for b in boxes],
                        schema=SCHEMAS["detections"],
                    )
                )

    def flush():
        nonlocal detector
        if not pending:
            return
        if detector is None:
            detector = make_detector(config)
        error = None
        values = None
        attempts = 0
        for attempts in range(1, config.detector.max_attempts + 1):
            try:
                values = detector.predict([root / im["source_path"] for im, key in pending])
                if len(values) != len(pending):
                    raise RuntimeError("Incomplete detector batch")
                break
            except RuntimeError as exc:
                error = exc
                if "out of memory" in str(exc).lower() and len(pending) > 1:
                    # Split explicitly; each result records the actual execution profile in the event log.
                    try:
                        values = [detector.predict([root / im["source_path"]])[0] for im, key in pending]
                        event(work, "infer", event="oom_batch_fallback", effective_batch_size=1)
                        break
                    except RuntimeError as inner:
                        error = inner
            except (OSError, ValueError) as exc:
                error = exc
                break
        if values is None:
            # Isolate failures so one bad image does not discard healthy neighbors.
            for im, key in pending:
                try:
                    value = detector.predict([root / im["source_path"]])[0]
                    cache.put_batch({key: value})
                    success(im, key, value, False, attempts + 1)
                except (RuntimeError, OSError, ValueError) as exc:
                    outcomes.append(
                        dict(
                            image_id=im["image_id"],
                            status="failed",
                            attempt_count=attempts + 1,
                            error_code=type(exc).__name__,
                            error_message=str(exc)[:500],
                            cache_hit=False,
                        )
                    )
                    event(
                        work, "infer", image_id=im["image_id"], status="failed", error=str(error or exc)[:500]
                    )
        else:
            batch = {key: value for (im, key), value in zip(pending, values)}
            cache.put_batch(batch)
            for (im, key), value in zip(pending, values):
                success(im, key, value, False, attempts)
        event(work, "infer", completed=len(outcomes), batch_size=len(pending))
        pending.clear()
        if len(outcomes) % 100 < config.detector.batch_size:
            print(f"Inference: {len(outcomes)}/{len(images)} accounted for", flush=True)

    try:
        for image in images:
            obs = observations[image["observation_id"]]
            q = quality[image["image_id"]]
            if obs["identity_status"] == "conflict":
                status = "skipped_conflict"
            elif obs["selected_image_id"] != image["image_id"]:
                status = "skipped_alias"
            elif q["quality_status"] == "invalid":
                status = "skipped_invalid"
            elif q["quality_status"] == "review":
                status = "skipped_review"
            elif not match_regions(image, regions):
                status = "unresolved"
            else:
                status = None
            if status:
                outcomes.append(
                    dict(image_id=image["image_id"], status=status, attempt_count=0, cache_hit=False)
                )
                continue
            key = inference_key(image, config)
            cached = cache.get(key)
            if cached is not None:
                success(image, key, cached, True, 0)
            else:
                pending.append((image, key))
                if len(pending) >= config.detector.batch_size:
                    flush()
        flush()
    finally:
        writer.close()
    temporary.replace(output_path)
    write_table(work, "inference_results", outcomes, run_id)
    atomic_json(
        work / "inference_summary.json",
        dict(
            elapsed_seconds=time.perf_counter() - started,
            successes=sum(r["status"] == "success" for r in outcomes),
            cache_hits=sum(r["cache_hit"] for r in outcomes),
            unique_inference_outputs=len(seen_keys),
            outcomes=len(outcomes),
        ),
    )

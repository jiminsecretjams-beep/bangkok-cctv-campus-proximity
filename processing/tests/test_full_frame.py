"""Scope and missingness contracts for the whole-image rerun."""

import json
from datetime import datetime, timezone

import pandas as pd

from bkk_cctv.config import load_config
from bkk_cctv.full_frame import (
    CLASSES,
    assemble_frame,
    connect,
    count_boxes,
    infer,
    output_key,
)


def test_whole_image_counts_include_all_positions_and_five_classes():
    # A high box that would have been outside a foreground road polygon counts.
    boxes = [dict(mapped_class=c, confidence=0.8, x1=0, y1=0, x2=4, y2=4) for c in CLASSES]
    boxes += [dict(mapped_class="car", confidence=0.249), dict(mapped_class="person", confidence=0.99)]
    counts = count_boxes(boxes)
    assert counts["vehicle_count"] == 5
    assert all(counts[c + "_count"] == 1 for c in CLASSES)


def test_quality_does_not_remove_predictions_and_failures_are_not_zero():
    timestamp = datetime(2026, 8, 29, 13, tzinfo=timezone.utc)
    images = [
        dict(
            image_id=str(i),
            camera_id="1",
            content_sha256=str(i),
            observation_id=str(i),
            decode_status="success",
            timestamp_status="verified",
            observed_at_utc=timestamp,
        )
        for i in range(4)
    ]
    quality = [
        dict(
            image_id=str(i),
            quality_status="invalid" if i == 1 else "usable",
            quality_flags=["blank_white"] if i == 1 else [],
            decision_reason="test",
        )
        for i in range(4)
    ]
    obs = [
        dict(observation_id=str(i), selected_image_id=str(i), identity_status="verified") for i in range(4)
    ]
    outputs = {
        output_key(images[i], "profile"): ("success", dict(boxes=[dict(mapped_class="car", confidence=0.9)]))
        for i in [0, 1]
    }
    outputs[output_key(images[2], "profile")] = ("failed", dict(error="test error"))
    df = assemble_frame(images, quality, obs, outputs, "profile", "test").set_index("image_id")
    assert df.loc["0", "vehicle_count"] == 1
    assert df.loc["1", "raw_vehicle_count"] == 1
    assert pd.isna(df.loc["1", "vehicle_count"])
    assert pd.isna(df.loc["2", "raw_vehicle_count"])
    assert pd.isna(df.loc["3", "raw_vehicle_count"])
    assert df.loc["3", "inference_status"] == "pending"
    assert df.loc["0", "hour_local"] == 20
    assert bool(df.loc["0", "is_weekend"])


def test_inference_has_no_region_or_quality_gate_and_resumes(sample_project, tmp_path):
    config = load_config(sample_project)
    images = [
        dict(image_id=str(i), content_sha256=str(i), decode_status="success", source_path=f"{i}.jpg")
        for i in range(3)
    ]
    images.append(dict(image_id="bad", content_sha256="bad", decode_status="failed", source_path="bad.jpg"))

    class FakeDetector:
        calls = []

        def predict(self, paths):
            self.calls.extend(paths)
            return [dict(boxes=[], timing={}) for _ in paths]

    model = FakeDetector()
    result = infer(config, tmp_path, images, tmp_path, "profile", detector=model)
    assert result["newly_predicted"] == 3
    assert len(model.calls) == 3
    again = infer(config, tmp_path, images, tmp_path, "profile", detector=model)
    assert again["reused_images"] == 3
    assert len(model.calls) == 3
    db = connect(tmp_path / "checkpoints.sqlite")
    assert all(json.loads(p)["boxes"] == [] for (p,) in db.execute("SELECT payload FROM outputs"))
    db.close()

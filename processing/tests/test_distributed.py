"""Distributed handoff contracts: tiny fixtures only, no GPU or real-model runs."""

import json
import hashlib
import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from bkk_cctv.config import file_hash, load_config
from bkk_cctv.distributed import (
    load_job,
    make_job,
    merge_bundles,
    package_batch,
    run_batch,
    safe_id,
    validate_frame,
    verify_bundle,
)
from bkk_cctv.storage import atomic_json


class TinyDetector:
    """Explicit fixture; never used by the public CLI."""

    calls = 0

    def predict(self, paths):
        self.calls += len(paths)
        return [dict(boxes=[], timing={}) for _ in paths]


def job_fixture(sample_project):
    cfg = load_config(sample_project)
    root = Path(cfg.project_root)
    (root / "uv.lock").write_text("test fixture lock\n")
    atomic_json(
        root / "sources.json",
        dict(
            archive=dict(
                file_id="fixture", path="data/raw/images.zip", sha256=file_hash(cfg.path(cfg.source.archive))
            ),
            metadata=dict(file_id="fixture", path="data/raw/cctv_info.csv"),
        ),
    )
    job_path = root / "configs/job.json"
    make_job(sample_project, cfg.path(cfg.source.metadata), job_path, batches=2)
    return job_path


def test_job_is_balanced_disjoint_and_frozen(sample_project):
    path = job_fixture(sample_project)
    job = load_job(path)
    ids = [c for b in job["batches"].values() for c in b["camera_ids"]]
    assert sorted(ids) == ["1", "2", "3"] and len(ids) == len(set(ids))
    assert sum(b["expected_images"] for b in job["batches"].values()) == 5
    job["batches"]["batch-01"]["expected_images"] += 1
    atomic_json(path, job)
    with pytest.raises(ValueError, match="edited"):
        load_job(path)


def test_worker_resumes_without_predicting_again_and_returns_nulls(sample_project):
    path = job_fixture(sample_project)
    model = TinyDetector()
    result = run_batch(path, "batch-01", "fixture", download=False, detector=model, verify_environment=False)
    assert result["status"] == "completed" and model.calls == 2  # two unique pixels: road and blank
    original = model.calls
    run_batch(path, "batch-01", "fixture", download=False, detector=model, verify_environment=False)
    assert model.calls == original
    folder = Path(result["folder"])
    df = pd.read_parquet(folder / "traffic_full_frame.parquet")
    assert len(df) == 3 and df.raw_vehicle_count.eq(0).all()
    assert df.loc[~df.eligible_analysis, "vehicle_count"].isna().all()
    assert df.loc[df.eligible_analysis, "vehicle_count"].eq(0).all()
    bundle = package_batch(path, "batch-01")
    manifest = verify_bundle(bundle, load_job(path), folder.parent / "verified")
    assert manifest["batch_id"] == "batch-01"
    with zipfile.ZipFile(bundle) as z:
        assert not any(name.endswith((".jpg", ".pt", ".sqlite")) for name in z.namelist())


def test_partial_batch_cannot_be_returned(sample_project):
    path = job_fixture(sample_project)
    result = run_batch(
        path,
        "batch-01",
        "fixture",
        download=False,
        limit=0,
        detector=TinyDetector(),
        verify_environment=False,
    )
    assert result["status"] == "partial"
    with pytest.raises(ValueError, match="partial"):
        package_batch(path, "batch-01")


def test_return_bundle_round_trip_and_disjoint_merge(sample_project, tmp_path):
    path = job_fixture(sample_project)
    bundles = []
    for batch in ["batch-01", "batch-02"]:
        run_batch(path, batch, "fixture", download=False, detector=TinyDetector(), verify_environment=False)
        bundles.append(package_batch(path, batch))
    with pytest.raises(ValueError, match="Missing batches"):
        merge_bundles(path, bundles[:1], tmp_path / "missing")
    with pytest.raises(ValueError, match="Duplicate batch"):
        merge_bundles(path, [bundles[0], bundles[0]], tmp_path / "duplicate")
    out = tmp_path / "combined"
    result = merge_bundles(path, bundles, out)
    assert result["status"] == "completed" and result["images"] == 5 and result["cameras"] == 3
    frame = pd.read_parquet(out / "traffic_full_frame.parquet")
    assert frame.image_id.is_unique
    assert frame.loc[frame.inference_status.eq("decode_failure"), "raw_vehicle_count"].isna().all()
    assert len(pd.read_parquet(out / "detections.parquet")) == 0


def test_tampered_or_unexpected_results_are_rejected(sample_project, tmp_path):
    path = job_fixture(sample_project)
    run_batch(path, "batch-01", "fixture", download=False, detector=TinyDetector(), verify_environment=False)
    bundle = package_batch(path, "batch-01")
    altered = tmp_path / "bad.zip"
    with zipfile.ZipFile(bundle) as src, zipfile.ZipFile(altered, "w") as dst:
        for name in src.namelist():
            dst.writestr(name, b"tampered" if name == "summary.json" else src.read(name))
    with pytest.raises(ValueError, match="checksum"):
        verify_bundle(altered, load_job(path), tmp_path / "bad-import")
    with zipfile.ZipFile(altered, "w") as z:
        z.writestr("../escape", b"bad")
    with pytest.raises(ValueError, match="Unexpected"):
        verify_bundle(altered, load_job(path), tmp_path / "traversal")


def test_old_job_and_cross_worker_resume_are_rejected(sample_project, tmp_path):
    path = job_fixture(sample_project)
    run_batch(path, "batch-01", "one", download=False, detector=TinyDetector(), verify_environment=False)
    with pytest.raises(ValueError, match="different worker"):
        run_batch(path, "batch-01", "two", download=False, detector=TinyDetector(), verify_environment=False)
    bundle = package_batch(path, "batch-01")
    wrong = load_job(path)
    wrong["job_id"] = "different-job"
    with pytest.raises(ValueError, match="different frozen job"):
        verify_bundle(bundle, wrong, tmp_path / "different")


@pytest.mark.parametrize("value", ["../bad", "x;echo bad", "name with spaces", "-option", ""])
def test_worker_identifiers_are_safe(value):
    with pytest.raises(ValueError):
        safe_id(value)


def test_zero_cannot_replace_missing_predictions():
    df = pd.DataFrame(
        dict(image_id=["a"], camera_id=["1"], counting_scope=["full_image"], inference_status=["pending"])
    )
    with pytest.raises(ValueError, match="Incomplete"):
        validate_frame(df, {"1": 1})


def test_nonempty_boxes_are_reconciled_even_if_checksums_are_recreated(sample_project, tmp_path):
    path = job_fixture(sample_project)

    class BoxDetector:
        def predict(self, paths):
            return [
                dict(
                    boxes=[
                        dict(
                            mapped_class="car",
                            confidence=0.9,
                            x1=10.0,
                            y1=10.0,
                            x2=50.0,
                            y2=50.0,
                            detection_index=0,
                            source_class_id=2,
                            source_class_name="car",
                        )
                    ],
                    timing={},
                )
                for _ in paths
            ]

    run_batch(path, "batch-01", "fixture", download=False, detector=BoxDetector(), verify_environment=False)
    original = package_batch(path, "batch-01")
    verify_bundle(original, load_job(path), tmp_path / "valid-boxes")
    with zipfile.ZipFile(original) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    frame = pd.read_parquet(io.BytesIO(files["traffic_full_frame.parquet"]))
    frame.loc[0, "raw_car_count"] += 1
    frame.loc[0, "raw_vehicle_count"] += 1
    buf = io.BytesIO()
    frame.to_parquet(buf, index=False)
    files["traffic_full_frame.parquet"] = buf.getvalue()
    manifest = json.loads(files["result_manifest.json"])
    manifest["files"]["traffic_full_frame.parquet"] = hashlib.sha256(buf.getvalue()).hexdigest()
    files["result_manifest.json"] = json.dumps(manifest).encode()
    altered = tmp_path / "internally-inconsistent.zip"
    with zipfile.ZipFile(altered, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    with pytest.raises(ValueError, match="reconcile"):
        verify_bundle(altered, load_job(path), tmp_path / "inconsistent-boxes")

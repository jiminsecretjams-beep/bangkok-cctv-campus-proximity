"""Optional whole-image inference using the original intake, quality and detector code."""

from __future__ import annotations

import json
from pathlib import Path


def run(root: Path, output: Path, limit=None):
    # Imported here so the normal analysis/dashboard does not need PyTorch.
    import pandas as pd
    from .image_processing.config import digest, file_hash, load_config
    from .image_processing.detector import verify_model
    from .image_processing.full_frame import assemble_frame, connect, infer, profile, write_frame
    from .image_processing.inventory import inventory_sources
    from .image_processing.quality import assess_quality
    from .image_processing.storage import atomic_json, read_table

    config = load_config(root / "inputs/image_model.yaml")
    archive = config.path(config.source.archive)
    if file_hash(archive) != "2a088ff18129120d2b2b81b270b3cb879328ba9ad9ccb69c8fb2b1febf88ce87":
        raise ValueError("Image archive differs from the study archive")
    verify_model(config)
    settings = profile(config)
    frozen = dict(
        profile=settings,
        source_sha256=file_hash(archive),
        metadata_sha256=file_hash(config.path(config.source.metadata)),
        config=config.model_dump(),
        code={p.name: file_hash(p) for p in sorted((root / "workflow/image_processing").glob("*.py"))},
    )
    fingerprint = digest(frozen)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "run.json"
    if manifest.exists() and json.loads(manifest.read_text())["fingerprint"] != fingerprint:
        raise ValueError("Source, code, settings or runtime changed; use a different --output folder")
    atomic_json(manifest, dict(fingerprint=fingerprint, **frozen))
    inventory = inventory_sources(config, output, output.name)
    images = read_table(output, "images")
    # No polygons or manual overrides are applied in this study.
    assess_quality(config, output, output.name)
    model_profile = digest(settings)

    def export():
        with connect(output / "checkpoints.sqlite") as db:
            predictions = {
                key: (status, json.loads(payload))
                for key, status, payload in db.execute("SELECT key,status,payload FROM outputs")
            }
        frame = assemble_frame(
            images,
            read_table(output, "quality_results"),
            read_table(output, "observations"),
            predictions,
            model_profile,
            output.name,
        )
        cameras = pd.DataFrame(read_table(output, "cameras")).drop(columns=["schema_version", "run_id"])
        frame = frame.merge(cameras, on="camera_id", how="left", validate="many_to_one")
        write_frame(frame, output / "traffic_full_frame.parquet")
        successful = frame.inference_status.eq("success")
        if frame.image_id.duplicated().any():
            raise ValueError("Duplicate image keys")
        classes = ["car", "motorcycle", "bus", "truck", "bicycle"]
        if (
            not frame.loc[successful, ["raw_" + c + "_count" for c in classes]]
            .sum(axis=1)
            .eq(frame.loc[successful, "raw_vehicle_count"])
            .all()
        ):
            raise ValueError("Raw class counts do not add to the total")
        count_columns = [c + "_count" for c in classes] + ["vehicle_count"]
        if frame.loc[~frame.eligible_temporal, count_columns].notna().any().any():
            raise ValueError("Excluded counts must be null")
        summary = dict(
            status="complete" if successful.all() else "partial",
            images=len(frame),
            success=int(successful.sum()),
            usable_images=int(frame.eligible_temporal.sum()),
            quality=frame.quality_status.value_counts().to_dict(),
            counts_sha256=file_hash(output / "traffic_full_frame.parquet"),
        )
        atomic_json(output / "summary.json", summary)
        return summary

    stats = infer(
        config,
        output,
        sorted(images, key=lambda im: (str(im["observed_at_utc"] or ""), im["camera_id"], im["image_id"])),
        Path(inventory["raw_root"]),
        model_profile,
        limit=limit,
        snapshot=export,
    )
    print(json.dumps(dict(**export(), session=stats), indent=2))

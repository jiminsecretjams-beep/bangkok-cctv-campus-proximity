"""Conservative quality flags; suspected freezing is a review state, not proof."""

from __future__ import annotations

import json
import hashlib
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .config import PipelineConfig, Quality, digest
from .imaging import load_rgb
from .regions import match_regions, vertices
from .storage import atomic_json, event, read_json, read_table, write_table


def test_pattern_score(rgb: np.ndarray) -> float:
    """Compare upper color bars against the conventional seven-color layout."""
    expected = np.array(
        [[255, 255, 255], [255, 255, 0], [0, 255, 255], [0, 255, 0], [255, 0, 255], [255, 0, 0], [0, 0, 255]],
        dtype=float,
    )
    h, w = rgb.shape[:2]
    best = 0.0
    for margin in [0.0, 0.02, 0.04, 0.06]:
        left, right = int(w * margin), int(w * (1 - margin))
        centers = np.linspace(left, right, 15)[1::2].astype(int)
        for level in [0.2, 0.35, 0.5, 0.6]:
            y = int(h * level)
            patches = [
                rgb[max(0, y - 2) : min(h, y + 3), max(0, x - 2) : min(w, x + 3)].mean((0, 1))
                for x in centers
            ]
            # Every bar must match: a single vivid object is insufficient.
            distances = np.max(np.abs(np.array(patches) - expected), axis=1)
            score = float(np.mean(np.clip(1 - distances / 100, 0, 1)))
            best = max(best, score)
    return best


def image_metrics(rgb: np.ndarray, policy: Quality) -> tuple[dict, np.ndarray]:
    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (policy.diagnostic_width, max(1, round(h * policy.diagnostic_width / w))))
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    thumb = cv2.resize(gray, (32, 32)).astype(np.float32)
    dct = cv2.dct(thumb)[:8, :8].flatten()[1:]
    bits = dct > np.median(dct)
    phash = hex(int("".join("1" if x else "0" for x in bits), 2))[2:].zfill(16)
    return dict(
        mean_luma=float(gray.mean()),
        luma_std=float(gray.std()),
        dark_fraction=float(np.mean(gray <= policy.dark_pixel_max)),
        bright_fraction=float(np.mean(gray >= policy.bright_pixel_min)),
        laplacian_variance=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        test_pattern_score=test_pattern_score(small),
        road_phash=phash,
    ), thumb


def base_decision(metrics: dict, width: int, height: int, policy: Quality) -> tuple[str, list[str]]:
    flags = []
    invalid = False
    if metrics["bright_fraction"] >= policy.blank_fraction and metrics["luma_std"] <= policy.blank_std:
        flags.append("blank_white")
        invalid = True
    if metrics["dark_fraction"] >= policy.blank_fraction and metrics["luma_std"] <= policy.blank_std:
        flags.append("blank_black")
        invalid = True
    if metrics["test_pattern_score"] >= policy.test_pattern_min_score:
        flags.append("offline_color_bars")
        invalid = True
    if width < policy.min_width or height < policy.min_height:
        flags.append("low_resolution")
    if metrics["laplacian_variance"] < policy.blur_review_threshold and not invalid:
        flags.append("low_detail")
    if metrics["dark_fraction"] >= policy.dark_review_fraction:
        flags.append("severe_darkness")
    if metrics["mean_luma"] < policy.low_luma_review_threshold and not policy.night_validated:
        flags.append("unvalidated_low_light")
    if metrics["bright_fraction"] >= policy.glare_review_fraction:
        flags.append("severe_glare")
    review = set(flags) - ({"severe_darkness"} if policy.night_validated else set())
    return ("invalid" if invalid else "review" if review else "usable"), flags


def assess_quality(config: PipelineConfig, work: Path, run_id: str, regions: list[dict]) -> None:
    images = read_table(work, "images")
    summary = read_json(work / "inventory_summary.json")
    root = Path(summary["raw_root"])
    overrides_path = config.path(config.overrides)
    override_bytes = overrides_path.read_text() if overrides_path.exists() else ""
    version = digest(
        dict(
            policy=config.quality.model_dump(),
            regions=regions,
            overrides=override_bytes,
            context=[
                (r["image_id"], r["content_sha256"], r["observed_at_utc"], r["observation_id"])
                for r in images
            ],
            algorithm="quality-v1",
        )
    )
    results = {}
    comparisons = {}
    for i, image in enumerate(images):
        row = dict(
            image_id=image["image_id"],
            quality_version=version,
            quality_status="invalid",
            quality_flags=["decode_failure"],
            decision_source="automatic",
            decision_reason="decode failure",
        )
        if image["decode_status"] == "success":
            rgb = load_rgb(root / image["source_path"])
            visual_hash = hashlib.sha256(
                rgb[int(rgb.shape[0] * 0.12) : int(rgb.shape[0] * 0.9)].tobytes()
            ).hexdigest()
            comparisons[image["image_id"]] = (None, None, visual_hash)
            metrics, _ = image_metrics(rgb, config.quality)
            status, flags = base_decision(metrics, image["width_px"], image["height_px"], config.quality)
            row.update(
                metrics,
                quality_status=status,
                quality_flags=flags,
                decision_reason=", ".join(flags) or "passed automatic checks",
            )
            matches = match_regions(image, regions)
            if matches:
                # Compare road content only, excluding configured overlays and a conservative clock strip.
                mask = np.zeros(rgb.shape[:2], np.uint8)
                for region in matches:
                    pts = np.array(
                        [
                            (x * image["width_px"], y * image["height_px"])
                            for x, y in vertices(region["polygon_normalized"])
                        ],
                        np.int32,
                    )
                    cv2.fillPoly(mask, [pts], 255)
                    for overlay in region["overlay_masks_normalized"]:
                        pts = np.array(
                            [(x * image["width_px"], y * image["height_px"]) for x, y in vertices(overlay)],
                            np.int32,
                        )
                        cv2.fillPoly(mask, [pts], 0)
                mask[int(mask.shape[0] * 0.9) :] = 0
                mask[: int(mask.shape[0] * 0.12)] = 0
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                compare = cv2.resize(np.where(mask, gray, 0), (32, 32))
                comparisons[image["image_id"]] = (
                    digest([r["region_version_id"] for r in matches]),
                    compare,
                    visual_hash,
                )
        results[image["image_id"]] = row
        if (i + 1) % 1000 == 0:
            event(work, "quality", processed=i + 1)
            print(f"Quality: {i + 1} images", flush=True)
    by_camera = defaultdict(list)
    selected = {o["selected_image_id"] for o in read_table(work, "observations") if o["selected_image_id"]}
    for im in images:
        if im["image_id"] in selected and im["timestamp_status"] == "verified" and im["observed_at_utc"]:
            by_camera[im["camera_id"]].append(im)
    for group in by_camera.values():
        group.sort(key=lambda r: r["observed_at_utc"])
        run = []

        def flush():
            if len(run) < config.quality.frozen_min_frames:
                return
            seconds = (run[-1]["observed_at_utc"] - run[0]["observed_at_utc"]).total_seconds()
            if seconds < config.quality.frozen_min_seconds:
                return
            gid = digest([r["image_id"] for r in run])
            for im in run:
                r = results[im["image_id"]]
                r["quality_flags"].append("suspected_frozen")
                r.update(stale_group_id=gid, stale_run_seconds=seconds)
                if r["quality_status"] != "invalid":
                    r["quality_status"] = "review"
                r["decision_reason"] = ", ".join(r["quality_flags"])

        for im in group:
            if not run:
                run = [im]
                continue
            prev = run[-1]
            gap = (im["observed_at_utc"] - prev["observed_at_utc"]).total_seconds()
            same = im["content_sha256"] == prev["content_sha256"]
            similarity = 1.0 if same else None
            a, b = comparisons.get(prev["image_id"]), comparisons.get(im["image_id"])
            if a is not None and b is not None:
                if a[2] == b[2]:
                    similarity = 1.0
                elif not same and a[0] is not None and a[0] == b[0]:
                    similarity = 1 - float(np.mean(np.abs(a[1].astype(float) - b[1].astype(float)))) / 255
            if similarity is not None:
                results[im["image_id"]].update(
                    previous_image_id=prev["image_id"], similarity_previous=similarity
                )
            if (
                0 < gap <= config.quality.frozen_max_gap_seconds
                and similarity is not None
                and similarity >= config.quality.similarity_review
            ):
                run.append(im)
            else:
                flush()
                run = [im]
        flush()
    image_index = {r["image_id"]: r for r in images}
    for line in override_bytes.splitlines():
        if not line.strip():
            continue
        override = json.loads(line)
        target = override["image_id"]
        if target not in results:
            continue
        if override["input_content_sha256"] != image_index[target]["content_sha256"]:
            raise ValueError("Override content checksum mismatch")
        if override["quality_status"] not in {"usable", "review", "invalid"} or not override.get("reason"):
            raise ValueError("Invalid review override")
        results[target].update(
            quality_status=override["quality_status"],
            decision_source="manual",
            decision_reason=override["reason"],
            override_id=override["override_id"],
        )
    rows = list(results.values())
    write_table(work, "quality_results", rows, run_id)
    counts = {s: sum(r["quality_status"] == s for r in rows) for s in ["usable", "review", "invalid"]}
    flags = defaultdict(int)
    for r in rows:
        for flag in r["quality_flags"]:
            flags[flag] += 1
    atomic_json(
        work / "quality_summary.json", dict(counts=counts, flags=dict(flags), quality_version=version)
    )

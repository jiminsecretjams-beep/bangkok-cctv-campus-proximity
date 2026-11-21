"""Versioned road polygons, boundary-inclusive point tests and geometry checks."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from shapely.geometry import Point, Polygon


def dt(value):
    if value is None or isinstance(value, datetime):
        return value
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Region validity timestamps must include timezone")
    return parsed.astimezone(timezone.utc)


def vertices(value):
    return [(p["x"], p["y"]) if isinstance(p, dict) else tuple(p) for p in value]


def load_regions(path: Path) -> list[dict]:
    rows = json.loads(path.read_text())
    keys = set()
    for row in rows:
        if row["region_version_id"] in keys:
            raise ValueError("Duplicate region version ID")
        keys.add(row["region_version_id"])
        points = vertices(row["polygon_normalized"])
        polygon = Polygon(points)
        if (
            not polygon.is_valid
            or polygon.area <= 0
            or any(not 0 <= x <= 1 or not 0 <= y <= 1 for x, y in points)
        ):
            raise ValueError(f"Invalid normalized road polygon: {row['region_version_id']}")
        row["polygon_normalized"] = [dict(x=x, y=y) for x, y in points]
        row["overlay_masks_normalized"] = [
            [dict(x=x, y=y) for x, y in vertices(mask)] for mask in row.get("overlay_masks_normalized", [])
        ]
        row["valid_from_utc"] = dt(row.get("valid_from_utc"))
        row["valid_to_utc"] = dt(row.get("valid_to_utc"))
        if row["valid_from_utc"] and row["valid_to_utc"] and row["valid_from_utc"] >= row["valid_to_utc"]:
            raise ValueError("Empty region validity interval")
        row.setdefault("member_image_ids", [])
        row.setdefault("review_status", "draft")
        row.setdefault("direction_label", None)
        row.setdefault("region_group", "road_total")
    # An overlap is checked on each actual image as well; reject obvious global overlaps here.
    for i, a in enumerate(rows):
        for b in rows[i + 1 :]:
            if (a["camera_id"], a["roi_id"]) != (b["camera_id"], b["roi_id"]):
                continue
            if a["review_status"] != "approved" or b["review_status"] != "approved":
                continue
            if a["member_image_ids"] or b["member_image_ids"]:
                if set(a["member_image_ids"]) & set(b["member_image_ids"]):
                    raise ValueError("Overlapping explicit ROI assignments")
                continue
            low = max(
                a["valid_from_utc"] or datetime.min.replace(tzinfo=timezone.utc),
                b["valid_from_utc"] or datetime.min.replace(tzinfo=timezone.utc),
            )
            high = min(
                a["valid_to_utc"] or datetime.max.replace(tzinfo=timezone.utc),
                b["valid_to_utc"] or datetime.max.replace(tzinfo=timezone.utc),
            )
            if low < high:
                raise ValueError("Overlapping approved ROI validity versions")
    return rows


def match_regions(image: dict, regions: list[dict], roi_id: str | None = None) -> list[dict]:
    matches = []
    for row in regions:
        if row["camera_id"] != image["camera_id"] or row["review_status"] != "approved":
            continue
        if roi_id is not None and row["roi_id"] != roi_id:
            continue
        if row["member_image_ids"]:
            if image["image_id"] not in row["member_image_ids"]:
                continue
        else:
            time = image["observed_at_utc"]
            if time is None:
                continue
            if row["valid_from_utc"] and time < row["valid_from_utc"]:
                continue
            if row["valid_to_utc"] and time >= row["valid_to_utc"]:
                continue
        if image["width_px"] is None:
            continue
        ratio = image["width_px"] / image["height_px"]
        if abs(ratio / (row["reference_width_px"] / row["reference_height_px"]) - 1) > 0.01:
            continue
        matches.append(row)
    ids = [r["roi_id"] for r in matches]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Ambiguous ROI versions for image {image['image_id']}")
    return matches


def contains_detection(box: dict, region: dict, width: int, height: int) -> bool:
    point = Point((box["x1"] + box["x2"]) / (2 * width), box["y2"] / height)
    return Polygon(vertices(region["polygon_normalized"])).covers(point)

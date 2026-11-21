"""Source-specific identity parsing with explicit trust, no mtime fallback."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import PurePosixPath
from zoneinfo import ZoneInfo

from .config import Source, digest


def parse_source_identity(member_path: str, source: Source) -> dict:
    match = re.search(source.filename_regex, PurePosixPath(member_path).name)
    camera = None
    fallback = re.match(r"(\d+)[_-]", PurePosixPath(member_path).name)
    if fallback:
        camera = fallback.group(1)
    result = dict(
        camera_id=camera,
        raw_timestamp_text=None,
        timestamp_source="unknown",
        timestamp_status="missing",
        acquired_at_utc=None,
        camera_at_utc=None,
        observed_at_utc=None,
        time_basis=None,
        parser_version=digest(
            source.model_dump(
                exclude={"archive", "metadata", "cameras", "max_images_per_camera", "max_uncompressed_gb"}
            )
        ),
    )
    if not match:
        return result
    groups = match.groupdict()
    result["camera_id"] = groups.get("camera_id") or camera
    text = groups.get("timestamp")
    result.update(raw_timestamp_text=text, timestamp_source="filename")
    if not text:
        return result
    try:
        parsed = datetime.strptime(text, source.timestamp_format).replace(tzinfo=ZoneInfo(source.timezone))
    except ValueError:
        result["timestamp_status"] = "invalid"
        return result
    utc = parsed.astimezone(timezone.utc)
    result.update(
        timestamp_status="verified" if source.timestamp_verified else "provisional",
        observed_at_utc=utc,
        time_basis=source.time_basis,
    )
    result["acquired_at_utc" if source.time_basis == "acquired" else "camera_at_utc"] = utc
    return result

"""One acquisition may have multiple source-file aliases; never merge stale captures."""

from __future__ import annotations

from collections import defaultdict

from .config import digest


def resolve_observations(images: list[dict], acquisition_verified: bool, version: str) -> list[dict]:
    groups = defaultdict(list)
    for image in images:
        known = acquisition_verified and image["observed_at_utc"] is not None and image["camera_id"]
        token = f"{image['camera_id']}:{image['observed_at_utc'].isoformat()}" if known else None
        key = token or image["image_id"]
        groups[key].append(image)
    observations = []
    for token, members in sorted(groups.items()):
        members.sort(key=lambda r: r["source_path"])
        first = members[0]
        obs_id = digest({"dataset": first["dataset_id"], "acquisition": token, "mapping": version})
        for row in members:
            row["observation_id"] = obs_id
        conflict = len({r["content_sha256"] for r in members}) > 1
        known = acquisition_verified and first["observed_at_utc"] is not None and first["camera_id"]
        observations.append(
            dict(
                observation_id=obs_id,
                camera_id=first["camera_id"],
                acquisition_token=token if known else None,
                identity_status="conflict" if conflict else "verified" if known else "provisional",
                selected_image_id=None if conflict else first["image_id"],
                member_count=len(members),
                observed_at_utc=first["observed_at_utc"],
                timestamp_status=first["timestamp_status"],
                mapping_version=version,
                resolution_reason="conflicting bytes at one acquisition"
                if conflict
                else "deterministic source-path selection; aliases retained",
            )
        )
    return observations

"""Recreate the five-question CCTV failure report from the team's saved inputs.

Run: python run.py
The only report-writing code lives in report.py, beside this file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def load_data(folder):
    """Keep missing pictures as missing data; join actual pictures exactly once."""
    sources = json.loads((folder / "sources.json").read_text())
    for name, expected in sources["input_sha256"].items():
        actual = hashlib.sha256((folder / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(
                f"Input checksum changed: {name}. Check the source before updating sources.json."
            )

    with ZipFile(folder / "failure_kit.zip") as archive:

        def read(name):
            with archive.open(f"failure_kit/{name}.csv") as file:
                return pd.read_csv(file, dtype={"camera_id": str})

        slots, cameras, episodes = read("slots"), read("cameras"), read("episodes")
    vehicles = pd.read_csv(folder / "vehicle_counts.csv.gz", dtype={"camera_id": str})
    if not vehicles.rel_path.is_unique or not slots.rel_path.dropna().is_unique:
        raise ValueError("An actual image appears more than once.")
    if not cameras.camera_id.is_unique or not episodes.episode_id.is_unique:
        raise ValueError("Camera or episode IDs are duplicated.")

    slots = slots.merge(
        vehicles.rename(columns={"camera_id": "model_camera_id"}),
        on="rel_path",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    # Repeated null paths are the inferred missing slots, not duplicated images.
    observed = slots.rel_path.notna()
    if not slots.loc[observed, "_merge"].eq("both").all():
        raise ValueError("Some pictures have no matching vehicle-count row.")
    if not slots.loc[observed, "camera_id"].eq(slots.loc[observed, "model_camera_id"]).all():
        raise ValueError("A picture was joined to the wrong camera.")
    if not slots.loc[~observed, "failure_type"].eq("missing").all():
        raise ValueError("Only missing slots may have an empty image path.")
    if not slots.loc[~observed, "vehicle_count"].isna().all():
        raise ValueError("Missing pictures must not become zero traffic.")
    if not slots.is_failure.eq(slots.failure_type.ne("ok")).all():
        raise ValueError("Failure labels disagree with is_failure.")

    totals = slots.groupby("camera_id").is_failure.agg(["size", "sum"]).sort_index()
    supplied = cameras.set_index("camera_id")[["n_slots", "n_failed"]].sort_index()
    if not np.array_equal(totals.to_numpy(), supplied.to_numpy()):
        raise ValueError("Camera totals do not reconcile with slots.csv.")
    episode_counts = slots[slots.is_failure].groupby("episode_id").size()
    expected = episodes.set_index("episode_id").n_slots.reindex(episode_counts.index)
    if not np.array_equal(episode_counts.to_numpy(), expected.to_numpy()):
        raise ValueError("Episode totals do not reconcile with slots.csv.")

    slots = slots.drop(columns=["_merge", "model_camera_id"])
    slots["time"] = pd.to_datetime(slots.slot_at)  # Source timestamps are Bangkok local.
    slots = slots.merge(
        cameras[["camera_id", "district", "subnet"]],
        on="camera_id",
        how="left",
        validate="many_to_one",
    )
    return slots, cameras, episodes


def failure_rates(slots, group):
    table = slots.groupby(group).agg(checks=("is_failure", "size"), failed=("is_failure", "sum"))
    table["percent"] = 100 * table.failed / table.checks
    return table


def freezing_pairs(slots, cameras):
    """Compare only ten-minute periods when both cameras delivered a picture."""
    delivered = slots[slots.rel_path.notna()].copy()
    delivered["period"] = delivered.time.dt.floor("10min")
    delivered["frozen"] = delivered.failure_type.eq("duplicate frame")
    grid = delivered.groupby(["period", "camera_id"]).frozen.max().unstack()
    counts = delivered[delivered.frozen].camera_id.value_counts()
    candidates = counts[counts.ge(5)].index.tolist()
    locations = cameras.set_index("camera_id")
    pairs = []
    for i, first in enumerate(candidates):
        for second in candidates[i + 1 :]:
            common = grid[[first, second]].dropna().astype(bool)
            together = int((common[first] & common[second]).sum())
            either = int((common[first] | common[second]).sum())
            if not either:
                continue
            a, b = locations.loc[first], locations.loc[second]
            lat1, lat2 = map(math.radians, [a.latitude, b.latitude])
            delta_lon = math.radians(a.longitude - b.longitude)
            distance = (
                2
                * 6371
                * math.asin(
                    math.sqrt(
                        math.sin((lat1 - lat2) / 2) ** 2
                        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
                    )
                )
            )
            pairs.append(
                dict(
                    camera_a=first,
                    camera_b=second,
                    cofreeze_bins=together,
                    freeze_union_bins=either,
                    common_observed_bins=len(common),
                    jaccard=together / either,
                    distance_km=distance,
                    same_subnet=bool(a.subnet == b.subnet),
                )
            )
    return (
        pd.DataFrame(pairs).sort_values(["cofreeze_bins", "jaccard"], ascending=False)
        if pairs
        else pd.DataFrame()
    )


def traffic_before_freezing(slots, episodes):
    """Same-camera, same-clock weekday comparison, before the first flagged repeat.

    Require two usable preceding pictures and three other weekdays. Reduce repeat
    events to camera medians so a frequently failing camera does not dominate.
    """
    good = slots[
        slots.failure_type.eq("ok") & slots.vehicle_count.notna() & slots.eligible_temporal.eq(True)
    ].copy()
    good["clock_minute"] = good.time.dt.hour * 60 + good.time.dt.minute
    good = good[good.time.dt.dayofweek.lt(5)]
    rows = []
    for episode in episodes[episodes.failure_type.eq("duplicate frame")].itertuples():
        start = pd.Timestamp(episode.start)
        if start.dayofweek >= 5 or episode.open_censored:
            continue
        # No normal pre-failure comparison during the large shared morning outage.
        if pd.Timestamp("2026-08-27 06:00") <= start < pd.Timestamp("2026-08-27 09:00"):
            continue
        camera = good[good.camera_id.eq(episode.camera_id)]
        before = camera[camera.time.ge(start - pd.Timedelta(minutes=30)) & camera.time.lt(start)]
        if len(before) < 2:
            continue
        minute = start.hour * 60 + start.minute
        control = camera[
            camera.date.ne(episode.date)
            & camera.clock_minute.ge(minute - 30)
            & camera.clock_minute.lt(minute)
        ]
        daily = control.groupby("date").vehicle_count.mean()
        if len(daily) < 3 or daily.mean() <= 0:
            continue
        rows.append(
            dict(
                episode_id=episode.episode_id,
                camera_id=episode.camera_id,
                start=episode.start,
                preceding_images=len(before),
                comparison_days=len(daily),
                before_mean=float(before.vehicle_count.mean()),
                baseline_mean=float(daily.mean()),
                ratio=float(before.vehicle_count.mean() / daily.mean()),
            )
        )
    cases = pd.DataFrame(rows)
    if cases.empty:
        return cases, dict(
            freeze_episodes=0,
            freeze_cameras=0,
            median_camera_ratio=None,
            camera_bootstrap_95pct=[],
        )
    per_camera = cases.groupby("camera_id").ratio.median().to_numpy()
    rng = np.random.default_rng(20261122)
    bootstrap = [np.median(rng.choice(per_camera, len(per_camera), replace=True)) for _ in range(3000)]
    return cases, dict(
        freeze_episodes=len(cases),
        freeze_cameras=len(per_camera),
        median_camera_ratio=float(np.median(per_camera)),
        camera_bootstrap_95pct=np.quantile(bootstrap, [0.025, 0.975]).tolist(),
    )


def analyse(slots, cameras, episodes):
    daily = failure_rates(slots, "date")
    ranked_cameras = failure_rates(slots, "camera_id").nlargest(5, "percent")
    districts = failure_rates(slots, "district")
    districts["cameras"] = slots.groupby("district").camera_id.nunique()
    districts = districts.nlargest(5, "percent")
    frozen = slots[slots.failure_type.eq("duplicate frame")].camera_id.value_counts()
    pairs = freezing_pairs(slots, cameras)
    cases, traffic = traffic_before_freezing(slots, episodes)
    duration = episodes.groupby("failure_type").agg(
        episodes=("episode_id", "size"),
        median_min=("duration_min", "median"),
        max_min=("duration_min", "max"),
    )
    types = {}
    for kind, group in slots[slots.is_failure].groupby("failure_type"):
        types[kind] = dict(
            images=len(group),
            min_hour=int(group.hour.min()),
            max_hour=int(group.hour.max()),
            days=group.date.nunique(),
        )
    longest = episodes[episodes.failure_type.eq("duplicate frame")].sort_values("duration_min").iloc[-1]
    metrics = dict(
        slots=len(slots),
        failures=int(slots.is_failure.sum()),
        cameras_total=len(cameras),
        dates=sorted(slots.date.unique().tolist()),
        failure_percent=100 * slots.is_failure.mean(),
        day=daily.reset_index().to_dict("records"),
        cameras=ranked_cameras.reset_index().to_dict("records"),
        districts=districts.reset_index().to_dict("records"),
        freeze_images=int(frozen.sum()),
        freeze_top5_images=int(frozen.reindex(ranked_cameras.index).fillna(0).sum()),
        freeze_top5_percent=100 * frozen.reindex(ranked_cameras.index).fillna(0).sum() / frozen.sum(),
        lighting=slots.groupby("lighting").is_failure.mean().mul(100).to_dict(),
        types=types,
        pair=None if pairs.empty else pairs.iloc[0].to_dict(),
        traffic=traffic,
        durations=duration.reset_index().to_dict("records"),
        longest_freeze=dict(
            camera_id=longest.camera_id,
            minutes=float(longest.duration_min),
            date=longest.date,
            ongoing_at_end=bool(longest.close_censored),
        ),
    )
    return metrics, dict(
        daily=daily,
        cameras=ranked_cameras,
        districts=districts,
        cofreezing=pairs,
        traffic_cases=cases,
        durations=duration,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "inputs")
    parser.add_argument("--output", type=Path, default=HERE / "output")
    args = parser.parse_args()
    slots, cameras, episodes = load_data(args.data)
    metrics, tables = analyse(slots, cameras, episodes)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(metrics, indent=2, default=str, allow_nan=False))
    for name, table in tables.items():
        table.to_csv(
            args.output / f"{name}.csv",
            index=name not in {"cofreezing", "traffic_cases"},
        )
    slots.to_csv(args.output / "slots_with_traffic.csv.gz", index=False)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output / ".matplotlib"))
    from report import write_report

    pdf = write_report(metrics, args.output)
    print(
        f"Joined {slots.rel_path.notna().sum():,} images; kept {slots.rel_path.isna().sum():,} missing counts empty."
    )
    print(f"Report: {pdf.resolve()}")


if __name__ == "__main__":
    main()

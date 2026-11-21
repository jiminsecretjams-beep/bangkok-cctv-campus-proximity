"""Run one portable camera batch, upload its bundle, or merge returned results."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

from bkk_cctv.config import load_config
from bkk_cctv.distributed import (
    ensure_inputs,
    load_job,
    make_job,
    merge_bundles,
    package_batch,
    run_batch,
    upload_bundle,
    verify_job_environment,
)
from bkk_cctv.storage import read_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "command", choices=["prepare-job", "doctor", "run", "status", "package", "upload", "import"]
    )
    p.add_argument("--job", type=Path, default=Path("configs/distributed_job.json"))
    p.add_argument("--config", type=Path, default=Path("configs/distributed.yaml"))
    p.add_argument("--metadata", type=Path, default=Path("data/raw/cctv_info.csv"))
    p.add_argument("--batches", type=int, default=4)
    p.add_argument("--batch")
    p.add_argument("--worker", help="Your name/handle using letters, numbers, hyphens or underscores")
    p.add_argument(
        "--limit",
        type=int,
        help="Diagnostic: process at most this many new images; not returnable until complete",
    )
    p.add_argument("--offline", action="store_true")
    p.add_argument(
        "--download",
        action="store_true",
        help="Doctor may download the pinned inputs/selected model, but never runs inference",
    )
    p.add_argument("--repo", help="GitHub repository you can write to: owner/name, including your own fork")
    p.add_argument("--bundles", type=Path, nargs="+")
    p.add_argument("--output", type=Path)
    p.add_argument("--allow-partial", action="store_true")
    args = p.parse_args()
    if args.command == "prepare-job":
        result = make_job(args.config, args.metadata, args.job, args.batches)
    elif args.command == "doctor":
        job = load_job(args.job)
        root = args.job.resolve().parent.parent
        config = load_config(root / job["config_path"])
        verify_job_environment(job, config, root)
        if args.download:
            ensure_inputs(job, config, root)
        import torch
        import psutil

        result = dict(
            job_id=job["job_id"],
            platform=platform.platform(),
            architecture=platform.machine(),
            mps_available=torch.backends.mps.is_available(),
            total_memory_gib=round(psutil.virtual_memory().total / 1024**3, 1),
            batches=job["batches"],
            inference_started=False,
        )
        if not result["mps_available"]:
            raise RuntimeError(
                "This frozen job requires Apple Silicon MPS. Run on an M1-or-newer MacBook, or ask the coordinator to freeze a separate CPU job; do not silently change model/device settings."
            )
    elif args.command == "run":
        if not args.batch or not args.worker:
            p.error("run requires --batch and --worker")
        result = run_batch(args.job, args.batch, args.worker, download=not args.offline, limit=args.limit)
    elif args.command == "status":
        if not args.batch:
            p.error("status requires --batch")
        job = load_job(args.job)
        folder = args.job.resolve().parent.parent / "data/distributed" / job["job_id"] / args.batch
        result = {
            name: read_json(folder / name)
            for name in ["worker_identity.json", "progress.json", "summary.json"]
            if (folder / name).exists()
        }
    elif args.command in {"package", "upload"}:
        if not args.batch:
            p.error("choose --batch")
        if args.command == "upload":
            if not args.repo:
                p.error("upload requires --repo owner/name (a repository you can write to)")
            result = upload_bundle(args.job, args.batch, args.repo)
        else:
            result = dict(bundle=str(package_batch(args.job, args.batch)))
    else:
        if not args.bundles or not args.output:
            p.error("import requires --bundles and --output")
        result = merge_bundles(args.job, args.bundles, args.output, args.allow_partial)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

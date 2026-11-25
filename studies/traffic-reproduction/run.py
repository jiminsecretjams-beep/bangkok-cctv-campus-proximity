"""Run the Bangkok school-traffic study from this folder or any working directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def check_inputs():
    manifest = json.loads((ROOT / "inputs/manifest.json").read_text())
    for name, record in manifest["files"].items():
        path = ROOT / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(
                f"Input changed: {name}. Restore the supplied input before reproducing this study."
            )
    print(f"Verified {len(manifest['files'])} input files.", flush=True)


def result_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT / "results"):
        raise ValueError("Keep generated output inside results/, for example results/rebuilt.")
    return path


def rebuild(output):
    from workflow import analysis, calendar, dashboard_bundle, verify

    check_inputs()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Results already exist. Use --output results/rebuilt (or another new folder).")
    output.mkdir(parents=True, exist_ok=True)
    print("1/4 Building calendar labels", flush=True)
    calendar.build(ROOT / "inputs/counts.parquet", ROOT / "inputs/campus", output / "calendar")
    print("2/4 Rebuilding hourly tables, comparisons and charts", flush=True)
    analysis.run(output / "analysis", output / "calendar")
    print("3/4 Independently checking per-image and headline results", flush=True)
    verify.verify(output / "analysis", output / "calendar")
    print("4/4 Preparing dashboard data", flush=True)
    dashboard_bundle.build(output / "analysis", output / "calendar", output / "dashboard")
    check(output)
    print(f"Finished: {output.relative_to(ROOT)}", flush=True)


def check(output):
    from workflow.verify import verify, verify_proximity

    check_inputs()
    verify_proximity()
    verify(output / "analysis", output / "calendar")
    sys.path.insert(0, str(ROOT / "dashboard"))
    import numpy as np
    from dashboard_data import load_bundle, paired_periods, period_summary

    data = load_bundle(output / "dashboard")
    frozen = data["frozen_periods"].set_index(["camera_id", "period"]).sort_index()
    for metric in ["vehicle", "car", "motorcycle", "bus", "truck", "bicycle"]:
        actual = paired_periods(data["daily"], data["context"].camera_id, data["manifest"]["dates"], metric)
        actual = actual.set_index(["camera_id", "period"]).sort_index()
        if not actual.index.equals(frozen.index):
            raise ValueError("Dashboard camera/period keys differ from the analysis")
        for actual_name, frozen_name in [
            ("relative_pct", metric + "_relative_pct"),
            ("weekday_mean", "weekday_" + metric + "_mean"),
            ("weekend_mean", "weekend_" + metric + "_mean"),
        ]:
            np.testing.assert_allclose(actual[actual_name], frozen[frozen_name], rtol=1e-10, equal_nan=True)
    headline = period_summary(
        paired_periods(data["daily"], data["context"].camera_id, data["manifest"]["dates"])
    ).set_index("period")
    for row in data["summary"]["primary_periods"]:
        if not np.isclose(headline.loc[row["period"], "median_pct"], row["median_relative_pct"]):
            raise ValueError("Dashboard headline differs from independently checked analysis")
    print("PASS: inputs, all six class comparisons, hourly data and dashboard headlines.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["rebuild", "check", "dashboard", "infer"])
    parser.add_argument("--output", default="results", help="Results folder, relative to this project")
    parser.add_argument("--port", type=int, default=8501, help="Dashboard port")
    parser.add_argument(
        "--limit", type=int, help="Optional cap on newly inferred images; output stays partial"
    )
    args = parser.parse_args()
    output = result_path(args.output)
    if args.command == "rebuild":
        rebuild(output)
    elif args.command == "check":
        check(output)
    elif args.command == "dashboard":
        environment = {**os.environ, "BKK_DASHBOARD_DATA": str(output / "dashboard")}
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(ROOT / "dashboard/app.py"),
                "--server.port",
                str(args.port),
            ],
            cwd=ROOT,
            env=environment,
            check=True,
        )
    else:
        from workflow.inference import run

        run(ROOT, output / "images", args.limit)


if __name__ == "__main__":
    main()

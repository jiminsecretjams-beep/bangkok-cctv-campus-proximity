# Process a Bangkok CCTV batch on your MacBook

This is the **whole-image** processing handoff. It produces vehicle-count DataFrames that can be returned through GitHub and combined on the coordinator's computer. No road boundaries or reviewer packs are needed. Parked vehicles and every visible road level/direction are included.

The existing campus-proximity project remains in the repository root. Work from this `processing/` folder for the commands below.

## What each person receives

The frozen job in `configs/distributed_job.json` divides all **100 cameras / 61,319 source images** into four disjoint batches: `batch-01` through `batch-04`. Each batch contains 25 cameras and approximately 15,330 images. The coordinator assigns a different batch to each laptop. One person can run more than one batch sequentially if only three laptops are available. Do not edit camera assignments.

| Batch | Cameras | Source images |
|---|---:|---:|
| `batch-01` | 25 | 15,333 |
| `batch-02` | 25 | 15,331 |
| `batch-03` | 25 | 15,332 |
| `batch-04` | 25 | 15,323 |

The model is pinned **YOLO26x**, 960-pixel input, batch size 1, Apple Silicon MPS, 30% accelerator-memory cap. Confidence retention is 0.10 and counting confidence is 0.25. The five target classes are car, motorcycle, bus, truck and bicycle, matching the selected full-image pipeline. `vehicle_count` includes all five; `motor_vehicle_count` excludes bicycles.

## Requirements

- An Apple Silicon MacBook (M1 or newer), preferably 16 GB RAM, plugged in with its lid open.
- At least 10 GB free storage recommended, plus internet for the first download. Each laptop downloads the same 1.59 GB course archive, but extracts only its assigned cameras. The model is downloaded once and checksum-verified.
- GitHub CLI (`gh`) and `uv`. If Homebrew is already installed: `brew install gh uv`.
- A GitHub account is needed to upload the result. You can use your own fork; write access to the coordinator's repository is optional.

An Intel Mac or a change to CPU/model/thresholds requires a separately frozen job. Do not silently alter the pinned setup to make a worker run. Model inference can take hours; the progress file estimates remaining time from observed processing speed.

## Quick start for a friend

Ask the coordinator which batch to run. This example uses `batch-01` and worker name `friend1`; use your assigned batch and your own short name/handle instead.

```bash
gh repo clone Panyawat-Phothiphanich/bangkok-cctv-campus-proximity
cd bangkok-cctv-campus-proximity/processing
uv sync --frozen --python 3.11
uv run --frozen python scripts/distributed_worker.py doctor
```

`doctor` checks the frozen code/environment and MPS availability. It does **not** run the detector. The following command downloads missing public inputs, inventories the selected cameras, checks image quality, and starts whole-image counting:

If an agent reports MPS unavailable on an Apple Silicon Mac, run the same command with normal Mac GPU access or in Terminal. Keep the pinned model/profile; this is often an execution-access issue rather than a reason to change the experiment.

```bash
caffeinate -i uv run --frozen python scripts/distributed_worker.py run --batch batch-01 --worker friend1
```

The SQLite checkpoint and progress file are under `data/distributed/<job-id>/batch-01/`. To inspect them without starting inference:

```bash
uv run --frozen python scripts/distributed_worker.py status --batch batch-01
```

To stop, press Ctrl+C. **Resume using the identical `run` command**, worker name and batch ID. Successfully checkpointed predictions are reused. Never run two processes for the same batch folder. Do not update/edit processing code or the dependency lock during a batch; the job deliberately rejects drift.

Optional short diagnostic: add `--limit 5` to process at most five new unique images, then resume without the limit. A diagnostic/unfinished batch is explicitly partial and cannot be packaged as a completed result.

### Download trouble

The worker only downloads the selected YOLO26x model, not the historical baseline models. If Drive rate limits the archive, download the course file manually from [the supplied course folder](https://drive.google.com/drive/folders/1k2-KxXe4wbDmFGcGXI-pK7TdSXjBHTni), place `cctv_images.zip` in `processing/data/raw/`, and retry. Place its `cctv_info.csv` alongside it. The worker verifies the exact archive checksum and camera metadata values before using them. Do not use a third-party replacement dataset.

## Return your results through GitHub

After the command reports `status: completed`, package the batch:

```bash
uv run --frozen python scripts/distributed_worker.py package --batch batch-01
```

This creates one ZIP containing the Parquet traffic DataFrame, detection table, profile, summary and checksummed manifest. It excludes original CCTV images, model weights, environments, credentials and checkpoints. Do not commit the archive/model/results directories to Git.

If you have write access to the central repository, upload there:

```bash
gh auth login
uv run --frozen python scripts/distributed_worker.py upload --batch batch-01 --repo Panyawat-Phothiphanich/bangkok-cctv-campus-proximity
```

Otherwise, create/use your own fork and upload the result as a Release asset on that fork:

```bash
gh auth login
gh repo fork Panyawat-Phothiphanich/bangkok-cctv-campus-proximity --clone=false
gh api user --jq .login
# Substitute the GitHub username printed above:
uv run --frozen python scripts/distributed_worker.py upload --batch batch-01 --repo YOUR_USERNAME/bangkok-cctv-campus-proximity
```

The upload command packages/validates again, creates a uniquely named result Release, uploads only the ZIP, and prints its link. Send that link to the coordinator. An upload failure leaves the completed local bundle intact; authenticate/fix repository access and retry. Do not send tokens, entire project folders or screenshots instead of the result bundle. If one person processes multiple batches, upload one bundle per batch.

## Coordinator: collect and combine returned batches

Use a clean clone of this repository and the same frozen job. If your existing clone has local edits, use a separate folder instead of resetting those edits. Set up the environment as above; importing does **not** run YOLO, download models, or require an accelerator.

Download each friend's result Release asset (use the repository and release tag from their link):

```bash
mkdir -p returned
gh release download RESULT_RELEASE_TAG --repo FRIEND_USERNAME/bangkok-cctv-campus-proximity --pattern '*.zip' --dir returned
```

After all four different batches have returned:

```bash
uv run --frozen python scripts/distributed_worker.py import --bundles returned/*.zip --output data/combined/full_image_job
```

The importer checks the frozen job, raw archive, model/runtime profile, file hashes, camera/image counts, unique image IDs, and zero/null contracts. Missing/duplicate batches, different models, edited jobs and incomplete results are rejected. Import exactly one final ZIP per batch; do not pass both an older export and its replacement. Failed imports leave a `.importing` folder for inspection and never publish an apparently successful combined result. Use a new output path to retry after resolving the error.

For an explicitly partial inspection, add `--allow-partial`; its manifest lists missing batches. Do not describe that as full-dataset completion.

Successful full import produces:

- `traffic_full_frame.parquet` — authoritative DataFrame, one row per source image.
- `traffic_full_frame.csv` — portable companion export.
- `detections.parquet` — retained boxes, keyed to observations without cross-batch key collisions.
- `import_manifest.json` — provenance, returned manifests, coverage and output checksums.
- `batches/` — preserved returned batch tables/manifests.

Load and continue on the coordinator's computer:

```python
import pandas as pd

df = pd.read_parquet("data/combined/full_image_job/traffic_full_frame.parquet")
analysis = df[df["eligible_temporal"]].copy()
# df includes all input rows and quality flags; analysis has usable unique acquisitions.
# Join campus data using camera_id as a string and a many-to-one left join.
```

Campus distances/types and verified institution calendars can be joined after import. The workers are responsible for image processing only, so calendar research and the final dashboard do not have to be duplicated on each laptop.

## Counting and quality rules

- Count all five visible target classes across the entire decoded image, including parked/stopped vehicles. `motor_vehicle_count` is also supplied for the four motor-vehicle classes.
- Every source image is accounted for. Predictions are attempted for all decodable frames, even if quality rules flag them; raw counts and quality-gated analysis counts remain separate.
- Failed/undecodable/pending processing has null counts; successful inference can have genuine zero raw counts. An invalid/review-quality image keeps raw predictions but has null analysis counts.
- Filenames represent acquisition time in Asia/Bangkok. UTC/local time and weekday/weekend fields are retained; unknown time is not invented.
- Counts are visible vehicle instances per snapshot, not vehicles per hour or unique journeys. Keep coverage/quality flags and compare within-camera patterns. Model/quality validation status remains explicit; this handoff does not manufacture reviewer labels or accuracy scores.

## Copy this prompt into ChatGPT/Codex on a friend's MacBook

> Read processing/AGENTS.md and processing/README.md. I am assigned BATCH_ID and my worker name is MY_HANDLE. Set up the frozen Python 3.11 environment, verify the job and download the pinned course inputs/model if missing. Run only my assigned whole-image batch, resume it until completed, validate/package it, and upload the result ZIP as a GitHub Release asset to a repository I can write to (use my own fork if central write access is unavailable). Give me the result Release link. Do not change the model/classes/thresholds, camera allocation or quality policy; do not generate human labels, upload raw images/weights/credentials, reset unrelated work, or process other batches unless I ask.

Replace `BATCH_ID` and `MY_HANDLE` before sending. The agent should read any runtime error and fix setup/access issues without changing the scientific job definition.

## Tests without real model processing

```bash
uv run --frozen python -m pytest tests -q
uv run --frozen python -m ruff check src scripts tests
```

Tests use tiny fixtures and an explicit fake detector. They exercise batch allocation, resume, incomplete-result rejection, integrity checks, duplicate/missing batches and a complete package/import round trip. No real CCTV inference is started by these tests. The detector adapter is shared with the existing executed full-image pipeline; per-laptop performance and upload permissions must be checked on the actual worker.

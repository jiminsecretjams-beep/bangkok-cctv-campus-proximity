# Working on this project

This folder is the traffic reproduction study. Start with its README.md.

## Commands

```sh
uv sync --frozen --python 3.11
uv run --frozen python run.py check
uv run --frozen pytest -q
uv run --frozen ruff check run.py workflow dashboard tests
uv run --frozen python run.py dashboard
```

The optional image-processing test module is skipped when its extra dependencies are absent; that is expected for a normal dashboard setup. Setup uses Python 3.11 and the committed lock. Do not update dependencies just to open the project.

## Working rules

- Use the included finished data for setup, calendar/analysis work and the dashboard. Run expensive image inference only when the user requests it. No boundary drawing or reviewer interface is required.
- The measurement is whole-image visible vehicle sightings, including parked vehicles. It is not flow, unique trips, speed or school-caused traffic. Read METHODS.md, FINDINGS.md and SOURCES.md before changing interpretations.
- Preserve frozen inputs, class definitions, thresholds, quality flags and missing values. Unknown calendar/closure states stay unknown. Successful empty detections are zero; excluded or unavailable measurements are not zero.
- Rebuild into a new empty folder under results/, for example `python run.py rebuild --output results/rebuilt`. Avoid touching existing results, another server or uncommitted teammate work. The raw inference command uses its own resumable output; follow README.md if explicitly requested.
- Keep edits small and readable. Use run.py as the entry point, workflow/ for calculations and dashboard/ for the interface. Avoid adding a parallel pipeline or duplicate configuration.
- For requested code changes, run relevant tests and checks. Frozen analysis manifests intentionally detect code/input changes. Rebuild outputs and verify them when changing the calculation code; do not merely replace hashes to hide a changed result. If packaging a new submission, regenerate submission_manifest.json for the intended files, excluding itself, caches and environments.
- Create a separate branch for requested changes. Use the repository explicitly named by the user for GitHub operations. Publish or merge only within the user's instructions. Do not message teammates automatically.
- Commit source, reviewed small inputs and intentional result artifacts. Do not commit raw images, model weights, environments, checkpoints, credentials or temporary rebuilds. Report actual test results and remaining limits honestly.

The previous project and original collection scripts remain in archive/pre-submission-20261122; the released model dataset is linked from README.md.

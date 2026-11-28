# Bangkok School Zone dashboard

A working Streamlit dashboard for the completed 25-31 August 2026 study. All 61,319 observations and 100 camera views are included in a small, verified portable data bundle. The app does not run YOLO or require source images, model weights, credentials, the main project environment or a GPU.

## Open on this Mac

Double-click `start.command`, or run from the dashboard folder:

```sh
./start.command
```

Then open **http://127.0.0.1:8501**. If that port is occupied by another app, use `./start.command 8502`. The launcher does not stop other programs. Stop the server with Control-C in its terminal.

## Run on another laptop

Copy or extract the entire dashboard folder, including `data/`, `pyproject.toml`, `uv.lock` and `.streamlit/`. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if it is unavailable. From that folder:

```sh
uv sync --frozen --python 3.11
uv run --frozen streamlit run app.py
```

Optional standalone data check: `uv run --frozen python verify_data.py`.
The portable folder also includes its tests: `uv run --frozen pytest -q`.

Python 3.11 and all dashboard dependencies are pinned by `uv.lock`. The environment is separate from the model pipeline. The data bundle is approximately 3.4 MB; dependency installation needs additional disk space and an initial internet connection. Analysis charts and coordinate view work offline after installation. The street basemap uses public CARTO tiles and therefore needs internet; switch to **Coordinate view** when offline. No map API key is required.

## Views

- **Overview:** median within-camera weekday/weekend changes, camera variation, an interactive camera map and matched-hour profiles. Clicking a camera marker opens that camera's explorer.
- **Camera explorer:** choose any filtered view; compare hourly means or individual snapshots; inspect coverage, vehicle mix and excluded raw predictions; download the selected observations.
- **School context:** inspect institutions within the existing 1.5 km reference screen, scoped calendar evidence and source links. Change the timetable window duration and hypothetical clock shift to compare both adjacent baselines.
- **Coverage & sources:** see missing camera-hours, interpretation rules, reference robustness results, source links and filtered CSV/JSON/report downloads.
- **Findings & workflow:** see the full-study result, inspect the held-out Monday peak check and download the Graphviz workflow source for the presentation. This view always uses the full study.

## What filters mean

Vehicle class, study dates, day type, school distance and university/college context filter the current data. Specific-camera selection is optional; when enabled, an empty list means no cameras, not all cameras. Reset restores the full original study.

The main percentage comparison uses the exact accepted eight-hour days from the verified analysis. Each camera needs at least three complete weekdays and both weekend days. The original full-study all-vehicle cards must show **+61.2% morning**, **+0.9% midday**, **+28.7% afternoon**, and **89 balanced camera views**.

For stable ratios, weekend means must be >=1 total vehicle or >=0.25 for a single class. A network percentage needs at least ten stable camera ratios. Smaller selections can still be explored as raw view-level time series and table comparisons. Missing denominators, absent weekend dates, rejected coverage and unsupported timetable cases are marked unavailable rather than zero.

School-distance filters use screened catalogue reference points, not gates. The screened catalogue still contains ambiguous institutions and is not a formal-school census. University/college context overlaps with school context. Unknown terms, school opening, closure and attendance remain unknown; programme/cohort evidence is never applied to an entire institution. Weekdays are not automatically called school days.

The hour-by-hour panel has its own sample sizes and is separate from the stricter primary cards. The robustness expander is labelled as a **full-study reference** and does not change with sidebar filters. Vehicle mix describes all five classes in the selected usable captures, regardless of which single class is selected for the main comparison.

## Tests

From this dashboard folder:

```sh
uv run --frozen python verify_data.py
uv run --frozen pytest -q
uv run --frozen ruff check .
```

The tests reconcile all six class comparisons against frozen verified results, check empty/missing/sparse cases, prevent lines bridging missing hours, preserve calendar scopes and exercise all pages and key widget flows with Streamlit AppTest. Browser inspection separately checks actual chart/map rendering, marker navigation, downloads and the responsive layout.

## Data provenance and updates

`data/manifest.json` contains a dataset ID, source hashes and file hashes. The app verifies the bundle before displaying data. It contains selected observation fields, complete hourly cells, accepted-day period tables, camera/institution context, scoped calendars and school-anchor evidence. It excludes source images, model files, private IP addresses and environment credentials.

From the project root, these commands rebuild the analysis and its companion dashboard bundle from the saved counts.

```sh
cd studies/traffic-reproduction
uv sync --frozen --no-dev --python 3.11
uv run --frozen --no-dev python run.py rebuild --output results/rebuilt
```

Choose an empty output folder for each rebuild. The command writes the companion dashboard data to `results/rebuilt/dashboard`. The presentation dashboard opens with the saved tables included in its own `data/` folder. [Study rebuild guide](../studies/traffic-reproduction/README.md#recreate-the-results)

The findings download links to the published evidence and works as a standalone Markdown file with an internet connection. The study report in the source ZIP links directly to the charts and tables included there. [Findings and bundled evidence](../studies/traffic-reproduction/FINDINGS.md)

## Limits and scope

This is a local dashboard, not a publicly hosted site. It binds to this laptop only. Whole-image model counts include parked vehicles and multiple directions; they measure repeated visible sightings, not flow, unique trips, congestion or school causation. The study covers one week and one weekend, no observed public holiday, incomplete school calendars, model rain and unverified exposure-time clocks. Model accuracy remains provisional. No reviewer forms were changed.

Implementation references: [Streamlit chart selection](https://docs.streamlit.io/1.51.0/develop/api-reference/charts/st.pydeck_chart), [AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest), [Streamlit themes](https://docs.streamlit.io/develop/api-reference/configuration/config.toml).

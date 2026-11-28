# Bangkok School Zone Commute Tracker

Course 2190513 Data Science, 2026/1. Study period 25–31 August 2026, Bangkok time.

## Run the dashboard

Extract the ZIP and open `dashboard/start.command` on macOS, or run these commands with uv installed:

```sh
cd dashboard
uv sync --frozen --python 3.11
uv run --frozen python verify_data.py
uv run --frozen streamlit run app.py
```

Open http://127.0.0.1:8501. The dashboard includes the verified portable data bundle. Use Coordinate view for an offline map. Use **Findings & workflow** for the presentation diagram, primary results, and held-out Monday check.

## How the components connect

| Component | Code and evidence |
|---|---|
| Data engineering | `processing/` contains the frozen image inventory, quality checks, YOLO26x inference, and batch import code from commit `a0128c67985bd3e6f92d1e28f8f671dfc6c8a8a3`. `scripts/campus_pipeline.py` cleans location references and applies the 1.5 km Haversine screen. |
| Data science | `scripts/build_analysis_dataset.py` joins the four batches to camera and campus context. `scripts/add_public_holidays.py` adds scoped holiday-reference flags. `dashboard/dashboard_data.py` recomputes the period comparisons from accepted daily tables and calculates matched-hour profiles. `dashboard/data/METHODS.md` describes the design and exclusions. |
| Visualization | `dashboard/app.py` serves Streamlit charts, filters, maps, and coverage views. `dashboard/workflow.dot` is the editable diagram source. |
| Insights | The app reports the full-week comparison. Among 89 balanced camera views, the median within-camera weekday versus weekend difference was +61.2% in the morning, +0.9% at midday, and +28.7% in the afternoon. 48 of 57 selected weekday peak candidates repeated direction on the held-out Monday. [Findings and bundled evidence](studies/traffic-reproduction/FINDINGS.md) |

The dashboard reports visible vehicles per snapshot, including parked vehicles. School attendance and the contribution of school commutes remain unverified. The study week contains no listed public holiday.

## Human count validation

Teammates counted 40 morning frames from 20 cameras, with a weekday and weekend frame matched by camera and clock hour. The report explains the counting protocol, the differences by day type, and how to check the results against the submitted worksheets and saved YOLO26x counts. [Human count validation report](docs/human-count-spot-check.md#matched-morning-review)

## Additional data collection

Web scraping and API collection are separate from the data engineering component. This inventory records the source material and where it is used.

| Source | Collection and use | Evidence in this package |
|---|---|---|
| OpenStreetMap | Overpass API queries collected education locations and outlines for the campus screen. | `scripts/refresh_osm.py`, stored Overpass queries and snapshots in `data/reference/`. |
| Thai public holiday reference | The team's script downloaded JSON from a community mirror. The holiday join records date membership and preserves the mirror's scope. | `data/raw/webscraping/scripts/03_public_holidays.py`, `data/reference/public_holidays/source_manifest.json` and its pinned snapshot. |
| Chulalongkorn University BBA International | The supplied calendar records describe one programme. The dashboard reports scoped calendar evidence. | `data/raw/webscraping/data/academic_calendar.csv`, `dashboard/data/sources.json`, and `dashboard/data/calendar.parquet`. |
| Tourism Authority of Thailand calendar and Horwang School | The dashboard's source records document a holiday cross-check and a school timetable case. | Source URLs, scope and recorded hashes in `dashboard/data/sources.json`, with processed evidence in the dashboard bundle. |
| OBEC and Bangkok Metropolitan Administration school records | Supplied directory and coordinates seed the school catalogue. | Original team files under `data/raw/webscraping/` and reference integration in `scripts/campus_pipeline.py`. |

The source records list attribution and exact source URLs. The portable dashboard bundle records hashes for the inputs and includes prepared tables, methods, and findings. The rebuild workflow recreates the research tables, charts, and data for the companion dashboard. [School traffic rebuild](studies/traffic-reproduction/README.md)

The [camera reliability study](studies/camera-reliability/README.md) is a companion analysis of missing, frozen, and unusable frames from the same study week. Its report explains the failure patterns and the limits of drawing conclusions about their causes.

## Upstream reproduction

Run `python3 scripts/campus_pipeline.py` to rebuild the campus tables from cached references. The submission copy removes camera IP fields from the camera metadata and old comparison flags. The rebuilt 1.5 km classifications use the same coordinates.

The four original batch ZIPs and joined dataset are available in the [project Releases](https://github.com/jiminsecretjams-beep/bangkok-cctv-campus-proximity/releases). To recombine the batches without rerunning inference, follow `processing/README.md` and give the resulting folder to the join script's `--traffic-folder` argument. For example, after importing to `processing/data/combined/full_image_job`:

```sh
processing/.venv/bin/python scripts/build_analysis_dataset.py --traffic-folder processing/data/combined/full_image_job
processing/.venv/bin/python scripts/add_public_holidays.py
```

The course archive and pinned model are separate downloads for an authorized rerun using the frozen worker. The ZIP carries code, references and packaged analysis tables.

The study uses the course images and recorded external sources. The accuracy of the detector remains provisional.

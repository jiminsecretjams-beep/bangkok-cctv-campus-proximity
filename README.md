# Bangkok School Zone Commute Tracker

2190513 Data Science · Group project · Bangkok, 25–31 August 2026

We used **61,319 snapshots from 100 Bangkok CCTV cameras** to see how visible vehicle counts near educational institutions changed by time of day on weekdays and weekends.

## Try the dashboard

Download and extract the [course source bundle](https://github.com/jiminsecretjams-beep/bangkok-cctv-campus-proximity/releases/tag/course-source-bundle-v2). On macOS, open `dashboard/start.command`. On another platform, install uv and run these commands from the extracted project folder:

```sh
cd dashboard
uv sync --frozen --python 3.11
uv run --frozen streamlit run app.py
```

Open [localhost:8501](http://127.0.0.1:8501). Start with **Overview**, pick a camera in **Camera explorer**, then open **Findings & workflow** for the results and project diagram. The dashboard data is included in the bundle.

## What we found

Among **89 camera views** with balanced coverage, the median within-camera weekday count was **61.2% higher in the morning**, **0.9% higher at midday**, and **28.7% higher in the afternoon** than the matching weekend windows. In a separate check, **48 of 57** selected weekday peaks stayed above their baseline on the held-out Monday.

The numbers describe vehicles visible per snapshot, including parked vehicles. The study covers one week, and school attendance and the share of traffic caused by school commutes remain unverified. See the [findings](dashboard/data/FINDINGS.md) and [methods](dashboard/data/METHODS.md).

In a [human spot check of eight frames](docs/human-count-spot-check.md), YOLO differed from a teammate’s count by an average of **2.75 vehicles per frame** in absolute terms.

## Where everything is

| Course component | Where to look |
|---|---|
| Data engineering | [Image processing](processing/README.md) covers YOLO counts, timestamps, quality checks, and merging batches. [Campus proximity](docs/campus-proximity.md) explains the 1.5 km location screen. |
| Data science | The [analysis scripts](scripts/) build the joined traffic table, and the [dashboard calculations](dashboard/dashboard_data.py) reproduce the comparison from packaged tables. |
| Visualization | The [Streamlit app](dashboard/README.md) includes the camera map, hourly charts, and coverage views. |
| Insights | The [findings report](dashboard/data/FINDINGS.md) explains the morning pattern, the Monday check, and the limits of the study. |
| Additional data collection | The [source inventory](SUBMISSION.md#additional-data-collection) explains how OpenStreetMap, holiday records, and academic calendars were used. |

The [submission guide](SUBMISSION.md) explains the full workflow, reproduction steps, and which original research-table scripts were unavailable. The [batch results and joined dataset](https://github.com/jiminsecretjams-beep/bangkok-cctv-campus-proximity/releases) are available separately.

**AI use.** Our group used Codex to help implement and check the pipeline and dashboard, develop analysis code and charts, and prepare the documentation and source package. See the [submission guide](SUBMISSION.md) for the disclosure.

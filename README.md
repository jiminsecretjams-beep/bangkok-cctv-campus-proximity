# Bangkok School Zone Commute Tracker

2190513 Data Science · Group project · Bangkok, 25–31 August 2026

We used **61,319 snapshots from 100 Bangkok CCTV cameras** to see how visible vehicle counts near educational institutions changed by time of day on weekdays and weekends.

## Try the dashboard

Download and extract the [current source ZIP](https://github.com/jiminsecretjams-beep/bangkok-cctv-campus-proximity/archive/refs/heads/main.zip). On macOS, open `dashboard/start.command`. On another platform, install uv and run these commands from the extracted project folder:

```sh
cd dashboard
uv sync --frozen --python 3.11
uv run --frozen streamlit run app.py
```

Open [localhost:8501](http://127.0.0.1:8501). Start with **Overview**, pick a camera in **Camera explorer**, then open **Findings & workflow** for the results and project diagram. The dashboard data is included in the bundle.

## What we found

Among **89 camera views** with balanced coverage, the median within-camera weekday count was **61.2% higher in the morning**, **0.9% higher at midday**, and **28.7% higher in the afternoon** than the matching weekend windows. In a separate check, **48 of 57** selected weekday peaks stayed above their baseline on the held-out Monday.

The numbers describe vehicles visible per snapshot, including parked vehicles. The study covers one week, and school attendance and the share of traffic caused by school commutes remain unverified. See the [findings and evidence](studies/traffic-reproduction/FINDINGS.md) and [methods](dashboard/data/METHODS.md).

Teammates reviewed **40 matched morning frames**. Manual counts were higher on weekdays for **15 of 20 cameras**. Compared with the reviewers, YOLO counted an average of **6.90 fewer vehicles per weekday frame** and **2.85 fewer per weekend frame** in this sample. [Human count review and evidence](docs/human-count-spot-check.md#matched-morning-review)

## Reproduce the analysis

- [School traffic rebuild](studies/traffic-reproduction/README.md) recreates the hourly tables, calendar joins, charts, and dashboard data from saved counts and documented inputs.
- [Camera reliability study](studies/camera-reliability/README.md) looks at missing, frozen, and unusable snapshots during the same week. It includes the five-question report and a CPU-only rebuild command.

The dashboard quick start above opens the school traffic presentation app.

## Where everything is

| Course component | Where to look |
|---|---|
| Data engineering | [Image processing](processing/README.md) covers YOLO counts, timestamps, quality checks, and merging batches. [Campus proximity](docs/campus-proximity.md) explains the 1.5 km location screen. |
| Data science | The [analysis scripts](scripts/) build the joined traffic table, and the [dashboard calculations](dashboard/dashboard_data.py) reproduce the comparison from packaged tables. |
| Visualization | The [Streamlit app](dashboard/README.md) includes the camera map, hourly charts, and coverage views. |
| Insights | The [findings report](studies/traffic-reproduction/FINDINGS.md) explains the morning pattern, the Monday check, and the limits of the study. |
| Additional data collection | The [source inventory](SUBMISSION.md#additional-data-collection) explains how OpenStreetMap, holiday records, and academic calendars were used. |

The [submission guide](SUBMISSION.md) explains the project workflow and reproduction steps. The [batch results and joined dataset](https://github.com/jiminsecretjams-beep/bangkok-cctv-campus-proximity/releases) are available separately.

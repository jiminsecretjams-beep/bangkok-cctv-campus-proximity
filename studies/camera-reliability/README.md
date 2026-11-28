# Bangkok camera reliability study

We analysed one week of snapshots from 100 Bangkok traffic cameras, 25–31 August 2026. We ask when pictures go missing, freeze or become unusable, which cameras are affected, and whether traffic helps explain it.

**[Read the five-question report](results/CCTV_Failures_Five_Questions.pdf)** · [See the calculated numbers](results/summary.json)

The strongest pattern is a large interruption affecting many cameras together, followed by persistent freezes on a few feeds. The faulty equipment remains unknown.

## Recreate the report

Use Python 3.11. From the root of this repository:

```bash
cd studies/camera-reliability
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py
```

On Windows, activate with `.venv\Scripts\activate` instead of the `source` line.

After setup, **`python run.py` is the only command needed to rerun the workflow.** It checks the inputs and joins, calculates the results, draws six charts and writes the five-page PDF. No GPU, API key or additional download is needed after the Python packages are installed.

Everything is written under `output/`:

- `CCTV_Failures_Five_Questions.pdf`: one readable page per question.
- `summary.json` and CSV tables: the numbers behind the report.
- `charts/`: six PNG charts for slides.
- `slots_with_traffic.csv.gz`: the joined dataframe, with missing vehicle counts left empty.

To write somewhere else: `python run.py --output my_results`.

## In Google Colab

Run these lines in a notebook cell. A CPU runtime is enough.

```python
!git clone https://github.com/jiminsecretjams-beep/bangkok-cctv-campus-proximity.git
%cd bangkok-cctv-campus-proximity/studies/camera-reliability
!pip install -q -r requirements.txt
!python run.py
```

## The code

There are two working Python files, plus one test file:

| File or folder | Purpose |
| --- | --- |
| `run.py` | Loads/checks the inputs, joins vehicle counts, answers the five questions and saves results. |
| `report.py` | Turns those calculated results into charts and the PDF. |
| `test_analysis.py` | Checks joins, missing counts, shared freezing, traffic comparisons and durations. |
| `requirements.txt` | The package versions used for the clean reproduction check. |
| `inputs/` | The original failure kit, a small vehicle-count export and their source/checksum record. |
| `results/` | A saved report and summary, readable without running code. |
| [School traffic rebuild](../traffic-reproduction/README.md) | The companion analysis of visible vehicles near schools. |

Start with `run.py`; its functions follow the five questions. There is no service or framework to configure.

## What the analysis does

1. **When do failures happen?** Calculate daily rates and describe hours/lighting for each error type. Timing and ranking charts include the whole week, including 27 August's large outage.
2. **Which cameras or districts fail most?** Rank failed checks divided by all checks. Show the number of sampled cameras beside district rates.
3. **Do cameras fail together?** Compare freezing in ten-minute periods when both cameras delivered a picture. Check the strongest pair’s distance and recorded address group. Shared equipment remains unverified.
4. **Does traffic matter?** Compare usable pictures in the 30 minutes before the first flagged freeze with the same camera and clock time on other weekdays. Require two preceding images and three comparison weekdays. Summarise repeated events by camera, then resample cameras with a fixed seed.
5. **How long do errors last?** Summarise the supplied episode durations. These are snapshot-based estimates; the first duplicate follows the actual freeze onset, and some faults continue beyond the recording period.

The traffic comparison excludes weekend onsets, opening-boundary episodes and onsets during 27 August 06:00–09:00 because they lack an adequate normal comparison. This filter applies to the traffic question, not the timing or ranking charts.

## Inputs and scope

- **`inputs/failure_kit.zip`** contains the team’s failure labels: 63,129 camera checks, including 61,319 actual images and 1,810 inferred missing pictures. It contains `slots.csv`, `cameras.csv`, `episodes.csv` and the team's definitions. The ZIP is read directly. This copy omits camera IP addresses and uses opaque labels for the original address groups. Failure slots, episodes and group membership are preserved.
- **`inputs/vehicle_counts.csv.gz`** keeps four columns from the saved **YOLO26x whole-image** output: image path, camera ID, vehicle count and analysis eligibility. Null counts and real zeroes are preserved. The model used a confidence threshold of 0.25.
- **`inputs/sources.json`** records input hashes, the original model dataframe hash and its [public download](https://github.com/Panyawat-Phothiphanich/bangkok-cctv-campus-proximity/releases/tag/yolo26x-full-frame-20261109).

This submission recreates the analysis/report **from the supplied failure labels and saved model counts**. It does not regenerate the team's labels from raw images or rerun YOLO. Raw images and model weights are not included. Human detection precision/recall and hardware root causes have not been established.

Every actual picture must join exactly once on `rel_path`; unavailable pictures remain null rather than zero traffic. The checks also reconcile camera and episode totals. The study covers one week and a non-random camera sample, so district rankings and possible causes need cautious interpretation.

## Check the results

```bash
python -m unittest -v test_analysis
```

Seven tests cover the image join, missing counts, whole-week rates, shared freezing, matched traffic comparisons and duration/sample-size results. One deliberately changes failed-image vehicle counts to confirm they cannot affect the traffic result.

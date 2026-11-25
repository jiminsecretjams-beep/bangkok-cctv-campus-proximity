# School-zone commute tracker: traffic findings

Study week: 25-31 August 2026. Analysis completed 14 November 2026.

**The strongest result is a widespread weekday commute-like signature in school-nearby camera views: mornings show substantially more visible vehicles than the same weekend hours, afternoons show a smaller increase, and midday is broadly similar.** This supports the project's temporal-pattern objective. It does not identify how much traffic schools cause.

The primary comparison uses the same **89 cameras**, the same eight clock hours and the same accepted dates within each camera. Each camera contributes 3-4 complete weekdays and both weekend days. These are visible vehicle sightings per snapshot, including parked vehicles, rather than vehicles passing per hour, unique journeys or congestion delay. Percentages below are **medians of within-camera percentage differences**, not a percentage change in citywide traffic.

## 1. The weekday excess concentrates in commute-like hours

| Bangkok collection-time window | Median weekday versus weekend difference | Middle 50% of camera differences | Cameras with higher weekday counts |
|---|---:|---:|---:|
| Morning, 07:00-09:00 | **+61.2%** | +35.7% to +100.3% | **85 / 89** |
| Midday, 10:00-14:00 | **+0.9%** | -8.6% to +11.3% | 46 / 89 |
| Afternoon, 15:00-17:00 | **+28.7%** | +13.9% to +52.1% | **82 / 89** |

The morning/midday ratio is also larger on weekdays than on weekends: the median within-camera ratio-of-ratios excess is **+58.8%**. This separates a morning-specific pattern from a camera simply having higher weekday counts at every time. It is a descriptive interaction, not a causal difference-in-differences estimate.

The hour-by-hour comparison is most pronounced at 07:00-07:59, where the median weekday/weekend ratio is about **1.95**. That profile uses up to 99 matched cameras and has a less restrictive per-hour eligibility rule than the 89-camera primary table; do not mix its denominator with the primary table.

![Same-clock profile](01_same_clock_hour.png)
![Camera-level period differences](02_camera_period_distribution.png)

**Project implication:** the dashboard should compare weekday and weekend observations at the same camera and clock time. A single all-day average would hide the most relevant pattern.

Evidence: [primary table](headline_period_evidence.csv), [camera effects](camera_commute_effects.csv), [hourly profile](matched_hourly_profile.csv), [balanced daily values](balanced_camera_day_period.csv).

## 2. The direction survives several reasonable checks; the exact size varies

| Check | Morning median difference | Cameras | Interpretation |
|---|---:|---:|---|
| Primary quality and balanced-day rules | +61.2% | 89 | Main result |
| Stricter hourly coverage | +64.6% | 44 | Similar direction in a smaller, better-covered cohort |
| Include all successful raw predictions, including quality flags | +58.6% | 97 with stable morning denominators | A sensitivity check, not a recommendation to use bad images |
| Exclude audited clock-discrepancy locations | +57.6% | 85 | Pattern is not driven by those four views |
| Exclude known metadata label discrepancies | +61.3% | 87 | Similar direction |
| Exclude bicycles from the total | +61.3% | 89 | Five-class versus four-motor-class choice is immaterial here |

Leaving out one weekday at a time gives morning medians of approximately **+49.9% to +63.1%**, with cohort sizes changing from 70 to 89. Using Saturday alone gives **+50.5%** among the original 89 cameras; using Sunday alone gives **+83.1%**. The weekend baseline matters, although both days preserve the direction.

Hypothetically moving collection timestamps by +/-15 or +/-30 minutes leaves a positive morning contrast; across those variants, the median for cameras also in the primary cohort ranges from approximately **+42.6% to +73.9%**. These are stress tests, not corrections to the camera clocks.

Nearby views are not independent locations. Grouping supplied coordinates within 50 m produces 81 clusters in the full sample and 70 in the primary cohort. The median cluster-level morning contrast is **+57.4%**. Resampling these 70 locations gives a conditional 2.5th-97.5th percentile range of **+46.2% to +78.4%**. This describes location resampling within this particular week, not uncertainty about school causation or future weeks.

**Weather remains unresolved:** both the complete-day and separate-period dry-only screens lack enough matched weekdays and both weekend days. In particular, every Saturday observation at 07:00-08:59 has model rain above the 0.1 mm dry threshold. Their estimates are explicitly unavailable, not zero. The Sunday-only comparison still has a positive morning contrast, but we cannot claim the weekday contrast has been isolated from rainfall. The weather input is gridded model precipitation, not street-level rain measurements.

Evidence: [sensitivity checks](sensitivity_checks.csv), [location resampling](location_resampling.csv), [model rain coverage](model_rain_coverage.csv).

## 3. Weekday peaks often repeat on a day not used to select them

For each camera, a two-hour weekday peak is selected using Tuesday-Friday only. Monday is withheld until after selection. A candidate must have at least a +20% discovery contrast against common non-peak hours and a positive difference on at least three discovery days.

- **57 cameras** pass this discovery screen.
- All 57 have the required Monday comparison.
- **48 / 57 (84.2%)** still have a positive peak-versus-baseline difference on Monday.
- Among the 57 discovery candidates, 20 select 07:00-09:00 and 25 select 15:00-17:00. Other selected windows occur throughout the day.

The morning weekday/weekend contrast being strongest does not mean every view's absolute weekday peak is in the morning. Afternoon peaks are common too. This is a useful reason to retain camera-specific time profiles.

This is a one-day directional check, not validation of causal school effects, detector accuracy or long-term forecasting. Selection across multiple windows inflates discovery contrasts; the held-out day must not be used to switch to a different window.

Evidence: [selected peaks and Monday results](weekday_peak_holdout.csv), [all candidate/day evidence](peak_candidate_daily_evidence.csv).

## 4. Individual school-adjacent views have different signatures

These four cases were included because of existing school-context documentation, not selected as the largest observed effects. The school names identify context; a confirmed school-gate view is not established.

| Camera / context | Morning weekday excess | Midday difference | Afternoon difference |
|---|---:|---:|---:|
| 139 / Horwang | +109.5% | -8.5% | +33.2% |
| 150 / Samsen | +50.1% | -6.6% | +42.7% |
| 239 / Triam Udom | +31.3% | -28.6% | **-34.5%** |
| 1060 / Saipanya | +84.9% | +16.7% | +137.2% |

Triam Udom's view is a useful counterexample: a weekday morning increase coexists with lower weekday afternoon counts than on the weekend. Saipanya has a particularly pronounced afternoon contrast. Neither establishes school attendance as the cause; road layout, destinations, parking, retail and camera visibility can differ.

![Four school-context views](07_school_case_profiles.png)

**Project implication:** present local exceptions alongside the overall pattern. A universal statement that every school has a weekday afternoon spike would contradict the data.

Evidence: [camera effects](camera_commute_effects.csv), [daily period values](balanced_camera_day_period.csv), existing school-context and timestamp audits referenced in the dataset guide.

## 5. A documented timetable is useful context, but precise school-time claims remain sensitive

| Clock reference and 60-minute window | Earlier-hour baseline | Later-hour baseline | Accepted days |
|---|---:|---:|---:|
| Horwang, 06:40-07:40 before arrival deadline | Outside usable acquisition span | **-21.5%** | 4 |
| Samsen M1/1, 07:10-08:10 before first period | **+33.6%** | **+4.0%** | 4 |
| Samsen M1/1, 16:40-17:40 after the class end | **+14.9%** | **+21.3%** | 3 |
| Samsen M1/1 Thursday 15:00 end | Insufficient paired coverage | Insufficient paired coverage | 0 |

Horwang's broad 07:00-09:00 weekday excess does not imply that the hour immediately before its arrival deadline exceeds the following hour: it is lower in the observed collection-time data. At Samsen's first-period reference, moving the comparison from the preceding to the following hour changes the contrast from +33.6% to +4.0%. A rising morning profile can look like a timetable-specific spike if the baseline is chosen selectively.

Samsen's 16:40 case is a candidate for discussion: the mean count is higher than either adjacent hour. However, one of three days is lower than the earlier-hour baseline, the later comparison extends into 17:40-18:40 when darkness can affect detection, and shifting the anchor by -30 minutes changes the later-baseline contrast to **-1.6%**. It is not a stable school-attributable effect.

Horwang has calendar-supported term context but unknown daily attendance. Its 25-28 August competition involves a subgroup; it is not a whole-school closure. The Samsen timetable is for one class, with its exact August effective interval unverified. Camera-overlay audits also show discrepancies from collection time. No clocks have been silently corrected, and no programme/class schedule is broadcast to all students.

![School baseline comparison](06_school_baseline_sensitivity.png)

Evidence: [headline case values](school_case_headline_evidence.csv), [30/60/90-minute and shifted-anchor summaries](school_anchor_summary.csv), [daily case evidence](school_anchor_daily_sensitivity.csv), [calendar integration](../academic_calendar_20261114/README.md).

## 6. Vehicle mix adds detail: the morning pattern is not a uniform truck increase

Across the same 89 cameras, the median paired weekday-minus-weekend change in morning class counts is **+2.42 cars**, **+0.76 motorcycles**, and **+0.09 trucks per snapshot**. These are separate medians and must not be added as an accounting decomposition.

The median camera's motorcycle share is **2.45 percentage points higher** on weekday mornings and **5.76 points higher** in the afternoon. Truck share is **1.97 points lower** in the morning. A median of share changes need not sum to zero across classes.

Class-specific percentage ratios use smaller cohorts when weekend class counts are sparse: cars +60.2% (89 cameras), motorcycles +92.9% (68), and trucks +9.3% (71) in the morning. Those percentages do not use identical class cohorts. Bus and bicycle ratios are too sparse for a strong general narrative; bicycle percentage summaries with fewer than ten stable reference cameras are suppressed.

![Morning mix changes](05_morning_vehicle_mix.png)

**Project implication:** show class composition alongside total counts. The results are compatible with a commute-like car/motorcycle pattern. The model's vehicle labels do not identify passengers, trip purpose, factory origin or whether a vehicle is parked.

Evidence: [vehicle-class comparisons](vehicle_class_comparisons.csv).

## 7. Proximity alone does not establish a school effect

All 100 cameras have a catalogue school candidate within 1.5 km, including after removing six obvious yoga/language/tutoring/training entries from the candidate screen. There is no school-free group in this sample. Ninety-four cameras are also near a university/college according to the original catalogue; these categories overlap rather than defining clean comparison groups.

The analysis keeps the original catalogue and distance fields intact and adds a screened nearest-school candidate. Six records are explicitly excluded by the existing project's non-general-school name screen; eight cameras change nearest candidate. For example, cameras 411 and 412 originally pointed to Baanchao Yoga about 12-13 m away; their screened candidate is Wat Sangwet about 253-255 m away. The screen is incomplete: remaining specialist or ambiguous facilities are not automatically verified as general schools.

After excluding two known metadata-label discrepancy cases, exploratory distance bands contain 31, 27 and 29 eligible cameras. Their median morning weekday excesses are **+48.8% at 0-250 m**, **+57.6% at 250-500 m**, and **+85.6% at 500-1,500 m**. The descriptive rank correlation of distance with the morning-specific excess is approximately **+0.20**. The observed pattern is not the simple claim that closer schools always have a stronger excess.

These reference-point distances are not entrance distances; road hierarchy, view scale, parked-vehicle baseline and mapping completeness remain confounded. Road-stratified summaries are provided for inspection, without calling them causal controls or fitted risk estimates.

![Proximity context](04_proximity_context.png)

Evidence: [screened catalogue](school_type_screen.csv), [camera context](camera_context.csv), [distance bands](proximity_band_comparisons.csv), [road strata](road_class_proximity_context.csv).

## 8. Data coverage prevents a false traffic-collapse story

On **27 August, no camera-hour passes the coverage rule at 06:00, 07:00 or 08:00**. At 09:00, 99 cameras pass. Treating those excluded early hours as zero would manufacture a morning collapse and distort weekday means. The balanced primary comparison excludes that day's incomplete windows.

Out of 10,500 full camera/day/hour cells from 06:00-21:00, 10,037 pass the coverage rule. The 21:00 endpoint hour is excluded from full-hour analyses. All source images remain accounted for in the input; an excluded cell means insufficient usable data, not no traffic.

![Coverage](03_coverage.png)

Evidence: [coverage table](coverage_by_date_hour.csv), [hour cube](camera_day_hour.parquet).

## What the project can say

> During the observed week, school-nearby Bangkok camera views showed a widespread weekday morning increase in visible vehicle presence relative to matched weekend hours, a smaller afternoon increase, and little median midday difference. The pattern varied by location and remained directionally similar under several coverage and time-alignment checks. The data support descriptive commute-pattern analysis; they do not isolate traffic caused by schools.

This is an association within one week and one weekend. There are no observed public-holiday dates, and the limited calendar coverage does not support a term-time versus school-break comparison. Full-image counts include parked vehicles and multiple visible roads. Detector accuracy remains provisional; no independent accuracy percentage is invented here. More observed vehicles in a snapshot is not automatically greater flow, worse congestion or greater safety risk.

## Dashboard priorities supported by this analysis

1. A same-camera weekday/weekend time comparison, with primary sample sizes and coverage visible.
2. A map that distinguishes raw location metadata, screened school candidates and unverified gate relationships.
3. Vehicle-class shares and local exceptions, rather than a single network-wide count total.
4. School-clock references with earlier/later baselines and an explicit schedule scope/uncertainty label.
5. A coverage toggle and sensitivity panel showing when a comparison becomes unavailable.

The final dashboard, slides and video are separate deliverables. This stage supplies the analysed tables, figures and evidence for them.

## AI assistance and reproducibility

Codex assisted with Python implementation, analytical design, chart generation, consistency checks and drafting interpretations. The underlying YOLO26x detections were already completed. No source images were re-inferred, no teammate review forms were changed, and AI-generated judgments are not presented as independent human validation.

See [METHODS.md](METHODS.md), [README.md](README.md), [verification.json](verification.json) and [manifest.json](manifest.json) for assumptions, commands and provenance.

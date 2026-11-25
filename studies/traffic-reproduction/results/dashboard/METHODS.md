# Methods and interpretation contract

## Questions and scope

The project question is whether camera observations near educational institutions show useful temporal patterns around commute-like hours. The analysis uses the completed 61,319-row, 100-camera whole-image dataset with calendar integration. It does not require the roadmap's suggested narrative that a particular result must be proven. The course requirement is meaningful spatial and/or temporal analysis; a negative or mixed school-specific finding is valid.

## Measurement

`vehicle_count` is the sum of car, motorcycle, bus, truck and bicycle counts at the existing confidence threshold. `eligible_temporal` preserves the completed pipeline's quality/acquisition rules. Failed/flagged analysis counts remain null; successful zero counts remain zero. Raw predictions are used only in the explicitly named raw sensitivity variant. Whole-image counts include parked vehicles and all visible road directions or levels.

Clock hours use `observed_at_utc` converted to Asia/Bangkok. They are filename-derived collection times, not independently verified exposure timestamps. No exact gate arrival, speed, queue delay, unique trip, pedestrian conflict or driver purpose is measured.

## Hourly cube and missingness

1. Build every camera x each of seven dates x each of 15 full hours, 06:00-20:59. Keep absent cells.
2. Estimate each camera's nominal cadence from the median positive 60-1,800-second gap across the week, bounded to 120-1,200 seconds. Typical cadence is 600 seconds.
3. Count acquired and usable images separately. An hour passes if it has at least three usable images, at least 50% expected cadence coverage, an observation within its first 21 minutes, and one within its final 21 minutes.
4. Means and sums for rejected cells are null. The strict sensitivity requires at least five usable images, 80% coverage, and observations within the first/final 15 minutes.
5. Do not interpret source gaps, dark/frozen imagery or rejected coverage as zero traffic. Do not use the sparse 21:00 endpoint as a full hour.

## Primary balanced comparison

Windows are fixed before inspecting their effect sizes: morning hours 7 and 8, midday hours 10-13, and afternoon hours 15 and 16. They are general clock windows, not invented official schedules for every school.

A camera/day is accepted only if all eight required hours pass. Every period for that camera uses the same accepted days. Require at least three accepted weekdays and both Saturday and Sunday. For each period:

```
day_mean = arithmetic mean of accepted hour means in that period
weekday_mean = arithmetic mean of weekday day_means
weekend_mean = arithmetic mean of Saturday and Sunday day_means
camera_relative_pct = 100 * (weekday_mean / weekend_mean - 1)
headline = median camera_relative_pct across eligible cameras
```

Require the weekend total-vehicle mean >=1 sighting/image for a stable relative denominator; otherwise retain absolute differences and mark that ratio unavailable. Equal-day weights avoid treating five weekdays as directly comparable to the sum of two weekend days, or a highly sampled day as more important.

The accepted 89-camera cohort includes 70 cameras with four weekdays and 19 with three; all have both weekend days. The 27 August early-morning gap means no complete primary day for that date. Independent per-hour profiles instead require >=3 weekday dates and both weekend dates in each camera/hour; that less restrictive panel includes 99 cameras and varying numbers by hour. It is separate from the fixed 89-camera headline cohort.

The morning-specific contrast is `(weekday_morning / weekday_midday) / (weekend_morning / weekend_midday) - 1`, multiplied by 100, then summarized across cameras. It is not a causal difference-in-differences estimate because there is no randomized exposure, untreated group, pre-treatment period or established parallel trend.

## Sensitivity and uncertainty

- Recalculate the full comparison for stricter coverage, successful raw predictions, four motor-vehicle classes, leave-one-date-out variants and hypothetical global +/-15/30-minute timestamp shifts.
- Clock shifts are stress tests, not exposure-time corrections. Also exclude the audited discrepancy locations: 139, 150, 546 and its nearby overlapping view 1368.
- Exclude the known label-discrepancy cases 1188 and 529 in a separate sensitivity and in spatial summaries.
- Attempt dry-only comparisons using the existing preceding-hour-aligned gridded model rain <=0.1 mm. Both complete-day and period-specific attempts fail matched-day requirements and remain explicitly unavailable. Weather is not controlled away.
- Cohort sizes change with sensitivity rules. `common_primary_median_pct` is the variant's median among cameras also in the original cohort, not a new primary estimate or a fixed common cohort shared by every variant.
- Camera coordinates within 50 m are connected into location components without examining counts. Take the median effect within each component, then summarize components. Bootstrap these component values 2,000 times with seed 20261114. The percentile range is conditional spatial resampling within this week; it does not address temporal generalization, common shocks, inaccurate coordinates or model error.
- No snapshot-level independence, causal p-values or citywide confidence interval is claimed. Thousands of sequential images do not become thousands of independent school-day experiments.

## Held-out weekday check

Use Tuesday-Friday only to select a two-hour window starting 07:00-15:00. Require at least three discovery days with both hours observed, and at least five common non-peak baseline hours between 07:00 and 16:59. Baseline hours are fixed for each candidate from the discovery days. For each candidate, compare its mean peak hours to its common non-peak baseline on each day. Select the maximum average discovery percentage, breaking ties by earlier start; do not inspect Monday in selection.

A discovery candidate requires >=20% mean excess and positive excess on >=3 discovery days. Monday is evaluated on that fixed peak/baseline with the same hour requirements. Report a direction check, not an unbiased estimate of the selected discovery magnitude or a forecast. Multiple-window selection inflates discovery contrasts; one held-out Monday is limited validation. The chosen windows are weekday peaks relative to other weekday hours, not estimates of school effects.

## School schedule cases

Only the existing `documented...` school schedule references are used. Scenario-only and older/undated references cannot become official dates. Horwang retains its calendar-supported term context; Samsen records retain their class scope and unverified exact August effective interval. Daily attendance remains unknown.

Compare 30/60/90-minute windows before an arrival/first-period anchor or after a class-end anchor to both adjacent windows. Keep weekdays specified in the reference. Apply coverage rules using the nominal ten-minute cadence; require at least two observations, at least three observations per hour of interval, >=50% cadence coverage, and observations within each interval's first/final 35%. Store failed comparisons with null results. Re-run anchors at -30/-15/0/+15/+30 minutes without calling these clock corrections.

School case means are equally weighted over accepted days and reported with day counts and daily ranges. The later Samsen dismissal baseline extends after 18:00 and may be affected by light. An activity is not a whole-school closure. A class timetable does not establish a common school dismissal or gate arrival time.

## Vehicle class and spatial context

Class share is class mean / total mean for each camera/day type/period; share changes are paired before taking the median. Class-specific relative ratios require weekend class means >=0.25. Suppress general percentage summaries if fewer than ten stable class cameras remain. Separate class medians and medians of shares are not additive decompositions.

Use original 1.5 km candidate links, not an image-by-all-institutions join. An explicit existing name screen flags obvious yoga/language/tutoring/training institutions; original catalogue fields are retained. This heuristic does not verify all remaining facilities as general schools. Distances remain supplied/reference-point distances rather than verified entrances.

Spatial band summaries and rank correlations are exploratory. All 100 cameras are within the retained school-candidate screen and 94 have overlapping university/college proximity, so these are not clean school/no-school or university/school groups. Road-stratified tables do not eliminate confounding by view scale, road capacity or incomplete mapping.

## Scope boundaries

The week contains no observed public holiday, and institutional calendar coverage is limited. No holiday-versus-working-day or term-versus-break contrast is estimable. No forecasting, school-caused congestion, safety risk, freight route, factory attribution, vehicle-trip totals or population-wide Bangkok effect is inferred. Existing reviewer forms and original model data remain unchanged.

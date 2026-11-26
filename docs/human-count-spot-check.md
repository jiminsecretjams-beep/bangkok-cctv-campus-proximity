# Human vehicle count review

## Matched morning review

Teammates counted vehicles in **40 morning frames from 20 Bangkok CCTV cameras** captured from 25 to 31 August 2026. Each camera contributed one weekday and one weekend frame from the same local clock hour between 07:00 and 09:00. Reviewers received the images and a blank worksheet. The worksheet asked them to count cars, motorcycles, buses, trucks, and bicycles across the whole frame, including parked vehicles, and to record uncertain objects separately from the class totals.

| Day type | Frames | Manual vehicles per frame | YOLO vehicles per frame | YOLO minus manual |
|---|---:|---:|---:|---:|
| Weekday morning | 20 | 17.70 | 10.80 | −6.90 |
| Weekend morning | 20 | 9.70 | 6.85 | −2.85 |

Manual counts were higher on weekdays for **15 of the 20 cameras**. The mean weekday minus weekend difference was **+8.00 vehicles per frame** by manual count and **+3.95** by YOLO26x. Both showed a weekday increase in the sample, and the shortfall relative to manual counts was larger on weekdays.

Across the 40 frames, YOLO counted fewer vehicles than the reviewers on **27**, matched on **8**, and counted more on **5**. The mean absolute count difference was **5.18 vehicles per frame**.

### Sampling and checks

- We drew 20 distinct cameras from the 81 eligible cameras in the balanced cohort after excluding cameras from the earlier spot check. We used seed `20261126` and sampled eligible frames from accepted days, matching weekday and weekend frames by hour. Selection used coverage and timing criteria. [Sampling record](../data/validation/morning_review/sampling.json)
- Each submitted sample ID matched one packet image and one record in the saved YOLO26x table. We checked the image checksum, camera, date, local hour, and class totals. The comparison uses the saved whole-frame counts at the 0.25 confidence threshold. [Frame comparison](../data/validation/morning_review/comparison.csv) · [Saved model table manifest](../studies/traffic-reproduction/inputs/manifest.json)
- We used the submitted totals and preserved 17 blank uncertainty fields. Notes on `A14` and `B15` flagged tuk-tuk classification. The note on `A17` raised a possible difference in which part of the frame was counted. After setting aside the three affected pairs, the mean weekday increase was **7.41 vehicles per frame** by manual count and **3.82** by YOLO. [Pack A worksheet](../data/validation/morning_review/human_review_a.csv) · [Pack B worksheet](../data/validation/morning_review/human_review_b.csv)

The sample assesses count agreement for eligible morning frames in this study week. The full-study result of **+61.2%** is the median of the percentage differences across the 89 views with balanced coverage. Codex assisted with sample preparation, data matching, and analysis. Teammates supplied the manual counts. [Full-study methods](../dashboard/data/METHODS.md)

## Eight-frame image quality check

A teammate counted vehicles in eight CCTV frames from eight Bangkok cameras. We compared those counts with the saved YOLO26x results. Both counts cover the whole frame and include visible cars, motorcycles, buses, trucks, and bicycles, including parked vehicles.

YOLO matched the reviewer’s total on **2 of 8 frames**, counted fewer vehicles on **5**, and counted more on **1**. The mean absolute difference was **2.75 vehicles per frame**. The reviewer counted 73 vehicles in total and YOLO counted 55. The largest gap was on camera 120 in a busy scene with 27 vehicles counted by the reviewer and 17 by YOLO.

| Frame | Camera | Human count | YOLO count | YOLO − human | Used in time analysis |
|---|---:|---:|---:|---:|---|
| `daylight-1` | 1070 | 6 | 8 | +2 | Yes |
| `daylight-5` | 233 | 4 | 2 | −2 | Yes |
| `lower_light-3` | 301 | 0 | 0 | 0 | No |
| `lower_light-5` | 1028 | 2 | 1 | −1 | No |
| `lower_light-4` | 18 | 6 | 3 | −3 | No |
| `possible_wet-4` | 604 | 7 | 7 | 0 | Yes |
| `potentially_busy-1` | 1071 | 21 | 17 | −4 | Yes |
| `potentially_busy-6` | 120 | 27 | 17 | −10 | Yes |

### Method and scope

- The review covered selected daylight, dim, wet-looking, and busy scenes. The busy-frame nomination used saved model counts as one selection aid. This small, purposive sample describes disagreement in these eight frames. Estimating accuracy across the full dataset would require a representative sample.
- We used the revised worksheet after two entered totals were reconciled with their class counts. A teammate’s note marked one motorcycle in `lower_light-5` as uncertain. The project owner subsequently confirmed it, giving a reviewer total of two for that frame.
- The review images matched the frozen [CCTV batch results](https://github.com/jiminsecretjams-beep/bangkok-cctv-campus-proximity/releases) by camera, archive path, and image checksum. The comparison uses saved raw model counts, including three frames excluded from the main time analysis by its quality rules.
- Codex assisted with sample preparation, matching the returned counts to the saved predictions, and computing the comparison. A teammate supplied the human counts.

# Human vehicle count spot check

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

## Method and scope

- The review covered selected daylight, dim, wet-looking, and busy scenes. The busy-frame nomination used saved model counts as one selection aid. This small, purposive sample describes disagreement in these eight frames. Estimating accuracy across the full dataset would require a representative sample.
- We used the revised worksheet after two entered totals were reconciled with their class counts. A teammate’s note marked one motorcycle in `lower_light-5` as uncertain. The project owner subsequently confirmed it, giving a reviewer total of two for that frame.
- The review images matched the frozen [CCTV batch results](https://github.com/jiminsecretjams-beep/bangkok-cctv-campus-proximity/releases) by camera, archive path, and image checksum. The comparison uses saved raw model counts, including three frames excluded from the main time analysis by its quality rules.
- Codex assisted with sample preparation, matching the returned counts to the saved predictions, and computing the comparison. A teammate supplied the human counts.

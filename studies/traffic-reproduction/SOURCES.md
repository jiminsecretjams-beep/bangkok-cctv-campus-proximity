# Data sources and reproduction boundaries

The normal workflow starts from the completed, frozen image-count table and the curated institution/context inputs. It reconstructs the calendar joins, quantitative analysis, figures and dashboard without the original research folders or a sibling repository. It does not repeat every historical collection or human source-review decision.

## CCTV and model

- **Course CCTV**: [supplied Drive folder](https://drive.google.com/drive/folders/1k2-KxXe4wbDmFGcGXI-pK7TdSXjBHTni), [camera CSV](https://drive.google.com/file/d/1VSCzWeNOV7ds_d9BNJ9i20_9JzyncL2F/view). The user confirmed filename timestamps are Bangkok collection time. Exposure-time accuracy is not guaranteed.
- **Completed counts**: [public dataset release](https://github.com/Panyawat-Phothiphanich/bangkok-cctv-campus-proximity/releases/tag/yolo26x-full-frame-20261109), retained exactly as `inputs/counts.parquet`. SHA256: `eb77e8cd0d98e0042f7c2fb56d747066bf264f16e9b94e3be4ad265d64f32c09`.
- **Image archive SHA256**: `2a088ff18129120d2b2b81b270b3cb879328ba9ad9ccb69c8fb2b1febf88ce87`.
- **YOLO26x**: [versioned weights](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt). SHA256: `9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92`. Configuration and runtime versions are pinned in this submission. No image bytes or weights are embedded in the package.

## Institution catalogue and proximity

`inputs/campus/` preserves the earlier team's cleaned catalogue and 1.5 km links: 1,154 institutions, 100 cameras and 2,306 pairs. Its `sources`, `source_urls`, `official_url` and point-role columns retain record-level provenance.

| Provider | Collection / role | Limitation |
|---|---|---|
| [OBEC / schooladdress](https://data.go.th/en/dataset/schooladdress) | Team-supplied school coordinate CSV | Historical collection is an upstream input; this package does not rescrape OBEC. |
| BMA | Team-supplied workbook; explicit columns used to recover school records | Original workbook download URL was not supplied. It must not be invented or described as newly scraped here. |
| [OpenStreetMap / Overpass](https://overpass-api.de/api/interpreter) | API records for school, university and college locations, merged with the above | Map completeness and institutional classification vary; mapped reference points are not verified gates. |

The catalogue also includes commercial/ambiguous education records. The analysis preserves the original types and applies the documented, limited name screen in `workflow/analysis.py`; this is not a certified school census. Camera 1188/529 location-label concerns are retained. Proximity is association context, not evidence that traffic belongs to a school.

The original catalogue preparation source remains in the [archived catalogue-preparation project](https://github.com/Panyawat-Phothiphanich/bangkok-cctv-campus-proximity/tree/archive/pre-submission-20261122). This submission freezes its outputs rather than bundling the older project's separate cleaning/map toolchain. `run.py check` independently recalculates the distance links from the included camera/institution coordinates.

## Calendar collection and interpretation

The exact four reviewed source snapshots, retrieval metadata, URLs, SHA256 values and rules are included in `inputs/snapshots/` and `inputs/calendar/`.

| Source | Method | Use |
|---|---|---|
| [Thai public-holiday JSON mirror](https://raw.githubusercontent.com/ppraserts/thailand-open-data/main/data/thai-public-holidays/all.json) | HTTP JSON download; adapted from a teammate's holiday collector | 2025/2026 holiday records. A community mirror attributed to government sources, not a direct official feed. |
| [TAT holiday calendar](https://www.thailandtravel.or.jp/about/calendar/) | Saved HTML and independent date crosscheck | Negative public-holiday labels restricted to the seven crosschecked dates. |
| [Chula BBA calendar](https://bba.acc.chula.ac.th/academic-calendar) | Saved HTML; reviewed programme/date extraction | BBA International only; does not represent all Chula programmes. |
| [Horwang 2026 calendar](https://www.horwang.ac.th/temp/calendar-m69-1.pdf) | Saved PDF, text/visual review of pages 33, 43 and 45 | Term context inferred from documented opening/exam milestones; cohort activities are not whole-school closures. |

Calendar extraction includes human-reviewed rules. Reproduction applies those rules and verifies the snapshots, rather than guessing new rules from today's website. New dates or updated pages require a new review. Weekends do not automatically imply school closure; absence of an event does not prove an instructional day. No holiday or term-break contrast is available in this study week.

These collection methods and evidence should be cited in the slides. The course's source-count requirement and the original BMA collection evidence still need a final team check; three downloaded pages should not be presented as three independent provider organizations.

## Other analysis inputs

- **School schedules**: `inputs/context/school_schedules.csv` preserves 28 clock-reference definitions and their source URLs/scopes. Historical, undated and scenario schedules remain reference-only. The included Samsen M1/1 timetable image/HTML supports the narrow 2026 class case; it does not establish an all-school dismissal time. Other large handbooks are linked and hashed in the upstream registry, not bundled here.
- **Road context**: saved historical OpenStreetMap/Overpass-derived camera road classes. The historical query requested 25 August 2026 map state. Source response/query hashes are retained in `inputs/manifest.json`; the large response is omitted.
- **Rain**: `inputs/context/rainfall.parquet` keeps only image ID and precipitation from the earlier [Open-Meteo historical-weather API](https://archive-api.open-meteo.com/v1/archive) join. The original joined table hash and provider metadata are in `inputs/manifest.json`. Gridded preceding-hour model rain is not a road gauge. A matched dry-only comparison is unavailable, so this analysis does not claim rain adjustment.
- **Clock audit**: `inputs/context/camera_timestamp_audit.csv` preserves observed offsets/warnings. Clock-shift checks are sensitivity scenarios, not corrected ground truth.

Input hashes identify the exact frozen study. They establish file identity, not factual accuracy of the source. Changing inputs intentionally requires a new release and reviewed source rules.

# Bangkok CCTV: campus proximity

Local data engineering handoff for classifying the 100 supplied CCTV cameras within **1,500 metres** of schools, universities and colleges.

## Open the results

- `data/processed/cctv_campus_flags.csv`: main camera table, one row per camera.
- `reports/campus_proximity_map.html`: open directly in a browser. Search cameras, filter groups, click a camera to view its matches. The point/grid view works offline; street tiles load only when you click **Load street map** and require internet.
- `reports/quality_report.md`: results, methods and material limitations.
- `data/processed/camera_institution_matches.csv`: all within-radius camera/institution pairs.
- `data/processed/institutions_clean.csv`: cleaned reference locations.
- `data/processed/institution_provenance.csv`: original source IDs and coordinates for each retained institution.
- `data/processed/deduplication_log.csv` and `campus_components.csv`: merge decisions and faculty/building records collapsed into parent campuses.
- `data/processed/borderline_matches.csv`: pairs within 100 m either side of the cutoff, including outside-radius records.
- `reports/camera_review.csv` and `reports/finalization_review.md`: targeted review of the original 13 university/college no-matches and school matches for cameras 224/225.
- `data/processed/reference_corrections.csv`: evidence-backed type/name corrections. The additional naval nursing college comes from its official contact page; its embedded-map viewport centre is explicitly approximate.

## Rerun locally

From this project folder, using Python 3.9 or newer:

```sh
python3 scripts/campus_pipeline.py
python3 -m unittest discover -s tests -v
python3 scripts/validate_outputs.py
```

The pipeline uses the Python standard library only. It reads cached input data, regenerates tables/report/map, and does not query the internet or download CCTV images. `config.json` contains the radius and input paths. The cached map reference covers the original study extent; expanding that extent requires a new source lookup. A run records source checksums and configuration in `reports/run_manifest.json`. Regenerating outputs changes the run timestamp; substantive data stays stable for identical inputs/configuration.

Optional explicit source refresh:

```sh
python3 scripts/refresh_osm.py
python3 scripts/campus_pipeline.py
```

Refresh requires internet and uses the saved queries, preserving provenance. An expanded radius/study area requires editing and reviewing the queries before refreshing.

## Main CSV dictionary

| Field | Meaning |
|---|---|
| `camera_id` | Stable camera key. Treat as a string when joining. |
| `location*`, `latitude`, `longitude`, `image_count` | Camera fields preserved from the supplied metadata; image_count is reported, not verified against the ZIP. |
| `radius_m` | 1500 for this study; inclusive threshold on unrounded distance. |
| `near_any_education` | True if a retained institution reference point falls within the radius. |
| `near_school` | Same rule for records classified as school. |
| `near_university_college` | Same rule for university/college records; blank if supplementary coverage is unavailable. |
| `institutions_within_radius`, `schools_within_radius`, `universities_colleges_within_radius` | Counts of retained reference entities, after conservative deduplication; unresolved aliases can inflate counts. |
| `nearest_institution_*`, `nearest_distance_m` | Nearest retained reference of any type in the cached catalog. Distances are in metres, displayed to 3 decimal places, without implying coordinate accuracy at that precision. |
| `nearest_school_*`, `nearest_university_college_*` | Nearest reference for each category, even if beyond 1.5 km. The map supplement is geographically bounded, so these are not certified global nearest institutions. |
| `nearest_point_role` | `source_point_unverified`, `mapped_node`, or `bounding_box_center`. |
| `coordinate_status` | Camera-coordinate validity, not a survey-grade location verification. |
| `match_status` | `matched`, `no_match_in_catalog`, or `unknown`. |
| `reference_coverage_status` | `incomplete_public_catalog`: references cannot establish that no institution exists in an area. |
| `review_required`, `review_reason` | Outstanding reference-point/threshold checks. Current rows remain provisional pending location review. |
| `targeted_review_status`, `school_identity_review_status` | Completed targeted checks and their limitations. `not_targeted` means the row was outside the requested 15-camera review. |
| `nearest_mapped_university_boundary_*`, `within_radius_of_mapped_university_boundary` | Auxiliary mapped-footprint sensitivity check for targeted cameras. This uses a local tangent-plane approximation and does not replace the primary point-distance rule. Blank outside the review set or if unavailable. Includes colleges despite the shorter column name. |
| `university_comparison_status` | Warns about point exposure, campus-edge sensitivity, near-cutoff uncertainty and incomplete-catalog no-matches. None certifies a causal comparison group. |

CSV files use UTF-8 with BOM for Thai-text compatibility. Booleans are `True`/`False`; blank flags mean unknown. A false flag means **no match in the catalog**, not confirmed absence. Invalid camera coordinates remain in the table with unknown flags. Joining the main table to traffic observations on camera_id must preserve the observation count.

## Source decisions

The unchanged team package is under `data/raw/webscraping/`. Recover 437 BMA school records missed by the supplied importer and reuse the 159 OBEC coordinate rows. The coordinate-free OPEC enrolment file and programme-specific academic calendar remain raw supporting material; neither is silently geocoded or applied to all institutions.

The supplementary OpenStreetMap snapshot and exact queries are under `data/reference/`. Data retrieved via the VK Maps Overpass mirror after other public endpoints failed. Campus outlines are used to suppress internal faculty/building duplicates, **not** to change the point-distance rule. A fallback Wikidata query was cached during source discovery but is not used to calculate final flags.

Camera CSV was fetched from Google Drive through the connector as readable CSV. Its values are preserved; line endings are normalized to LF. The source-byte archive was not downloaded. Manifest checksums refer to the local cached files.

## Limits relevant to analysis

All flags describe straight-line distance to recorded points, not route distance, road direction, verified campus gates, class schedules or commuter counts. Source freshness/coordinate roles vary. Institution names and types from public mapping may still need corrections. Current outputs do not establish causality.

If all cameras are within 1.5 km of a school, that binary flag has no unexposed comparison group. The separate higher-education flag and continuous nearest distances may support an appropriately designed downstream analysis.

Browser preview permission was declined. JavaScript syntax and output consistency were checked, but visual rendering was not verified in a browser.

After the user requested the preview again, the tool reported a saved permission or managed policy blocking the local origin. The visual check remains unperformed. To inspect it yourself, double-click `reports/campus_proximity_map.html`. The data review was completed independently of the browser restriction.

## Join completed traffic counts to campus flags

Download the completed [traffic and campus analysis handoff](https://github.com/jiminsecretjams-beep/bangkok-cctv-campus-proximity/releases/tag/analysis-campus-traffic-20260825-20260831-v1). Extract its ZIP to get the full 61,319-row CSV/Parquet table, the 59,834-row temporal-eligible Parquet subset, camera/hourly coverage, calendar evidence, data dictionary and validation/provenance reports. These are derived results; original CCTV images, weights, environments and checkpoints are excluded. The four original processing-batch ZIPs remain in their separate GitHub Releases.

The code is in this repository; large generated analysis tables are distributed through the Release asset. To recreate the join, use the [pinned processing environment and import instructions](https://github.com/Panyawat-Phothiphanich/bangkok-cctv-campus-proximity/blob/a0128c67985bd3e6f92d1e28f8f671dfc6c8a8a3/processing/README.md) with the original combined traffic import. Downloading and using the completed joined dataset does not require rerunning image inference.

After all four whole-image processing batches are imported, build the downstream analysis handoff with the existing frozen worker environment:

```sh
bangkok-cctv-worker-batch-01/processing/.venv/bin/python scripts/build_analysis_dataset.py
bangkok-cctv-worker-batch-01/processing/.venv/bin/python -m unittest discover -s tests/analysis -v
```

The default input is the completed local combined job `whole-image-f4ba6aa4fcd2ca80`; use `--traffic-folder` for another verified import location. Outputs are under `data/analysis/campus_traffic/`, excluded from Git because they contain the full observation dataset. The builder preserves all original traffic rows, counts, nulls, time fields, quality flags and batch provenance, and enforces a many-to-one join to the 100-row campus dimension. It does not rerun inference or modify the frozen processing checkout. An identical rerun verifies existing output checksums; different inputs/code require a fresh `--output` path.

The handoff contains Parquet/CSV observations, the original temporal-eligible subset, camera coverage, hourly snapshot means/medians with missing-hour coverage, supplied calendar evidence, a dictionary and validation/provenance manifests. Camera metadata uses `camera_` names and campus reference fields use `campus_` names. No camera IP addresses are copied into the analysis table.

The capture period is 25–31 August 2026. The supplied calendar is Chula BBA International programme evidence, not a verified calendar for every camera's nearby institutions; `teaching_day` remains unknown. Weekday/weekend fields describe Bangkok calendar dates only. All cameras have a nearby recorded school, and the higher-education comparison fields retain the original boundary/threshold/catalog cautions. Snapshot counts include parked vehicles and do not establish traffic flow or campus causality. See the generated handoff README and validation report for coverage and limitations.

## Attribution

Camera data: supplied course Google Drive metadata. School data: web scraping team handoff, with source paths/years retained. OpenStreetMap data: **© OpenStreetMap contributors**, [ODbL](https://opendatacommons.org/licenses/odbl/1-0/), [copyright and attribution](https://www.openstreetmap.org/copyright). Review source terms before redistributing the combined reference.

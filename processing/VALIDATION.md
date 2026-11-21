# Handoff verification — 9 November 2026

- 16 focused tests pass in the self-contained processing package. Tests use tiny source fixtures and explicit fake detectors, not the real CCTV dataset or GPU.
- Verified disjoint frozen allocations: four batches, 25 cameras each, 61,319 source images in total.
- Verified checkpoint resume without new detector calls, whole-image class counting, zero versus null behavior, failed/incomplete result rejection, ZIP checksums, invalid paths, wrong-job rejection, duplicate/missing batch rejection, and a complete package/import round trip.
- Independently reconciled returned detection boxes with raw count columns, including a test that recreates file checksums but supplies inconsistent counts.
- Ruff passes for the included source, script and tests.
- `uv sync --frozen --dry-run --python 3.11 --offline` resolves the included lock for a fresh Apple Silicon project environment. Installation was not performed again on the coordinator's occupied computer.
- The frozen job's source/configuration/runtime fingerprints match the prepared checkout.
- No new real-image inference, GPU model run, human label generation or processed-data upload was performed to prepare this handoff. Real processing and upload permissions are exercised on each assigned worker's own laptop/account.

The detector adapter and full-image counting functions are reused from the existing local YOLO26x project. Software tests do not establish empirical counting accuracy. The handoff preserves the explicit provisional validation status and all source/quality information.

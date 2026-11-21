"""Verify the standalone data bundle and its reconciliation with frozen results."""

from pathlib import Path

import numpy as np

from dashboard_data import load_bundle, paired_periods, period_summary

data = load_bundle(Path(__file__).resolve().parent / "data")
frozen = data["frozen_periods"].set_index(["camera_id", "period"]).sort_index()
ids = data["context"].camera_id
dates = data["manifest"]["dates"]
for metric in ["vehicle", "car", "motorcycle", "truck", "bus", "bicycle"]:
    actual = paired_periods(data["daily"], ids, dates, metric).set_index(["camera_id", "period"]).sort_index()
    assert actual.index.equals(frozen.index)
    np.testing.assert_allclose(
        actual.relative_pct, frozen[metric + "_relative_pct"], rtol=1e-10, equal_nan=True
    )
    np.testing.assert_allclose(actual.weekday_mean, frozen["weekday_" + metric + "_mean"], rtol=1e-10)
    np.testing.assert_allclose(actual.weekend_mean, frozen["weekend_" + metric + "_mean"], rtol=1e-10)
summary = period_summary(paired_periods(data["daily"], ids, dates)).set_index("period")
for row in data["summary"]["primary_periods"]:
    assert np.isclose(summary.loc[row["period"], "median_pct"], row["median_relative_pct"])
assert data["frames"].image_id.is_unique and len(data["frames"]) == 61319
assert len(data["context"]) == 100 and len(data["hourly"]) == 10500
print("PASS: file hashes, 61,319 observations, 100 cameras, all six class comparisons and headline values.")

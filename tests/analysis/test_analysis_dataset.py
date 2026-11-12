import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build_analysis_dataset import hourly_summary, join_observations, nullable_boolean, parse_event_dates, read_campus_flags


def observations():
    utc = pd.Series(pd.to_datetime(["2026-08-24T23:05:00Z", "2026-08-24T23:15:00Z", "2026-08-25T00:05:00Z"]))
    local = utc.dt.tz_convert("Asia/Bangkok")
    return pd.DataFrame({
        "camera_id": ["1"] * 3, "image_id": ["a", "b", "c"],
        "observed_at_utc": utc, "observed_at_local": local,
        "date_local": local.dt.strftime("%Y-%m-%d"),
        "hour_local": local.dt.hour.astype("Int64"),
        "is_weekend": local.dt.dayofweek.ge(5).astype("boolean"),
        "quality_status": ["usable", "usable", "invalid"],
        "eligible_temporal": [True, True, False],
        "vehicle_count": pd.Series([0, 2, pd.NA], dtype="Int64"),
        "motor_vehicle_count": pd.Series([0, 2, pd.NA], dtype="Int64"),
        "raw_vehicle_count": pd.Series([0, 2, 5], dtype="Int64"),
    })


class AnalysisJoinContracts(unittest.TestCase):
    def test_duplicate_reference_keys_cannot_multiply_observations(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            join_observations(observations(), pd.DataFrame({"camera_id": ["1", "1"]}))

    def test_unmatched_camera_fails_instead_of_dropping_rows(self):
        with self.assertRaisesRegex(ValueError, "camera sets differ"):
            join_observations(observations(), pd.DataFrame({"camera_id": ["2"]}))

    def test_join_preserves_order_raw_counts_zeroes_nulls_and_eligibility(self):
        original = observations()
        result = join_observations(original, pd.DataFrame({"camera_id": ["1"], "campus_near_school": [True]}))
        pd.testing.assert_frame_equal(result[original.columns], original)
        self.assertEqual(result.vehicle_count.iloc[0], 0)
        self.assertTrue(pd.isna(result.vehicle_count.iloc[2]))
        self.assertEqual(result.raw_vehicle_count.iloc[2], 5)
        self.assertTrue(result.teaching_day.isna().all())

    def test_unknown_reference_boolean_remains_unknown(self):
        values = nullable_boolean(pd.Series(["True", "False", ""]))
        self.assertTrue(values.iloc[0])
        self.assertFalse(values.iloc[1])
        self.assertTrue(pd.isna(values.iloc[2]))
        with self.assertRaises(ValueError):
            nullable_boolean(pd.Series(["maybe"]))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "flags.csv"
            path.write_text("camera_id,radius_m,near_school,near_university_college\n1,1500,True,\n")
            dimension = read_campus_flags(path)
            self.assertTrue(pd.isna(dimension.campus_near_university_college.iloc[0]))

    def test_hourly_mean_uses_snapshots_and_does_not_zero_fill_absent_hours(self):
        hourly = hourly_summary(observations()).set_index("hour_local")
        self.assertEqual(len(hourly), 24)
        self.assertEqual(hourly.loc[6, "mean_visible_vehicle_count"], 1)
        self.assertEqual(hourly.loc[6, "source_images"], 2)
        self.assertTrue(pd.isna(hourly.loc[7, "mean_visible_vehicle_count"]))
        self.assertEqual(hourly.loc[7, "coverage_status"], "no_eligible_observations")
        self.assertEqual(hourly.loc[8, "source_images"], 0)
        self.assertTrue(pd.isna(hourly.loc[8, "mean_visible_vehicle_count"]))
        self.assertEqual(hourly.loc[8, "coverage_status"], "no_observations")

    def test_source_time_mismatch_is_rejected(self):
        frame = observations()
        frame.loc[0, "hour_local"] = 23  # UTC hour incorrectly substituted for Bangkok hour.
        with self.assertRaisesRegex(ValueError, "local hours disagree"):
            join_observations(frame, pd.DataFrame({"camera_id": ["1"]}))

    def test_calendar_ranges_preserve_month_and_year_boundaries(self):
        self.assertEqual(parse_event_dates("Monday, November 23 - Friday, December 4, 2026"), (date(2026, 11, 23), date(2026, 12, 4)))
        self.assertEqual(parse_event_dates("December 30 – January 2, 2027"), (date(2026, 12, 30), date(2027, 1, 2)))
        with self.assertRaises(ValueError):
            parse_event_dates("January 4")


if __name__ == "__main__":
    unittest.main()

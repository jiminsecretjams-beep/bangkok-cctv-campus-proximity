import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from add_public_holidays import attach_holidays, build, validate_reference
from build_analysis_dataset import ROOT


def reference():
    return pd.DataFrame({"date": ["2026-08-12"], "year": ["2026"], "buddhist_year": ["2569"],
                         "name_en": ["Mother's Day"], "name_th": ["วันแม่"], "is_substitution": ["False"]})


class PublicHolidayContracts(unittest.TestCase):
    def test_external_baseline_reaches_content_validation(self):
        with TemporaryDirectory() as directory:
            baseline = Path(directory)
            (baseline / "manifest.json").write_text('{"status": "partial"}\n')
            with self.assertRaisesRegex(ValueError, "Expected the completed baseline analysis handoff"):
                build(
                    baseline,
                    ROOT / "data/raw/webscraping/data/public_holidays.csv",
                    ROOT / "data/reference/public_holidays/source_manifest.json",
                    baseline / "output",
                )

    def test_uncovered_year_is_unknown_not_false(self):
        frame = pd.DataFrame({"date": ["2027-08-12"]})
        result = attach_holidays(frame, frame.date, reference(), {2026})
        self.assertTrue(pd.isna(result.public_holiday_listed.iloc[0]))
        self.assertTrue(pd.isna(result.public_holiday_names.iloc[0]))

    def test_multiple_holidays_on_a_date_do_not_multiply_rows(self):
        frame = pd.DataFrame({"date": ["2026-08-12", "2026-08-25"]})
        refs = pd.concat([reference(), reference().assign(name_en="Another event")], ignore_index=True)
        result = attach_holidays(frame, frame.date, refs, {2026})
        self.assertEqual(len(result), 2)
        self.assertTrue(result.public_holiday_listed.iloc[0])
        self.assertFalse(result.public_holiday_listed.iloc[1])
        self.assertIn("Another event", result.public_holiday_names.iloc[0])

    def test_bangkok_date_crosses_utc_midnight(self):
        frame = pd.DataFrame({"utc": pd.to_datetime(["2026-08-11T18:00:00Z"])})
        dates = frame.utc.dt.tz_convert("Asia/Bangkok").dt.strftime("%Y-%m-%d")
        self.assertTrue(attach_holidays(frame, dates, reference(), {2026}).public_holiday_listed.iloc[0])

    def test_original_nulls_zeroes_and_teaching_day_are_preserved(self):
        frame = pd.DataFrame({"date": ["2026-08-12", "2026-08-25"],
                              "vehicle_count": pd.Series([0, pd.NA], dtype="Int64"),
                              "teaching_day": pd.Series([pd.NA, pd.NA], dtype="boolean")})
        result = attach_holidays(frame, frame.date, reference(), {2026})
        pd.testing.assert_frame_equal(result[frame.columns], frame)

    def test_year_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, "date/year mismatch"):
            validate_reference(reference().assign(year="2025"), {"available_years": [2026], "rows": 1})

    def test_bad_substitution_value_fails(self):
        with self.assertRaisesRegex(ValueError, "boolean"):
            validate_reference(reference().assign(is_substitution="no"), {"available_years": [2026], "rows": 1})


if __name__ == "__main__":
    unittest.main()

"""Regression checks for the joins and comparisons used in the report.

Run from the repository root:
    python -m unittest -v test_analysis
"""

import unittest

from run import HERE, analyse, load_data, traffic_before_freezing


class FailureAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.slots, cls.cameras, cls.episodes = load_data(HERE / "inputs")
        cls.metrics, cls.tables = analyse(cls.slots, cls.cameras, cls.episodes)

    def test_actual_images_join_once_and_missing_is_not_zero(self):
        observed = self.slots[self.slots.rel_path.notna()]
        missing = self.slots[self.slots.rel_path.isna()]
        self.assertEqual(len(self.slots), 63129)
        self.assertEqual(len(observed), 61319)
        self.assertTrue(observed.rel_path.is_unique)
        self.assertEqual(len(missing), 1810)
        self.assertTrue(missing.vehicle_count.isna().all())

    def test_whole_week_rates_keep_the_large_outage(self):
        self.assertEqual(self.metrics["failures"], 2966)
        worst = max(self.metrics["day"], key=lambda row: row["percent"])
        self.assertEqual(worst["date"], "2026-08-27")
        self.assertEqual((worst["failed"], worst["checks"]), (1759, 8985))
        self.assertEqual(self.metrics["cameras"][0]["camera_id"], "150")
        self.assertAlmostEqual(self.metrics["cameras"][0]["percent"], 100 * 132 / 635)

    def test_pair_overlap_uses_common_observed_periods(self):
        pair = self.metrics["pair"]
        self.assertEqual({pair["camera_a"], pair["camera_b"]}, {"938", "939"})
        self.assertEqual((pair["cofreeze_bins"], pair["freeze_union_bins"]), (92, 100))
        self.assertEqual(pair["common_observed_bins"], 607)
        self.assertTrue(pair["same_subnet"])
        self.assertAlmostEqual(pair["distance_km"], 2.488987, places=5)

    def test_traffic_comparison_reproduces_the_reviewed_result(self):
        traffic = self.metrics["traffic"]
        self.assertEqual((traffic["freeze_episodes"], traffic["freeze_cameras"]), (33, 15))
        self.assertAlmostEqual(traffic["median_camera_ratio"], 1.0688659793814432)
        self.assertEqual([round(100 * v) for v in traffic["camera_bootstrap_95pct"]], [88, 135])
        self.assertTrue(self.tables["traffic_cases"].preceding_images.ge(2).all())
        self.assertTrue(self.tables["traffic_cases"].comparison_days.ge(3).all())

    def test_failed_image_counts_cannot_change_the_traffic_result(self):
        changed = self.slots.copy()
        changed.loc[changed.is_failure, "vehicle_count"] = 100000
        _, traffic = traffic_before_freezing(changed, self.episodes)
        self.assertEqual(traffic, self.metrics["traffic"])

    def test_no_usable_controls_is_not_reported_as_zero_traffic(self):
        empty = self.slots.iloc[:0].copy()
        cases, traffic = traffic_before_freezing(empty, self.episodes)
        self.assertTrue(cases.empty)
        self.assertIsNone(traffic["median_camera_ratio"])

    def test_durations_and_sample_sizes_match_the_supplied_episodes(self):
        durations = self.tables["durations"]
        self.assertEqual(durations.loc["duplicate frame", "episodes"], 65)
        self.assertEqual(durations.loc["duplicate frame", "median_min"], 29.6)
        self.assertEqual(self.metrics["longest_freeze"]["minutes"], 804)
        self.assertTrue(self.metrics["longest_freeze"]["ongoing_at_end"])
        self.assertEqual(durations.loc["test pattern", "episodes"], 3)


if __name__ == "__main__":
    unittest.main()

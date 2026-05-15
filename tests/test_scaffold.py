import unittest

from topogenesis.analysis.metrics import aggregate_summaries
from topogenesis.core.experiments import PRESETS, preset_args


class ScaffoldTests(unittest.TestCase):
    def test_smoke_preset_exists(self):
        self.assertIn("smoke", PRESETS)
        self.assertIn("--steps", preset_args("smoke"))

    def test_aggregate_summaries_marks_alive_run_stable(self):
        health = aggregate_summaries([
            {
                "population_final": 1,
                "births_total": 0,
                "viability_mean": 0.7,
                "viability_min": 0.6,
                "energy_min": 0.5,
                "membrane_min": 0.8,
            }
        ])
        self.assertTrue(health.stable)


if __name__ == "__main__":
    unittest.main()

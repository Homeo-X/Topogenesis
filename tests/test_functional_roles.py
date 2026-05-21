import unittest

from topogenesis.core.experiments import PRESETS
from topogenesis.engine import ABLATION_FLAGS, TopogenesisConfig, apply_ablations


class FunctionalRoleStandardTests(unittest.TestCase):
    def test_named_ablations_disable_expected_config_flags(self):
        config = apply_ablations(
            TopogenesisConfig(),
            ["affect", "memory", "world_model", "communication"],
        )

        self.assertFalse(config.use_affect)
        self.assertFalse(config.use_memory)
        self.assertFalse(config.use_world_model)
        self.assertFalse(config.use_communication)
        self.assertTrue(config.use_reflex)

    def test_unknown_ablation_fails_fast(self):
        with self.assertRaises(ValueError):
            apply_ablations(TopogenesisConfig(), ["decorative_label"])

    def test_ablation_presets_are_exposed(self):
        self.assertIn("ablation_no_affect", PRESETS)
        self.assertIn("ablation_no_memory", PRESETS)
        self.assertIn("ablation_reflex_only", PRESETS)

        reflex_args = PRESETS["ablation_reflex_only"].args
        self.assertIn("--ablate", reflex_args)
        self.assertIn("world_model", reflex_args)

    def test_every_ablation_name_maps_to_existing_config_flag(self):
        config = TopogenesisConfig()
        for flag_name in ABLATION_FLAGS.values():
            self.assertTrue(hasattr(config, flag_name), flag_name)


if __name__ == "__main__":
    unittest.main()

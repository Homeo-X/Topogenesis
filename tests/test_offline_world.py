import tempfile
import unittest
from pathlib import Path

from topogenesis.world import (
    OfflineConfig,
    OfflineSimulator,
    PopulationConfig,
    PopulationManager,
    WorldState,
)


class OfflineWorldTests(unittest.TestCase):
    def test_offline_sim_runs_200_npcs_without_extinction(self):
        world = WorldState.default(ticks_per_day=24, seed=7)
        manager = PopulationManager(
            PopulationConfig(initial_population=200, max_population=250),
            seed=7,
        )
        sim = OfflineSimulator(
            world=world,
            population_manager=manager,
            config=OfflineConfig(ticks_per_day=24, metrics_interval=24, seed=7),
        )

        summary = sim.run_days(10)

        self.assertTrue(summary.stable)
        self.assertGreaterEqual(summary.population_final, 180)
        self.assertGreater(summary.mean_energy_final, 0.05)

    def test_offline_sim_save_load_roundtrip(self):
        sim = OfflineSimulator(
            population_manager=PopulationManager(
                PopulationConfig(initial_population=32, max_population=64),
                seed=3,
            ),
            config=OfflineConfig(ticks_per_day=12, metrics_interval=12, seed=3),
        )
        sim.run_days(2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "offline.pkl"
            sim.save(path)
            loaded = OfflineSimulator.load(path)

        self.assertEqual(loaded.world.clock.tick, sim.world.clock.tick)
        self.assertEqual(loaded.population.size, sim.population.size)
        self.assertEqual(
            loaded.metrics.summarize().population_final,
            sim.metrics.summarize().population_final,
        )

    def test_population_exposes_viability_and_need_sample(self):
        world = WorldState.default(ticks_per_day=12, seed=1)
        manager = PopulationManager(
            PopulationConfig(initial_population=12, max_population=20),
            seed=1,
        )
        batch = manager.create_population(world)
        manager.step(world, batch)

        viability, needs = manager.sample_viability(batch, 0)

        self.assertGreaterEqual(viability.energy, 0.0)
        self.assertLessEqual(viability.energy, 1.0)
        self.assertGreaterEqual(needs.total, 0.0)
        self.assertLessEqual(needs.total, 1.0)

    def test_default_world_is_regional_and_resource_typed(self):
        world = WorldState.default(ticks_per_day=24, seed=2)

        self.assertGreaterEqual(world.carrying_capacity, 800)
        self.assertGreaterEqual(len([loc for loc in world.locations if loc.kind == "home"]), 20)
        self.assertGreaterEqual(len([loc for loc in world.locations if loc.kind == "food"]), 5)
        self.assertGreaterEqual(len([loc for loc in world.locations if loc.kind == "hazard"]), 4)

        food_locations = [loc for loc in world.locations if loc.kind == "food"]
        before = sum(loc.stock for loc in food_locations)
        positions = food_locations[0].position.reshape(1, 2)
        gained = world.consume_resource_near(positions, "food", demand=[1.0])
        after = sum(loc.stock for loc in food_locations)

        self.assertGreater(float(gained[0]), 0.0)
        self.assertLess(after, before)
        self.assertGreater(world.stock_fraction("food"), 0.0)

    def test_hazard_profile_separates_damage_and_confusion(self):
        world = WorldState.default(ticks_per_day=24, seed=4)
        ruin = next(loc for loc in world.locations if loc.name == "old_ruins")
        profile = world.hazard_profile_at(ruin.position.reshape(1, 2))

        self.assertGreater(float(profile["danger"][0]), 0.0)
        self.assertGreater(float(profile["confusion"][0]), 0.0)
        self.assertGreaterEqual(float(profile["damage"][0]), 0.0)

    def test_population_spawns_as_small_bands(self):
        world = WorldState.default(ticks_per_day=24, seed=9)
        manager = PopulationManager(
            PopulationConfig(initial_population=240, max_population=300),
            seed=9,
        )
        batch = manager.create_population(world)

        home_counts = {}
        for home in batch.home_location.tolist():
            home_counts[int(home)] = home_counts.get(int(home), 0) + 1

        self.assertGreaterEqual(len(home_counts), 20)
        self.assertLessEqual(max(home_counts.values()), 14)

    def test_snapshot_samples_across_bands(self):
        sim = OfflineSimulator(
            world=WorldState.default(ticks_per_day=24, seed=12),
            population_manager=PopulationManager(
                PopulationConfig(initial_population=240, max_population=300),
                seed=12,
            ),
            config=OfflineConfig(ticks_per_day=24, metrics_interval=24, seed=12),
        )
        sim.run_ticks(1)
        snapshot = sim.snapshot(visible_limit=48)

        homes = {npc["home_location"] for npc in snapshot["npcs"]}
        self.assertGreaterEqual(len(homes), 20)
        self.assertEqual(snapshot["world"]["visible_population"], len(snapshot["npcs"]))
        self.assertGreaterEqual(snapshot["world"]["band_count"], 20)


if __name__ == "__main__":
    unittest.main()

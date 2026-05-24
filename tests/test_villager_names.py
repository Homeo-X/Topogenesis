import unittest

from topogenesis.world.names import villager_calling, villager_display_name
from topogenesis.world.offline_sim import OfflineConfig, OfflineSimulator
from topogenesis.world.population import PopulationConfig, PopulationManager
from topogenesis.world.world_state import WorldState


class VillagerNameTests(unittest.TestCase):
    def test_names_are_stable_and_not_generic(self):
        first = villager_display_name(7, lineage=123, generation=1)
        second = villager_display_name(7, lineage=123, generation=1)
        self.assertEqual(first, second)
        self.assertNotIn("Villager", first)
        self.assertGreaterEqual(len(first.split()), 2)

    def test_offline_snapshot_includes_names_and_callings(self):
        world = WorldState.default(seed=3)
        manager = PopulationManager(PopulationConfig(initial_population=8), seed=3)
        sim = OfflineSimulator(
            world=world,
            population_manager=manager,
            config=OfflineConfig(seed=3),
        )
        snapshot = sim.snapshot(visible_limit=8)
        names = [npc["display_name"] for npc in snapshot["npcs"]]
        callings = [npc["calling"] for npc in snapshot["npcs"]]
        self.assertTrue(all("Villager" not in name for name in names))
        self.assertTrue(all(villager_calling(int(npc["id"]), int(npc["lineage"]), int(npc["generation"])) in callings for npc in snapshot["npcs"]))
        self.assertGreater(len(set(names)), 4)


if __name__ == "__main__":
    unittest.main()

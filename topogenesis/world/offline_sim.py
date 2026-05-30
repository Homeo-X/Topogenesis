from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .metrics import OfflineMetrics, OfflineSummary
from .names import villager_calling, villager_display_name
from .population import NPCBatch, PopulationConfig, PopulationManager
from .world_state import WorldState


def _round_vec2(values) -> list[float]:
    return [round(float(values[0]), 4), round(float(values[1]), 4)]


@dataclass(frozen=True)
class OfflineConfig:
    ticks_per_day: int = 60
    acceleration: float = 1.0
    metrics_interval: int = 60
    seed: int = 0


class OfflineSimulator:
    def __init__(
        self,
        world: Optional[WorldState] = None,
        population: Optional[NPCBatch] = None,
        population_manager: Optional[PopulationManager] = None,
        config: OfflineConfig = OfflineConfig(),
    ) -> None:
        self.config = config
        self.world = world or WorldState.default(
            ticks_per_day=config.ticks_per_day,
            seed=config.seed,
        )
        self.population_manager = population_manager or PopulationManager(
            PopulationConfig(),
            seed=config.seed,
        )
        self.population = population or self.population_manager.create_population(self.world)
        self.metrics = OfflineMetrics()

    def run_ticks(self, num_ticks: int) -> OfflineSummary:
        for _ in range(max(0, int(num_ticks))):
            sample = self.population_manager.step(self.world, self.population)
            if (
                self.world.clock.tick % max(1, self.config.metrics_interval) == 0
                or not self.metrics.samples
            ):
                self.metrics.record(sample)
        if not self.metrics.samples:
            self.metrics.record(self.population_manager._metrics(
                self.world,
                self.population,
                self.population.energy * 0.0,
            ))
        return self.metrics.summarize()

    def run_days(self, days: int) -> OfflineSummary:
        return self.run_ticks(days * self.world.clock.ticks_per_day)

    def snapshot(self, *, visible_limit: int = 48) -> dict:
        alive_idx = self._visible_indices_by_band(max(0, int(visible_limit)))
        locations = []
        for idx, loc in enumerate(self.world.locations):
            locations.append({
                "id": idx,
                "name": loc.name,
                "kind": loc.kind,
                "position": _round_vec2(loc.position),
                "radius": round(float(loc.radius), 4),
                "resource_rate": round(float(loc.resource_rate), 4),
                "stock": round(float(loc.stock), 4),
                "capacity": round(float(loc.capacity), 4),
                "regen_rate": round(float(loc.regen_rate), 4),
                "danger": round(float(loc.danger), 4),
                "damage_rate": round(float(loc.damage_rate), 4),
                "confusion_rate": round(float(loc.confusion_rate), 4),
                "hazard_kind": loc.hazard_kind,
                "social_pull": round(float(loc.social_pull), 4),
            })
        npcs = []
        for idx in alive_idx:
            viability, needs = self.population_manager.sample_viability(self.population, idx)
            npc_id = int(self.population.ids[idx])
            lineage = int(self.population.lineage[idx])
            generation = int(self.population.generation[idx])
            npcs.append({
                "id": npc_id,
                "display_name": villager_display_name(npc_id, lineage, generation),
                "calling": villager_calling(npc_id, lineage, generation),
                "lineage": lineage,
                "generation": generation,
                "age_days": round(float(self.population.age_days[idx]), 4),
                "position": _round_vec2(self.population.position[idx]),
                "target_location": int(self.population.target_location[idx]),
                "home_location": int(self.population.home_location[idx]),
                "target_location_name": self.world.locations[int(self.population.target_location[idx])].name,
                "home_location_name": self.world.locations[int(self.population.home_location[idx])].name,
                "energy": round(float(self.population.energy[idx]), 4),
                "hydration": round(float(self.population.hydration[idx]), 4),
                "bodily_integrity": round(float(self.population.bodily_integrity[idx]), 4),
                "prediction_coherence": round(float(self.population.prediction_coherence[idx]), 4),
                "social_stability": round(float(self.population.social_stability[idx]), 4),
                "environmental_safety": round(float(self.population.environmental_safety[idx]), 4),
                "affect_stability": round(float(self.population.affect_stability[idx]), 4),
                "need_total": round(float(needs.total), 4),
                "dominant_need": needs.dominant,
            })
        alive = self.population.alive
        living_count = int(sum(bool(v) for v in alive))
        average_affect = (
            float(self.population.affect_stability[alive].mean())
            if living_count > 0 else 0.0
        )
        if living_count > 0:
            need_values = [
                self.population_manager.sample_viability(self.population, idx)[1].total
                for idx, alive_now in enumerate(alive) if bool(alive_now)
            ]
            average_need = float(sum(need_values) / max(1, len(need_values)))
        else:
            average_need = 1.0
        return {
            "version": 1,
            "mode": "offline_world",
            "clock": {
                "tick": self.world.clock.tick,
                "day": self.world.clock.day,
                "phase": round(float(self.world.clock.phase), 4),
                "ticks_per_day": self.world.clock.ticks_per_day,
            },
            "world": {
                "half_extent": round(float(self.world.half_extent), 4),
                "carrying_capacity": int(self.world.carrying_capacity),
                "population": living_count,
                "visible_population": len(npcs),
                "band_count": len({int(self.population.home_location[idx]) for idx, v in enumerate(alive) if bool(v)}),
                "average_affect": round(average_affect, 4),
                "average_need": round(average_need, 4),
                "food_stock": self.world.stock_fraction("food"),
                "water_stock": self.world.stock_fraction("water"),
                "material_stock": self.world.stock_fraction("materials"),
                "food_ratio": self.world.stock_fraction("food"),
                "water_ratio": self.world.stock_fraction("water"),
            },
            "locations": locations,
            "npcs": npcs,
            "summary": self.metrics.summarize().__dict__,
        }

    def _visible_indices_by_band(self, visible_limit: int) -> list[int]:
        """Sample visible NPCs across home bands instead of first-id crowding."""
        if visible_limit <= 0:
            return []
        by_home: dict[int, list[int]] = {}
        for idx, alive in enumerate(self.population.alive):
            if not bool(alive):
                continue
            home = int(self.population.home_location[idx])
            by_home.setdefault(home, []).append(int(idx))
        if not by_home:
            return []
        ordered_homes = sorted(by_home)
        selected: list[int] = []
        cursor = 0
        while len(selected) < visible_limit and any(by_home.values()):
            home = ordered_homes[cursor % len(ordered_homes)]
            bucket = by_home[home]
            if bucket:
                selected.append(bucket.pop(0))
            cursor += 1
        return selected

    def save(self, path: str | Path) -> None:
        payload = {
            "config": self.config,
            "world": self.world,
            "population": self.population,
            "population_manager": self.population_manager,
            "metrics": self.metrics,
        }
        with Path(path).open("wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load(cls, path: str | Path) -> "OfflineSimulator":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        sim = cls(
            world=payload["world"],
            population=payload["population"],
            population_manager=payload["population_manager"],
            config=payload["config"],
        )
        sim.metrics = payload["metrics"]
        return sim


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run Topogenesis offline population simulation.")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--population", type=int, default=120)
    parser.add_argument("--max-population", type=int, default=300)
    parser.add_argument("--ticks-per-day", type=int, default=60)
    parser.add_argument("--metrics-interval", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", default="")
    args = parser.parse_args(argv)

    config = OfflineConfig(
        ticks_per_day=args.ticks_per_day,
        metrics_interval=args.metrics_interval,
        seed=args.seed,
    )
    world = WorldState.default(ticks_per_day=args.ticks_per_day, seed=args.seed)
    manager = PopulationManager(
        PopulationConfig(
            initial_population=args.population,
            max_population=args.max_population,
        ),
        seed=args.seed,
    )
    sim = OfflineSimulator(world=world, population_manager=manager, config=config)
    summary = sim.run_days(args.days)
    print("[topogenesis-offline] Summary " + json.dumps(summary.__dict__, sort_keys=True))
    if args.save:
        sim.save(args.save)
        print(f"[topogenesis-offline] Saved {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

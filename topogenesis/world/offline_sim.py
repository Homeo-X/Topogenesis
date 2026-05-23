from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .metrics import OfflineMetrics, OfflineSummary
from .population import NPCBatch, PopulationConfig, PopulationManager
from .world_state import WorldState


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

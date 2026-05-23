from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class OfflineSummary:
    ticks: int
    days: int
    population_final: int
    births_total: int
    deaths_total: int
    mean_energy_final: float
    mean_need_final: float
    mean_affect_final: float
    lineage_count_final: int
    food_stock_final: float = 1.0
    water_stock_final: float = 1.0
    material_stock_final: float = 1.0

    @property
    def stable(self) -> bool:
        return self.population_final > 0 and self.mean_energy_final > 0.05


@dataclass
class OfflineMetrics:
    samples: List[Dict[str, float]] = field(default_factory=list)

    def record(self, sample: Dict[str, float]) -> None:
        self.samples.append(dict(sample))

    def latest(self) -> Dict[str, float]:
        return self.samples[-1] if self.samples else {}

    def summarize(self) -> OfflineSummary:
        latest = self.latest()
        return OfflineSummary(
            ticks=int(latest.get("tick", 0.0)),
            days=int(latest.get("day", 0.0)),
            population_final=int(latest.get("population", 0.0)),
            births_total=int(latest.get("births_total", 0.0)),
            deaths_total=int(latest.get("deaths_total", 0.0)),
            mean_energy_final=float(latest.get("mean_energy", 0.0)),
            mean_need_final=float(latest.get("mean_need", 1.0)),
            mean_affect_final=float(latest.get("mean_affect_stability", 0.0)),
            lineage_count_final=int(latest.get("lineage_count", 0.0)),
            food_stock_final=float(latest.get("food_stock", 1.0)),
            water_stock_final=float(latest.get("water_stock", 1.0)),
            material_stock_final=float(latest.get("material_stock", 1.0)),
        )

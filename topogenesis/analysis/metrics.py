from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping


@dataclass
class RunHealth:
    population_final: int = 0
    births_total: int = 0
    viability_mean: float = 0.0
    viability_min: float = 0.0
    energy_min: float = 0.0
    membrane_min: float = 0.0

    @property
    def stable(self) -> bool:
        return (
            self.population_final > 0
            and self.viability_min > 0.05
            and self.energy_min >= 0.0
            and self.membrane_min >= 0.0
        )


def aggregate_summaries(summaries: Iterable[Mapping[str, object]]) -> RunHealth:
    rows = list(summaries)
    if not rows:
        return RunHealth()

    def mean_key(key: str) -> float:
        vals = [float(r.get(key, 0.0)) for r in rows]
        return sum(vals) / max(1, len(vals))

    return RunHealth(
        population_final=max(int(r.get("population_final", 0)) for r in rows),
        births_total=max(int(r.get("births_total", 0)) for r in rows),
        viability_mean=mean_key("viability_mean"),
        viability_min=min(float(r.get("viability_min", 0.0)) for r in rows),
        energy_min=min(float(r.get("energy_min", 0.0)) for r in rows),
        membrane_min=min(float(r.get("membrane_min", 0.0)) for r in rows),
    )


OEE_METRIC_CONTRACT: Dict[str, str] = {
    "population": "population size, births, deaths, lifespan, generation",
    "genetics": "genome diversity, mutation retention, lineage clusters",
    "behavior": "resource strategies, hazard avoidance, travel range, niches",
    "cognition": "prediction error, memory usefulness, symbolic stability",
    "ecology": "field diversity, resource cycling, territory persistence",
    "open_endedness": "novelty rate, complexity trend, diversity trend",
}


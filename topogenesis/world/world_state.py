from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np


@dataclass(frozen=True)
class Location:
    name: str
    kind: str
    position: np.ndarray
    radius: float = 6.0
    resource_rate: float = 0.0
    danger: float = 0.0
    social_pull: float = 0.0

    @classmethod
    def make(
        cls,
        name: str,
        kind: str,
        position: Iterable[float],
        *,
        radius: float = 6.0,
        resource_rate: float = 0.0,
        danger: float = 0.0,
        social_pull: float = 0.0,
    ) -> "Location":
        return cls(
            name=name,
            kind=kind,
            position=np.asarray(tuple(position), dtype=np.float32),
            radius=radius,
            resource_rate=resource_rate,
            danger=danger,
            social_pull=social_pull,
        )


@dataclass
class WorldClock:
    tick: int = 0
    ticks_per_day: int = 60
    day_length_hours: float = 24.0

    @property
    def day(self) -> int:
        return self.tick // max(1, self.ticks_per_day)

    @property
    def phase(self) -> float:
        return (self.tick % max(1, self.ticks_per_day)) / max(1, self.ticks_per_day)

    def advance(self, ticks: int = 1) -> None:
        self.tick += max(0, int(ticks))


@dataclass
class WorldState:
    clock: WorldClock = field(default_factory=WorldClock)
    locations: List[Location] = field(default_factory=list)
    resources: Dict[str, float] = field(default_factory=lambda: {
        "food": 1_000.0,
        "water": 1_000.0,
        "materials": 300.0,
    })
    carrying_capacity: int = 300
    rng_seed: int = 0

    @classmethod
    def default(cls, *, ticks_per_day: int = 60, seed: int = 0) -> "WorldState":
        return cls(
            clock=WorldClock(ticks_per_day=ticks_per_day),
            rng_seed=seed,
            locations=[
                Location.make("hearth", "home", (0.0, 0.0), radius=12.0, social_pull=0.25),
                Location.make("market", "social", (9.0, -4.0), radius=10.0, social_pull=0.65),
                Location.make("fields", "food", (22.0, 18.0), radius=13.0, resource_rate=3.0),
                Location.make("grove", "food", (-24.0, -18.0), radius=12.0, resource_rate=2.2),
                Location.make("river", "water", (-6.0, 26.0), radius=10.0, resource_rate=2.0),
                Location.make("fen", "hazard", (29.0, -24.0), radius=16.0, danger=0.85),
                Location.make("ruins", "hazard", (34.0, 24.0), radius=10.0, danger=0.45),
            ],
        )

    def positions_by_kind(self, kind: str) -> np.ndarray:
        positions = [loc.position for loc in self.locations if loc.kind == kind]
        if not positions:
            return np.zeros((0, 2), dtype=np.float32)
        return np.stack(positions).astype(np.float32)

    def nearest_location_index(self, position: np.ndarray, kind: Optional[str] = None) -> int:
        candidates = [
            (idx, loc) for idx, loc in enumerate(self.locations)
            if kind is None or loc.kind == kind
        ]
        if not candidates:
            return 0
        pos = np.asarray(position, dtype=np.float32)
        return min(candidates, key=lambda item: float(np.linalg.norm(item[1].position - pos)))[0]

    def location_positions(self) -> np.ndarray:
        return np.stack([loc.position for loc in self.locations]).astype(np.float32)

    def resource_pressure_at(self, positions: np.ndarray) -> np.ndarray:
        value = np.zeros(len(positions), dtype=np.float32)
        for loc in self.locations:
            if loc.resource_rate <= 0.0:
                continue
            dist = np.linalg.norm(positions - loc.position[None, :], axis=1)
            value = np.maximum(value, np.clip(1.0 - dist / loc.radius, 0.0, 1.0) * loc.resource_rate)
        return np.clip(value / 3.0, 0.0, 1.0)

    def danger_at(self, positions: np.ndarray) -> np.ndarray:
        value = np.zeros(len(positions), dtype=np.float32)
        for loc in self.locations:
            if loc.danger <= 0.0:
                continue
            dist = np.linalg.norm(positions - loc.position[None, :], axis=1)
            value = np.maximum(value, np.clip(1.0 - dist / loc.radius, 0.0, 1.0) * loc.danger)
        return np.clip(value, 0.0, 1.0)

    def social_pull_at(self, positions: np.ndarray) -> np.ndarray:
        value = np.zeros(len(positions), dtype=np.float32)
        for loc in self.locations:
            if loc.social_pull <= 0.0:
                continue
            dist = np.linalg.norm(positions - loc.position[None, :], axis=1)
            value = np.maximum(value, np.clip(1.0 - dist / loc.radius, 0.0, 1.0) * loc.social_pull)
        return np.clip(value, 0.0, 1.0)

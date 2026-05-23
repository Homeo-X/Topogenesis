from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np


@dataclass
class Location:
    name: str
    kind: str
    position: np.ndarray
    radius: float = 6.0
    resource_rate: float = 0.0
    stock: float = 0.0
    capacity: float = 0.0
    regen_rate: float = 0.0
    danger: float = 0.0
    damage_rate: float = 0.0
    confusion_rate: float = 0.0
    hazard_kind: str = ""
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
        stock: float = 0.0,
        capacity: float = 0.0,
        regen_rate: float = 0.0,
        danger: float = 0.0,
        damage_rate: float = 0.0,
        confusion_rate: float = 0.0,
        hazard_kind: str = "",
        social_pull: float = 0.0,
    ) -> "Location":
        initial_capacity = max(capacity, stock)
        return cls(
            name=name,
            kind=kind,
            position=np.asarray(tuple(position), dtype=np.float32),
            radius=radius,
            resource_rate=resource_rate,
            stock=stock,
            capacity=initial_capacity,
            regen_rate=regen_rate,
            danger=danger,
            damage_rate=damage_rate,
            confusion_rate=confusion_rate,
            hazard_kind=hazard_kind,
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
    carrying_capacity: int = 800
    half_extent: float = 160.0
    rng_seed: int = 0

    @classmethod
    def default(cls, *, ticks_per_day: int = 60, seed: int = 0) -> "WorldState":
        return cls(
            clock=WorldClock(ticks_per_day=ticks_per_day),
            rng_seed=seed,
            locations=[
                Location.make("hearth", "home", (0.0, 0.0), radius=15.0, social_pull=0.30),
                Location.make("north_hamlet", "home", (-62.0, 58.0), radius=14.0, social_pull=0.22),
                Location.make("river_hamlet", "home", (76.0, 34.0), radius=14.0, social_pull=0.20),
                Location.make("market", "social", (9.0, -4.0), radius=15.0, social_pull=0.70),
                Location.make("chapel_green", "social", (-52.0, 48.0), radius=12.0, social_pull=0.50),
                Location.make("river_mill", "social", (82.0, 30.0), radius=12.0, social_pull=0.42),
                Location.make("south_fields", "food", (35.0, 24.0), radius=22.0, resource_rate=3.0, stock=700.0, capacity=900.0, regen_rate=5.0),
                Location.make("north_barley", "food", (-72.0, 76.0), radius=20.0, resource_rate=2.6, stock=520.0, capacity=720.0, regen_rate=4.2),
                Location.make("orchard", "food", (-42.0, -64.0), radius=18.0, resource_rate=2.1, stock=360.0, capacity=520.0, regen_rate=3.4),
                Location.make("hunting_wood", "food", (96.0, -72.0), radius=28.0, resource_rate=1.5, stock=420.0, capacity=680.0, regen_rate=2.1, danger=0.22, damage_rate=0.10, hazard_kind="wildlife"),
                Location.make("mushroom_grove", "food", (-106.0, -36.0), radius=16.0, resource_rate=1.7, stock=260.0, capacity=400.0, regen_rate=2.8, danger=0.18, confusion_rate=0.22, hazard_kind="toxins"),
                Location.make("river", "water", (-6.0, 72.0), radius=18.0, resource_rate=2.5, stock=900.0, capacity=1_200.0, regen_rate=8.0),
                Location.make("east_spring", "water", (116.0, 20.0), radius=12.0, resource_rate=2.0, stock=420.0, capacity=520.0, regen_rate=4.5),
                Location.make("quarry", "materials", (-118.0, 88.0), radius=18.0, resource_rate=1.4, stock=350.0, capacity=500.0, regen_rate=0.8, danger=0.25, damage_rate=0.18, hazard_kind="rockfall"),
                Location.make("fen", "hazard", (72.0, -98.0), radius=30.0, danger=0.90, damage_rate=0.18, confusion_rate=0.30, hazard_kind="disease"),
                Location.make("old_ruins", "hazard", (112.0, 92.0), radius=22.0, danger=0.58, damage_rate=0.10, confusion_rate=0.45, hazard_kind="uncertainty"),
                Location.make("wolf_pass", "hazard", (-92.0, -112.0), radius=26.0, danger=0.72, damage_rate=0.30, confusion_rate=0.12, hazard_kind="predation"),
                Location.make("burnt_wood", "hazard", (18.0, -132.0), radius=20.0, danger=0.46, damage_rate=0.22, confusion_rate=0.20, hazard_kind="scarcity"),
            ],
        )

    def advance_ecology(self) -> None:
        for loc in self.locations:
            if loc.capacity > 0.0 and loc.regen_rate > 0.0:
                loc.stock = min(loc.capacity, loc.stock + loc.regen_rate / max(1, self.clock.ticks_per_day))

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

    def best_resource_location_index(self, position: np.ndarray, kind: str = "food") -> int:
        candidates = [
            (idx, loc) for idx, loc in enumerate(self.locations)
            if loc.kind == kind and loc.resource_rate > 0.0 and loc.stock > 0.0
        ]
        if not candidates:
            return self.nearest_location_index(position, kind)
        pos = np.asarray(position, dtype=np.float32)
        return max(candidates, key=lambda item: (
            item[1].resource_rate * (item[1].stock / max(1.0, item[1].capacity))
            - 0.01 * float(np.linalg.norm(item[1].position - pos))
            - item[1].danger
        ))[0]

    def location_positions(self) -> np.ndarray:
        return np.stack([loc.position for loc in self.locations]).astype(np.float32)

    def resource_pressure_at(self, positions: np.ndarray, kind: Optional[str] = None) -> np.ndarray:
        value = np.zeros(len(positions), dtype=np.float32)
        for loc in self.locations:
            if kind is not None and loc.kind != kind:
                continue
            if loc.resource_rate <= 0.0:
                continue
            stock_factor = 1.0 if loc.capacity <= 0.0 else np.clip(loc.stock / loc.capacity, 0.0, 1.0)
            dist = np.linalg.norm(positions - loc.position[None, :], axis=1)
            value = np.maximum(value, np.clip(1.0 - dist / loc.radius, 0.0, 1.0) * loc.resource_rate * stock_factor)
        return np.clip(value / 3.0, 0.0, 1.0)

    def consume_resource_near(self, positions: np.ndarray, kind: str, demand: np.ndarray) -> np.ndarray:
        demand = np.asarray(demand, dtype=np.float32)
        gained = np.zeros(len(positions), dtype=np.float32)
        for loc in self.locations:
            if loc.kind != kind or loc.stock <= 0.0 or loc.resource_rate <= 0.0:
                continue
            dist = np.linalg.norm(positions - loc.position[None, :], axis=1)
            access = np.clip(1.0 - dist / loc.radius, 0.0, 1.0)
            requested = demand * access
            total_requested = float(np.sum(requested))
            if total_requested <= 1e-8:
                continue
            available = min(loc.stock, total_requested)
            scale = available / total_requested
            actual = requested * scale
            loc.stock = max(0.0, loc.stock - float(np.sum(actual)))
            gained += actual
        return gained

    def danger_at(self, positions: np.ndarray) -> np.ndarray:
        value = np.zeros(len(positions), dtype=np.float32)
        for loc in self.locations:
            if loc.danger <= 0.0:
                continue
            dist = np.linalg.norm(positions - loc.position[None, :], axis=1)
            value = np.maximum(value, np.clip(1.0 - dist / loc.radius, 0.0, 1.0) * loc.danger)
        return np.clip(value, 0.0, 1.0)

    def hazard_profile_at(self, positions: np.ndarray) -> Dict[str, np.ndarray]:
        danger = np.zeros(len(positions), dtype=np.float32)
        damage = np.zeros(len(positions), dtype=np.float32)
        confusion = np.zeros(len(positions), dtype=np.float32)
        for loc in self.locations:
            if loc.danger <= 0.0:
                continue
            dist = np.linalg.norm(positions - loc.position[None, :], axis=1)
            exposure = np.clip(1.0 - dist / loc.radius, 0.0, 1.0) * loc.danger
            danger = np.maximum(danger, exposure)
            damage = np.maximum(damage, exposure * loc.damage_rate)
            confusion = np.maximum(confusion, exposure * loc.confusion_rate)
        return {
            "danger": np.clip(danger, 0.0, 1.0),
            "damage": np.clip(damage, 0.0, 1.0),
            "confusion": np.clip(confusion, 0.0, 1.0),
        }

    def social_pull_at(self, positions: np.ndarray) -> np.ndarray:
        value = np.zeros(len(positions), dtype=np.float32)
        for loc in self.locations:
            if loc.social_pull <= 0.0:
                continue
            dist = np.linalg.norm(positions - loc.position[None, :], axis=1)
            value = np.maximum(value, np.clip(1.0 - dist / loc.radius, 0.0, 1.0) * loc.social_pull)
        return np.clip(value, 0.0, 1.0)

    def stock_fraction(self, kind: str) -> float:
        caps = [loc.capacity for loc in self.locations if loc.kind == kind and loc.capacity > 0.0]
        if not caps:
            return 1.0
        stock = sum(loc.stock for loc in self.locations if loc.kind == kind and loc.capacity > 0.0)
        return float(np.clip(stock / max(1.0, sum(caps)), 0.0, 1.0))

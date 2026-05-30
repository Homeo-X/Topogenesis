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
    half_extent: float = 220.0
    rng_seed: int = 0

    @classmethod
    def default(cls, *, ticks_per_day: int = 60, seed: int = 0) -> "WorldState":
        return cls(
            clock=WorldClock(ticks_per_day=ticks_per_day),
            rng_seed=seed,
            locations=[
                # Small-band home sites. These replace the single-village pull
                # with several local hearths for early hominin style groups.
                Location.make("ember_camp", "home", (0.0, 0.0), radius=12.0, social_pull=0.26),
                Location.make("river_cave", "home", (-86.0, 62.0), radius=12.0, social_pull=0.25),
                Location.make("ridge_shelter", "home", (92.0, 58.0), radius=12.0, social_pull=0.24),
                Location.make("birch_camp", "home", (-132.0, -24.0), radius=12.0, social_pull=0.23),
                Location.make("southern_hearth", "home", (42.0, -116.0), radius=12.0, social_pull=0.24),
                Location.make("mammoth_bluff", "home", (154.0, -76.0), radius=12.0, social_pull=0.22),
                Location.make("stone_ring", "home", (-42.0, -148.0), radius=12.0, social_pull=0.22),
                Location.make("willow_bank", "home", (142.0, 20.0), radius=12.0, social_pull=0.23),
                Location.make("flint_hollow", "home", (-160.0, 112.0), radius=12.0, social_pull=0.22),
                Location.make("red_cliff_camp", "home", (8.0, 152.0), radius=12.0, social_pull=0.24),
                Location.make("ash_grove_camp", "home", (176.0, 132.0), radius=12.0, social_pull=0.21),
                Location.make("fen_edge_camp", "home", (-116.0, -132.0), radius=12.0, social_pull=0.20),
                Location.make("sunken_cave", "home", (-204.0, -18.0), radius=11.0, social_pull=0.20),
                Location.make("eagle_ledge", "home", (206.0, -12.0), radius=11.0, social_pull=0.20),
                Location.make("fern_shelter", "home", (-72.0, 176.0), radius=11.0, social_pull=0.21),
                Location.make("bone_hearth", "home", (82.0, 188.0), radius=11.0, social_pull=0.21),
                Location.make("low_marsh_camp", "home", (-196.0, -176.0), radius=11.0, social_pull=0.18),
                Location.make("dry_creek_camp", "home", (196.0, -174.0), radius=11.0, social_pull=0.18),
                Location.make("mist_cave", "home", (-28.0, 206.0), radius=11.0, social_pull=0.20),
                Location.make("split_oak_camp", "home", (188.0, 72.0), radius=11.0, social_pull=0.20),
                Location.make("lichen_shelter", "home", (-188.0, 68.0), radius=11.0, social_pull=0.20),
                Location.make("ravine_hearth", "home", (22.0, -204.0), radius=11.0, social_pull=0.19),
                Location.make("deer_track_camp", "home", (116.0, -196.0), radius=11.0, social_pull=0.19),
                Location.make("amber_bank", "home", (-102.0, 202.0), radius=11.0, social_pull=0.20),

                # Local gathering sites. Each band can seek its nearest social
                # place instead of all agents converging on one market.
                Location.make("ember_story_circle", "social", (9.0, -4.0), radius=11.0, social_pull=0.68),
                Location.make("river_story_circle", "social", (-78.0, 72.0), radius=10.0, social_pull=0.62),
                Location.make("ridge_story_circle", "social", (82.0, 70.0), radius=10.0, social_pull=0.60),
                Location.make("birch_story_circle", "social", (-140.0, -10.0), radius=10.0, social_pull=0.58),
                Location.make("southern_story_circle", "social", (52.0, -106.0), radius=10.0, social_pull=0.58),
                Location.make("bluff_story_circle", "social", (144.0, -66.0), radius=10.0, social_pull=0.56),
                Location.make("stone_story_circle", "social", (-34.0, -138.0), radius=10.0, social_pull=0.56),
                Location.make("willow_story_circle", "social", (132.0, 32.0), radius=10.0, social_pull=0.56),
                Location.make("ledge_story_circle", "social", (198.0, -2.0), radius=9.0, social_pull=0.52),
                Location.make("fern_story_circle", "social", (-66.0, 166.0), radius=9.0, social_pull=0.54),
                Location.make("bone_story_circle", "social", (74.0, 178.0), radius=9.0, social_pull=0.54),
                Location.make("marsh_story_circle", "social", (-186.0, -166.0), radius=9.0, social_pull=0.48),
                Location.make("creek_story_circle", "social", (186.0, -164.0), radius=9.0, social_pull=0.48),
                Location.make("mist_story_circle", "social", (-18.0, 196.0), radius=9.0, social_pull=0.52),
                Location.make("ravine_story_circle", "social", (30.0, -194.0), radius=9.0, social_pull=0.50),
                Location.make("amber_story_circle", "social", (-96.0, 192.0), radius=9.0, social_pull=0.52),

                # Food is regionally distributed so bands forage locally unless
                # pressure becomes severe enough to travel farther.
                Location.make("berry_thicket", "food", (35.0, 24.0), radius=20.0, resource_rate=2.6, stock=620.0, capacity=760.0, regen_rate=4.8),
                Location.make("north_nut_grove", "food", (-92.0, 88.0), radius=19.0, resource_rate=2.4, stock=520.0, capacity=680.0, regen_rate=4.1),
                Location.make("marrow_field", "food", (96.0, -72.0), radius=26.0, resource_rate=1.7, stock=460.0, capacity=680.0, regen_rate=2.2, danger=0.18, damage_rate=0.08, hazard_kind="large_game"),
                Location.make("mushroom_grove", "food", (-148.0, -42.0), radius=16.0, resource_rate=1.6, stock=300.0, capacity=430.0, regen_rate=2.8, danger=0.16, confusion_rate=0.20, hazard_kind="toxins"),
                Location.make("root_basin", "food", (-38.0, -166.0), radius=18.0, resource_rate=2.0, stock=420.0, capacity=580.0, regen_rate=3.3),
                Location.make("reed_eggs", "food", (150.0, 42.0), radius=17.0, resource_rate=2.0, stock=390.0, capacity=540.0, regen_rate=3.4, danger=0.12, damage_rate=0.04, hazard_kind="snakes"),
                Location.make("highland_hunt", "food", (172.0, 142.0), radius=24.0, resource_rate=1.5, stock=420.0, capacity=640.0, regen_rate=1.9, danger=0.24, damage_rate=0.12, hazard_kind="wildlife"),
                Location.make("fen_roots", "food", (-124.0, -150.0), radius=18.0, resource_rate=1.8, stock=340.0, capacity=500.0, regen_rate=3.0, danger=0.24, confusion_rate=0.18, hazard_kind="bog"),

                Location.make("north_river", "water", (-6.0, 72.0), radius=18.0, resource_rate=2.6, stock=900.0, capacity=1_200.0, regen_rate=8.0),
                Location.make("east_spring", "water", (116.0, 20.0), radius=12.0, resource_rate=2.0, stock=420.0, capacity=520.0, regen_rate=4.5),
                Location.make("south_spring", "water", (30.0, -134.0), radius=12.0, resource_rate=2.0, stock=420.0, capacity=520.0, regen_rate=4.5),
                Location.make("west_creek", "water", (-154.0, -70.0), radius=13.0, resource_rate=1.9, stock=390.0, capacity=520.0, regen_rate=4.1),
                Location.make("cliff_drip", "water", (166.0, 108.0), radius=10.0, resource_rate=1.6, stock=260.0, capacity=360.0, regen_rate=3.0),

                Location.make("flint_knap_site", "materials", (-118.0, 88.0), radius=16.0, resource_rate=1.4, stock=350.0, capacity=500.0, regen_rate=0.8, danger=0.18, damage_rate=0.12, hazard_kind="rockfall"),
                Location.make("clay_bank", "materials", (128.0, -20.0), radius=16.0, resource_rate=1.2, stock=300.0, capacity=450.0, regen_rate=0.9),
                Location.make("bone_pile", "materials", (66.0, -150.0), radius=13.0, resource_rate=1.0, stock=210.0, capacity=320.0, regen_rate=0.6, danger=0.22, damage_rate=0.10, hazard_kind="scavengers"),

                Location.make("fen", "hazard", (72.0, -98.0), radius=30.0, danger=0.82, damage_rate=0.16, confusion_rate=0.30, hazard_kind="disease"),
                Location.make("old_ruins", "hazard", (112.0, 92.0), radius=22.0, danger=0.54, damage_rate=0.08, confusion_rate=0.45, hazard_kind="uncertainty"),
                Location.make("wolf_pass", "hazard", (-92.0, -112.0), radius=26.0, danger=0.68, damage_rate=0.26, confusion_rate=0.12, hazard_kind="predation"),
                Location.make("burnt_wood", "hazard", (18.0, -132.0), radius=20.0, danger=0.44, damage_rate=0.20, confusion_rate=0.20, hazard_kind="scarcity"),
                Location.make("sinkhole_field", "hazard", (-174.0, 28.0), radius=22.0, danger=0.48, damage_rate=0.22, confusion_rate=0.08, hazard_kind="terrain"),
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

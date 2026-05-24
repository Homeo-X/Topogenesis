from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from topogenesis.npc import NeedPressure, ViabilityState
from .world_state import WorldState


@dataclass(frozen=True)
class PopulationConfig:
    initial_population: int = 120
    max_population: int = 300
    active_fraction: float = 0.15
    active_update_interval: int = 1
    background_update_interval: int = 8
    movement_rate: float = 0.28
    energy_decay: float = 0.00055
    water_decay: float = 0.00035
    social_decay: float = 0.002
    reproduction_min_age_days: float = 18.0 * 365.0
    reproduction_energy: float = 0.78
    reproduction_bond: float = 0.58
    reproduction_probability: float = 0.00004
    old_age_days: float = 72.0 * 365.0


@dataclass
class NPCBatch:
    ids: np.ndarray
    lineage: np.ndarray
    generation: np.ndarray
    age_days: np.ndarray
    position: np.ndarray
    energy: np.ndarray
    bodily_integrity: np.ndarray
    memory_integrity: np.ndarray
    prediction_coherence: np.ndarray
    social_stability: np.ndarray
    attachment_integrity: np.ndarray
    environmental_safety: np.ndarray
    hydration: np.ndarray
    repair_material: np.ndarray
    affect_stability: np.ndarray
    social_bond: np.ndarray
    home_location: np.ndarray
    target_location: np.ndarray
    alive: np.ndarray

    @property
    def size(self) -> int:
        return int(self.ids.shape[0])

    @classmethod
    def spawn(cls, count: int, world: WorldState, *, seed: int = 0) -> "NPCBatch":
        rng = np.random.default_rng(seed)
        loc_positions = world.location_positions()
        home_indices = np.asarray([
            idx for idx, loc in enumerate(world.locations) if loc.kind == "home"
        ], dtype=np.int32)
        home_idx = rng.choice(home_indices if len(home_indices) else np.asarray([0]), size=count).astype(np.int32)
        jitter = rng.normal(0.0, 7.0, size=(count, 2)).astype(np.float32)
        position = loc_positions[home_idx] + jitter
        return cls(
            ids=np.arange(count, dtype=np.int32),
            lineage=rng.integers(0, 1_000_000, size=count, dtype=np.int32),
            generation=np.zeros(count, dtype=np.int32),
            age_days=rng.uniform(14.0 * 365.0, 45.0 * 365.0, size=count).astype(np.float32),
            position=position.astype(np.float32),
            energy=rng.uniform(0.55, 0.95, size=count).astype(np.float32),
            bodily_integrity=rng.uniform(0.80, 1.0, size=count).astype(np.float32),
            memory_integrity=rng.uniform(0.70, 1.0, size=count).astype(np.float32),
            prediction_coherence=rng.uniform(0.60, 0.95, size=count).astype(np.float32),
            social_stability=rng.uniform(0.45, 0.85, size=count).astype(np.float32),
            attachment_integrity=rng.uniform(0.45, 0.85, size=count).astype(np.float32),
            environmental_safety=np.ones(count, dtype=np.float32) * 0.75,
            hydration=rng.uniform(0.60, 1.0, size=count).astype(np.float32),
            repair_material=rng.uniform(0.20, 0.70, size=count).astype(np.float32),
            affect_stability=rng.uniform(0.55, 0.85, size=count).astype(np.float32),
            social_bond=rng.uniform(0.15, 0.55, size=count).astype(np.float32),
            home_location=home_idx,
            target_location=home_idx.copy(),
            alive=np.ones(count, dtype=bool),
        )

    def append_child(self, parent_idx: int, child_id: int, world: WorldState, rng: np.random.Generator) -> None:
        self.ids = np.append(self.ids, np.int32(child_id))
        self.lineage = np.append(self.lineage, self.lineage[parent_idx])
        self.generation = np.append(self.generation, self.generation[parent_idx] + 1)
        self.age_days = np.append(self.age_days, np.float32(0.0))
        self.position = np.vstack([
            self.position,
            self.position[parent_idx] + rng.normal(0.0, 1.2, size=2).astype(np.float32),
        ])
        self.energy = np.append(self.energy, np.float32(0.72))
        self.bodily_integrity = np.append(self.bodily_integrity, np.float32(0.95))
        self.memory_integrity = np.append(self.memory_integrity, np.float32(0.72))
        self.prediction_coherence = np.append(self.prediction_coherence, np.float32(0.70))
        self.social_stability = np.append(self.social_stability, np.float32(0.65))
        self.attachment_integrity = np.append(self.attachment_integrity, np.float32(0.70))
        self.environmental_safety = np.append(self.environmental_safety, np.float32(0.75))
        self.hydration = np.append(self.hydration, np.float32(0.82))
        self.repair_material = np.append(self.repair_material, np.float32(0.20))
        self.affect_stability = np.append(self.affect_stability, np.float32(0.68))
        self.social_bond = np.append(self.social_bond, np.float32(0.25))
        self.home_location = np.append(self.home_location, self.home_location[parent_idx])
        self.target_location = np.append(self.target_location, self.home_location[parent_idx])
        self.alive = np.append(self.alive, True)


@jax.jit
def _batch_need_total(
    energy: jnp.ndarray,
    bodily_integrity: jnp.ndarray,
    memory_integrity: jnp.ndarray,
    prediction_coherence: jnp.ndarray,
    social_stability: jnp.ndarray,
    attachment_integrity: jnp.ndarray,
    environmental_safety: jnp.ndarray,
) -> jnp.ndarray:
    return (
        0.18 * (1.0 - energy)
        + 0.18 * (1.0 - bodily_integrity)
        + 0.12 * (1.0 - memory_integrity)
        + 0.16 * (1.0 - prediction_coherence)
        + 0.11 * (1.0 - social_stability)
        + 0.10 * (1.0 - attachment_integrity)
        + 0.15 * (1.0 - environmental_safety)
    )


class PopulationManager:
    def __init__(self, config: PopulationConfig = PopulationConfig(), *, seed: int = 0) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.births_total = 0
        self.deaths_total = 0
        self.events: list[dict[str, object]] = []

    def create_population(self, world: WorldState) -> NPCBatch:
        return NPCBatch.spawn(self.config.initial_population, world, seed=world.rng_seed)

    def step(self, world: WorldState, batch: NPCBatch) -> Dict[str, float]:
        cfg = self.config
        alive = batch.alive
        if not np.any(alive):
            world.clock.advance()
            return self._metrics(world, batch, np.zeros(batch.size, dtype=np.float32))

        world.advance_ecology()
        food = world.resource_pressure_at(batch.position, "food")
        water = world.resource_pressure_at(batch.position, "water")
        materials = world.resource_pressure_at(batch.position, "materials")
        hazard = world.hazard_profile_at(batch.position)
        danger = hazard["danger"]
        damage = hazard["damage"]
        confusion = hazard["confusion"]
        social_pull = world.social_pull_at(batch.position)
        day_step = 1.0 / max(1, world.clock.ticks_per_day)

        batch.age_days[alive] += day_step
        food_gain = world.consume_resource_near(
            batch.position, "food", np.where(alive, 0.030 * (1.0 - batch.energy), 0.0))
        water_gain = world.consume_resource_near(
            batch.position, "water", np.where(alive, 0.025 * (1.0 - batch.hydration), 0.0))
        material_gain = world.consume_resource_near(
            batch.position, "materials", np.where(alive, 0.010 * (1.0 - batch.repair_material), 0.0))
        batch.energy[alive] = np.clip(
            batch.energy[alive] - cfg.energy_decay + 0.018 * food[alive] + food_gain[alive],
            0.0, 1.0,
        )
        batch.hydration[alive] = np.clip(
            batch.hydration[alive] - cfg.water_decay + 0.014 * water[alive] + water_gain[alive],
            0.0, 1.0,
        )
        home_reserve = np.clip(0.00045 * world.stock_fraction("food"), 0.0, 0.00045)
        water_reserve = np.clip(0.00030 * world.stock_fraction("water"), 0.0, 0.00030)
        reserve_mask = alive & (batch.environmental_safety > 0.55)
        batch.energy[reserve_mask] = np.clip(batch.energy[reserve_mask] + home_reserve, 0.0, 1.0)
        batch.hydration[reserve_mask] = np.clip(batch.hydration[reserve_mask] + water_reserve, 0.0, 1.0)
        batch.repair_material[alive] = np.clip(
            batch.repair_material[alive] + 0.010 * materials[alive] + material_gain[alive],
            0.0, 1.0,
        )
        batch.environmental_safety[alive] = np.clip(1.0 - danger[alive], 0.0, 1.0)
        batch.social_stability[alive] = np.clip(
            batch.social_stability[alive] * (1.0 - cfg.social_decay)
            + social_pull[alive] * 0.015
            + batch.social_bond[alive] * 0.004,
            0.0, 1.0,
        )
        batch.prediction_coherence[alive] = np.clip(
            batch.prediction_coherence[alive] * 0.998
            + (1.0 - danger[alive]) * 0.002
            - confusion[alive] * 0.010,
            0.0, 1.0,
        )
        batch.bodily_integrity[alive] = np.clip(
            batch.bodily_integrity[alive]
            - 0.010 * damage[alive]
            + 0.003 * batch.repair_material[alive],
            0.0, 1.0,
        )

        need_total = np.asarray(_batch_need_total(
            jnp.asarray(batch.energy),
            jnp.asarray(batch.bodily_integrity),
            jnp.asarray(np.minimum(batch.memory_integrity, batch.hydration)),
            jnp.asarray(batch.prediction_coherence),
            jnp.asarray(batch.social_stability),
            jnp.asarray(batch.attachment_integrity),
            jnp.asarray(batch.environmental_safety),
        ))
        batch.affect_stability[alive] = np.clip(
            0.96 * batch.affect_stability[alive] + 0.04 * (1.0 - need_total[alive]),
            0.0, 1.0,
        )

        self._choose_targets(world, batch, need_total)
        self._move_toward_targets(world, batch)
        self._handle_social_resonance(batch)
        self._handle_births_and_deaths(world, batch)
        if need_total.shape[0] != batch.size:
            need_total = np.asarray(_batch_need_total(
                jnp.asarray(batch.energy),
                jnp.asarray(batch.bodily_integrity),
                jnp.asarray(np.minimum(batch.memory_integrity, batch.hydration)),
                jnp.asarray(batch.prediction_coherence),
                jnp.asarray(batch.social_stability),
                jnp.asarray(batch.attachment_integrity),
                jnp.asarray(batch.environmental_safety),
            ))
        world.clock.advance()
        return self._metrics(world, batch, need_total)

    def _choose_targets(self, world: WorldState, batch: NPCBatch, need_total: np.ndarray) -> None:
        material_idx = world.nearest_location_index(np.zeros(2, dtype=np.float32), "materials")
        social_idx = world.nearest_location_index(np.zeros(2, dtype=np.float32), "social")
        home_idx = world.nearest_location_index(np.zeros(2, dtype=np.float32), "home")
        hunger_pressure = 1.0 - batch.energy
        thirst_pressure = 1.0 - batch.hydration
        hungry = batch.energy < 0.70
        thirsty = batch.hydration < 0.68
        critical_hunger = batch.energy < 0.28
        critical_thirst = batch.hydration < 0.38
        damaged = (batch.bodily_integrity < 0.72) | (batch.repair_material < 0.25)
        unsafe = batch.environmental_safety < 0.45
        isolated = batch.social_stability < 0.42
        water_priority = thirsty & (critical_thirst | (thirst_pressure > hunger_pressure + 0.08))
        food_priority = hungry & ~water_priority
        food_priority = food_priority | (critical_hunger & ~critical_thirst)
        for idx in np.flatnonzero(food_priority):
            batch.target_location[idx] = world.best_resource_location_index(batch.position[idx], "food")
        for idx in np.flatnonzero(water_priority):
            batch.target_location[idx] = world.best_resource_location_index(batch.position[idx], "water")
        urgent_viability = food_priority | water_priority
        batch.target_location = np.where(damaged & ~urgent_viability, material_idx, batch.target_location)
        batch.target_location = np.where(isolated & ~(urgent_viability | damaged), social_idx, batch.target_location)
        batch.target_location = np.where(unsafe, home_idx, batch.target_location)
        calm = (need_total < 0.22) & (self.rng.random(batch.size) < 0.02)
        batch.target_location = np.where(calm, batch.home_location, batch.target_location)

    def _move_toward_targets(self, world: WorldState, batch: NPCBatch) -> None:
        targets = world.location_positions()[batch.target_location]
        delta = targets - batch.position
        distance = np.linalg.norm(delta, axis=1, keepdims=True) + 1e-6
        direction = delta / distance
        speed = self.config.movement_rate * np.clip(batch.energy[:, None] + 0.15, 0.15, 1.0)
        batch.position = batch.position + direction * speed * batch.alive[:, None]

    def _handle_social_resonance(self, batch: NPCBatch) -> None:
        if batch.size < 2:
            return
        same_target = batch.target_location[:, None] == batch.target_location[None, :]
        close = np.linalg.norm(batch.position[:, None, :] - batch.position[None, :, :], axis=2) < 3.0
        contacts = np.maximum(0, np.sum(same_target & close & batch.alive[:, None], axis=1) - 1)
        batch.social_bond = np.clip(batch.social_bond + contacts * 0.0008, 0.0, 1.0)
        batch.attachment_integrity = np.clip(batch.attachment_integrity + contacts * 0.0004, 0.0, 1.0)

    def _handle_births_and_deaths(self, world: WorldState, batch: NPCBatch) -> None:
        alive_before = batch.alive.copy()
        old_age_pressure = np.clip((batch.age_days - self.config.old_age_days) / 3650.0, 0.0, 1.0)
        death_random = self.rng.random(batch.size)
        dying = (
            (batch.energy <= 0.02)
            | (batch.hydration <= 0.02)
            | (batch.bodily_integrity <= 0.03)
            | (death_random < old_age_pressure * 0.0005)
        ) & batch.alive
        batch.alive[dying] = False
        self.deaths_total += int(np.sum(alive_before & ~batch.alive))

        if batch.size >= min(self.config.max_population, world.carrying_capacity):
            return
        eligible = np.flatnonzero(
            batch.alive
            & (batch.age_days >= self.config.reproduction_min_age_days)
            & (batch.energy >= self.config.reproduction_energy)
            & (batch.hydration >= 0.70)
            & (batch.social_bond >= self.config.reproduction_bond)
        )
        for idx in eligible:
            if batch.size >= min(self.config.max_population, world.carrying_capacity):
                break
            if self.rng.random() > self.config.reproduction_probability:
                continue
            child_id = int(np.max(batch.ids)) + 1
            batch.append_child(int(idx), child_id, world, self.rng)
            batch.energy[idx] = max(0.15, batch.energy[idx] - 0.18)
            self.births_total += 1
            self.events.append({
                "tick": world.clock.tick,
                "kind": "birth",
                "parent": int(batch.ids[idx]),
                "child": child_id,
            })

    def _metrics(self, world: WorldState, batch: NPCBatch, need_total: np.ndarray) -> Dict[str, float]:
        alive = batch.alive
        population = int(np.sum(alive))
        if population == 0:
            return {
                "tick": float(world.clock.tick),
                "day": float(world.clock.day),
                "population": 0.0,
                "births_total": float(self.births_total),
                "deaths_total": float(self.deaths_total),
                "mean_need": 1.0,
                "mean_energy": 0.0,
                "mean_hydration": 0.0,
                "mean_affect_stability": 0.0,
                "food_stock": world.stock_fraction("food"),
                "water_stock": world.stock_fraction("water"),
                "material_stock": world.stock_fraction("materials"),
            }
        return {
            "tick": float(world.clock.tick),
            "day": float(world.clock.day),
            "population": float(population),
            "births_total": float(self.births_total),
            "deaths_total": float(self.deaths_total),
            "mean_need": float(np.mean(need_total[alive])),
            "mean_energy": float(np.mean(batch.energy[alive])),
            "min_energy": float(np.min(batch.energy[alive])),
            "mean_hydration": float(np.mean(batch.hydration[alive])),
            "min_hydration": float(np.min(batch.hydration[alive])),
            "mean_repair_material": float(np.mean(batch.repair_material[alive])),
            "mean_safety": float(np.mean(batch.environmental_safety[alive])),
            "mean_social": float(np.mean(batch.social_stability[alive])),
            "mean_affect_stability": float(np.mean(batch.affect_stability[alive])),
            "lineage_count": float(len(set(batch.lineage[alive].tolist()))),
            "food_stock": world.stock_fraction("food"),
            "water_stock": world.stock_fraction("water"),
            "material_stock": world.stock_fraction("materials"),
        }

    def sample_viability(self, batch: NPCBatch, idx: int) -> Tuple[ViabilityState, NeedPressure]:
        viability = ViabilityState(
            energy=float(batch.energy[idx]),
            bodily_integrity=float(batch.bodily_integrity[idx]),
            memory_integrity=float(batch.memory_integrity[idx]),
            prediction_coherence=float(batch.prediction_coherence[idx]),
            social_stability=float(batch.social_stability[idx]),
            attachment_integrity=float(batch.attachment_integrity[idx]),
            environmental_safety=float(batch.environmental_safety[idx]),
        )
        return viability, NeedPressure.from_viability(viability)

"""3D world: terrain, resources, hazards, and field coupling.

Extracted from the integrated reference engine; behavior-preserving move.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Callable, Deque, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import jit, lax, random, vmap

from topogenesis.constants import ATTN_DIM
from topogenesis.fields.sigma import SigmaFieldGeometric
from topogenesis.body.body_state import AgentBodyPhys

class World3D:
    def __init__(self, size=(32, 32, 32), n_resources=20, n_hazards=12,
                 seed=42, mass=1.0, friction=0.1, gravity=9.8, n_decoys=0,
                 membrane_repair_rate=0.03, membrane_decay_rate=0.002,
                 ground_resource_frac=0.85, max_spawn_height=4,
                 ground_locomotion_gain=3.0, interaction_radius=1.35,
                 safe_spawn_radius=5.0, starter_resource_patch=6,
                 energy_decay=0.012, resource_energy_gain=0.65,
                 resource_repair_gain=0.25, force_metabolic_cost=0.0005,
                 resource_regen_interval=25, resource_regen_count=2,
                 starter_regen_count=3) -> None:
        self.size        = size
        self.n_resources = n_resources
        self.n_hazards   = n_hazards
        self.n_decoys    = n_decoys
        self.rng         = np.random.default_rng(seed)
        self.mass        = mass
        self.friction    = friction
        self.gravity     = gravity
        self.energy_decay = energy_decay
        self.resource_energy_gain = resource_energy_gain
        self.resource_repair_gain = resource_repair_gain
        self.force_metabolic_cost = force_metabolic_cost
        self.membrane_repair_rate = membrane_repair_rate
        self.membrane_decay_rate  = membrane_decay_rate
        self.ground_resource_frac = ground_resource_frac
        self.max_spawn_height = max_spawn_height
        self.ground_locomotion_gain = ground_locomotion_gain
        self.interaction_radius = interaction_radius
        self.safe_spawn_radius = safe_spawn_radius
        self.starter_resource_patch = starter_resource_patch
        self.resource_regen_interval = resource_regen_interval
        self.resource_regen_count = resource_regen_count
        self.starter_regen_count = starter_regen_count
        self._world_step = 0
        self.field       = SigmaFieldGeometric(size)
        self.reset()

    def reset(self) -> None:
        self.voxels = np.zeros(self.size, dtype=np.int32)
        self.voxels[:, :, 0] = 1
        self._place_items(2, self.n_resources)
        self._place_items(3, self.n_hazards)
        self._place_items(4, self.n_decoys)
        self._seed_starter_resources()
        self.field = SigmaFieldGeometric(self.size)
        self._world_step = 0

    def _seed_starter_resources(self) -> None:
        cx, cy = self.size[0] // 2, self.size[1] // 2
        offsets = [(1, 0), (-1, 0), (0, 1), (0, -1),
                   (2, 1), (-2, -1), (1, -2), (-1, 2)]
        for dx, dy in offsets[:self.starter_resource_patch]:
            x = int(np.clip(cx + dx, 0, self.size[0] - 1))
            y = int(np.clip(cy + dy, 0, self.size[1] - 1))
            self.voxels[x, y, 1] = 2

    def _regen_starter_resources(self, n: int) -> int:
        cx, cy = self.size[0] // 2, self.size[1] // 2
        offsets = [(1, 0), (-1, 0), (0, 1), (0, -1),
                   (2, 1), (-2, -1), (1, -2), (-1, 2)]
        placed = 0
        for dx, dy in offsets:
            if placed >= n:
                break
            x = int(np.clip(cx + dx, 0, self.size[0] - 1))
            y = int(np.clip(cy + dy, 0, self.size[1] - 1))
            if self.voxels[x, y, 1] == 0:
                self.voxels[x, y, 1] = 2
                placed += 1
        return placed

    def _place_items(self, vtype: int, n: int) -> None:
        placed = 0
        attempts = 0
        max_attempts = max(100, n * 50)
        while placed < n and attempts < max_attempts:
            attempts += 1
            x = int(self.rng.integers(0, self.size[0]))
            y = int(self.rng.integers(0, self.size[1]))
            if vtype == 2 and self.rng.random() < self.ground_resource_frac:
                z = 1
            else:
                z_hi = min(self.size[2], max(2, self.max_spawn_height + 1))
                z = int(self.rng.integers(1, z_hi))
            center = np.array([self.size[0] / 2, self.size[1] / 2, 1.0])
            if vtype == 3 and np.linalg.norm(np.array([x, y, z]) - center) < self.safe_spawn_radius:
                continue
            if self.voxels[x, y, z] == 0:
                self.voxels[x, y, z] = vtype
                placed += 1

    def _nearest_of_type(self, vtype: int, pos: np.ndarray):
        mask   = self.voxels == vtype
        if not np.any(mask):
            return np.zeros(3), 1e9
        coords = np.argwhere(mask).astype(np.float32)
        dists  = np.linalg.norm(coords - np.array(pos), axis=1)
        idx    = np.argmin(dists)
        return coords[idx], dists[idx]

    def _resource_count(self) -> int:
        return int(np.count_nonzero(self.voxels == 2))

    def regenerate_resources(self) -> None:
        if self.resource_regen_interval <= 0:
            return
        if self._world_step % self.resource_regen_interval != 0:
            return
        missing = max(0, self.n_resources + self.starter_resource_patch - self._resource_count())
        if missing <= 0:
            return
        starter_placed = self._regen_starter_resources(min(self.starter_regen_count, missing))
        remaining = max(0, missing - starter_placed)
        if remaining > 0:
            self._place_items(2, min(self.resource_regen_count, remaining))

    def step_body_only(self, force_torque: np.ndarray, body: AgentBodyPhys):
        self._world_step += 1
        self.regenerate_resources()
        dt    = 0.05
        force = np.nan_to_num(np.array(force_torque[:3], dtype=np.float32),
                              nan=0.0, posinf=1.0, neginf=-1.0)
        force = np.clip(force, -10.0, 10.0)
        body.pos = np.nan_to_num(body.pos, nan=0.0)
        body.vel = np.nan_to_num(body.vel, nan=0.0)
        grounded = float(body.pos[2]) <= 0.08
        if grounded:
            force[:2] *= self.ground_locomotion_gain
            force[2] = max(force[2], self.mass * self.gravity * 0.85)
        acc      = force / self.mass
        acc[2]  -= self.gravity
        body.vel = body.vel + acc * dt
        body.pos = body.pos + body.vel * dt
        body.vel[:2] = body.vel[:2] * (1.0 - self.friction)
        body.vel[2] = body.vel[2] * (1.0 - 0.5 * self.friction)
        for i in range(3):
            body.pos[i] = np.clip(body.pos[i], 0.0, self.size[i] - 1e-3)
            if body.pos[i] <= 0.0 or body.pos[i] >= self.size[i] - 1e-3:
                body.vel[i] = 0.0
        body.pos[2] = max(0.0, float(body.pos[2]))
        if body.pos[2] <= 0.0:
            body.vel[2] = max(0.0, float(body.vel[2]))

        # ── Homeostatic baseline (snapshot before interactions) ───────────────
        _e0 = body.energy
        _h0 = body.health
        _m0 = body.membrane_integrity

        r_pos, r_dist = self._nearest_of_type(2, body.pos)
        r_contact = (np.linalg.norm(r_pos[:2] - body.pos[:2]) < self.interaction_radius
                     and abs(float(r_pos[2] - body.pos[2])) <= 1.6)
        if r_contact and body.inventory < self.n_resources:
            ix = tuple(np.clip(np.round(r_pos).astype(int),
                               [0, 0, 0], [s - 1 for s in self.size]))
            if self.voxels[ix] == 2:
                self.voxels[ix] = 0
                body.inventory += 1
                body.energy = min(1.0, body.energy + self.resource_energy_gain)
                body.repair_budget = min(1.0, body.repair_budget + self.resource_repair_gain)
                # ── Autopoiesis: fraction of resource directly synthesises membrane ──
                synth = min(0.12, 1.0 - body.membrane_integrity)
                body.membrane_integrity = min(1.0, body.membrane_integrity + synth)
                body.repair_budget = max(0.0, body.repair_budget - synth * 0.5)
                # Resource → biosynthetic budget (produces structural components)
                body.biosynthetic_budget = min(
                    1.0, body.biosynthetic_budget + 0.15 * self.resource_energy_gain)
        h_pos, h_dist = self._nearest_of_type(3, body.pos)
        h_contact = (np.linalg.norm(h_pos[:2] - body.pos[:2]) < self.interaction_radius
                     and abs(float(h_pos[2] - body.pos[2])) <= 1.6)
        if h_contact:
            body.health  = max(0.0, body.health - 0.1)
            body.membrane_integrity = max(0.0, body.membrane_integrity - 0.08)
        d_pos, d_dist = self._nearest_of_type(4, body.pos)
        d_contact = (np.linalg.norm(d_pos[:2] - body.pos[:2]) < self.interaction_radius
                     and abs(float(d_pos[2] - body.pos[2])) <= 1.6)
        if d_contact:
            body.energy  = max(0.0, body.energy - 0.03)
        metabolic_cost = self.force_metabolic_cost * float(np.linalg.norm(force))
        body.energy = max(0.0, body.energy - self.energy_decay - metabolic_cost)
        repair = min(body.repair_budget, body.energy, self.membrane_repair_rate)
        if repair > 0.0 and body.membrane_integrity < 1.0:
            body.membrane_integrity = min(1.0, body.membrane_integrity + repair)
            body.repair_budget = max(0.0, body.repair_budget - repair)
            body.energy = max(0.0, body.energy - 0.5 * repair)
        membrane_decay = self.membrane_decay_rate + 0.012 * max(0.0, 0.2 - body.energy)
        body.membrane_integrity = max(0.0, body.membrane_integrity - membrane_decay)
        if body.membrane_integrity <= 0.2:
            leak = 0.03 * (0.2 - body.membrane_integrity)
            body.energy = max(0.0, body.energy - leak)
            body.health = max(0.0, body.health - 0.01)
        if body.energy <= 0.05:
            body.health = max(0.0, body.health - 0.04)

        # ── Intrinsic homeostatic reward: improvement in viability state ──────
        homeostasis_target = 0.35
        _deviation_before = abs(_e0 - homeostasis_target) + abs(_m0 - 0.8) + abs(_h0 - 0.8)
        _deviation_after  = abs(body.energy - homeostasis_target) + abs(body.membrane_integrity - 0.8) + abs(body.health - 0.8)
        reward = float(_deviation_before - _deviation_after)  # positive = moved toward viability
        reward -= 0.002  # small persistent metabolic cost signal
        # Structural decay: cognitive substrate degrades faster when starved.
        _edef = max(0.0, 0.25 - body.energy)
        for _mod in list(body.structural_integrity):
            body.structural_integrity[_mod] = max(
                0.02, body.structural_integrity[_mod] - 8e-4 * (1.0 + 4.0 * _edef))
        # Structural repair: biosynthetic budget → component production
        _repair = min(body.biosynthetic_budget, 0.015)
        if _repair > 1e-6:
            _n = len(body.structural_integrity)
            for _mod in list(body.structural_integrity):
                _gap = 1.0 - body.structural_integrity[_mod]
                _act = min(_repair / _n, _gap)
                body.structural_integrity[_mod] += _act
            body.biosynthetic_budget = max(0.0, body.biosynthetic_budget - _repair)
        _mean_si = float(np.mean(list(body.structural_integrity.values())))
        body.last_reward = reward
        body.t          += 1
        body.age        += 1
        # Structural collapse (mean integrity < 4%) is terminal.
        return reward, body.health <= 0.0 or body.membrane_integrity <= 0.0 or _mean_si < 0.04

    def advance_field(self, bodies: List[AgentBodyPhys]) -> None:
        if bodies:
            pos_arr = jnp.array(np.stack([b.pos for b in bodies]), dtype=jnp.float32)
            eng_arr = jnp.array([b.energy for b in bodies],        dtype=jnp.float32)
        else:
            # Extinct population: the field still evolves, with no pump.
            pos_arr = jnp.zeros((0, 3), dtype=jnp.float32)
            eng_arr = jnp.zeros((0,),   dtype=jnp.float32)
        self.field.step(pos_arr, eng_arr)
        for b in bodies:
            b.last_q_prev = getattr(b, 'last_q', 1.0)
            b.last_q = self.field.topological_charge_at(
                int(np.clip(round(float(b.pos[2])), 0, self.size[2] - 1)))

    def obs_dict(self, body: AgentBodyPhys) -> dict:
        _, r_dist = self._nearest_of_type(2, body.pos)
        _, h_dist = self._nearest_of_type(3, body.pos)
        _, d_dist = self._nearest_of_type(4, body.pos)
        wmax = float(max(self.size))
        return {
            'energy':        float(body.energy),
            'health':        float(body.health),
            'inventory':     body.inventory / max(1, self.n_resources),
            'membrane':      float(getattr(body, 'membrane_integrity', 1.0)),
            'resource_dist': float(r_dist / wmax) if r_dist < 1e9 else 1.0,
            'hazard_dist':   float(h_dist / wmax) if h_dist < 1e9 else 1.0,
            'decoy_dist':    float(d_dist / wmax) if d_dist < 1e9 else 1.0,
        }

    def affordance_context(self, body: AgentBodyPhys) -> jnp.ndarray:
        """Encode local resource/hazard affordances into the attention channel."""
        r_pos, r_dist = self._nearest_of_type(2, body.pos)
        h_pos, h_dist = self._nearest_of_type(3, body.pos)
        wmax = float(max(self.size))

        def unit_and_proximity(target, dist):
            if dist >= 1e8:
                return np.zeros(3, dtype=np.float32), 0.0
            vec = np.array(target - body.pos, dtype=np.float32)
            norm = float(np.linalg.norm(vec)) + 1e-8
            unit = vec / norm
            prox = float(np.clip(1.0 - dist / wmax, 0.0, 1.0))
            return unit, prox

        r_unit, r_prox = unit_and_proximity(r_pos, r_dist)
        h_unit, h_prox = unit_and_proximity(h_pos, h_dist)
        ctx = np.zeros(ATTN_DIM, dtype=np.float32)
        ctx[0:3] = r_unit
        ctx[3]   = r_prox
        ctx[4:7] = h_unit
        ctx[7]   = h_prox
        ctx[8]   = float(body.energy)
        ctx[9]   = float(body.health)
        ctx[10]  = float(getattr(body, 'membrane_integrity', 1.0))
        ctx[11]  = float(getattr(body, 'repair_budget', 0.0))
        ctx[12]  = min(1.0, float(getattr(body, 'death_count', 0)) / 10.0)
        return jnp.array(ctx)

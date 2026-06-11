"""Embodiment: body physics, metabolism, rich body state, observation vector.

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

from topogenesis.npc import (
    AffectField as NpcAffectField,
    CommunicationIntent,
    NeedPressure,
    OtherMindModel,
    SocialMemory,
    ViabilityState,
    interpret_intent,
    simulate_future,
)

from topogenesis.constants import (
    ATTN_DIM,
    FIELD_PATCH_DIM,
    MAX_JOINTS,
    MAX_MOTORS,
    MAX_TACTILE,
    MOTOR_DIM,
    N_JOINTS,
    TACTILE_ZONES,
)

class CognitiveMetabolism:
    """
    Thermodynamic supervenience: every cognitive operation costs energy.

    Accumulates energy debits during a single step() cycle; self_maintain()
    flushes the total to body.energy, so the thermodynamic reservoir
    constrains what is computationally possible.  An agent that cannot
    afford the energy bill of a complex cognitive act literally cannot
    execute it at full fidelity.

    Cost schedule (tuned so total ≈ 0.5–1.5× energy_decay per step):
      GRU layers:    COST_GRU_PER_DIM_LAYER × hidden_dim × n_layers
      Slot attention: COST_ATTN_PER_SLOT_ITER × n_slots × n_iters × (slot_dim/64)
      Anderson DEQ:  COST_DEQ_PER_DIM_ITER  × deter_dim × n_iters
      Policy sample: COST_POLICY_SAMPLE  (flat per forward pass)
      Memory add:    COST_MEM_ADD        (per episodic write)
      Consolidation: COST_MEM_CONSOLIDATE (per consolidation cycle)
      Retrieval:     COST_MEM_RETRIEVE   (per query)
      Learning step: COST_LEARNING_STEP  × n_params/10 000
      Workspace:     COST_WORKSPACE_UPDATE × workspace_dim/32
    """
    COST_GRU_PER_DIM_LAYER  = 1.0e-6
    COST_ATTN_PER_SLOT_ITER = 2.0e-5
    COST_DEQ_PER_DIM_ITER   = 5.0e-7
    COST_POLICY_SAMPLE      = 5.0e-5
    COST_MEM_ADD            = 1.0e-5
    COST_MEM_CONSOLIDATE    = 2.5e-4
    COST_MEM_RETRIEVE       = 5.0e-6
    COST_LEARNING_STEP      = 3.0e-5
    COST_WORKSPACE_UPDATE   = 1.0e-5

    def __init__(self) -> None:
        self._pending_cost:  float = 0.0
        self.total_cost_ema: float = 0.0
        self._n_ops:         int   = 0

    def charge(self, cost: float) -> None:
        """Debit a positive energy cost; accumulates until flush()."""
        self._pending_cost += max(0.0, float(cost))
        self._n_ops += 1

    def flush(self) -> float:
        """Return total pending cost and reset accumulator."""
        cost = self._pending_cost
        self.total_cost_ema = 0.99 * self.total_cost_ema + 0.01 * cost
        self._pending_cost  = 0.0
        self._n_ops         = 0
        return cost

    # ── Cost calculators ────────────────────────────────────────────────────

    def gru_cost(self, hidden_dim: int, n_layers: int = 3) -> float:
        return self.COST_GRU_PER_DIM_LAYER * hidden_dim * n_layers

    def attention_cost(self, n_slots: int, n_iters: int,
                       slot_dim: int = 64) -> float:
        return self.COST_ATTN_PER_SLOT_ITER * n_slots * n_iters * (slot_dim / 64.0)

    def deq_cost(self, deter_dim: int, n_iters: int) -> float:
        return self.COST_DEQ_PER_DIM_ITER * deter_dim * max(1, n_iters)

    def policy_cost(self) -> float:
        return self.COST_POLICY_SAMPLE

    def memory_add_cost(self) -> float:
        return self.COST_MEM_ADD

    def memory_consolidate_cost(self) -> float:
        return self.COST_MEM_CONSOLIDATE

    def memory_retrieve_cost(self) -> float:
        return self.COST_MEM_RETRIEVE

    def learning_cost(self, n_params: int = 10_000) -> float:
        return self.COST_LEARNING_STEP * max(1, n_params // 10_000)

    def workspace_cost(self, workspace_dim: int = 32) -> float:
        return self.COST_WORKSPACE_UPDATE * (workspace_dim / 32.0)

    def snapshot(self) -> dict:
        return {'cog_cost_ema': round(self.total_cost_ema, 7)}

class RichBodyState:
    def __init__(self, pos, vel, quat, ang_vel, joints, joint_vel,
                 tactile, intero, efference, motor_noise=0.0) -> None:
        self.pos              = jnp.asarray(pos,       dtype=jnp.float32)
        self.vel              = jnp.asarray(vel,       dtype=jnp.float32)
        self.quat             = jnp.asarray(quat,      dtype=jnp.float32)
        self.ang_vel          = jnp.asarray(ang_vel,   dtype=jnp.float32)
        self.joint_angles     = jnp.asarray(joints,    dtype=jnp.float32)
        self.joint_vel        = jnp.asarray(joint_vel, dtype=jnp.float32)
        self.tactile          = jnp.asarray(tactile,   dtype=jnp.float32)
        self.interoception    = jnp.asarray(intero,    dtype=jnp.float32)
        self.efference_copy   = jnp.asarray(efference, dtype=jnp.float32)
        self.motor_noise_sigma = motor_noise

    def to_vector(self) -> jnp.ndarray:
        return jnp.concatenate([
            self.pos, self.vel, self.quat, self.ang_vel,
            self.joint_angles, self.joint_vel, self.tactile,
            self.interoception, self.efference_copy,
        ])

def build_rich_body(body, efference=None) -> RichBodyState:
    # Pad joints/tactile/efference to MAX dims so all agents share observation shape
    tactile_raw = jnp.zeros(body.n_tactile).at[0].set(
        1.0 if float(body.pos[2]) < 0.1 else 0.0)
    tactile = jnp.pad(tactile_raw, (0, MAX_TACTILE - body.n_tactile))

    joints_padded   = jnp.pad(jnp.asarray(body.joint_angles, dtype=jnp.float32),
                               (0, MAX_JOINTS - body.n_joints))
    joint_vel_padded = jnp.pad(jnp.asarray(body.joint_vel, dtype=jnp.float32),
                                (0, MAX_JOINTS - body.n_joints))

    intero = jnp.array([
        float(body.energy), 1.0 - float(body.energy),
        float(body.health), 1.0 - float(body.health),
        float(getattr(body, 'membrane_integrity', 1.0)),
        1.0 - float(getattr(body, 'membrane_integrity', 1.0)),
        float(getattr(body, 'repair_budget', 0.0)),
        min(1.0, float(getattr(body, 'age', 0)) / 1000.0),
    ])

    if efference is None:
        efference_raw = jnp.zeros(body.n_motors)
    else:
        efference_raw = jnp.asarray(efference[:body.n_motors], dtype=jnp.float32)
    efference_padded = jnp.pad(efference_raw, (0, MAX_MOTORS - body.n_motors))

    return RichBodyState(
        body.pos, body.vel, body.quat, body.ang_vel,
        joints_padded, joint_vel_padded,
        tactile, intero, efference_padded,
        motor_noise=0.1 if float(body.energy) < 0.3 else 0.0,
    )

def observe_full_vector(rich, energy, health, inventory,
                        field_patch=None, q_scalar=0.0,
                        field_grad=None, topo_stability=0.0,
                        attn_context=None) -> jnp.ndarray:
    vec   = rich.to_vector()
    extra = jnp.array([energy, health, inventory, rich.motor_noise_sigma])
    if field_patch is None:
        field_patch = jnp.zeros(FIELD_PATCH_DIM)
    if field_grad is None:
        field_grad  = jnp.zeros(3)
    if attn_context is None:
        attn_context = jnp.zeros(ATTN_DIM)
    field_features = jnp.concatenate([
        field_patch, jnp.array([float(q_scalar)]),
        field_grad,  jnp.array([float(topo_stability)]),
    ])
    obs = jnp.concatenate([vec, extra, attn_context, field_features])
    return jnp.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

class AgentBodyPhys:
    def __init__(self, start_pos=(16, 16, 1),
                 n_joints: int = N_JOINTS,
                 n_motors: int = MOTOR_DIM,
                 n_tactile: int = TACTILE_ZONES) -> None:
        self.start_pos    = tuple(start_pos)
        self.n_joints     = int(np.clip(n_joints,  4, MAX_JOINTS))
        self.n_motors     = int(np.clip(n_motors,  3, MAX_MOTORS))
        self.n_tactile    = int(np.clip(n_tactile, 4, MAX_TACTILE))
        self.pos          = np.array(start_pos, dtype=np.float32)
        self.vel          = np.zeros(3,         dtype=np.float32)
        self.quat         = np.array([1., 0., 0., 0.], dtype=np.float32)
        self.ang_vel      = np.zeros(3,         dtype=np.float32)
        self.joint_angles = np.zeros(self.n_joints, dtype=np.float32)
        self.joint_vel    = np.zeros(self.n_joints, dtype=np.float32)
        self.energy       = 1.0
        self.health       = 1.0
        self.inventory    = 0
        self.membrane_integrity = 1.0
        self.repair_budget = 0.0
        # Structural production closure: cognitive substrate must be
        # continuously re-synthesised from metabolism.  Each module has an
        # integrity ∈ [0, 1]; it decays when starved and is replenished by the
        # biosynthetic budget produced by resource consumption.
        self.structural_integrity: Dict[str, float] = {
            'policy': 1.0, 'world_model': 1.0, 'affect': 1.0,
            'symbolic': 1.0, 'viability': 1.0, 'misc': 1.0,
        }
        self.biosynthetic_budget: float = 0.0
        # Fidelity of genome encoding in the sigma field (0=lost, 1=intact)
        self.genome_field_fidelity: float = 1.0
        self.age          = 0
        self.death_count  = 0
        self.generation   = 0
        self.lineage_id   = int(np.random.default_rng().integers(0, 1_000_000))
        self.parent_id    = None
        self.repro_cooldown = 0
        self.last_reward  = 0.0
        self.t            = 0
        self.last_q       = 1.0
        self.last_q_prev  = 1.0

    def reset(self, start_pos=None) -> None:
        sp = start_pos if start_pos is not None else self.start_pos
        deaths     = getattr(self, 'death_count', 0) + 1
        generation = getattr(self, 'generation', 0)
        lineage_id = getattr(self, 'lineage_id', None)
        parent_id  = getattr(self, 'parent_id', None)
        n_joints   = getattr(self, 'n_joints',  N_JOINTS)
        n_motors   = getattr(self, 'n_motors',  MOTOR_DIM)
        n_tactile  = getattr(self, 'n_tactile', TACTILE_ZONES)
        self.__init__(sp, n_joints=n_joints, n_motors=n_motors, n_tactile=n_tactile)
        self.death_count = deaths
        self.generation  = generation
        if lineage_id is not None:
            self.lineage_id = lineage_id
        self.parent_id = parent_id

#!/usr/bin/env python3

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

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

N_JOINTS      = 12   # default / minimum meaningful
TACTILE_ZONES = 12
INTERO_DIM    = 8
MOTOR_DIM     = 6

# Maximum body plan dimensions — all networks observe padded-to-max vectors
# so every agent shares the same architecture regardless of evolved body plan.
MAX_JOINTS    = 24
MAX_MOTORS    = 8
MAX_TACTILE   = 24

BODY_VEC_LEN  = 3 + 3 + 4 + 3 + MAX_JOINTS + MAX_JOINTS + MAX_TACTILE + INTERO_DIM + MAX_MOTORS
ATTN_DIM      = 32
FIELD_PATCH_DIM = 64
FIELD_OBS_START = BODY_VEC_LEN + 4 + ATTN_DIM
FIELD_Q_IDX     = FIELD_OBS_START + FIELD_PATCH_DIM
FIELD_GRAD_IDX  = FIELD_Q_IDX + 1
FIELD_STAB_IDX  = FIELD_Q_IDX + 4
FIELD_OBS_DIM   = FIELD_PATCH_DIM + 5
sigmoid = jax.nn.sigmoid
GENOME_DIM = 256

# ── Genome growth parameters ──────────────────────────────────────────────────
GENOME_MAX_MODULE_DIM = 512   # hard cap per module — prevents unbounded growth
GENOME_DUP_PROB       = 0.02  # probability of segment duplication per module per birth
GENOME_DEL_PROB       = 0.01  # probability of segment deletion per module per birth
GENOME_DUP_SEGMENT    = 8     # elements copied / removed per event

# ── Genome field embodiment ──────────────────────────────────────────────────
# The genome is a physical pattern in the sigma field.  Agents must actively
# pump the field to maintain their genome; field decay degrades it for free.
GENOME_LOCI_PER_MODULE  = 4      # sigma-field positions encoding each genome module
GENOME_FIELD_MAINT_COST = 3e-4   # energy / locus / step (metabolic cost of heredity)
GENOME_FIELD_STRENGTH   = 0.35   # pump magnitude when writing genome to field
GENOME_FIELD_RADIUS     = 2      # spatial radius of genome loci around body centre

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AffectConfig:
    valence_dim:   int   = 16
    arousal_dim:   int   = 16
    affect_decay:  float = 0.95
    w_valence:     float = 0.05
    w_distress:    float = 0.1

@dataclass(frozen=True)
class MemoryConfig:
    episodic_capacity:        int   = 2000
    semantic_capacity:        int   = 200
    consolidation_interval:   int   = 100
    pruning_threshold:        float = 1e-3

@dataclass(frozen=True)
class CognitiveConfig:
    # ── World model / agent geometry ─────────────────────────────────────
    deter_dim:               int   = 128
    stoch_dim:               int   = 64
    n_slots:                 int   = 16
    slot_dim:                int   = 64
    causal_dim:              int   = 64
    symbolic_dim:            int   = 64
    planner_dim:             int   = 64
    embed_dim:               int   = 128
    action_dim:              int   = 6
    world_size:              int   = 32
    object_world_size:       int   = 32
    world_depth:             int   = 32
    # ── DEQ / Anderson ──────────────────────────────────────────────────
    max_fp_iter:             int   = 15
    fp_tol:                  float = 1e-4
    anderson_memory:         int   = 5
    anderson_ridge:          float = 1e-5
    anderson_damping:        float = 0.5
    # ── Workspace ───────────────────────────────────────────────────────
    workspace_dim:           int   = 32
    workspace_decay:         float = 0.85
    workspace_temperature_base: float = 0.7
    workspace_action_gain:   float = 0.15
    # ── Affect ──────────────────────────────────────────────────────────
    affect_decay:            float = 0.9
    affect_action_gain:      float = 0.1
    affect_consolidation_boost: float = 2.0
    affect_salience_weight:  float = 0.15
    # ── Dynamical stability / Lyapunov ───────────────────────────────────
    lyapunov_n_vectors:      int   = 8
    lyapunov_renorm_steps:   int   = 10
    sparsity_target:         float = 0.1
    sparsity_beta:           float = 0.01
    convergence_window:      int   = 5
    # ── Emergent metastability ───────────────────────────────────────────
    eoc_target:              float = 1.0
    kuramoto_target_lo:      float = 0.3
    kuramoto_target_hi:      float = 0.7
    soc_tau_target:          float = 1.5
    soc_tau_lo:              float = 1.3
    soc_tau_hi:              float = 1.7
    soc_s_max_init:          float = 50.0
    soc_s_max_floor:         float = 5.0
    soc_contraction_lr:      float = 0.05
    hopf_mu_lr:              float = 0.01
    # ── Compositional symbolics / HRR ────────────────────────────────────
    hrr_dim:                 int   = 64
    hrr_capacity:            int   = 32
    hrr_noise_floor:         float = 0.05
    # ── Causal learning ──────────────────────────────────────────────────
    causal_lr:               float = 0.05
    granger_window:          int   = 20
    intervention_p:          float = 0.05
    causal_threshold:        float = 0.1
    # ── Self-development ─────────────────────────────────────────────────
    competency_beta:         float = 0.98
    competency_thresh:       float = 0.4
    mastery_var_eps:         float = 0.02
    mastery_window:          int   = 50
    advance_tau:             int   = 20
    regress_tau:             int   = 40
    entropy_delta:           float = 0.5
    # ── Field physics ────────────────────────────────────────────────────
    field_pump_gain:         float = 0.25
    field_diffusion:         float = 0.15
    field_decay_rate:        float = 0.008
    # ── Memory ──────────────────────────────────────────────────────────
    episodic_capacity:       int   = 4096
    semantic_capacity:       int   = 512
    causal_capacity:         int   = 1024
    autobio_capacity:        int   = 2048
    symbolic_mem_capacity:   int   = 512
    replay_steps:            int   = 50
    smm_n_modules:           int   = 8
    smm_module_dim:          int   = 64
    smm_k_active:            int   = 2
    smm_episodic_capacity:   int   = 4096
    smm_semantic_capacity:   int   = 512
    smm_hash_bits:           int   = 8
    smm_n_tables:            int   = 4
    smm_forget_threshold:    float = 0.05
    smm_forget_halflife:     float = 500.0
    smm_consolidation_interval: int = 100
    smm_uncertainty_bins:    int   = 64
    smm_context_blend:       float = 0.12
    smm_causal_ema:          float = 0.10
    smm_replay_priority_frac: float = 0.50
    schema_cooccur_threshold: int  = 8
    schema_l1_cooccur_threshold: int = 4
    # ── World dynamics ───────────────────────────────────────────────────
    homeostasis_target_norm: float = 0.35
    homeostasis_weight:      float = 0.08
    competence_ema:          float = 0.98
    n_resources:             int   = 20
    n_hazards:               int   = 12
    n_decoys:                int   = 0
    energy_decay:            float = 0.006
    force_metabolic_cost:    float = 0.0005
    resource_energy_gain:    float = 0.65
    resource_repair_gain:    float = 0.25
    # ── Entity attention ─────────────────────────────────────────────────
    entity_attn_dim:         int   = 32
    entity_max_count:        int   = 20
    # ── HierarchicalGRU ──────────────────────────────────────────────────
    time_embed_dim:          int   = 8
    spatial_attn_out:        int   = 32
    K_medium:                int   = 10
    K_slow:                  int   = 100
    medium_buffer_size:      int   = 50
    slow_buffer_size:        int   = 500
    use_stochastic_latent:   bool  = True
    latent_kl_weight:        float = 0.01
    # ── Policy / critic ──────────────────────────────────────────────────
    policy_net_hidden:       int   = 128
    critic_lr:               float = 1e-4
    critic_discount:         float = 0.99
    policy_online_lr:        float = 1e-4
    sensorimotor_lr:         float = 3e-4
    sensorimotor_hidden:     int   = 128
    # ── Concept / relational ─────────────────────────────────────────────
    concept_enc_dim:         int   = 64
    concept_wm_inject:       bool  = True
    goal_net_concept_dim:    int   = 64
    goal_net_field_feat_dim: int   = 64
    goal_net_hidden:         int   = 64
    relational_net_hidden:   int   = 128
    relational_action_proj_dim: int = 16
    relnet_tscs_weight:      float = 0.1
    vq_n_codes:              int   = 64
    # ── Misc ─────────────────────────────────────────────────────────────
    n_drives:                int   = 6
    drive_decay:             float = 0.99
    min_policy_entropy:      float = 0.5
    entropy_penalty:         float = 0.5
    viability_lr:            float = 0.03
    viability_actor_decay:   float = 0.995
    viability_reflex_gain:   float = 1.25
    enactive_memory_gain:    float = 0.20
    enactive_actor_lr:       float = 3e-4
    enactive_discount:       float = 0.97
    enactive_action_sigma:   float = 0.55
    membrane_repair_rate:    float = 0.03
    membrane_decay_rate:     float = 0.002
    timescale_gate_hidden:   int   = 32
    motor_noise_base:        float = 0.05
    physics_gravity:         float = 9.8
    ground_resource_frac:    float = 0.85
    max_spawn_height:        int   = 4
    ground_locomotion_gain:  float = 3.0
    interaction_radius:      float = 1.35
    safe_spawn_radius:       float = 5.0
    starter_resource_patch:  int   = 6
    resource_regen_interval: int   = 25
    resource_regen_count:    int   = 4
    starter_regen_count:     int   = 3
    low_viability_policy_suppression: float = 0.30
    reproduction_energy:     float = 0.72
    reproduction_membrane:   float = 0.94
    reproduction_inventory:  int   = 4
    reproduction_min_age:    int   = 35
    reproduction_cooldown:   int   = 80
    offspring_mutation_sigma: float = 0.015
    reproduction_energy_cost: float = 0.35
    reproduction_inventory_cost: int = 3
    death_structural_min:    float = 0.04
    death_genome_fidelity_min: float = 0.02
    juvenile_death_threshold_scale: float = 0.50
    juvenile_age:           int   = 100
    adolescent_age:         int   = 300
    adult_age:              int   = 600
    developmental_min_viability: float = 0.55
    developmental_memory_interval: int = 25
    stage_transition_stability: float = 0.04
    body_mass:               float = 1.0
    friction_coeff:          float = 0.1
    proprioceptive_delay:    int   = 2

@dataclass(frozen=True)
class TopogenesisConfig:
    d_E:        int   = BODY_VEC_LEN + 4 + ATTN_DIM + FIELD_OBS_DIM
    d_D:        int   = 128
    d_I:        int   = 64
    latent_dim: int   = 512
    hidden_dim: int   = 1024
    tau_max:    int   = 32
    dt:         float = 1.0
    lambda_:    float = 0.2
    A_rank:     int   = 16
    alpha_goal: float = 0.1
    lr:         float = 3e-4
    wm_jac_reg: float = 0.01
    contraction_target:  float = 0.9
    contraction_penalty: float = 10.0
    w_curiosity:     float = 0.25
    curiosity_scale: float = 1.0
    w_entropy:       float = 0.02
    w_compression:   float = 0.25
    w_competence:    float = 0.25
    w_survival:      float = 0.25
    grad_clip_norm:  float = 1.0
    num_agents:      int   = 2
    coupling_strength: float = 0.1
    use_affect:        bool = True
    use_memory_consolidation: bool = True
    use_developmental_growth: bool = False
    affect:    AffectConfig   = field(default_factory=AffectConfig)
    memory:    MemoryConfig   = field(default_factory=MemoryConfig)
    cognition: CognitiveConfig = field(default_factory=CognitiveConfig)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def spectral_normalize(W: jnp.ndarray, sigma_max: float = 1.0) -> jnp.ndarray:
    u, s, vt = jnp.linalg.svd(W, full_matrices=False)
    return W * (sigma_max / (jnp.max(s) + 1e-8))

def get_time_encoding(t_vec: jnp.ndarray,
                      periods: jnp.ndarray,
                      out_dim: int) -> jnp.ndarray:
    """Sine/cosine positional encoding across logarithmic time scales."""
    n = len(periods)
    sins = jnp.sin(2 * jnp.pi * t_vec[:, None] / periods[None, :])
    coss = jnp.cos(2 * jnp.pi * t_vec[:, None] / periods[None, :])
    enc  = jnp.concatenate([sins, coss], axis=-1).flatten()
    if enc.shape[0] >= out_dim:
        return enc[:out_dim]
    return jnp.pad(enc, (0, out_dim - enc.shape[0]))

def xavier(rng: jax.random.KeyArray,
           shape: Tuple[int, ...],
           scale: float = 1.0) -> jnp.ndarray:
    """Xavier/Glorot uniform initialisation."""
    fan_in  = shape[-1] if len(shape) >= 2 else shape[0]
    fan_out = shape[-2] if len(shape) >= 2 else shape[0]
    limit   = scale * math.sqrt(6.0 / (fan_in + fan_out + 1e-8))
    return random.uniform(rng, shape, minval=-limit, maxval=limit)

# ─────────────────────────────────────────────────────────────────────────────
# THERMODYNAMIC RESERVOIR
# ─────────────────────────────────────────────────────────────────────────────

class ThermodynamicReservoir:
    """
    Heat bath implementing the Fluctuation-Dissipation Theorem.

    noise_amplitude = sqrt(2 · D · k_B · T / dt)
    dT/dt = Q_dot/C - cooling_rate * (T - T0)
    dS_prod = Q_dot * dt / T  (Clausius entropy production)
    """

    def __init__(self, T0: float = 1.0, capacity: float = 50.0,
                 cooling_rate: float = 0.001) -> None:
        self.T            = T0
        self.T0           = T0
        self.C            = capacity
        self.cooling_rate = cooling_rate
        self.entropy_produced = 0.0
        self._dissipation_ema = 0.0

    def exchange(self, dissipation: float, dt: float) -> float:
        Q_dot = float(dissipation)
        dT    = (Q_dot / self.C - self.cooling_rate * (self.T - self.T0)) * dt
        self.T = max(1e-3, self.T + dT)
        self.entropy_produced += Q_dot * dt / max(self.T, 1e-3)
        self._dissipation_ema  = 0.99 * self._dissipation_ema + 0.01 * Q_dot
        return self.T

    def noise_amplitude(self, D: float, dt: float) -> float:
        return float(np.sqrt(max(0.0, 2.0 * D * self.T / max(dt, 1e-6))))

    def snapshot(self) -> dict:
        return {
            'reservoir_T':      round(self.T, 4),
            'entropy_produced': round(self.entropy_produced, 4),
            'dissipation_ema':  round(self._dissipation_ema, 6),
        }

# ─────────────────────────────────────────────────────────────────────────────
# SUPERVENIENCE SUBSYSTEMS
# ─────────────────────────────────────────────────────────────────────────────

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


class FieldSupervenience:
    """
    Field supervenience: cognitive state supervenes on the sigma field.

    The sigma field at the agent's location provides a *neural gain* —
    a multiplicative gate applied to GRU hidden states after each step.
    When the field is strong and coherent (actively maintained by the
    agent's slot pumping), hidden states express at full magnitude.
    When the field dissipates, hidden states are attenuated and the
    world model's predictive fidelity degrades.

    GRU *weights* remain fixed; what changes is how strongly those
    weights express in the field substrate.  This is analogous to
    neuromodulatory gain control.

    Genome fidelity co-modulates gain: hereditary information collapse
    degrades cognitive fidelity even when the local field is strong.
    """
    GAIN_FLOOR   = 0.10   # minimum expressible gain even in empty field
    GENOME_FLOOR = 0.25   # genome contributes at least this much to gain

    def __init__(self, ema_decay: float = 0.90) -> None:
        self._gain_ema: float = 1.0
        self._ema_decay = float(ema_decay)

    def compute_neural_gain(self, field: 'SigmaFieldGeometric',
                             body_pos: np.ndarray,
                             genome_fidelity: float) -> float:
        """
        Neural gain ∈ [GAIN_FLOOR, 1.0]:
          field_gain  = clip(||local_patch|| × 3, 0, 1)
          genome_gate = GENOME_FLOOR + (1 − GENOME_FLOOR) × genome_fidelity
          raw_gain    = clip(field_gain × genome_gate, GAIN_FLOOR, 1)
        Smoothed via EMA so gain can't drop catastrophically in a single step.
        """
        try:
            patch      = np.array(
                field.sample_patch(jnp.array(body_pos, dtype=jnp.float32), patch_size=4))
            local_str  = float(np.linalg.norm(patch)) / max(1.0, len(patch) ** 0.5)
            field_gain = float(np.clip(local_str * 3.0, 0.0, 1.0))
        except Exception:
            field_gain = 0.5
        genome_gate = self.GENOME_FLOOR + (1.0 - self.GENOME_FLOOR) * float(
            np.clip(genome_fidelity, 0.0, 1.0))
        raw_gain    = float(np.clip(field_gain * genome_gate, self.GAIN_FLOOR, 1.0))
        self._gain_ema = self._ema_decay * self._gain_ema + (1.0 - self._ema_decay) * raw_gain
        return self._gain_ema

    def gate_hidden(self, h_gru: jnp.ndarray, gain: float,
                    h_field: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Constitutive mixing: h = gain * h_gru + (1 − gain) * h_field.

        h_field is the substrate attractor — defaults to the zero vector
        (vacuum state), so a fully dissipated field drives h toward zero
        rather than toward a scaled version of what the GRU computed.

        In practice this method is called directly only in test/utility
        contexts; the main forward pass performs mixing inline in step()
        immediately after the GRU call to avoid an extra JAX dispatch.

        Args:
            h_gru:   GRU output hidden state.
            gain:    Neural gain ∈ [GAIN_FLOOR, 1.0] from compute_neural_gain.
            h_field: Field attractor vector (default: zero / vacuum).
        """
        g = float(gain)
        h_f = h_field if h_field is not None else jnp.zeros_like(h_gru)
        return g * h_gru + (1.0 - g) * h_f

    def snapshot(self) -> dict:
        return {'field_neural_gain': round(self._gain_ema, 4)}


class InformationalSupervenience:
    """
    Informational supervenience: higher-level representations supervene
    on lower-level neural states.

    Degradation cascade (bottom-up):
      structural_integrity['world_model'] × energy
        → neural_q    (quality of raw hidden states)
        → workspace_q (workspace supervenes on neural, with softer exponent)
        → symbolic_q  (HRR symbolic layer supervenes on workspace × symbolic SI)
        → concept_q   (concept/affect layer supervenes on symbolic × affect SI)

    When any level degrades, higher levels degrade in turn.  An agent
    with a damaged world-model substrate loses workspace coherence, then
    symbolic binding quality, then conceptual reasoning — even before
    motor output is affected.

    Workspace degradation is implemented as additive noise; symbolic
    degradation as multiplicative attenuation of the HRR vector.
    """

    def compute_quality(self, structural_integrity: Dict[str, float],
                        energy: float) -> Dict[str, float]:
        si = structural_integrity
        # Level 0 — raw neural (world model × policy SI × energy availability)
        neural_q   = float(np.clip(
            (si.get('world_model', 1.0) * si.get('policy', 1.0)) ** 0.5
            * (0.15 + 0.85 * max(0.0, energy)),
            0.0, 1.0))
        # Level 1 — workspace supervenes on neural (softer exponent)
        workspace_q = float(np.clip(neural_q ** 0.65, 0.0, 1.0))
        # Level 2 — symbolic supervenes on workspace × symbolic SI
        symbolic_q  = float(np.clip(
            workspace_q * si.get('symbolic', 1.0), 0.0, 1.0))
        # Level 3 — concept/affect supervenes on symbolic × affect SI
        concept_q   = float(np.clip(
            symbolic_q * si.get('affect', 1.0), 0.0, 1.0))
        return {
            'neural':    neural_q,
            'workspace': workspace_q,
            'symbolic':  symbolic_q,
            'concept':   concept_q,
        }

    def apply_workspace_noise(self, ws: jnp.ndarray,
                               quality: float,
                               rng: jax.random.KeyArray) -> jnp.ndarray:
        """
        Inject noise inversely proportional to workspace substrate quality.
        At quality=1 no noise; at quality=0, workspace is pure noise.
        """
        if quality >= 0.995:
            return ws
        noise_scale = (1.0 - quality) * 0.35
        noise       = jax.random.normal(rng, ws.shape) * noise_scale
        return ws * quality + noise

    def apply_symbolic_attenuation(self, sym: jnp.ndarray,
                                    quality: float) -> jnp.ndarray:
        """Attenuate HRR/symbolic vector by substrate quality."""
        return sym * max(0.02, float(quality))

    def snapshot(self, quality: Optional[Dict[str, float]] = None) -> dict:
        if quality is None:
            quality = {}
        return {
            'info_q_neural':    round(quality.get('neural',    1.0), 4),
            'info_q_workspace': round(quality.get('workspace', 1.0), 4),
            'info_q_symbolic':  round(quality.get('symbolic',  1.0), 4),
        }


class MetabolicSupervenience:
    """
    Metabolic supervenience: memory consolidation, online learning, and
    attentional breadth all require metabolic energy.

    Effects:
      learning_rate_scale  — scales all gradient updates (hungry → slow learner)
      attention_n_active   — narrows the attentional spotlight when starved
      consolidation_cycles — skips or reduces memory consolidation when starved
      memory_retrieval_k   — narrows retrieval breadth when starved

    The biosynthetic budget (produced by resource intake) independently
    gates synaptic maintenance: even a well-energised agent that hasn't
    eaten recently will have degraded consolidation capacity.
    """
    CONSOLIDATION_ENERGY_MIN = 0.08   # skip consolidation below this energy
    CONSOLIDATION_BIOSYN_MIN = 0.05   # skip consolidation below this biosyn
    LR_FLOOR                 = 0.05   # minimum learning rate fraction

    def learning_rate_scale(self, energy: float,
                             biosynthetic: float) -> float:
        """Multiplier ∈ [LR_FLOOR, 1.0] on all gradient updates."""
        return float(np.clip(
            self.LR_FLOOR + (1.0 - self.LR_FLOOR)
            * max(0.0, energy) * (0.4 + 0.6 * max(0.0, biosynthetic)),
            self.LR_FLOOR, 1.0))

    def attention_n_active(self, energy: float, n_slots: int) -> int:
        """
        Number of active attention slots.
        Full n_slots when well-fed; at least ceil(n_slots / 4) when critical.
        """
        frac = 0.25 + 0.75 * float(np.clip(energy, 0.0, 1.0))
        return max(1, int(math.ceil(n_slots * frac)))

    def consolidation_cycles(self, energy: float,
                              biosynthetic: float) -> int:
        """
        Number of consolidation replay cycles (0 = skip entirely).
        Full capacity requires both energy and biosynthetic budget.
        """
        if (energy < self.CONSOLIDATION_ENERGY_MIN or
                biosynthetic < self.CONSOLIDATION_BIOSYN_MIN):
            return 0
        return max(1, int(round(energy * biosynthetic * 3.0)))

    def memory_retrieval_k(self, energy: float, k_base: int = 4) -> int:
        """Narrow retrieval breadth when starved."""
        frac = 0.5 + 0.5 * float(np.clip(energy, 0.0, 1.0))
        return max(1, int(round(k_base * frac)))

# ─────────────────────────────────────────────────────────────────────────────
# SIGMA FIELD — JIT KERNELS + GEOMETRIC CLASS
# ─────────────────────────────────────────────────────────────────────────────

@partial(jit, static_argnums=(3, 4, 5, 6))
def stable_step_field_pde(field, agent_positions_arr, energies_arr,
                          dt=0.05, D=0.15, decay=0.008, pump_gain=0.25,
                          max_cfl=0.8, viscosity=0.02):
    """CFL-stable reaction-diffusion PDE for S² sigma field."""
    Wx, Wy, Wz, _ = field.shape
    dx      = 1.0
    D_safe  = jnp.maximum(jnp.asarray(D, dtype=jnp.float32), 1e-8)
    dt_safe = jnp.minimum(jnp.asarray(dt, dtype=jnp.float32),
                          max_cfl * dx**2 / (6.0 * D_safe + 1e-8))
    lap  = (jnp.roll(field, 1, 0) + jnp.roll(field, -1, 0)
          + jnp.roll(field, 1, 1) + jnp.roll(field, -1, 1)
          + jnp.roll(field, 1, 2) + jnp.roll(field, -1, 2) - 6.0 * field)
    dfield = D_safe * lap - decay * field
    nb = (jnp.roll(field, 1, 0) + jnp.roll(field, -1, 0)
        + jnp.roll(field, 1, 1) + jnp.roll(field, -1, 1)
        + jnp.roll(field, 1, 2) + jnp.roll(field, -1, 2)) / 6.0
    dfield = dfield + viscosity * nb * field * (1.0 - field)

    def pump_one(carry, agent):
        pump = carry
        pos, eng = agent[:3], agent[3]
        xi = jnp.clip(jnp.round(pos[0]).astype(jnp.int32), 0, Wx - 1)
        yi = jnp.clip(jnp.round(pos[1]).astype(jnp.int32), 0, Wy - 1)
        zi = jnp.clip(jnp.round(pos[2]).astype(jnp.int32), 0, Wz - 1)
        pump = pump.at[xi, yi, zi].add(
            jnp.array([1., 0., 0.]) * jnp.clip(eng, 0., 2.) * pump_gain)
        return pump, None

    agents = jnp.concatenate([agent_positions_arr, energies_arr[:, None]], axis=1)
    pump, _ = lax.scan(pump_one, jnp.zeros_like(field), agents)
    vacuum  = jnp.array([0., 0., 1.])
    new_phi = field + (dfield + pump) * dt_safe
    new_phi = jnp.clip(new_phi, -2.0, 2.0)
    new_phi = jnp.nan_to_num(new_phi, nan=0.0, posinf=2.0, neginf=-2.0)
    # Dirichlet vacuum boundary
    for ax in range(3):
        new_phi = new_phi.at[tuple(
            [slice(None)] * ax + [0] + [slice(None)] * (3 - ax))].set(vacuum)
        new_phi = new_phi.at[tuple(
            [slice(None)] * ax + [-1] + [slice(None)] * (3 - ax))].set(vacuum)
    norms = jnp.linalg.norm(new_phi, axis=-1, keepdims=True)
    projected = new_phi / jnp.maximum(norms, 0.1)
    return jnp.nan_to_num(projected, nan=0.0, posinf=1.0, neginf=-1.0)

@jit
def compute_q_all_z(field: jnp.ndarray) -> jnp.ndarray:
    """Topological charge Q per z-slice via triple scalar product."""
    def _q_slice(sl):
        dx      = jnp.roll(sl, -1, 0) - jnp.roll(sl, 1, 0)
        dy      = jnp.roll(sl, -1, 1) - jnp.roll(sl, 1, 1)
        cross   = jnp.cross(dx, dy)
        density = jnp.einsum('xyi,xyi->xy', sl, cross)
        return jnp.nan_to_num(jnp.sum(density) / (4.0 * jnp.pi), nan=0.0)
    slices = jnp.moveaxis(field, 2, 0)
    return vmap(_q_slice)(slices)

class SigmaFieldGeometric:
    """S² sigma-model with thermodynamic reservoir and agent coupling."""

    def __init__(self, shape: Tuple[int, int, int]) -> None:
        self.shape = shape
        phi          = np.zeros(shape + (3,), dtype=np.float32)
        phi[..., 2]  = 1.0
        self.phi     = jnp.array(phi)
        self._q_all_z = jnp.zeros(shape[2])
        self.reservoir = ThermodynamicReservoir()
        self.last_stability = {
            'field_finite': 1.0,
            'field_max_abs': 1.0,
            'field_dissipation': 0.0,
        }

    # ── Geometric primitives ────────────────────────────────────────────────

    @staticmethod
    def project_tangent(phi: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        return v - jnp.sum(phi * v, axis=-1, keepdims=True) * phi

    @staticmethod
    def geodesic_step(phi: jnp.ndarray, dphi: jnp.ndarray, dt: float) -> jnp.ndarray:
        angle = jnp.linalg.norm(dphi * dt, axis=-1, keepdims=True)
        axis  = dphi / (jnp.linalg.norm(dphi, axis=-1, keepdims=True) + 1e-8)
        new   = (phi * jnp.cos(angle)
                 + axis * jnp.sin(angle)
                 + phi * jnp.sum(phi * axis, axis=-1, keepdims=True) * (1.0 - jnp.cos(angle)))
        n     = jnp.linalg.norm(new, axis=-1, keepdims=True)
        return new / (n + 1e-8)

    # ── PDE step with thermodynamic coupling ───────────────────────────────

    def step(self, agent_positions: jnp.ndarray, agent_energies: jnp.ndarray,
             dt: float = 0.05, D: float = 0.15,
             decay: float = 0.008, pump_gain: float = 0.25) -> float:
        """Advance field one PDE step; returns Rayleigh dissipation."""
        self.phi      = stable_step_field_pde(
            self.phi, agent_positions, agent_energies, dt, D, decay, pump_gain)
        self._q_all_z = compute_q_all_z(self.phi)
        dphi_x = (jnp.roll(self.phi, -1, 0) - jnp.roll(self.phi, 1, 0)) * 0.5
        dphi_y = (jnp.roll(self.phi, -1, 1) - jnp.roll(self.phi, 1, 1)) * 0.5
        dphi_z = (jnp.roll(self.phi, -1, 2) - jnp.roll(self.phi, 1, 2)) * 0.5
        grad_sq    = jnp.sum(dphi_x**2 + dphi_y**2 + dphi_z**2)
        dissipation = float(D * jnp.mean(grad_sq))
        self.last_stability = {
            'field_finite': float(jnp.all(jnp.isfinite(self.phi))),
            'field_max_abs': float(jnp.max(jnp.abs(self.phi))),
            'field_dissipation': dissipation,
        }
        self.reservoir.exchange(dissipation, dt)
        return dissipation

    def topological_charge_at(self, z_index: int) -> float:
        return float(self._q_all_z[z_index])

    def total_charge(self) -> float:
        return float(jnp.sum(self._q_all_z))

    def angular_gradient_energy(self) -> float:
        phi    = self.phi
        dphi_x = (jnp.roll(phi, -1, 0) - jnp.roll(phi, 1, 0)) * 0.5
        dphi_y = (jnp.roll(phi, -1, 1) - jnp.roll(phi, 1, 1)) * 0.5
        dphi_z = (jnp.roll(phi, -1, 2) - jnp.roll(phi, 1, 2)) * 0.5
        return float(jnp.mean(jnp.sum(dphi_x**2 + dphi_y**2 + dphi_z**2, axis=-1)))

    def kuramoto_order(self) -> float:
        mean_phi = jnp.mean(self.phi.reshape(-1, 3), axis=0)
        return float(jnp.linalg.norm(mean_phi))

    def sample_patch(self, pos: jnp.ndarray, patch_size: int = 4) -> jnp.ndarray:
        Wx, Wy, Wz, _ = self.phi.shape
        half = patch_size // 2
        xi   = jnp.clip(jnp.round(pos[0]).astype(jnp.int32), half, Wx - patch_size + half - 1) - half
        yi   = jnp.clip(jnp.round(pos[1]).astype(jnp.int32), half, Wy - patch_size + half - 1) - half
        zi   = jnp.clip(jnp.round(pos[2]).astype(jnp.int32), half, Wz - patch_size + half - 1) - half
        patch = lax.dynamic_slice(self.phi, (xi, yi, zi, 0), (patch_size, patch_size, patch_size, 3))
        flat  = patch[..., 2].flatten()
        return jnp.clip(jnp.nan_to_num(flat, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)

    def field_gradient(self, pos: jnp.ndarray) -> jnp.ndarray:
        Wx, Wy, Wz, _ = self.phi.shape
        xi  = jnp.clip(jnp.round(pos[0]).astype(jnp.int32), 1, Wx - 2)
        yi  = jnp.clip(jnp.round(pos[1]).astype(jnp.int32), 1, Wy - 2)
        zi  = jnp.clip(jnp.round(pos[2]).astype(jnp.int32), 1, Wz - 2)
        orient = self.phi[..., 2]
        gx  = (orient[xi + 1, yi, zi] - orient[xi - 1, yi, zi]) * 0.5
        gy  = (orient[xi, yi + 1, zi] - orient[xi, yi - 1, zi]) * 0.5
        gz  = (orient[xi, yi, zi + 1] - orient[xi, yi, zi - 1]) * 0.5
        return jnp.nan_to_num(jnp.stack([gx, gy, gz]), nan=0.0, posinf=10.0, neginf=-10.0)

# ─────────────────────────────────────────────────────────────────────────────
# ENTITY ATTENTION (3D spatial)
# ─────────────────────────────────────────────────────────────────────────────

class EntityAttention:
    def __init__(self, config: TopogenesisConfig, rng: jax.random.KeyArray) -> None:
        k1, k2, k3 = random.split(rng, 3)
        d = config.cognition.entity_attn_dim
        self.query_W = xavier(k1, (d, 3), 0.5)
        self.key_W   = xavier(k2, (d, 3), 0.5)
        self.out_W   = xavier(k3, (d, d), 0.5)

    @staticmethod
    @jit
    def forward(query_pos, entity_positions, params):
        q     = query_pos @ params['query_W'].T
        k     = entity_positions @ params['key_W'].T
        attn  = jax.nn.softmax(jnp.dot(k, q))
        ctx   = jnp.dot(attn, k)
        return jnp.tanh(ctx @ params['out_W'].T), attn

    def to_params(self) -> dict:
        return {'query_W': self.query_W, 'key_W': self.key_W, 'out_W': self.out_W}

    def from_params(self, p: dict) -> None:
        self.query_W = p['query_W']
        self.key_W   = p['key_W']
        self.out_W   = p['out_W']

# ─────────────────────────────────────────────────────────────────────────────
# RICH BODY STATE + OBSERVATION BUILDER
# ─────────────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────────────
# WORLD + AGENT PHYSICS
# ─────────────────────────────────────────────────────────────────────────────

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
        pos_arr = jnp.array(np.stack([b.pos for b in bodies]), dtype=jnp.float32)
        eng_arr = jnp.array([b.energy for b in bodies],        dtype=jnp.float32)
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


# ─────────────────────────────────────────────────────────────────────────────
# GENOME FIELD INTERFACE
# The genome is a physical pattern etched into the sigma field.
# • Agents pay energy each step to pump the field and maintain the pattern.
# • Field diffusion + decay erode the pattern unless actively maintained.
# • At birth the parent writes the child's genome into the field.
# • Genome fidelity < threshold ⟹ organism can no longer replicate ⟹ terminal.
# ─────────────────────────────────────────────────────────────────────────────

class GenomeFieldInterface:
    """Maps genome modules to S² field positions; reads/writes genome as field topology."""

    _MODULE_NAMES = ('policy', 'world_model', 'affect', 'symbolic', 'viability', 'misc')

    def __init__(self, n_loci: int = GENOME_LOCI_PER_MODULE) -> None:
        self.n_loci    = n_loci
        self.n_modules = len(self._MODULE_NAMES)
        self.total     = self.n_modules * n_loci
        # Pre-compute fixed spatial offsets (helix) for genome loci relative to body
        self._offsets  = self._helix_offsets()

    def _helix_offsets(self) -> np.ndarray:
        offs = []
        for m in range(self.n_modules):
            for l in range(self.n_loci):
                angle = 2.0 * np.pi * (m * self.n_loci + l) / max(1, self.total)
                offs.append([
                    GENOME_FIELD_RADIUS * np.cos(angle),
                    GENOME_FIELD_RADIUS * np.sin(angle),
                    float(m) * 0.5,   # z-stack: each module at different height
                ])
        return np.array(offs, dtype=np.float32)   # (total, 3)

    def _module_fingerprint(self, genome: 'Genome') -> np.ndarray:
        """Compress each module to n_loci target φ_z values in (−1, 1)."""
        targets = []
        for mod_name in self._MODULE_NAMES:
            mod = genome.modules.get(mod_name, np.zeros(1, dtype=np.float32))
            for l in range(self.n_loci):
                step = max(1, len(mod) // self.n_loci)
                val  = float(np.tanh(mod[(l * step) % len(mod)]))
                targets.append(val)
        return np.array(targets, dtype=np.float32)   # (total,)

    def _locus_voxel(self, body_pos: np.ndarray, offset: np.ndarray,
                     shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        pos = body_pos + offset
        xi  = int(np.clip(round(pos[0]), 0, shape[0] - 1))
        yi  = int(np.clip(round(pos[1]), 0, shape[1] - 1))
        zi  = int(np.clip(round(pos[2]), 0, shape[2] - 1))
        return xi, yi, zi

    def write_to_field(self, genome: 'Genome', body: 'AgentBodyPhys',
                       field_obj: 'SigmaFieldGeometric') -> float:
        """Pump field at loci toward genome encoding.  Returns energy cost."""
        targets = self._module_fingerprint(genome)
        phi     = np.array(field_obj.phi)
        shape   = field_obj.shape
        for idx, (off, tz) in enumerate(zip(self._offsets, targets)):
            xi, yi, zi = self._locus_voxel(body.pos, off, shape)
            target_vec = np.array([np.sqrt(max(0.0, 1.0 - tz * tz)), 0.0, tz],
                                  dtype=np.float32)
            phi[xi, yi, zi] += GENOME_FIELD_STRENGTH * (target_vec - phi[xi, yi, zi])
            n = np.linalg.norm(phi[xi, yi, zi])
            if n > 1e-8:
                phi[xi, yi, zi] /= n
        field_obj.phi = jnp.array(phi)
        return GENOME_FIELD_MAINT_COST * self.total

    def genome_fidelity(self, genome: 'Genome', body: 'AgentBodyPhys',
                        field_obj: 'SigmaFieldGeometric') -> float:
        """Mean squared deviation between current field and encoded genome; mapped to [0,1]."""
        targets = self._module_fingerprint(genome)
        phi     = np.array(field_obj.phi)
        shape   = field_obj.shape
        errs    = []
        for idx, (off, tz) in enumerate(zip(self._offsets, targets)):
            xi, yi, zi = self._locus_voxel(body.pos, off, shape)
            errs.append((float(phi[xi, yi, zi, 2]) - tz) ** 2)
        return float(1.0 - np.clip(np.mean(errs), 0.0, 1.0))

    def write_offspring_genome(self, child_genome: 'Genome',
                               child_body: 'AgentBodyPhys',
                               field_obj: 'SigmaFieldGeometric') -> None:
        """Parent writes child genome fully into field at birth — initial condition."""
        targets = self._module_fingerprint(child_genome)
        phi     = np.array(field_obj.phi)
        shape   = field_obj.shape
        for idx, (off, tz) in enumerate(zip(self._offsets, targets)):
            xi, yi, zi = self._locus_voxel(child_body.pos, off, shape)
            target_vec = np.array([np.sqrt(max(0.0, 1.0 - tz * tz)), 0.0, tz],
                                  dtype=np.float32)
            phi[xi, yi, zi] = target_vec   # hard write at birth
            n = np.linalg.norm(phi[xi, yi, zi])
            if n > 1e-8:
                phi[xi, yi, zi] /= n
        field_obj.phi = jnp.array(phi)

# ─────────────────────────────────────────────────────────────────────────────
# OBJECT LAYER — slot attention + registry
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ObjectBus:
    slots:     jnp.ndarray    # (n_slots, slot_dim)
    mask:      jnp.ndarray    # (n_slots,) — persistence weights
    positions: jnp.ndarray    # (n_slots, 3)
    velocities: jnp.ndarray   # (n_slots, 3)

class SlotAttention:
    def __init__(self, num_slots: int, slot_dim: int, feature_dim: int,
                 iters: int = 3, epsilon: float = 1e-8) -> None:
        self.num_slots   = num_slots
        self.slot_dim    = slot_dim
        self.feature_dim = feature_dim
        self.iters       = iters
        self.epsilon     = epsilon

    def __call__(self, features, rng, slots_init=None):
        B, N, D = features.shape
        slots   = (slots_init if slots_init is not None else
                   jax.random.normal(rng, (B, self.num_slots, self.slot_dim)) * 0.02)
        for _ in range(self.iters):
            logits  = jnp.einsum('bsd,bnd->bsn', slots, features)
            weights = jax.nn.softmax(logits, axis=1)
            weights = weights / (weights.sum(-1, keepdims=True) + self.epsilon)
            updates = jnp.einsum('bsn,bnd->bsd', weights, features)
            slots   = slots + 0.1 * (updates - slots)
        return slots, weights

# ─────────────────────────────────────────────────────────────────────────────
# CAUSAL LEARNER (do-calculus, Granger test, interventions)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CausalBus:
    nodes:     jnp.ndarray    # (n_slots, causal_dim)
    adjacency: jnp.ndarray    # (n_slots, n_slots)

class CausalLearner:
    """
    Intervention-based causal discovery on slot embeddings.

    C[i,j] = online causal strength i→j (temporal-difference correlation)
    G[i,j] = Granger causality estimate (OLS residual variance ratio)
    Final adjacency = 0.7·C + 0.3·G

    Do-intervention: freeze one slot at its previous value with probability
    p_intervention and measure downstream change in C update.
    """

    def __init__(self, n_slots: int, cfg: 'CognitiveConfig') -> None:
        self.n   = n_slots
        self.cfg = cfg
        self.C   = np.zeros((n_slots, n_slots), dtype=np.float32)
        self._buf:    Deque[np.ndarray] = collections.deque(
            maxlen=max(3, getattr(cfg, 'granger_window', 20)))
        self._prev_z: Optional[np.ndarray] = None
        self._do_slot:  Optional[int]       = None
        self._do_value: Optional[np.ndarray] = None

    def update(self, slots: np.ndarray) -> None:
        if self._prev_z is None:
            self._prev_z = slots.copy()
            self._buf.append(slots.copy())
            return
        delta_new  = slots        - self._prev_z
        if len(self._buf) >= 2:
            delta_prev = self._buf[-1] - self._buf[-2]
        else:
            delta_prev = np.zeros_like(self._prev_z)
        dn = np.linalg.norm(delta_new,  axis=-1)
        dp = np.linalg.norm(delta_prev, axis=-1)
        lr = getattr(self.cfg, 'causal_lr', 0.05)
        for i in range(self.n):
            if dp[i] < 1e-8:
                continue
            for j in range(self.n):
                if i == j:
                    continue
                corr = float(np.dot(delta_new[j], delta_prev[i]) /
                             (np.linalg.norm(delta_new[j]) * dp[i] + 1e-8))
                self.C[i, j] += lr * (corr - self.C[i, j] * dp[i]**2)
        thr = getattr(self.cfg, 'causal_threshold', 0.1)
        self.C = np.clip(self.C, 0.0, 1.0)
        self.C[self.C < thr] = 0.0
        np.fill_diagonal(self.C, 0.0)
        self._buf.append(slots.copy())
        self._prev_z = slots.copy()

    def granger_matrix(self) -> np.ndarray:
        if len(self._buf) < 3:
            return np.zeros((self.n, self.n), dtype=np.float32)
        history = np.array(self._buf)
        z       = np.linalg.norm(history, axis=-1)
        G       = np.zeros((self.n, self.n), dtype=np.float32)
        for j in range(self.n):
            var_j = np.var(np.diff(z[:, j]))
            if var_j < 1e-8:
                continue
            for i in range(self.n):
                if i == j:
                    continue
                dj    = np.diff(z[:, j])
                di    = np.diff(z[:, i])
                beta  = np.dot(di, dj) / (np.dot(di, di) + 1e-8)
                resid = dj - beta * di
                G[i, j] = float(np.clip(
                    (var_j - np.var(resid)) / (var_j + 1e-8), 0., 1.))
        return G

    def maybe_intervene(self, slots: np.ndarray,
                        rng: np.random.Generator) -> Tuple[np.ndarray, Optional[int]]:
        p = getattr(self.cfg, 'intervention_p', 0.05)
        if rng.random() < p and self._prev_z is not None:
            idx          = int(rng.integers(0, self.n))
            modified     = slots.copy()
            modified[idx] = self._prev_z[idx]
            return modified, idx
        return slots, None

    def to_bus(self, slots: jnp.ndarray) -> CausalBus:
        G   = self.granger_matrix()
        adj = jnp.array(0.7 * self.C + 0.3 * G)
        n, d = slots.shape
        cdim = getattr(self.cfg, 'causal_dim', 64)
        if d == cdim:
            nodes = slots
        else:
            pad   = max(0, cdim - d)
            nodes = jnp.concatenate(
                [slots[:, :min(d, cdim)], jnp.zeros((n, pad))], axis=-1)
        return CausalBus(nodes=nodes, adjacency=adj)

# ─────────────────────────────────────────────────────────────────────────────
# HIERARCHICAL GRU + STOCHASTIC LATENTS
# ─────────────────────────────────────────────────────────────────────────────

def gru_cell_stochastic(x, h, p, rng=None, stochastic=True):
    z     = sigmoid(h @ p['W_z'] + x @ p['U_z'] + p['b_z'])
    r     = sigmoid(h @ p['W_r'] + x @ p['U_r'] + p['b_r'])
    h_til = jnp.tanh(r * (h @ p['W_h']) + x @ p['U_h'] + p['b_h'])
    h_new = (1 - z) * h + z * h_til
    if stochastic and 'mu_W' in p:
        mu     = h_new @ p['mu_W'] + p['mu_b']
        logvar = jnp.clip(h_new @ p['logvar_W'] + p['logvar_b'], -10.0, 2.0)
        noise  = random.normal(rng, mu.shape) if rng is not None else jnp.zeros_like(mu)
        z_lat  = mu + jnp.exp(0.5 * logvar) * noise
        kl     = -0.5 * jnp.sum(1 + logvar - mu**2 - jnp.exp(logvar))
        return z_lat, h_new, kl
    return h_new, h_new, jnp.zeros(())

def timescale_gate(h_fast, h_medium, h_slow, wm_params):
    gp   = wm_params['timescale_gate']
    feat = jnp.concatenate([h_fast, h_medium, h_slow])
    h    = jnp.tanh(feat @ gp['W1'].T + gp['b1'])
    g    = sigmoid(h @ gp['W2'].T + gp['b2'])
    return g[0], g[1]

def hierarchical_wm_step(x, h_f, h_m, h_s, t, wm_params,
                          K_m, K_s, rng, stochastic, vq_token=None):
    z_f, h_f2, kl_f = gru_cell_stochastic(x,   h_f, wm_params['fast'],   rng, stochastic)
    gm, gs           = timescale_gate(h_f2, h_m, h_s, wm_params)
    z_m, h_m2, kl_m  = gru_cell_stochastic(z_f, h_m, wm_params['medium'], rng, stochastic)
    h_m_gate         = gm * h_m2 + (1.0 - gm) * h_m
    z_s, h_s2, kl_s  = gru_cell_stochastic(z_m, h_s, wm_params['slow'],   rng, stochastic)
    h_s_gate         = gs * h_s2 + (1.0 - gs) * h_s
    concat = jnp.concatenate([z_f, z_m, z_s])
    if vq_token is not None:
        concat = jnp.concatenate([concat, vq_token])
    elif concat.shape[0] < wm_params['dec_W'].shape[0]:
        concat = jnp.pad(concat, (0, wm_params['dec_W'].shape[0] - concat.shape[0]))
    S_next    = concat @ wm_params['dec_W'] + wm_params['dec_b']
    gate_ent  = -jnp.sum(
        jnp.stack([gm, gs]) * jnp.log(jnp.stack([gm, gs]) + 1e-8) +
        (1 - jnp.stack([gm, gs])) * jnp.log(1 - jnp.stack([gm, gs]) + 1e-8))
    return S_next, h_f2, h_m_gate, h_s_gate, kl_f + gm * kl_m + gs * kl_s, gate_ent

class GRUCell:
    _KEYS = ['W_z', 'U_z', 'b_z', 'W_r', 'U_r', 'b_r', 'W_h', 'U_h', 'b_h',
             'mu_W', 'mu_b', 'logvar_W', 'logvar_b']

    def __init__(self, rng: jax.random.KeyArray, input_dim: int,
                 hidden_dim: int, stochastic: bool = True) -> None:
        keys = random.split(rng, 13)
        self.W_z   = xavier(keys[0],  (hidden_dim, hidden_dim))
        self.U_z   = xavier(keys[1],  (input_dim,  hidden_dim))
        self.b_z   = jnp.zeros(hidden_dim)
        self.W_r   = xavier(keys[2],  (hidden_dim, hidden_dim))
        self.U_r   = xavier(keys[3],  (input_dim,  hidden_dim))
        self.b_r   = jnp.zeros(hidden_dim)
        self.W_h   = xavier(keys[4],  (hidden_dim, hidden_dim))
        self.U_h   = xavier(keys[5],  (input_dim,  hidden_dim))
        self.b_h   = jnp.zeros(hidden_dim)
        self.mu_W      = xavier(keys[6],  (hidden_dim, hidden_dim), 0.3)
        self.mu_b      = jnp.zeros(hidden_dim)
        self.logvar_W  = xavier(keys[7],  (hidden_dim, hidden_dim), 0.3)
        self.logvar_b  = jnp.zeros(hidden_dim) - 2.0   # small init variance

    def to_params(self) -> dict:
        return {k: getattr(self, k) for k in self._KEYS}

    def from_params(self, p: dict) -> None:
        for k in self._KEYS:
            setattr(self, k, p[k])

class HierarchicalGRU:
    def __init__(self, rng: jax.random.KeyArray, state_dim: int,
                 latent_dim: int, output_dim: int,
                 config: TopogenesisConfig) -> None:
        keys = random.split(rng, 6)
        cog  = config.cognition
        inp_dim = state_dim + cog.time_embed_dim + cog.spatial_attn_out
        self.stochastic = cog.use_stochastic_latent
        self.fast   = GRUCell(keys[0], inp_dim,    latent_dim, self.stochastic)
        self.medium = GRUCell(keys[1], latent_dim, latent_dim, self.stochastic)
        self.slow   = GRUCell(keys[2], latent_dim, latent_dim, self.stochastic)
        cdim        = cog.concept_enc_dim if cog.concept_wm_inject else 0
        self.dec_W  = xavier(keys[3], (3 * latent_dim + cdim, output_dim))
        self.dec_b  = jnp.zeros(output_dim)
        self.K_m    = cog.K_medium
        self.K_s    = cog.K_slow
        tg_h        = cog.timescale_gate_hidden
        self.tg_W1  = xavier(keys[4], (tg_h, 3 * latent_dim))
        self.tg_b1  = jnp.zeros(tg_h)
        self.tg_W2  = xavier(keys[5], (2, tg_h))
        self.tg_b2  = jnp.zeros(2)

    def to_params(self) -> dict:
        return {
            'fast': self.fast.to_params(), 'medium': self.medium.to_params(),
            'slow': self.slow.to_params(), 'dec_W': self.dec_W, 'dec_b': self.dec_b,
            'timescale_gate': {'W1': self.tg_W1, 'b1': self.tg_b1,
                               'W2': self.tg_W2, 'b2': self.tg_b2},
        }

    def from_params(self, p: dict) -> None:
        self.fast.from_params(p['fast'])
        self.medium.from_params(p['medium'])
        self.slow.from_params(p['slow'])
        self.dec_W, self.dec_b = p['dec_W'], p['dec_b']
        gp = p['timescale_gate']
        self.tg_W1, self.tg_b1 = gp['W1'], gp['b1']
        self.tg_W2, self.tg_b2 = gp['W2'], gp['b2']

    def step(self, x, h_f, h_m, h_s, t, rng, vq_token=None):
        return hierarchical_wm_step(
            x, h_f, h_m, h_s, t, self.to_params(),
            self.K_m, self.K_s, rng, self.stochastic, vq_token)

# ─────────────────────────────────────────────────────────────────────────────
# INFLUENCE TENSOR + ANDERSON SOLVER
# ─────────────────────────────────────────────────────────────────────────────

def init_A_params(rng: jax.random.KeyArray, state_dim: int,
                  d_I: int, rank: int) -> dict:
    k1, k2 = random.split(rng)
    return {
        'U': spectral_normalize(xavier(k1, (d_I, rank), 0.5), 0.9),
        'V': spectral_normalize(xavier(k2, (state_dim, rank), 0.5), 0.9),
    }

@jit
def apply_A(x: jnp.ndarray, A_params: dict) -> jnp.ndarray:
    return x @ A_params['V'] @ A_params['U'].T

def _anderson_mix(FX, R, ridge):
    m    = R.shape[0]
    gram = R @ R.T + ridge * jnp.eye(m)
    ones = jnp.ones((m, 1))
    kkt  = jnp.block([[gram, ones], [ones.T, jnp.zeros((1, 1))]])
    rhs  = jnp.concatenate([jnp.zeros(m), jnp.ones(1)])
    return jnp.linalg.solve(kkt, rhs)[:m] @ FX

@partial(jit, static_argnums=(0, 3, 4, 5, 6))
def anderson_solver(F, x0, args, max_iter, tol, memory, ridge, damping=0.5):
    dim = x0.shape[0]
    X   = jnp.zeros((memory, dim))
    R   = jnp.zeros((memory, dim))
    FX  = jnp.zeros((memory, dim))
    fx0 = F(x0, *args)
    r0  = fx0 - x0
    X   = X.at[0].set(x0)
    R   = R.at[0].set(r0)
    FX  = FX.at[0].set(fx0)
    conv = jnp.linalg.norm(r0) < tol

    def body(carry, i):
        x, X, R, FX, converged, steps = carry
        fx     = F(x, *args)
        r      = fx - x
        r_norm = jnp.linalg.norm(r)
        ptr    = i % memory
        X      = X.at[ptr].set(x)
        R      = R.at[ptr].set(r)
        FX     = FX.at[ptr].set(fx)
        cand   = lax.cond(i > 0, lambda: _anderson_mix(FX, R, ridge), lambda: fx)
        new_c  = r_norm < tol
        x_new  = jnp.where(converged, x, damping * x + (1 - damping) * cand)
        return (x_new, X, R, FX, converged | new_c,
                jnp.where(converged, steps, i + 1)), None

    (x_final, _, _, _, _, steps), _ = lax.scan(
        body, (x0, X, R, FX, conv, jnp.array(0, dtype=jnp.int32)),
        jnp.arange(max_iter))
    return x_final, steps

# ─────────────────────────────────────────────────────────────────────────────
# AUTOREGRESSIVE ROLLOUT
# ─────────────────────────────────────────────────────────────────────────────

class StateSpace:
    def __init__(self, d_E: int, d_D: int, d_I: int) -> None:
        self.d_E = d_E
        self.d_D = d_D
        self.d_I = d_I
        self.total_dim = d_E + d_D + d_I

    def decompose(self, S: jnp.ndarray):
        return S[:self.d_E], S[self.d_E:self.d_E+self.d_D], S[self.d_E+self.d_D:]

    def assemble(self, E, D, I) -> jnp.ndarray:
        return jnp.concatenate([E, D, I])

@partial(jit, static_argnums=(1, 3, 4, 5, 6, 8))
def autoregressive_rollout(S0, horizon, wm_params, config, dt, K_m, K_s,
                           rng, stochastic=True):
    latent_dim = config.latent_dim
    h0 = jnp.zeros(latent_dim)

    def step(carry, t):
        S, h_f, h_m, h_s, key = carry
        key, subkey = random.split(key)
        t_enc = get_time_encoding(jnp.array([t * dt]),
                                  jnp.array([10., 50., 200., 1000.]),
                                  config.cognition.time_embed_dim)
        field_ctx = S[FIELD_OBS_START:FIELD_OBS_START + config.cognition.spatial_attn_out]
        x = jnp.concatenate([S, t_enc, field_ctx])
        S_next, h_f2, h_m2, h_s2, kl, ge = hierarchical_wm_step(
            x, h_f, h_m, h_s, t, wm_params,
            K_m, K_s, subkey, stochastic, None)
        S_next = jnp.clip(S_next, -5.0, 5.0)
        return (S_next, h_f2, h_m2, h_s2, key), (S_next, kl, ge)

    S0c = jnp.clip(S0, -5.0, 5.0)
    (_, _, _, _, _), (traj, kls, gate_entropies) = lax.scan(
        step, (S0c, h0, h0, h0, rng), jnp.arange(horizon))
    return jnp.concatenate([S0c[None], traj], axis=0), jnp.sum(kls), jnp.mean(gate_entropies)

@partial(jit, static_argnums=(5, 6, 7, 8))
def compute_guidance(S, S_dagger, wm_params, A_params,
                     lambda_, tau_max, dt, K_m, K_s, rng, config):
    rollout, _, _ = autoregressive_rollout(
        S, tau_max, wm_params, config, dt, K_m, K_s, rng, stochastic=False)
    delta    = S_dagger - rollout
    tau      = jnp.arange(tau_max + 1) * dt
    integral = jnp.sum(delta * jnp.exp(-lambda_ * tau)[:, None], axis=0) * dt
    return apply_A(integral, A_params)

# ─────────────────────────────────────────────────────────────────────────────
# AFFECT ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def init_affect_params(rng: jax.random.KeyArray,
                       state_dim: int,
                       config: TopogenesisConfig) -> dict:
    k1, k2, k3, k4, k5 = random.split(rng, 5)
    aff = config.affect
    return {
        'W_valence':           xavier(k1, (aff.valence_dim, state_dim)),
        'W_arousal':           xavier(k2, (aff.arousal_dim, state_dim)),
        'W_gate':              xavier(k3, (aff.valence_dim, aff.arousal_dim)),
        'emotion_to_valence':  xavier(k4, (aff.valence_dim,
                                           config.cognition.n_drives), 0.3),
        'emotion_to_distress': xavier(k5, (1, config.cognition.n_drives), 0.3),
    }

@partial(jit, static_argnums=(6,))
def compute_affect(S, prediction_error, homeostasis_deviation, affect_params,
                   previous_affect_state, drive_vec, config):
    raw_val = S @ affect_params['W_valence'].T + drive_vec @ affect_params['emotion_to_valence'].T
    valence = jnp.tanh(raw_val - prediction_error)
    arousal = jnp.abs(S @ affect_params['W_arousal'].T) * sigmoid(
        affect_params['W_gate'] @ valence)
    distress_mod = jnp.tanh(
        drive_vec @ affect_params['emotion_to_distress'].T).squeeze()
    distress = -jnp.tanh(homeostasis_deviation * 5.0) * (1.0 + distress_mod) * 0.8
    new_aff  = (config.affect.affect_decay * previous_affect_state
                + (1 - config.affect.affect_decay) * (valence + distress))
    new_aff  = new_aff / jnp.maximum(jnp.linalg.norm(new_aff), 0.5)
    return valence, arousal, distress, new_aff

# ─────────────────────────────────────────────────────────────────────────────
# GAUSSIAN POLICY + CRITIC + RELATIONAL NET
# ─────────────────────────────────────────────────────────────────────────────

class GaussianPolicy:
    _KEYS = ('W1', 'b1', 'mean_W', 'mean_b', 'logstd_W', 'logstd_b')

    def __init__(self, rng: jax.random.KeyArray, latent_dim: int,
                 action_dim: int, config: TopogenesisConfig) -> None:
        hidden = config.cognition.policy_net_hidden
        k      = random.split(rng, 4)
        self.W1       = xavier(k[0], (hidden, latent_dim))
        self.b1       = jnp.zeros(hidden)
        self.mean_W   = xavier(k[1], (action_dim, hidden), 0.3)
        self.mean_b   = jnp.zeros(action_dim)
        self.logstd_W = xavier(k[2], (action_dim, hidden), 0.3)
        self.logstd_b = jnp.zeros(action_dim) - 1.0

    def to_params(self) -> dict:
        return {k: getattr(self, k) for k in self._KEYS}

    def from_params(self, p: dict) -> None:
        for k in self._KEYS:
            setattr(self, k, p[k])

    @staticmethod
    @jit
    def sample_and_log_prob(latent, rng, params):
        h      = jnp.tanh(latent @ params['W1'].T + params['b1'])
        mean   = h @ params['mean_W'].T + params['mean_b']
        logstd = jnp.clip(h @ params['logstd_W'].T + params['logstd_b'], -3.0, 1.0)
        action = mean + jnp.exp(logstd) * random.normal(rng, mean.shape)
        lp     = -0.5 * jnp.sum(
            ((action - mean) / (jnp.exp(logstd) + 1e-8))**2
            + 2 * logstd + jnp.log(2 * jnp.pi))
        entropy = 0.5 * jnp.sum(1 + logstd + jnp.log(2 * jnp.pi))
        return action, lp, entropy

@jit
def gaussian_policy_log_prob_entropy(latent, action, params):
    h      = jnp.tanh(latent @ params['W1'].T + params['b1'])
    mean   = h @ params['mean_W'].T + params['mean_b']
    logstd = jnp.clip(h @ params['logstd_W'].T + params['logstd_b'], -3.0, 1.0)
    lp     = -0.5 * jnp.sum(
        ((action - mean) / (jnp.exp(logstd) + 1e-8))**2
        + 2 * logstd + jnp.log(2 * jnp.pi))
    entropy = 0.5 * jnp.sum(1 + logstd + jnp.log(2 * jnp.pi))
    return lp, entropy

@partial(jit, static_argnums=(4,))
def gaussian_policy_online_loss(params, latent, action, advantage, config):
    lp, entropy = gaussian_policy_log_prob_entropy(latent, action, params)
    return -lax.stop_gradient(advantage) * lp - config.w_entropy * entropy

def init_critic_params(rng: jax.random.KeyArray, state_dim: int,
                       hidden: int = 64) -> dict:
    k1, k2 = random.split(rng)
    return {
        'W1': xavier(k1, (hidden, state_dim)),
        'b1': jnp.zeros(hidden),
        'W2': xavier(k2, (1, hidden), 0.3),
        'b2': jnp.zeros(1),
    }

def critic_forward(S: jnp.ndarray, params: dict) -> jnp.ndarray:
    h = jnp.tanh(S @ params['W1'].T + params['b1'])
    return (h @ params['W2'].T + params['b2']).squeeze(-1)

def init_sensorimotor_params(rng: jax.random.KeyArray,
                             state_dim: int,
                             action_dim: int,
                             hidden: int) -> dict:
    k1, k2, k3 = random.split(rng, 3)
    in_dim = state_dim + action_dim
    return {
        'W1': xavier(k1, (hidden, in_dim)),
        'b1': jnp.zeros(hidden),
        'W2': xavier(k2, (state_dim, hidden), 0.05),
        'b2': jnp.zeros(state_dim),
        'skip': xavier(k3, (state_dim, action_dim), 0.01),
    }

@jit
def sensorimotor_predict(params: dict,
                         S: jnp.ndarray,
                         action: jnp.ndarray) -> jnp.ndarray:
    x = jnp.concatenate([S, action])
    h = jnp.tanh(x @ params['W1'].T + params['b1'])
    delta = h @ params['W2'].T + params['b2'] + action @ params['skip'].T
    return jnp.clip(S + delta, -5.0, 5.0)

@jit
def sensorimotor_loss(params: dict,
                      S_prev: jnp.ndarray,
                      action_prev: jnp.ndarray,
                      S_now: jnp.ndarray):
    pred = sensorimotor_predict(params, S_prev, action_prev)
    err = pred - S_now
    loss = jnp.mean(err ** 2)
    return loss, {'mse': loss}

@partial(jit, static_argnums=(7,))
def wm_online_loss(wm_params: dict,
                   x_prev: jnp.ndarray,
                   h_f_prev: jnp.ndarray,
                   h_m_prev: jnp.ndarray,
                   h_s_prev: jnp.ndarray,
                   t_prev: int,
                   S_now: jnp.ndarray,
                   config: TopogenesisConfig):
    pred, _, _, _, kl, gate_ent = hierarchical_wm_step(
        x_prev, h_f_prev, h_m_prev, h_s_prev, t_prev,
        wm_params, config.cognition.K_medium, config.cognition.K_slow,
        jax.random.PRNGKey(0), False, None)
    pred = jnp.clip(pred, -5.0, 5.0)
    loss = jnp.mean((pred - S_now) ** 2) + config.cognition.latent_kl_weight * kl
    return loss, {'mse': jnp.mean((pred - S_now) ** 2),
                  'gate_entropy': gate_ent}

def init_enactive_ac_params(rng: jax.random.KeyArray,
                            feature_dim: int,
                            action_dim: int) -> dict:
    k1, k2 = random.split(rng)
    return {
        'actor_W': xavier(k1, (action_dim, feature_dim), 0.05),
        'actor_b': jnp.zeros(action_dim),
        'critic_W': xavier(k2, (feature_dim,), 0.05),
        'critic_b': jnp.zeros(()),
    }

def enactive_ac_mean(params: dict, feat: jnp.ndarray) -> jnp.ndarray:
    return jnp.tanh(params['actor_W'] @ feat + params['actor_b'])

def enactive_ac_value(params: dict, feat: jnp.ndarray) -> jnp.ndarray:
    return feat @ params['critic_W'] + params['critic_b']

@partial(jit, static_argnums=(5,))
def enactive_ac_loss(params: dict,
                     feat_prev: jnp.ndarray,
                     action_prev: jnp.ndarray,
                     reward: jnp.ndarray,
                     feat_now: jnp.ndarray,
                     config: TopogenesisConfig):
    sigma = config.cognition.enactive_action_sigma
    gamma = config.cognition.enactive_discount
    v_prev = enactive_ac_value(params, feat_prev)
    v_now = enactive_ac_value(params, feat_now)
    target = reward + gamma * lax.stop_gradient(v_now)
    td = target - v_prev
    mean = enactive_ac_mean(params, feat_prev)
    logp = -0.5 * jnp.sum(((action_prev - mean) / sigma) ** 2
                          + 2 * jnp.log(sigma)
                          + jnp.log(2 * jnp.pi))
    critic_loss = 0.5 * td ** 2
    actor_loss = -lax.stop_gradient(td) * logp
    action_reg = 1e-3 * jnp.mean(mean ** 2)
    loss = critic_loss + actor_loss + action_reg
    return loss, {'td': td, 'value': v_prev, 'mean_norm': jnp.linalg.norm(mean)}

class RelationalReasoningNet:
    _KEYS = ('W_a', 'b_a', 'W1', 'b1', 'W2', 'b2', 'W_res')

    def __init__(self, rng: jax.random.KeyArray, concept_dim: int,
                 action_dim: int, proj_dim: int, hidden_dim: int) -> None:
        k = random.split(rng, 7)
        self.W_a  = xavier(k[0], (proj_dim, action_dim))
        self.b_a  = jnp.zeros(proj_dim)
        self.W1   = xavier(k[1], (hidden_dim, 2 * concept_dim + proj_dim))
        self.b1   = jnp.zeros(hidden_dim)
        self.W2   = xavier(k[2], (concept_dim, hidden_dim), 0.3)
        self.b2   = jnp.zeros(concept_dim)
        self.W_res = xavier(k[3], (concept_dim, concept_dim), 0.1)

    def to_params(self) -> dict:
        return {k: getattr(self, k) for k in self._KEYS}

    def from_params(self, p: dict) -> None:
        for k in self._KEYS:
            setattr(self, k, p[k])

# ─────────────────────────────────────────────────────────────────────────────
# GOAL MANAGER + DRIVE SYSTEM + GOAL NET
# ─────────────────────────────────────────────────────────────────────────────

class GoalManager:
    def __init__(self) -> None:
        self.goals = {
            'stabilize_body':            1.0,
            'improve_prediction':        1.0,
            'preserve_self_continuity':  0.6,
            'explore_uncertainty':       0.5,
            'maintain_workspace_focus':  0.35,
        }

    def priorities(self, metrics: dict) -> dict:
        p      = dict(self.goals)
        wm_mse = float(metrics.get('wm_mse', 0))
        homeo  = float(metrics.get('homeostasis_deviation', 0))
        p['stabilize_body']           *= 1 + homeo
        p['improve_prediction']       *= 1 + wm_mse
        p['explore_uncertainty']      *= 1 + min(wm_mse, 2)
        return p

class DriveSystem:
    def __init__(self, config: TopogenesisConfig) -> None:
        self.drives = jnp.ones(config.cognition.n_drives) * 0.5
        self.decay  = config.cognition.drive_decay

    def update(self, obs: dict, reward: float,
               curiosity_signal: float, wm_mse: float) -> jnp.ndarray:
        targets = jnp.array([
            1 - obs.get('energy',       0.5),
            1 - min(obs.get('health', 0.5), obs.get('membrane', 1.0)),
            1 - obs.get('inventory',    0),
            curiosity_signal,
            1 - obs.get('hazard_dist',  1),
            float(jnp.exp(-wm_mse)),
        ])
        self.drives = jnp.clip(
            self.decay * self.drives + (1 - self.decay) * jnp.clip(targets, 0, 2),
            0.0, 2.0)
        return self.drives

def init_goal_net_params(rng: jax.random.KeyArray, E_dim: int,
                         n_drives: int, hidden: int,
                         concept_dim: int = 0, field_dim: int = 0) -> dict:
    k1, k2, k3 = random.split(rng, 3)
    in_dim = E_dim + n_drives + concept_dim + field_dim
    p = {
        'W1': xavier(k1, (hidden, in_dim)),
        'b1': jnp.zeros(hidden),
        'W2': xavier(k2, (E_dim, hidden), 0.3),
        'b2': jnp.zeros(E_dim),
    }
    if concept_dim > 0:
        p['W_concept_res'] = xavier(k3, (E_dim, concept_dim), 0.1)
    return p

@jit
def goal_net_predict(S_E, drives, params,
                     concept_ctx=None, field_feat=None) -> jnp.ndarray:
    inp = [S_E, drives]
    if concept_ctx is not None:
        inp.append(concept_ctx)
    if field_feat is not None:
        inp.append(field_feat)
    x   = jnp.concatenate(inp)
    h   = jnp.tanh(x @ params['W1'].T + params['b1'])
    out = jnp.tanh(h @ params['W2'].T + params['b2'])
    if concept_ctx is not None and 'W_concept_res' in params:
        out = out + 0.1 * jnp.tanh(concept_ctx @ params['W_concept_res'].T)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL WORKSPACE
# ─────────────────────────────────────────────────────────────────────────────

def init_workspace_params(rng: jax.random.KeyArray, state_dim: int,
                          d_D: int, config: TopogenesisConfig) -> dict:
    k1, k2 = random.split(rng)
    wdim = config.cognition.workspace_dim
    return {
        'W_in':        xavier(k1, (wdim, state_dim)),
        'affect_bias': xavier(k2, (wdim, config.affect.valence_dim), 0.1),
    }

@partial(jit, static_argnums=(4,))
def update_global_workspace(S, workspace_state, workspace_params,
                             affect_state, config):
    candidate    = jnp.tanh(S @ workspace_params['W_in'].T)
    salience_b   = workspace_params['affect_bias'] @ affect_state
    arousal_norm = jnp.linalg.norm(affect_state) / jnp.sqrt(affect_state.shape[0] + 1e-8)
    temperature  = jnp.maximum(
        config.cognition.workspace_temperature_base * (1.0 + arousal_norm * 0.5),
        0.1)
    logits    = (jnp.abs(candidate) + salience_b) / (temperature + 1e-8)
    salience  = jax.nn.softmax(logits)
    broadcast = salience * candidate
    new_state = (config.cognition.workspace_decay * workspace_state
                 + (1 - config.cognition.workspace_decay) * broadcast)
    return new_state, broadcast, jnp.max(salience), -jnp.sum(salience * jnp.log(salience + 1e-8))

# ─────────────────────────────────────────────────────────────────────────────
# SPARSE MODULAR MEMORY
# ─────────────────────────────────────────────────────────────────────────────

class LSHTable:
    def __init__(self, dim: int, n_bits: int, n_tables: int, seed: int = 0) -> None:
        rng         = np.random.default_rng(seed)
        self.planes = [rng.standard_normal((n_bits, dim)).astype(np.float32)
                       for _ in range(n_tables)]
        self.tables  = [defaultdict(list) for _ in range(n_tables)]
        self.entries: Dict[int, dict] = {}
        self._nid    = 0

    def _hash(self, v: np.ndarray, plane: np.ndarray) -> tuple:
        return tuple((v @ plane.T > 0).astype(np.int8).tolist())

    def add(self, v: np.ndarray, payload: dict) -> int:
        eid   = self._nid; self._nid += 1
        entry = {**payload, '_v': v.copy(), '_id': eid}
        self.entries[eid] = entry
        for plane, table in zip(self.planes, self.tables):
            table[self._hash(v, plane)].append(eid)
        return eid

    def query(self, v: np.ndarray, k: int = 8) -> list:
        candidates = set()
        for plane, table in zip(self.planes, self.tables):
            candidates.update(table.get(self._hash(v, plane), []))
        live = [c for c in candidates if c in self.entries]
        if not live:
            return []
        vecs = np.stack([self.entries[c]['_v'] for c in live])
        sims = vecs @ v / (np.linalg.norm(vecs, axis=1) * (np.linalg.norm(v) + 1e-8) + 1e-8)
        return [self.entries[live[i]] for i in np.argsort(sims)[-k:][::-1]]

    def remove(self, eid: int) -> None:
        if eid not in self.entries:
            return
        v = self.entries[eid]['_v']
        for plane, table in zip(self.planes, self.tables):
            h = self._hash(v, plane)
            if eid in table.get(h, []):
                table[h].remove(eid)
        del self.entries[eid]

class ModuleRouter:
    def __init__(self, n_modules: int, module_dim: int,
                 input_dim: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.n_modules  = n_modules
        proj = rng.standard_normal((input_dim, module_dim)).astype(np.float32)
        self.proj       = proj / (np.linalg.norm(proj, axis=0, keepdims=True) + 1e-8)
        proto = rng.standard_normal((n_modules, module_dim)).astype(np.float32)
        self.prototypes = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-8)
        self._lr        = 0.01

    def route(self, S: np.ndarray, k_active: int = 2) -> jnp.ndarray:
        d    = min(len(S), self.proj.shape[0])
        v    = np.array(S[:d])
        proj = v @ self.proj[:d]
        proj /= np.linalg.norm(proj) + 1e-8
        sims = self.prototypes @ proj
        mask = np.zeros(self.n_modules)
        top  = np.argsort(sims)[-k_active:]
        mask[top] = np.maximum(sims[top], 0.)
        mask /= (mask.sum() + 1e-8)
        return jnp.array(mask, dtype=jnp.float32)

    def update(self, S: np.ndarray, mask: np.ndarray) -> None:
        d    = min(len(S), self.proj.shape[0])
        v    = np.array(S[:d])
        proj = v @ self.proj[:d]
        proj /= np.linalg.norm(proj) + 1e-8
        for i in range(self.n_modules):
            if float(mask[i]) > 1e-6:
                self.prototypes[i] += self._lr * float(mask[i]) * (proj - self.prototypes[i])
                self.prototypes[i] /= np.linalg.norm(self.prototypes[i]) + 1e-8

class UncertaintyMap:
    def __init__(self, dim: int, n_bins: int = 64,
                 ema: float = 0.05, seed: int = 0) -> None:
        rng        = np.random.default_rng(seed)
        d          = min(dim, 128)
        self.proj  = rng.standard_normal(
            (d, int(np.ceil(np.log2(n_bins + 1))))).astype(np.float32)
        self.n_bins    = n_bins
        self.ema       = ema
        self.error_map = np.ones(n_bins, dtype=np.float32) * 0.5
        self.count_map = np.zeros(n_bins, dtype=np.int32)

    def _bin(self, S: np.ndarray) -> int:
        d    = min(len(S), self.proj.shape[0])
        bits = (S[:d] @ self.proj[:d] > 0).astype(np.int32)
        return int(np.sum(bits * (2**np.arange(len(bits))))) % self.n_bins

    def update(self, S: np.ndarray, err: float) -> None:
        b = self._bin(S)
        self.error_map[b] = (1 - self.ema) * self.error_map[b] + self.ema * abs(err)
        self.count_map[b] += 1

    def get(self, S: np.ndarray) -> float:
        return float(self.error_map[self._bin(S)])

class ConceptRegistry:
    MERGE_THRESHOLD     = 0.85
    DIVERSITY_THRESHOLD = 0.90

    def __init__(self, dim: int, capacity: int = 64, seed: int = 0) -> None:
        rng                  = np.random.default_rng(seed)
        self.dim             = dim
        self.capacity        = capacity
        self.prototypes      = np.zeros((capacity, dim), dtype=np.float32)
        self.anchors         = np.zeros((capacity, dim), dtype=np.float32)
        self.l1_prototypes   = np.zeros((capacity, dim), dtype=np.float32)
        self.counts          = np.zeros(capacity, dtype=np.int32)
        self.l1_counts       = np.zeros(capacity, dtype=np.int32)
        self.n_concepts      = 0
        self.n_l1            = 0
        self.l1_src_pairs    = [None] * capacity
        self.relation_W      = rng.standard_normal((dim, dim * 2)).astype(np.float32) * 0.02

    def _cosine_sims(self, vec: np.ndarray) -> np.ndarray:
        if self.n_concepts == 0:
            return np.array([])
        live  = self.prototypes[:self.n_concepts]
        norms = np.linalg.norm(live, axis=1) * (np.linalg.norm(vec) + 1e-8) + 1e-8
        return live @ vec / norms

    def add_or_update(self, vec: np.ndarray, ema: float = 0.05) -> int:
        sims = self._cosine_sims(vec)
        if len(sims) > 0:
            best = int(np.argmax(sims))
            if sims[best] >= self.MERGE_THRESHOLD:
                self.prototypes[best] = (1 - ema) * self.prototypes[best] + ema * vec
                self.prototypes[best] += 0.01 * (self.anchors[best] - self.prototypes[best])
                self.counts[best]     += 1
                return best
            if sims[best] >= self.DIVERSITY_THRESHOLD:
                return best
        if self.n_concepts >= self.capacity:
            evict = int(np.argmin(self.counts[:self.n_concepts]))
            self.prototypes[evict] = vec
            self.anchors[evict]    = vec.copy()
            self.counts[evict]     = 1
            return evict
        idx = self.n_concepts
        self.prototypes[idx] = vec
        self.anchors[idx]    = vec.copy()
        self.counts[idx]     = 1
        self.n_concepts      += 1
        return idx

    def lookup(self, vec: np.ndarray, k: int = 3) -> Tuple[list, list]:
        if self.n_concepts == 0:
            return [], []
        live = self.prototypes[:self.n_concepts]
        sims = live @ vec / (
            np.linalg.norm(live, axis=1) * (np.linalg.norm(vec) + 1e-8) + 1e-8)
        top  = np.argsort(sims)[-k:][::-1]
        return top.tolist(), sims[top].tolist()

class RelationalGraph:
    def __init__(self, capacity: int = 64) -> None:
        self.capacity    = capacity
        self.co_occur_l0 = np.zeros((capacity, capacity), dtype=np.int32)

    def record(self, top_ids: list) -> None:
        for i in range(len(top_ids)):
            for j in range(i + 1, len(top_ids)):
                a, b = int(top_ids[i]), int(top_ids[j])
                if a < self.capacity and b < self.capacity:
                    self.co_occur_l0[a, b] += 1
                    self.co_occur_l0[b, a] += 1

    def promote_all(self, registry: ConceptRegistry, thr: int = 8) -> None:
        n = registry.n_concepts
        if n < 2:
            return
        for a, b in np.argwhere(self.co_occur_l0[:n, :n] >= thr):
            if a >= b:
                continue
            composed = np.tanh(
                np.concatenate([registry.prototypes[a], registry.prototypes[b]])
                @ registry.relation_W.T)
            registry.add_or_update(composed)
            self.co_occur_l0[a, b] = 0
            self.co_occur_l0[b, a] = 0

class SparseModularMemory:
    def __init__(self, config: TopogenesisConfig, state_dim: int,
                 rng: jax.random.KeyArray) -> None:
        cog              = config.cognition
        self.config      = config
        self.state_dim   = state_dim
        self.step_count  = 0
        _sdim            = min(state_dim, 256)
        seed             = int(jax.random.randint(rng, (), 0, 100000))
        self.lsh         = LSHTable(_sdim, cog.smm_hash_bits, cog.smm_n_tables, seed)
        self.episodic:   List[dict]  = []
        self.semantic:   List[dict]  = []
        self.router      = ModuleRouter(cog.smm_n_modules, cog.smm_module_dim,
                                        min(state_dim, 512))
        self.unc_map     = UncertaintyMap(_sdim, cog.smm_uncertainty_bins)
        self.concept_reg = ConceptRegistry(_sdim)
        self.rel_graph   = RelationalGraph(capacity=self.concept_reg.capacity)
        self.hard_cap    = cog.smm_episodic_capacity

    def add(self, S: np.ndarray, S_next: np.ndarray, reward: float,
            prediction_error: float = 0.0, action=None,
            affect_state=None) -> None:
        self.step_count += 1
        sv = np.array(S[:min(self.state_dim, 256)], dtype=np.float32)
        if np.any(np.isnan(sv)) or np.linalg.norm(sv) > 1e4:
            return
        base_surprise = float(np.clip(abs(prediction_error), 0., 1.))
        if affect_state is not None:
            arousal = float(np.linalg.norm(np.array(affect_state))) / (
                np.sqrt(len(np.array(affect_state))) + 1e-8)
            boost = 1.0 + self.config.cognition.affect_consolidation_boost * np.clip(arousal, 0., 1.)
        else:
            boost = 1.0
        surprise = float(np.clip(base_surprise * boost, 0., 1.))
        aidx     = (int(np.argmax(np.array(action)))
                    if action is not None and hasattr(action, '__len__')
                    else int(action) if action is not None else None)
        action_vec = (np.array(action, dtype=np.float32).copy()
                      if action is not None and hasattr(action, '__len__')
                      else np.zeros(MAX_MOTORS, dtype=np.float32))
        entry  = {'S': S.copy(), 'S_next': S_next.copy(),
                  'reward': float(reward), 'surprise': surprise,
                  'action': aidx, 'action_vec': action_vec,
                  'timestamp': self.step_count, 'access_count': 0}
        entry['_eid'] = self.lsh.add(sv, entry)
        self.episodic.append(entry)
        self.unc_map.update(sv, prediction_error)
        top_ids, _ = self.concept_reg.lookup(sv, k=3)
        if top_ids:
            self.rel_graph.record(top_ids)
        if self.step_count % 50 == 0:
            self._forget()
        if self.step_count % self.config.cognition.smm_consolidation_interval == 0:
            self.consolidate()

    def _forget(self) -> None:
        while len(self.episodic) > self.hard_cap:
            scores   = np.array([
                e.get('surprise', 0.) + 0.001 * e.get('access_count', 0)
                for e in self.episodic])
            worst    = int(np.argmin(scores))
            entry    = self.episodic.pop(worst)
            if '_eid' in entry:
                self.lsh.remove(entry['_eid'])

    def consolidate(self, n_cycles: int = 1) -> None:
        if len(self.episodic) < 8:
            return
        cap = self.config.cognition.smm_semantic_capacity
        for _ in range(n_cycles):
            surprises = np.array([e.get('surprise', 0.1) + 0.1 for e in self.episodic])
            probs     = (surprises + 0.1) / (surprises + 0.1).sum()
            k         = min(8, len(self.episodic))
            idxs      = np.random.choice(len(self.episodic), k, replace=False, p=probs)
            proto     = np.mean([self.episodic[i]['S'] for i in idxs], axis=0).astype(np.float32)
            pnext     = np.mean([self.episodic[i]['S_next'] for i in idxs], axis=0).astype(np.float32)
            weight    = float(np.mean([self.episodic[i]['surprise'] for i in idxs]))
            rec       = {'S': proto, 'S_next': pnext, 'weight': weight}
            if self.semantic:
                exist = np.stack([s['S'][:len(proto)] for s in self.semantic])
                sims  = exist @ proto / (
                    np.linalg.norm(exist, axis=1) * (np.linalg.norm(proto) + 1e-8) + 1e-8)
                if float(np.max(sims)) > 0.92:
                    continue
            if len(self.semantic) >= cap:
                self.semantic[int(np.argmin([s['weight'] for s in self.semantic]))] = rec
            else:
                self.semantic.append(rec)
            pv = proto[:min(len(proto), self.concept_reg.dim)]
            self.concept_reg.add_or_update(pv)
        self.rel_graph.promote_all(self.concept_reg,
                                   thr=self.config.cognition.schema_cooccur_threshold)

    def retrieve_context(self, S: np.ndarray, k: int = 4) -> jnp.ndarray:
        sv      = np.array(S[:min(self.state_dim, 256)], dtype=np.float32)
        matches = self.lsh.query(sv, k=k)
        if not matches and self.semantic:
            protos = np.stack([s['S'][:len(sv)] for s in self.semantic])
            dists  = np.linalg.norm(protos - sv, axis=1)
            return jnp.array(self.semantic[int(np.argmin(dists))]['S'], dtype=jnp.float32)
        if not matches:
            return jnp.zeros(self.state_dim)
        t  = self.step_count
        w  = np.array([np.exp(-0.001 * (t - m.get('timestamp', 0))) *
                       (1. + m.get('surprise', 0.)) for m in matches], dtype=np.float32)
        w /= w.sum() + 1e-8
        return jnp.array(sum(wi * m['S'] for wi, m in zip(w, matches)), dtype=jnp.float32)

    def retrieve_action_prior(self, S: np.ndarray, k: int = 8) -> jnp.ndarray:
        sv = np.array(S[:min(self.state_dim, 256)], dtype=np.float32)
        matches = [m for m in self.lsh.query(sv, k=k) if 'action_vec' in m]
        if not matches:
            return jnp.zeros(MAX_MOTORS)
        rewards = np.array([m.get('reward', 0.0) for m in matches], dtype=np.float32)
        ages = np.array([self.step_count - m.get('timestamp', 0) for m in matches], dtype=np.float32)
        weights = np.exp(-0.001 * ages) * np.maximum(rewards, 0.0)
        if float(weights.sum()) <= 1e-8:
            return jnp.zeros(MAX_MOTORS)
        weights /= weights.sum() + 1e-8
        prior = sum(float(w) * np.array(m['action_vec'], dtype=np.float32)
                    for w, m in zip(weights, matches))
        return jnp.array(np.clip(prior, -3.0, 3.0), dtype=jnp.float32)

    def uncertainty(self, S: np.ndarray) -> float:
        return self.unc_map.get(np.array(S[:256], dtype=np.float32))

    def sample_batch(self, batch_size: int):
        pool = self.episodic
        if not pool:
            raise ValueError("Empty memory")
        idxs = np.random.choice(len(pool), batch_size,
                                 replace=len(pool) < batch_size)
        S     = jnp.stack([jnp.array(pool[i]['S'])      for i in idxs])
        S_next = jnp.stack([jnp.array(pool[i]['S_next']) for i in idxs])
        rewards = jnp.array([pool[i]['reward']           for i in idxs])
        return S, S_next, rewards

# ─────────────────────────────────────────────────────────────────────────────
# HRR CLEANUP MEMORY
# ─────────────────────────────────────────────────────────────────────────────

class HRRCleanupMemory:
    """
    Nearest-neighbour cleanup memory for HRR unbinding.

    All roles and fillers are stored as unit vectors.  After unbinding
    produces a noisy vector, cleanup() snaps it to the closest codebook
    entry by cosine similarity, preventing SNR collapse at depth > 1.
    """

    def __init__(self, d: int) -> None:
        self.d         = d
        self._codebook: Dict[str, np.ndarray] = {}

    def store(self, label: str, v: np.ndarray) -> None:
        norm = np.linalg.norm(v)
        if norm > 1e-8:
            self._codebook[label] = (v / norm).astype(np.float32)

    def cleanup(self, v: np.ndarray) -> np.ndarray:
        if not self._codebook:
            n = np.linalg.norm(v)
            return v / (n + 1e-8)
        keys   = list(self._codebook.keys())
        matrix = np.stack([self._codebook[k] for k in keys], axis=0)
        v_norm = v / (np.linalg.norm(v) + 1e-8)
        sims   = matrix @ v_norm
        return self._codebook[keys[int(np.argmax(sims))]].copy()

# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITIONAL SYMBOLIC SYSTEM (HRR)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SymbolicBus:
    structure:  jnp.ndarray
    n_bound:    int
    retrieval_q: float

class RecursiveBindingStack:
    def __init__(self, d: int, max_depth: int = 4) -> None:
        self.d         = d
        self.max_depth = max_depth
        self._stack:   List[Tuple[str, np.ndarray]] = []

    def push(self, scope: str, structure: np.ndarray) -> None:
        if len(self._stack) < self.max_depth:
            self._stack.append((scope, structure.copy()))

    def pop(self) -> Optional[Tuple[str, np.ndarray]]:
        return self._stack.pop() if self._stack else None

    @property
    def depth(self) -> int:
        return len(self._stack)

class CompositionalSymbolicSystem:
    """
    Holographic Reduced Representations (Plate 2003) with topology-aware
    cleanup memory to prevent SNR collapse at superposition depth > 1.
    """

    def __init__(self, cfg: 'TopogenesisConfig') -> None:
        self.d     = cfg.cognition.hrr_dim
        self.cap   = cfg.cognition.hrr_capacity
        self.noise = cfg.cognition.hrr_noise_floor
        self._roles:    Dict[str, np.ndarray] = {}
        self._rng       = np.random.default_rng(0)
        self.structure  = np.zeros(self.d, dtype=np.float32)
        self._bindings: Dict[str, np.ndarray] = {}
        self._n_bound   = 0
        self._rec_stack = RecursiveBindingStack(self.d, max_depth=4)
        self.cleanup_memory = HRRCleanupMemory(self.d)

    # ── Primitives ──────────────────────────────────────────────────────────

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b))).astype(np.float32)

    def unbind(self, structure: np.ndarray, role: np.ndarray) -> np.ndarray:
        return self.bind(structure, np.roll(role[::-1], 1))

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def normalise(self, v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v) + 1e-8)

    # ── Role management ─────────────────────────────────────────────────────

    def get_role(self, name: str) -> np.ndarray:
        if name not in self._roles:
            v = self._rng.standard_normal(self.d).astype(np.float32)
            self._roles[name] = v / (np.linalg.norm(v) + 1e-8)
            self.cleanup_memory.store(name, self._roles[name])
        return self._roles[name]

    # ── Binding / retrieval ─────────────────────────────────────────────────

    def bind_variable(self, role_name: str, filler: np.ndarray) -> None:
        if self._n_bound >= self.cap:
            oldest    = next(iter(self._bindings))
            old_role  = self.get_role(oldest)
            old_fill  = self._bindings.pop(oldest)
            self.structure -= self.bind(old_role, old_fill)
            self._n_bound   -= 1
        role      = self.get_role(role_name)
        filler_n  = self.normalise(filler)
        self.structure += self.bind(role, filler_n)
        self._bindings[role_name] = filler_n
        self._n_bound             += 1
        self.structure             = self.normalise(self.structure)
        self.cleanup_memory.store(role_name + ':filler', filler_n)

    def retrieve(self, role_name: str) -> Tuple[np.ndarray, float]:
        role      = self.get_role(role_name)
        retrieved = self.cleanup_memory.cleanup(
            self.unbind(self.structure, role))
        quality   = (self.similarity(retrieved, self._bindings[role_name])
                     if role_name in self._bindings else 0.0)
        return retrieved, quality

    def mean_retrieval_quality(self) -> float:
        if not self._bindings:
            return 0.0
        return float(np.mean([
            self.similarity(
                self.cleanup_memory.cleanup(
                    self.unbind(self.structure, self.get_role(r))),
                self._bindings[r])
            for r in self._bindings
        ]))

    def bind_objects(self, slots_np: np.ndarray, mask_np: np.ndarray) -> SymbolicBus:
        self._n_bound  = 0
        self._bindings.clear()
        self.structure = np.zeros(self.d, dtype=np.float32)
        for i in range(min(len(mask_np), len(slots_np))):
            if float(mask_np[i]) > 0.3:
                filler = (slots_np[i, :self.d] if slots_np.shape[1] >= self.d
                          else np.pad(slots_np[i], (0, self.d - slots_np.shape[1])))
                self.bind_variable(f'slot_{i}', filler)
        quality = self.mean_retrieval_quality()
        return SymbolicBus(
            structure    = jnp.array(self.structure),
            n_bound      = self._n_bound,
            retrieval_q  = quality,
        )

    def to_jnp(self) -> jnp.ndarray:
        return jnp.array(self.structure)

# ─────────────────────────────────────────────────────────────────────────────
# EMERGENT METASTABILITY FIELD
# ─────────────────────────────────────────────────────────────────────────────

class EmergentMetastabilityField:
    """
    Metastability as intrinsic dynamical property of the sigma field.

    Indicators: φ_eoc (angular gradient energy), r_kura (Kuramoto order),
    τ_soc (power-law exponent of activation avalanches).
    Regulation: skyrmion injection (ordered) / geodesic damping (chaotic).
    """

    def __init__(self, sigma_field: SigmaFieldGeometric,
                 cfg: TopogenesisConfig) -> None:
        self.field         = sigma_field
        self.cfg           = cfg
        self.mu:    float  = 0.0
        self.phi_eoc: float = 1.0
        self.r_kura:  float = 0.5
        self.tau_soc: float = 1.5
        self._avalanche_sizes: List[float] = []
        self.phase:     str   = 'metastable'
        self.s_max:     float = cfg.cognition.soc_s_max_init
        self.contraction_gain: float = 1.0

    def update_eoc(self) -> float:
        phi     = self.field.phi
        dphi_x  = (jnp.roll(phi, -1, 0) - jnp.roll(phi, 1, 0)) * 0.5
        dphi_y  = (jnp.roll(phi, -1, 1) - jnp.roll(phi, 1, 1)) * 0.5
        dphi_z  = (jnp.roll(phi, -1, 2) - jnp.roll(phi, 1, 2)) * 0.5
        self.phi_eoc = float(jnp.mean(
            jnp.sum(dphi_x**2 + dphi_y**2 + dphi_z**2, axis=-1)))
        return self.phi_eoc

    def update_kuramoto(self) -> float:
        mean_phi  = jnp.mean(self.field.phi.reshape(-1, 3), axis=0)
        self.r_kura = float(jnp.linalg.norm(mean_phi))
        return self.r_kura

    def update_hopf_mu(self) -> float:
        self.mu = self.field.angular_gradient_energy()
        return self.mu

    def record_avalanche(self, activation: jnp.ndarray) -> None:
        size = float(jnp.sum(jnp.abs(activation) > 0.5))
        if size > 0:
            self._avalanche_sizes.append(min(size, self.s_max))
        if len(self._avalanche_sizes) > 1000:
            self._avalanche_sizes.pop(0)

    def estimate_soc_exponent(self) -> float:
        if len(self._avalanche_sizes) < 20:
            return 1.5
        s     = np.array(self._avalanche_sizes, dtype=np.float64)
        s_min = max(1.0, np.min(s))
        s     = s[s >= s_min]
        if len(s) < 5:
            return 1.5
        self.tau_soc = float(
            1.0 + len(s) / np.sum(np.log(s / s_min + 1e-8)))
        return self.tau_soc

    def _regulate_criticality(self) -> None:
        cog   = self.cfg.cognition
        tau   = self.tau_soc
        lr    = cog.soc_contraction_lr
        if tau > cog.soc_tau_hi:
            overshoot = (tau - cog.soc_tau_hi) / max(cog.soc_tau_hi, 1e-6)
            self.s_max = max(cog.soc_s_max_floor,
                             self.s_max * (1.0 - lr * (1.0 + 3.0 * overshoot)))
            self.contraction_gain = min(4.0,
                                        self.contraction_gain + lr * 5.0 * overshoot)
        elif tau < cog.soc_tau_lo:
            self.s_max = min(cog.soc_s_max_init,
                             self.s_max * (1.0 + lr * 0.5))
            self.contraction_gain = max(1.0, self.contraction_gain - lr * 0.5)
        else:
            self.contraction_gain = max(1.0, self.contraction_gain - lr * 0.1)

    def classify_phase(self) -> str:
        cog = self.cfg.cognition
        if self.phi_eoc < 0.2 and self.r_kura > 0.8:
            return 'ordered'
        elif self.phi_eoc > 5.0 or self.r_kura < 0.1:
            return 'chaotic'
        elif (cog.kuramoto_target_lo <= self.r_kura <= cog.kuramoto_target_hi
              and abs(self.phi_eoc - cog.eoc_target) < 0.5):
            return 'metastable'
        return 'transitional'

    def field_perturbation_if_needed(self, rng_np: np.random.Generator) -> str:
        phase      = self.classify_phase()
        self.phase = phase
        if phase == 'ordered':
            X, Y, Z = self.field.shape
            cx, cy, cz = (int(rng_np.integers(2, d - 2)) for d in (X, Y, Z))
            r  = 2
            seed_phi = np.array(self.field.phi)
            seed_phi[max(0, cx-r):cx+r+1,
                     max(0, cy-r):cy+r+1,
                     max(0, cz-r):cz+r+1] = np.array([0., 0., -1.], dtype=np.float32)
            norms = np.linalg.norm(seed_phi, axis=-1, keepdims=True) + 1e-10
            self.field.phi = jnp.array(seed_phi / norms)
            return 'skyrmion_injected'
        elif phase == 'chaotic':
            north           = jnp.array([0., 0., 1.], dtype=jnp.float32)
            toward          = SigmaFieldGeometric.project_tangent(self.field.phi, north - self.field.phi)
            self.field.phi  = SigmaFieldGeometric.geodesic_step(self.field.phi, toward, dt=0.01)
            return 'geodesic_damping'
        return 'no_action'

    def update(self, activation: jnp.ndarray,
               rng_np: np.random.Generator) -> dict:
        self.update_eoc()
        self.update_kuramoto()
        self.update_hopf_mu()
        self.record_avalanche(activation)
        tau = self.estimate_soc_exponent()
        self._regulate_criticality()
        action = self.field_perturbation_if_needed(rng_np)
        return {
            'phi_eoc':          self.phi_eoc,
            'r_kura':           self.r_kura,
            'tau_soc':          tau,
            'hopf_mu':          self.mu,
            'phase':            self.phase,
            'action':           action,
            'contraction_gain': self.contraction_gain,
            's_max':            self.s_max,
        }

# ─────────────────────────────────────────────────────────────────────────────
# DYNAMICAL STABILITY MONITOR (Benettin QR Lyapunov)
# ─────────────────────────────────────────────────────────────────────────────

class DynamicalStabilityMonitor:
    def __init__(self, cfg: TopogenesisConfig, dim: int) -> None:
        self.cfg  = cfg
        self.dim  = dim
        n         = cfg.cognition.lyapunov_n_vectors
        self.Q:   np.ndarray = np.eye(dim, n, dtype=np.float32)
        self.lyapunov_sum    = np.zeros(n,  dtype=np.float64)
        self._n_renorm       = 0
        self._step_count     = 0
        self.sparsity_ema:  float = cfg.cognition.sparsity_target
        self._sparsity_history: Deque[float] = collections.deque(maxlen=100)
        self._deq_residuals:    Deque[float] = collections.deque(
            maxlen=cfg.cognition.convergence_window)
        self.convergence_r: float = 0.0
        self.phase:         str   = 'unknown'
        self.lambda_max:    float = 0.0

    def update_lyapunov(self, f: Callable,
                        z: jnp.ndarray) -> np.ndarray:
        n       = self.cfg.cognition.lyapunov_n_vectors
        Q_cols  = [jnp.array(self.Q[:, k]) for k in range(n)]
        new_cols = []
        for v in Q_cols:
            _, Jv = jax.jvp(f, (z,), (v,))
            new_cols.append(np.array(Jv.flatten()[:self.dim]))
        M         = np.stack(new_cols, axis=1)
        Q_new, R  = np.linalg.qr(M)
        self.Q    = Q_new[:, :n]
        self.lyapunov_sum += np.log(np.abs(np.diag(R)[:n]) + 1e-10)
        self._n_renorm    += 1
        self._step_count  += 1
        exps               = self.lyapunov_sum / max(1, self._n_renorm)
        self.lambda_max    = float(np.max(exps))
        return exps

    def classify_phase(self, exponents: np.ndarray) -> str:
        lmax = float(exponents[0]) if len(exponents) > 0 else 0.0
        if lmax < -0.1:
            self.phase = 'fixed_point'
        elif -0.1 <= lmax < 0.05:
            self.phase = 'edge_of_chaos'
        elif 0.05 <= lmax < 0.5:
            self.phase = 'limit_cycle'
        else:
            self.phase = 'chaotic'
        return self.phase

    def update_sparsity(self, z: jnp.ndarray) -> float:
        l1     = float(jnp.sum(jnp.abs(z)))
        l2     = float(jnp.linalg.norm(z)) + 1e-8
        sigma  = l1 / l2
        self.sparsity_ema = 0.95 * self.sparsity_ema + 0.05 * sigma
        self._sparsity_history.append(sigma)
        return sigma

    def record_deq_residual(self, residual: float) -> float:
        self._deq_residuals.append(residual)
        if len(self._deq_residuals) >= 2:
            r0 = self._deq_residuals[-2]
            r1 = self._deq_residuals[-1]
            self.convergence_r = float(r1 / (r0 + 1e-8))
        return self.convergence_r

    def to_bus(self) -> dict:
        return {
            'lambda_max':    self.lambda_max,
            'phase':         self.phase,
            'sparsity':      self.sparsity_ema,
            'convergence_r': self.convergence_r,
        }

# ─────────────────────────────────────────────────────────────────────────────
# ADAPTIVE FREE ENERGY FUNCTIONAL
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveCouplings:
    """
    Lagrange multipliers for the free energy partition function.
    w_i adapted via dual gradient ascent: ⟨E_i⟩_ema → target_i.
    """
    TARGETS = {
        'prediction':  0.10,
        'homeostasis': 0.05,
        'information': -1.00,
        'causal':      0.10,
        'structural':  0.00,
        'sparsity':    0.00,
    }
    W_MIN, W_MAX = 0.05, 20.0

    def __init__(self, lr: float = 0.005) -> None:
        self.lr    = lr
        self.w     = {k: 1.0 for k in self.TARGETS}
        self._ema  = {k: 0.0 for k in self.TARGETS}
        self._beta = 0.99

    def update(self, terms: dict) -> None:
        for k in self.w:
            v           = float(terms.get(k, 0.0))
            self._ema[k] = self._beta * self._ema[k] + (1.0 - self._beta) * v
            grad        = self._ema[k] - self.TARGETS[k]
            self.w[k]   = float(np.clip(self.w[k] + self.lr * grad,
                                         self.W_MIN, self.W_MAX))

    def snapshot(self) -> dict:
        return {f'coupling_{k}': round(v, 4) for k, v in self.w.items()}

class FreeEnergyFunctional:
    """
    F = Σ w_i · E_i — global variational principle; all subsystems minimise F.
    """

    def __init__(self, lr: float = 0.005) -> None:
        self.couplings = AdaptiveCouplings(lr=lr)
        self._history: Deque[dict] = collections.deque(maxlen=200)

    def compute(self, prediction_error: float,
                deter: jnp.ndarray, equilibrium: jnp.ndarray,
                entropy_composite: float,
                causal_adj: jnp.ndarray,
                topo_charge: float, sparsity: float,
                topo_target: float = 1.0,
                sparsity_target: float = 0.1) -> dict:
        E_pred   = float(prediction_error)
        E_homeo  = float(jnp.mean((deter - equilibrium)**2))
        E_info   = -float(entropy_composite)
        E_causal = float(jnp.mean(jnp.abs(causal_adj)))
        E_struct = abs(topo_charge - topo_target)
        E_sparse = abs(sparsity - sparsity_target)
        w = self.couplings.w
        F = (w['prediction']  * E_pred
           + w['homeostasis'] * E_homeo
           + w['information'] * E_info
           + w['causal']      * E_causal
           + w['structural']  * E_struct
           + w['sparsity']    * E_sparse)
        terms = {
            'prediction': E_pred, 'homeostasis': E_homeo,
            'information': E_info, 'causal': E_causal,
            'structural': E_struct, 'sparsity': E_sparse, 'total': F,
        }
        self.couplings.update(terms)
        self._history.append(terms)
        return terms

    def running_mean_F(self) -> float:
        if not self._history:
            return 0.0
        return float(np.mean([t['total'] for t in list(self._history)[-20:]]))

# ─────────────────────────────────────────────────────────────────────────────
# HEREDITARY CHANNEL (replicator dynamics on HRR structure)
# ─────────────────────────────────────────────────────────────────────────────

class HereditaryChannel:
    """
    Closes Darwin's three conditions on the symbolic substrate:
      Heredity  — HRR structure vector replicates to child
      Variation — Gaussian mutation with adaptive sigma (Eigen threshold)
      Selection — tournament on free energy F (lower = fitter)
    """

    def __init__(self, d: int, pop_size: int = 8,
                 mutation_sigma: float = 0.05) -> None:
        self.d         = d
        self.pop_size  = pop_size
        self.sigma     = mutation_sigma
        rng   = np.random.default_rng(42)
        raw   = rng.standard_normal((pop_size, d)).astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8
        self.population = raw / norms
        self.fitness    = np.zeros(pop_size, dtype=np.float32)
        self._generation = 0
        self._lineage:   List[float] = []

    def replicate(self, current_structure: np.ndarray,
                  free_energy: float) -> np.ndarray:
        if self._generation > 0:
            best_idx = int(np.argmin(self.fitness))
            self.fitness[best_idx] = (0.9 * self.fitness[best_idx]
                                      + 0.1 * free_energy)
        idxs       = np.random.choice(self.pop_size, size=2, replace=False)
        parent_idx = idxs[int(np.argmin(self.fitness[idxs]))]
        child_idx  = idxs[int(np.argmax(self.fitness[idxs]))]
        parent     = self.population[parent_idx]
        child      = parent + np.random.randn(self.d).astype(np.float32) * self.sigma
        child     /= np.linalg.norm(child) + 1e-8
        self.population[child_idx] = child
        self.fitness[child_idx]    = free_energy
        fitness_var = float(np.var(self.fitness))
        self.sigma  = float(np.clip(
            self.sigma * (1.0 + 0.1 * (fitness_var - 0.1)), 0.001, 0.5))
        self._generation += 1
        self._lineage.append(float(self.fitness[parent_idx]))
        return self.population[int(np.argmin(self.fitness))].copy()

    def inject(self, css: CompositionalSymbolicSystem) -> None:
        fittest       = self.population[int(np.argmin(self.fitness))]
        fittest       = fittest / (np.linalg.norm(fittest) + 1e-8)
        css.structure = 0.7 * css.structure + 0.3 * fittest
        css.structure /= np.linalg.norm(css.structure) + 1e-8

    def snapshot(self) -> dict:
        return {
            'heredity_generation':  self._generation,
            'heredity_sigma':       round(self.sigma, 6),
            'heredity_fitness_min': round(float(np.min(self.fitness)), 6),
            'heredity_fitness_var': round(float(np.var(self.fitness)), 6),
        }

# ─────────────────────────────────────────────────────────────────────────────
# AUXILIARY COGNITIVE MODULES
# ─────────────────────────────────────────────────────────────────────────────

class MetaObjectiveHypernetwork:
    """Generates adaptive loss weights from drive state."""
    def __init__(self, config, rng, n_drives, n_objectives):
        k  = random.split(rng)[0]
        h  = config.cognition.policy_net_hidden // 2
        self.in_dim = n_drives + h
        self.W = xavier(k, (n_objectives, n_drives + h))
        self.b = jnp.zeros(n_objectives)
    def to_params(self):   return {'W': self.W, 'b': self.b}
    def from_params(self, p): self.W, self.b = p['W'], p['b']
    def forward(self, drives, context=None):
        d = jnp.ravel(jnp.array(drives, dtype=jnp.float32))
        if context is None:
            c = jnp.zeros(max(0, self.in_dim - d.shape[0]))
        else:
            c = jnp.ravel(jnp.array(context, dtype=jnp.float32))
        x = jnp.concatenate([d, c])
        x = jnp.pad(x, (0, max(0, self.in_dim - x.shape[0])))[:self.in_dim]
        return jax.nn.softmax(x @ self.W.T + self.b)

class LanguageModule:
    """Maps concept embeddings to discrete tokens."""
    def __init__(self, rng, vocab_size, embed_dim):
        self.embed = xavier(rng, (vocab_size, embed_dim), 0.1)
        self.bigram = np.ones((vocab_size, vocab_size), dtype=np.float32) * 1e-3
        self.prev_token: Optional[int] = None
        self.last_token = 0
    def to_params(self):   return {'embed': self.embed}
    def from_params(self, p): self.embed = p['embed']
    def encode(self, vec):
        v = jnp.ravel(jnp.array(vec, dtype=jnp.float32))
        d = min(v.shape[0], self.embed.shape[1])
        v = jnp.pad(v[:d], (0, self.embed.shape[1] - d))
        sims = self.embed @ (v / (jnp.linalg.norm(v) + 1e-8))
        token = int(jnp.argmax(sims))
        if self.prev_token is not None:
            self.bigram[self.prev_token, token] += 1.0
        self.prev_token = token
        self.last_token = token
        return token
    def transition_confidence(self):
        row = self.bigram[self.prev_token if self.prev_token is not None else self.last_token]
        return float(np.max(row / (np.sum(row) + 1e-8)))
    def action_bias(self, token, action_dim=MAX_MOTORS):
        emb = np.array(self.embed[int(token) % self.embed.shape[0]], dtype=np.float32)
        return jnp.array(np.pad(emb[:action_dim], (0, max(0, action_dim - len(emb))))[:action_dim])

class TheoryOfMind:
    """Maintains belief models of peer agents."""
    def __init__(self, config, state_dim, n_peers=1):
        self.beliefs = [np.zeros(state_dim) for _ in range(n_peers)]
        self.prediction_error_ema = 0.0
    def _fit(self, peer_obs):
        v = np.ravel(np.array(peer_obs, dtype=np.float32))
        d = len(self.beliefs[0])
        return np.pad(v[:d], (0, max(0, d - len(v))))[:d]
    def update(self, peer_obs, peer_idx=0):
        if not self.beliefs:
            return 0.0
        peer_idx = int(peer_idx) % len(self.beliefs)
        v = self._fit(peer_obs)
        err = float(np.linalg.norm(v - self.beliefs[peer_idx]) / (np.sqrt(v.size) + 1e-8))
        self.prediction_error_ema = 0.98 * self.prediction_error_ema + 0.02 * err
        self.beliefs[peer_idx] = 0.9 * self.beliefs[peer_idx] + 0.1 * v
        return err
    def get_belief(self, peer_idx=0): return self.beliefs[peer_idx]
    def summary(self):
        if not self.beliefs:
            return {'peer_energy': 0.0, 'peer_need': 0.0, 'peer_count': 0}
        M = np.stack(self.beliefs)
        peer_energy = float(np.clip(np.mean(M[:, BODY_VEC_LEN]), 0.0, 1.0))
        return {
            'peer_energy': peer_energy,
            'peer_need': float(np.clip(1.0 - peer_energy, 0.0, 1.0)),
            'peer_count': len(self.beliefs),
            'tom_error': self.prediction_error_ema,
        }

class ImaginationPlanner:
    def __init__(self, config): self.config = config
    def plan(self, S, drives, world_summary=None):
        obs = jnp.array(S[:self.config.d_E], dtype=jnp.float32)
        attn_start = BODY_VEC_LEN + 4
        attn = obs[attn_start:attn_start + ATTN_DIM]
        energy = jnp.clip(obs[BODY_VEC_LEN], 0.0, 1.0)
        health = jnp.clip(obs[BODY_VEC_LEN + 1], 0.0, 1.0)
        hunger = 1.0 - energy
        injury = 1.0 - health
        resource = attn[0:3] * attn[3] * (0.35 + hunger)
        avoid = -attn[4:7] * attn[7] * (0.25 + injury)
        remembered_hazard = 0.0
        if world_summary is not None:
            remembered_hazard = float(world_summary.get('hazard_ema', 0.0))
        force = resource + avoid * (1.0 + remembered_hazard)
        return jnp.pad(jnp.clip(force, -1.5, 1.5), (0, MAX_MOTORS - 3))

class NoveltyDetector:
    def __init__(self, state_dim):
        self._mem   = collections.deque(maxlen=500)
        self._dim   = min(state_dim, 128)
    def score(self, S):
        v = np.array(S[:self._dim], dtype=np.float32)
        if not self._mem:
            self._mem.append(v); return 1.0
        M    = np.stack(list(self._mem))
        sims = M @ v / (np.linalg.norm(M, axis=1) * (np.linalg.norm(v) + 1e-8) + 1e-8)
        self._mem.append(v)
        return float(1.0 - np.max(sims))

class SelfModel:
    def __init__(self, config):
        self.decay  = config.cognition.self_model_decay if hasattr(
            config.cognition, 'self_model_decay') else 0.95
        self.state  = None
    def update(self, S): self.state = S if self.state is None else (
        self.decay * self.state + (1 - self.decay) * np.array(S))

class AutobiographicalMemory:
    def __init__(self, capacity=2048):
        self._log: List[dict] = []
        self.capacity = capacity
    def record(self, step, summary):
        self._log.append({'step': step, 'summary': summary})
        if len(self._log) > self.capacity:
            self._log.pop(0)

class IdentityStabilityMonitor:
    def __init__(self): self.drift_ema = 0.0
    def update(self, S, S_prev):
        diff = np.array(S) - np.array(S_prev)
        drift = float(np.linalg.norm(diff) / (np.sqrt(diff.size) + 1e-8))
        self.drift_ema = 0.99 * self.drift_ema + 0.01 * drift
        return self.drift_ema

class DevelopmentalGate:
    STAGE_NAMES = ('neonate', 'juvenile', 'adolescent', 'adult')

    def __init__(self, config):
        self.config = config
        self.stage = 0
        self.stage_age = 0
        self.transitions: List[dict] = []

    def check(self, metrics):
        cog = self.config.cognition
        age = int(metrics.get('age', 0))
        viability = float(metrics.get('viability', 0.0))
        identity_drift = float(metrics.get('identity_drift', 1.0))
        old = self.stage
        if (self.stage == 0 and age >= cog.juvenile_age
                and viability >= cog.developmental_min_viability):
            self.stage = 1
        if (self.stage == 1 and age >= cog.adolescent_age
                and viability >= cog.developmental_min_viability
                and identity_drift <= cog.stage_transition_stability):
            self.stage = 2
        if (self.stage == 2 and age >= cog.adult_age
                and viability >= cog.developmental_min_viability
                and identity_drift <= cog.stage_transition_stability):
            self.stage = 3
        if self.stage != old:
            self.stage_age = 0
            self.transitions.append({
                'age': age,
                'from': self.STAGE_NAMES[old],
                'to': self.STAGE_NAMES[self.stage],
                'viability': viability,
                'identity_drift': identity_drift,
            })
        else:
            self.stage_age += 1
        return self.stage

    def name(self):
        return self.STAGE_NAMES[self.stage]

class SocialEngine:
    def __init__(self, config):
        self.trust = {}
        self.affiliation_ema = 0.0
    def update(self, peer_id, reward):
        old = self.trust.get(peer_id, 0.0)
        self.trust[peer_id] = 0.95 * old + 0.05 * float(reward)
    def action_prior(self, peer_summary, viability_features):
        peer_need = float(peer_summary.get('peer_need', 0.0))
        trust = float(np.mean(list(self.trust.values()))) if self.trust else 0.0
        affiliation = np.clip(peer_need * (0.5 + trust), 0.0, 1.0)
        self.affiliation_ema = 0.98 * self.affiliation_ema + 0.02 * affiliation
        feat = jnp.array(viability_features, dtype=jnp.float32)
        social_pull = jnp.array([feat[3], feat[4], 0.0]) * affiliation * 0.4
        return jnp.pad(jnp.clip(social_pull, -0.8, 0.8), (0, MAX_MOTORS - 3))

class PersistentWorldModel:
    def __init__(self, config):
        self._state = {}
        self._ema = {}
    def update(self, key, val):
        x = float(val)
        self._state[key] = x
        self._ema[key] = 0.97 * self._ema.get(key, x) + 0.03 * x
    def summary(self):
        return dict(self._ema)

class IrreversibleConsequenceTracker:
    def __init__(self, config):
        self._events: List = []
        self.risk_ema = 0.0
    def record(self, event):
        self._events.append(event)
        if len(self._events) > 512:
            self._events.pop(0)
    def update(self, viability, reward, organism_obs):
        risk = float(np.clip(
            (1.0 - viability)
            + organism_obs.get('hazard_prox', 0.0)
            + max(0.0, -reward),
            0.0, 3.0) / 3.0)
        self.risk_ema = 0.98 * self.risk_ema + 0.02 * risk
        if risk > 0.65:
            self.record({'risk': risk, 'age': organism_obs.get('age', 0)})
        return self.risk_ema

class MetaStableSelfMonitor:
    def __init__(self, config): self.stability_ema = 1.0
    def update(self, val):
        self.stability_ema = 0.99 * self.stability_ema + 0.01 * float(val)
        return self.stability_ema

# ─────────────────────────────────────────────────────────────────────────────
# GENOME + DEVELOPMENTAL DECODER
# ─────────────────────────────────────────────────────────────────────────────

class Genome:
    """Variable-length heritable encoding with regulatory gating and recombination.

    Three structural mechanisms:

    1. Variable-length modules — duplication / deletion events during mutation
       allow the genome's representational capacity to grow (or shrink) over
       evolutionary time.  Each module is independently extensible up to
       GENOME_MAX_MODULE_DIM.

    2. Regulatory module — a small vector whose sigmoid-transformed values gate
       the expression strength of every other module during developmental
       decoding.  Same genome sequence → different phenotype magnitude depending
       on regulatory state, enabling context-sensitive expression without
       changing the structural genes.

    3. recombine() — uniform crossover between two parents, one element at a
       time per module.  Mismatched lengths are handled by inheriting the
       remainder from the longer parent.  Produces combinatorial novelty
       unreachable by point mutation alone.

    Body plan genes live in the first 3 floats of the misc module (unchanged):
      misc[0] → n_joints  ∈ [4, MAX_JOINTS]
      misc[1] → n_motors  ∈ [3, MAX_MOTORS]
      misc[2] → n_tactile ∈ [4, MAX_TACTILE]
    """

    _MODULE_NAMES = ('policy', 'world_model', 'affect', 'symbolic', 'viability', 'misc')

    def __init__(self, rng_np: np.random.Generator, dim: int = GENOME_DIM):
        base = dim // 8
        self.modules: Dict[str, np.ndarray] = {
            'policy':      rng_np.standard_normal(dim // 4).astype(np.float32),
            'world_model': rng_np.standard_normal(dim // 4).astype(np.float32),
            'affect':      rng_np.standard_normal(base).astype(np.float32),
            'symbolic':    rng_np.standard_normal(base).astype(np.float32),
            'viability':   rng_np.standard_normal(base).astype(np.float32),
            'misc':        rng_np.standard_normal(base).astype(np.float32),
            # Regulatory module: one gene per structural module, gates expression
            'regulatory':  rng_np.standard_normal(len(self._MODULE_NAMES)).astype(np.float32),
        }

    # ── Total genome size (now dynamic) ─────────────────────────────────────
    @property
    def dim(self) -> int:
        return sum(v.size for v in self.modules.values())

    # ── Regulatory expression gates: sigmoid → [0.05, 1.0] per module ───────
    @property
    def expression_gates(self) -> dict:
        """Per-module expression weight.  Regulatory module controls how strongly
        each structural module is expressed during developmental decoding."""
        reg = self.modules['regulatory']
        gates = {}
        for i, name in enumerate(self._MODULE_NAMES):
            raw = float(reg[i]) if i < len(reg) else 0.0
            gates[name] = float(np.clip(1.0 / (1.0 + np.exp(-raw)), 0.05, 1.0))
        return gates

    # ── Body plan properties (unchanged interface) ───────────────────────────
    @property
    def n_joints(self) -> int:
        raw = float(self.modules['misc'][0])
        return int(np.clip(round(4 + (MAX_JOINTS - 4) * (np.tanh(raw) * 0.5 + 0.5)), 4, MAX_JOINTS))

    @property
    def n_motors(self) -> int:
        raw = float(self.modules['misc'][1])
        return int(np.clip(round(3 + (MAX_MOTORS - 3) * (np.tanh(raw) * 0.5 + 0.5)), 3, MAX_MOTORS))

    @property
    def n_tactile(self) -> int:
        raw = float(self.modules['misc'][2])
        return int(np.clip(round(4 + (MAX_TACTILE - 4) * (np.tanh(raw) * 0.5 + 0.5)), 4, MAX_TACTILE))

    # ── Mutation: point noise + duplication / deletion ───────────────────────
    def mutate(self, rng_np: np.random.Generator, sigma: float = 0.015) -> 'Genome':
        child = Genome.__new__(Genome)
        child.modules = {}
        min_size = GENOME_DIM // 8

        for k, v in self.modules.items():
            # Point mutation (unchanged behaviour)
            mask  = rng_np.random(v.shape).astype(np.float32) < 0.10
            noise = rng_np.standard_normal(v.shape).astype(np.float32) * sigma
            new_v = v + mask * noise

            if k != 'regulatory':
                # Duplication: append a noisy copy of a random segment
                if (rng_np.random() < GENOME_DUP_PROB and
                        len(new_v) + GENOME_DUP_SEGMENT <= GENOME_MAX_MODULE_DIM):
                    start   = int(rng_np.integers(0, max(1, len(new_v) - GENOME_DUP_SEGMENT + 1)))
                    segment = new_v[start:start + GENOME_DUP_SEGMENT].copy()
                    segment += rng_np.standard_normal(segment.shape).astype(np.float32) * sigma * 2
                    new_v   = np.concatenate([new_v, segment])

                # Deletion: remove a random segment (never below min_size)
                if (rng_np.random() < GENOME_DEL_PROB and
                        len(new_v) > min_size + GENOME_DUP_SEGMENT):
                    start = int(rng_np.integers(0, len(new_v) - GENOME_DUP_SEGMENT))
                    new_v = np.concatenate([new_v[:start],
                                            new_v[start + GENOME_DUP_SEGMENT:]])

            child.modules[k] = new_v.astype(np.float32)
        return child

    # ── Recombination: uniform crossover per module ──────────────────────────
    @classmethod
    def recombine(cls, parent_a: 'Genome', parent_b: 'Genome',
                  rng_np: np.random.Generator) -> 'Genome':
        """One offspring via uniform crossover between two parents.

        Each element is drawn independently from either parent up to min length;
        the remainder is inherited from the longer parent, preserving any
        duplications acquired during that lineage's history.
        """
        child = cls.__new__(cls)
        child.modules = {}
        all_keys = list(dict.fromkeys(list(parent_a.modules) + list(parent_b.modules)))
        for k in all_keys:
            a = parent_a.modules.get(k, np.zeros(1, dtype=np.float32))
            b = parent_b.modules.get(k, np.zeros(1, dtype=np.float32))
            min_len = min(len(a), len(b))
            mask    = rng_np.random(min_len) < 0.5
            shared  = np.where(mask, a[:min_len], b[:min_len])
            # Remainder from the longer parent — inherits its duplications
            if len(a) > min_len:
                shared = np.concatenate([shared, a[min_len:]])
            elif len(b) > min_len:
                shared = np.concatenate([shared, b[min_len:]])
            child.modules[k] = shared.astype(np.float32)
        return child

    def to_dict(self) -> dict:
        d = {k: v.tolist() for k, v in self.modules.items()}
        d['_body_plan'] = {
            'n_joints':  self.n_joints,
            'n_motors':  self.n_motors,
            'n_tactile': self.n_tactile,
        }
        d['_expression_gates'] = self.expression_gates
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'Genome':
        g = cls.__new__(cls)
        g.modules = {k: np.array(v, dtype=np.float32)
                     for k, v in d.items() if not k.startswith('_')}
        return g

class DevelopmentalDecoder:
    """Low-rank hypernetwork with lazy basis regeneration and regulatory gating.

    Two structural mechanisms:

    1. Lazy basis regeneration — basis matrices are rebuilt automatically when
       a module's length changes due to duplication / deletion.  This makes the
       decoder forward-compatible with variable-length genomes without any
       external coordination.

    2. Regulatory gating — genome.expression_gates are applied as multiplicative
       scalars on the perturbation magnitude per region, so the same structural
       genome produces phenotypes of different magnitude depending on the
       regulatory module's state.  This is the minimal implementation of
       context-sensitive gene expression.
    """
    _REGION_MAP = {
        'policy':        'policy',
        'pol':           'policy',
        'wm':            'world_model',
        'world':         'world_model',
        'A_params':      'world_model',
        'viability':     'viability',
        'enactive':      'viability',
        'affect':        'affect',
        'workspace':     'affect',
        'symbol':        'symbolic',
        'concept':       'symbolic',
        'vq':            'symbolic',
        'entity':        'symbolic',
        'relational':    'symbolic',
        'goal':          'misc',
        'sensorimotor':  'misc',
    }
    _BASIS_COLS = 32

    def __init__(self, rng: jax.random.KeyArray, genome: Genome):
        self._rng_key = rng
        self._basis:       Dict[str, np.ndarray] = {}
        self._basis_sizes: Dict[str, int]        = {}
        self._rebuild_basis(genome)

    def _rebuild_basis(self, genome: Genome) -> None:
        """(Re)build orthonormal basis for any module whose length has changed.
        Called at init and again lazily inside decode_perturbation."""
        keys = jax.random.split(self._rng_key, len(genome.modules))
        for key, (mod_name, mod_vec) in zip(keys, genome.modules.items()):
            current_size = len(mod_vec)
            if (mod_name not in self._basis or
                    self._basis_sizes.get(mod_name) != current_size):
                cols = min(self._BASIS_COLS, current_size)
                raw  = jax.random.normal(key, (current_size, cols))
                q, _ = jnp.linalg.qr(raw)
                self._basis[mod_name]       = np.array(q[:, :cols])
                self._basis_sizes[mod_name] = current_size

    def _region_for_key(self, k: str) -> str:
        for prefix, region in self._REGION_MAP.items():
            if prefix in k:
                return region
        return 'misc'

    def decode_perturbation(self, genome: Genome, target_tree: dict,
                            rng_np: np.random.Generator) -> dict:
        """Returns tree of same structure as target_tree with genome-derived perturbations.

        Perturbation magnitude per region is modulated by genome.expression_gates,
        implementing regulatory control over phenotypic expression strength.
        """
        self._rebuild_basis(genome)   # no-op if all module sizes unchanged
        gates = genome.expression_gates

        def _perturb_leaf(x, region: str) -> jnp.ndarray:
            arr = np.array(x)
            if arr.dtype.kind not in ('f', 'c'):
                return x
            mod_vec = genome.modules[region]
            B       = self._basis[region]
            cols    = B.shape[1]
            seed    = mod_vec @ B                          # R^{cols}
            scale   = float(np.linalg.norm(mod_vec)) * 0.015 + 1e-5
            # ── Regulatory gate scales the perturbation magnitude ────────────
            scale  *= gates.get(region, 1.0)
            n_elem  = arr.size
            seed_tiled = np.tile(seed, n_elem // cols + 1)[:n_elem]
            residual   = rng_np.standard_normal(n_elem).astype(np.float32) * scale * 0.1
            delta = (seed_tiled * scale + residual).reshape(arr.shape).astype(arr.dtype)
            return jnp.array(arr + delta)

        out = {}
        for k, v in target_tree.items():
            region = self._region_for_key(k)
            if isinstance(v, dict):
                out[k] = jax.tree_util.tree_map(
                    lambda leaf: _perturb_leaf(leaf, region), v)
            else:
                out[k] = _perturb_leaf(v, region)
        return out
# ─────────────────────────────────────────────────────────────────────────────
# TOOL REQUEST BROKER
# ─────────────────────────────────────────────────────────────────────────────

class ToolRequestBroker:
    """
    Decision gate sitting between GlobalWorkspace and the world.

    Four exclusive modes per step:
      ACT    (0) — emit motor forces at full gain (existing only path)
      IMAGINE (1) — run internal autoregressive rollout; zero motor; imagined
                    terminal state stored as next-step GRU injection
      QUERY   (2) — retrieve concept context at L1/L2 level; normal motor;
                    retrieved vector stored as next-step GRU injection
      DEFER   (3) — hold workspace, suppress motor, accumulate signal;
                    current ws_final stored as next-step GRU injection

    Decision: argmax over mode logits from (ws_final ‖ drives ‖ [v, H, ε]).
    Survival pressure hard-gates IMAGINE and DEFER off when viability is low.
    Consecutive DEFER is capped at MAX_DEFER.

    The 'next-step injection' tensor produced by IMAGINE/QUERY/DEFER is fed
    into x_wm at the following step via W_broker_feedback (learned projection),
    closing the loop: GlobalWorkspace → ToolRequestBroker → GRU.
    """

    MODE_ACT     = 0
    MODE_IMAGINE = 1
    MODE_QUERY   = 2
    MODE_DEFER   = 3
    MODE_NAMES   = ('ACT', 'IMAGINE', 'QUERY', 'DEFER')
    MAX_DEFER    = 3

    def __init__(self, config: 'TopogenesisConfig', rng: jax.random.KeyArray) -> None:
        cog      = config.cognition
        feat_dim = cog.workspace_dim + cog.n_drives + 3   # ws ‖ drives ‖ [v, H, ε]
        self.W   = xavier(rng, (4, feat_dim), 0.15)
        # Priors: ACT preferred, DEFER costly
        self.b   = jnp.array([1.5, 0.0, 0.2, -1.0], dtype=jnp.float32)
        self._imagine_horizon: int  = 5
        self._defer_count:     int  = 0
        self._last_mode:       int  = self.MODE_ACT
        self._mode_ema: np.ndarray  = np.array([1., 0., 0., 0.], dtype=np.float32)

    def decide(self, ws_final: jnp.ndarray, drives: jnp.ndarray,
               viability: float, ws_entropy: float,
               pred_err: float, survival_pressure: float) -> int:
        feat   = jnp.concatenate([ws_final, drives,
                                  jnp.array([viability, ws_entropy, pred_err],
                                            dtype=jnp.float32)])
        logits = self.W @ feat + self.b
        # Desperate agents cannot afford to daydream or stall
        if survival_pressure > 0.7:
            logits = logits.at[self.MODE_IMAGINE].add(-5.0)
            logits = logits.at[self.MODE_DEFER].add(-8.0)
        if self._defer_count >= self.MAX_DEFER:
            logits = logits.at[self.MODE_DEFER].add(-20.0)
        mode = int(jnp.argmax(jax.nn.softmax(logits)))
        self._last_mode = mode
        self._mode_ema  = 0.95 * self._mode_ema
        self._mode_ema[mode] += 0.05
        self._defer_count = self._defer_count + 1 if mode == self.MODE_DEFER else 0
        return mode

    def imagine(self, S_full: jnp.ndarray, wm_params: dict,
                config: 'TopogenesisConfig', rng: jax.random.KeyArray) -> jnp.ndarray:
        """Run internal rollout; return imagined terminal state for GRU injection."""
        traj, _, _ = autoregressive_rollout(
            S_full, self._imagine_horizon, wm_params, config,
            config.dt, config.cognition.K_medium, config.cognition.K_slow,
            rng, stochastic=False)
        return traj[-1]

    def snapshot(self) -> dict:
        return {
            'broker_mode': self.MODE_NAMES[self._last_mode],
            'broker_defer_count': self._defer_count,
            **{f'broker_ema_{self.MODE_NAMES[i].lower()}': round(float(self._mode_ema[i]), 4)
               for i in range(4)},
        }

# ─────────────────────────────────────────────────────────────────────────────

class TopogenesisAgent:
    """
    Unified Synthetic Cognitive Physics Agent.

    Forward pass each step:
      1.  Encode observation → embodiment vector
      2.  Project to n_slots feature vectors
      3.  SlotAttention → ObjectBus
      4.  Pump slots into sigma field PDE
      5.  CausalLearner update + do-intervention
      6.  Hierarchical GRU world model step
      7.  Anderson/DEQ equilibrium solve
      8.  Metastability field update (Hopf/SOC/Kuramoto)
      9.  Dynamical stability (Lyapunov QR)
      10. Compositional HRR binding (with cleanup)
      11. Global workspace broadcast (non-zero projection)
      12. Affect (valence / distress / arousal)
      13. Policy sampling → action
      14. Free energy computation + Lagrange multiplier update
      15. Hereditary replication on symbolic structure
      16. Memory store (SparseModularMemory)
    """

    def __init__(self, config: TopogenesisConfig, rng: jax.random.KeyArray,
                 num_agents: int = 1, self_idx: int = 0) -> None:
        self.config   = config
        self.self_idx = self_idx
        cog           = config.cognition
        S_total       = config.d_E + config.d_D + config.d_I
        self.ss       = StateSpace(config.d_E, config.d_D, config.d_I)

        keys = random.split(rng, 24)
        (rng_wm, rng_A, rng_pol, rng_proj, rng_aff, rng_ws, rng_critic,
         rng_goal, rng_attn, rng_cenc, rng_rel, rng_vq, rng_meta,
         rng_lang, rng_featproj, rng_slotproj, rng_wsobj, rng_wscaus,
         rng_wssym, rng_obs2feat, rng_spare1, rng_spare2,
         rng_spare3, rng_spare4) = keys

        # ── World model ───────────────────────────────────────────────────
        self.wm = HierarchicalGRU(rng_wm, S_total, config.latent_dim, S_total, config)

        # ── Influence tensor ──────────────────────────────────────────────
        self.A_params = init_A_params(rng_A, S_total, config.d_I, config.A_rank)

        # ── Policy + critic ───────────────────────────────────────────────
        self.policy       = GaussianPolicy(rng_pol, config.latent_dim, MAX_MOTORS, config)
        self.critic_params = init_critic_params(rng_critic, S_total)
        self.W_pol_proj   = xavier(rng_proj, (config.d_D, config.d_I))
        self.policy_opt = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(config.cognition.policy_online_lr))
        self.policy_opt_state = self.policy_opt.init(self.policy.to_params())
        self.sensorimotor_params = init_sensorimotor_params(
            rng_spare2, S_total, MAX_MOTORS, cog.sensorimotor_hidden)
        self.sensorimotor_opt = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(config.cognition.sensorimotor_lr))
        self.sensorimotor_opt_state = self.sensorimotor_opt.init(
            self.sensorimotor_params)
        self.wm_online_opt = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(config.lr))
        self.wm_online_state = self.wm_online_opt.init(self.wm.to_params())
        self.enactive_ac_params = init_enactive_ac_params(rng_spare1, 16, MAX_MOTORS)
        self.enactive_ac_opt = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(config.cognition.enactive_actor_lr))
        self.enactive_ac_state = self.enactive_ac_opt.init(self.enactive_ac_params)

        # ── Workspace (non-zero projections) ─────────────────────────────
        self.workspace_state  = jnp.zeros(cog.workspace_dim)
        self.workspace_params = init_workspace_params(rng_ws, S_total, config.d_D, config)
        # Separate projections from object/causal/symbolic into workspace
        self.W_obj_to_ws  = xavier(rng_wsobj,  (cog.workspace_dim, cog.slot_dim))
        self.W_caus_to_ws = xavier(rng_wscaus, (cog.workspace_dim, cog.causal_dim))
        self.W_sym_to_ws  = xavier(rng_wssym,  (cog.workspace_dim, cog.hrr_dim))

        # ── Affect ───────────────────────────────────────────────────────
        if config.use_affect:
            self.affect_params = init_affect_params(rng_aff, S_total, config)
            self.affect_state  = jnp.zeros(config.affect.valence_dim)
        else:
            self.affect_params = None
            self.affect_state  = jnp.zeros(1)

        # ── Goal / drive ──────────────────────────────────────────────────
        self.goal_net_params = init_goal_net_params(
            rng_goal, config.d_E, cog.n_drives,
            cog.goal_net_hidden,
            concept_dim=cog.goal_net_concept_dim,
            field_dim=cog.goal_net_field_feat_dim)
        self.drive_system  = DriveSystem(config)
        self.goal_manager  = GoalManager()

        # ── Entity attention ──────────────────────────────────────────────
        self.entity_attn = EntityAttention(config, rng_attn)

        # ── Concept encoding + VQ ─────────────────────────────────────────
        cdim = cog.concept_enc_dim
        self.concept_enc_W = xavier(rng_cenc, (cdim, config.d_E))
        self.vq_codebook   = xavier(rng_vq,   (cog.vq_n_codes, cdim), 0.3)
        self.W_rel_proj    = xavier(rng_rel,   (config.d_D, cdim))
        self.relational_net = RelationalReasoningNet(
            rng_rel, cdim, MAX_MOTORS,
            cog.relational_action_proj_dim, cog.relational_net_hidden)

        # ── Projection from obs to n_slots feature vectors ─────────────────
        self.W_obs_to_feat = xavier(
            rng_obs2feat, (cog.n_slots, cog.slot_dim, config.d_E), 0.3)

        # ── Slot attention ────────────────────────────────────────────────
        self.slot_attn = SlotAttention(cog.n_slots, cog.slot_dim, cog.slot_dim)

        # ── Physics substrate ─────────────────────────────────────────────
        self.sigma_field       = SigmaFieldGeometric(
            (cog.object_world_size, cog.object_world_size, cog.world_depth))
        self.metastability_field = EmergentMetastabilityField(self.sigma_field, config)
        self.stability           = DynamicalStabilityMonitor(config, cog.deter_dim)

        # ── Causal learner ────────────────────────────────────────────────
        self.causal_learner = CausalLearner(cog.n_slots, cog)

        # ── Symbolic / hereditary ─────────────────────────────────────────
        self.symbolic_sys = CompositionalSymbolicSystem(config)
        self.hereditary   = HereditaryChannel(d=cog.hrr_dim, pop_size=8)

        # ── Free energy ───────────────────────────────────────────────────
        self.free_energy = FreeEnergyFunctional()

        # ── Memory (SMM) ──────────────────────────────────────────────────
        self.memory = SparseModularMemory(config, S_total, rng)

        # ── Auxiliary cognition ───────────────────────────────────────────
        self.meta_hypernet = MetaObjectiveHypernetwork(config, rng_meta, cog.n_drives, 5)
        self.language      = LanguageModule(rng_lang, cog.vq_n_codes, cdim)
        self.tom           = TheoryOfMind(config, S_total, n_peers=max(1, num_agents-1))
        self.planner       = ImaginationPlanner(config)
        self.novelty       = NoveltyDetector(S_total)
        self.self_model    = SelfModel(config)
        self.autobio       = AutobiographicalMemory(cog.autobio_capacity)
        self.identity_mon  = IdentityStabilityMonitor()
        self.dev_gate      = DevelopmentalGate(config)
        self.social        = SocialEngine(config)
        self.persist_wm    = PersistentWorldModel(config)
        self.irreversible  = IrreversibleConsequenceTracker(config)
        self.meta_self     = MetaStableSelfMonitor(config)

        # ── Causal loop projections ────────────────────────────────────────
        # Concept context (L1/L2 memory) → spatial_attn_out for x_wm injection
        self.W_concept_to_ctx = xavier(rng_spare3,
                                       (cog.spatial_attn_out,
                                        min(S_total, 256)), 0.1)
        # Broker injection (imagined S / concept / ws) → spatial_attn_out
        self.W_broker_feedback = xavier(rng_spare4,
                                        (cog.spatial_attn_out,
                                         min(S_total, 256)), 0.05)
        # z_star (Anderson equilibrium, deter_dim) → hrr_dim for symbolic injection
        self.W_z_to_concept = xavier(rng_spare3,
                                     (cog.hrr_dim, cog.deter_dim), 0.1)

        # ── Tool Request Broker ────────────────────────────────────────────
        rng, _rng_broker = random.split(rng)
        self.broker = ToolRequestBroker(config, _rng_broker)

        # ── Optimiser ─────────────────────────────────────────────────────
        schedule = optax.cosine_decay_schedule(
            init_value=config.lr, decay_steps=10000, alpha=0.01)
        self.opt      = optax.chain(
            optax.clip_by_global_norm(config.grad_clip_norm),
            optax.adam(learning_rate=schedule, eps=1e-6))
        params         = self._collect_params()
        self.opt_state = self.opt.init(
            {k: v for k, v in params.items()
             if v is not None and hasattr(v, 'shape')})

        # ── Runtime state ─────────────────────────────────────────────────
        self._step             = 0
        self._np_rng           = np.random.default_rng(42 + self_idx)
        self.h_fast            = jnp.zeros(config.latent_dim)
        self.h_medium          = jnp.zeros(config.latent_dim)
        self.h_slow            = jnp.zeros(config.latent_dim)
        self.deter_state       = jnp.zeros(cog.deter_dim)
        self.stoch_state       = jnp.zeros(cog.stoch_dim)
        self.equilibrium_state = jnp.zeros(cog.deter_dim)
        self.competence_ema    = 0.0
        self.last_S            = jnp.zeros(S_total)
        self.last_metrics:     dict = {}
        self.soft_failures     = collections.Counter()
        self.viability_actor_W = jnp.zeros((MAX_MOTORS, 16))
        self.prev_viability    = None
        self.prev_viability_features = None
        self.prev_action       = None
        self.prev_S_full       = None
        self.prev_prediction   = None
        self.prev_wm_train     = None
        self.prev_policy_latent = None
        self.prev_policy_action = None
        self.prev_slots        = None
        self.prev_ws_final     = None      # ws_final from last step → GRU feedback
        self.prev_broker_context: Optional[np.ndarray] = None  # broker injection → x_wm
        self._peer_observations: List[np.ndarray] = []
        self._last_peer_summary: dict = {'peer_energy': 0.0, 'peer_need': 0.0, 'peer_count': 0}
        self.survival_ema      = 0.0
        self.enactive_td_ema   = 0.0
        self.enactive_loss_ema = 0.0
        self.policy_loss_ema   = 0.0
        self.sensorimotor_mse_ema = 0.0
        self.wm_online_mse_ema = 0.0
        # ── Genome + developmental decoder ──────────────────────────────────
        _init_rng_np = np.random.default_rng(int(jax.random.randint(rng, (), 0, 1_000_000)))
        self.genome = Genome(_init_rng_np, dim=GENOME_DIM)
        _dev_rng, rng = jax.random.split(rng)
        self.dev_decoder = DevelopmentalDecoder(_dev_rng, self.genome)
        # Genome lives in the sigma field through a read/write/fidelity interface.
        self.genome_field_iface = GenomeFieldInterface()
        # Structural integrity side-channel (set by self_maintain each step).
        self._current_si: Optional[Dict[str, float]] = None

        # ── Supervenience subsystems ───────────────────────────────────────
        # These implement the five supervenience principles: every cognitive
        # operation costs energy, cognition supervenes on the field, actions
        # are constrained by physical limits, higher-level representations
        # supervene on lower-level states, and metabolic state gates learning.
        self.cog_metabolism      = CognitiveMetabolism()
        self.field_supervenience = FieldSupervenience()
        self.info_supervenience  = InformationalSupervenience()
        self.metabolic_super     = MetabolicSupervenience()
        # ── Body state cache — populated by self_maintain() before step() ──
        # step() reads these to gate supervenience without needing body access.
        # Defaults are fully-viable so the first step before self_maintain()
        # runs correctly if step() is called standalone.
        self._current_body_energy:     float = 1.0
        self._current_biosynthetic:    float = 0.0
        self._current_genome_fidelity: float = 1.0
        self._current_lr_mod:          float = 1.0
        # Thermodynamic supervenience: energy budget determines iteration counts.
        # Set as preconditions by self_maintain() before step() runs.
        # Full energy → full iteration budget; starvation truncates computation.
        self._thermo_max_fp_iter:          int = config.cognition.max_fp_iter
        self._thermo_n_timescale_layers:   int = 3

    # ── Parameter collect / apply ────────────────────────────────────────────

    def _collect_params(self) -> dict:
        p = {**self.wm.to_params(),
             'A': self.A_params,
             'policy': self.policy.to_params(),
             'W_pol_proj': self.W_pol_proj,
             'W_rel_proj': self.W_rel_proj,
             'W_obj_to_ws': self.W_obj_to_ws,
             'W_caus_to_ws': self.W_caus_to_ws,
             'W_sym_to_ws': self.W_sym_to_ws,
             'W_obs_to_feat': self.W_obs_to_feat,
             'workspace': self.workspace_params,
             'critic': self.critic_params,
             'goal_net': self.goal_net_params,
             'concept_enc_W': self.concept_enc_W,
             'vq_codebook': self.vq_codebook,
             'entity_attn': self.entity_attn.to_params(),
             'relational_net': self.relational_net.to_params(),
             'meta': self.meta_hypernet.to_params(),
             'lang': self.language.to_params()}
        if self.config.use_affect:
            p['affect'] = self.affect_params
        return p

    # ── Feature projection: obs → n_slots features   ─────────────────

    def _obs_to_features(self, obs_jnp: jnp.ndarray) -> jnp.ndarray:
        """
        Project scalar observation vector into n_slots feature vectors,
        each of slot_dim width, using a learned linear projection.
        Shape: (1, n_slots, slot_dim) — ready for SlotAttention.
        """
        # W_obs_to_feat: (n_slots, slot_dim, d_E)
        features = jnp.einsum('fsd,d->fs', self.W_obs_to_feat, obs_jnp)
        features = jnp.tanh(features)        # (n_slots, slot_dim)
        return features[None]                # (1, n_slots, slot_dim)

    def _record_soft_failure(self, subsystem: str, exc: Exception) -> None:
        """Track recoverable subsystem failures instead of hiding them."""
        self.soft_failures[subsystem] += 1
        self.soft_failures[f'{subsystem}:{type(exc).__name__}'] += 1

    def _viability_from_obs(self, obs_jnp: jnp.ndarray) -> Tuple[float, dict]:
        attn_start = BODY_VEC_LEN + 4
        attn = obs_jnp[attn_start:attn_start + ATTN_DIM]
        energy = float(jnp.clip(obs_jnp[BODY_VEC_LEN], 0.0, 1.0))
        health = float(jnp.clip(obs_jnp[BODY_VEC_LEN + 1], 0.0, 1.0))
        inventory = float(jnp.clip(obs_jnp[BODY_VEC_LEN + 2], 0.0, 1.0))
        membrane_idx = 3 + 3 + 4 + 3 + MAX_JOINTS + MAX_JOINTS + MAX_TACTILE + 4
        membrane = float(jnp.clip(obs_jnp[membrane_idx], 0.0, 1.0))
        hazard_prox = float(jnp.clip(attn[7], 0.0, 1.0))
        viability = float(np.clip(
            0.32 * energy + 0.32 * health + 0.20 * membrane
            + 0.16 * inventory - 0.18 * hazard_prox,
            0.0, 1.0))
        obs = {
            'energy': energy,
            'health': health,
            'membrane': membrane,
            'inventory': inventory,
            'death_count': int(round(float(jnp.clip(attn[12], 0.0, 1.0)) * 10.0)),
            'age': int(round(float(jnp.clip(obs_jnp[3 + 3 + 4 + 3 + MAX_JOINTS + MAX_JOINTS + MAX_TACTILE + 7], 0.0, 1.0)) * 1000.0)),
            'resource_dist': 1.0 - float(jnp.clip(attn[3], 0.0, 1.0)),
            'hazard_dist': 1.0 - hazard_prox,
            'hazard_prox': hazard_prox,
        }
        return viability, obs

    def _viability_features(self, obs_jnp: jnp.ndarray) -> jnp.ndarray:
        attn_start = BODY_VEC_LEN + 4
        attn = obs_jnp[attn_start:attn_start + ATTN_DIM]
        energy = jnp.clip(obs_jnp[BODY_VEC_LEN], 0.0, 1.0)
        health = jnp.clip(obs_jnp[BODY_VEC_LEN + 1], 0.0, 1.0)
        membrane_idx = 3 + 3 + 4 + 3 + MAX_JOINTS + MAX_JOINTS + MAX_TACTILE + 4
        membrane = jnp.clip(obs_jnp[membrane_idx], 0.0, 1.0)
        z_pos = jnp.clip(obs_jnp[2] / max(1.0, float(self.config.cognition.world_depth)), 0.0, 1.0)
        return jnp.array([
            1.0 - energy,
            1.0 - health,
            1.0 - membrane,
            attn[0], attn[1], attn[2], attn[3],
            -attn[4], -attn[5], -attn[6], attn[7],
            obs_jnp[FIELD_GRAD_IDX], obs_jnp[FIELD_GRAD_IDX + 1], obs_jnp[FIELD_GRAD_IDX + 2],
            z_pos,
            1.0,
        ], dtype=jnp.float32)

    def _viability_reflex(self, obs_jnp: jnp.ndarray) -> jnp.ndarray:
        feat = self._viability_features(obs_jnp)
        hunger = feat[0]
        injury = feat[1]
        resource_pull = feat[3:6] * feat[6] * (0.20 + 1.35 * hunger)
        hazard_push = feat[7:10] * feat[10] * (0.7 + injury)
        field_push = feat[11:14] * 0.15
        resource_pull = resource_pull.at[2].multiply(0.25)
        hazard_push = hazard_push.at[2].multiply(0.25)
        lift = jnp.array([0.0, 0.0, jnp.maximum(0.0, 0.04 - feat[14]) * 0.5])
        force = (resource_pull + hazard_push + field_push + lift)
        force = force * self.config.cognition.viability_reflex_gain
        return jnp.pad(jnp.clip(force, -2.5, 2.5), (0, MAX_MOTORS - 3))

    def _update_auxiliary_context(self, S_full, organism_obs, viability, reward):
        for idx, peer_obs in enumerate(self._peer_observations):
            self.tom.update(peer_obs, idx)
            self.social.update(idx, reward)
        peer_summary = self.tom.summary()
        self._last_peer_summary = peer_summary
        self.persist_wm.update('resource_prox', 1.0 - organism_obs['resource_dist'])
        self.persist_wm.update('hazard_ema', organism_obs['hazard_prox'])
        self.persist_wm.update('viability', viability)
        consequence_risk = self.irreversible.update(viability, reward, organism_obs)
        return peer_summary, self.persist_wm.summary(), consequence_risk

    def _adapt_viability_actor(self, viability: float, reward: float) -> None:
        if self.prev_viability is None:
            return
        delta_v = viability - float(self.prev_viability)
        reinforcement = float(np.clip(reward + 2.0 * delta_v, -1.0, 1.0))
        if abs(reinforcement) < 1e-6:
            return
        action = jnp.array(self.prev_action, dtype=jnp.float32)
        feat = jnp.array(self.prev_viability_features, dtype=jnp.float32)
        # ── Metabolic supervenience: viability actor LR scales with metabolism ─
        lr = self.config.cognition.viability_lr * self._current_lr_mod
        decay = self.config.cognition.viability_actor_decay
        self.viability_actor_W = (
            decay * self.viability_actor_W
            + lr * reinforcement * jnp.outer(action, feat))
        row_norm = jnp.linalg.norm(self.viability_actor_W, axis=1, keepdims=True)
        self.viability_actor_W = self.viability_actor_W / jnp.maximum(1.0, row_norm)

    def _scale_grads(self, grads):
        """
        Metabolic supervenience: scale gradient tree by current metabolic LR
        modulator.  Hungry agents (low energy / biosyn) learn more slowly —
        synaptic consolidation is a metabolically expensive process.
        """
        mod = float(self._current_lr_mod)
        if mod >= 0.999:
            return grads
        return jax.tree_util.tree_map(lambda g: g * mod, grads)

    def _update_enactive_actor_critic(self,
                                      viability: float,
                                      reward: float,
                                      viability_features: jnp.ndarray) -> None:
        if self.prev_viability_features is None or self.prev_action is None:
            return
        feat_prev = jnp.array(self.prev_viability_features, dtype=jnp.float32)
        action_prev = jnp.array(self.prev_action, dtype=jnp.float32)
        feat_now = jnp.array(viability_features, dtype=jnp.float32)
        organism_reward = jnp.array(
            reward + 0.5 * viability + 2.0 * (viability - float(self.prev_viability)),
            dtype=jnp.float32)
        (loss, aux), grads = jax.value_and_grad(
            enactive_ac_loss, has_aux=True)(
                self.enactive_ac_params, feat_prev, action_prev,
                organism_reward, feat_now, self.config)
        # ── Metabolic supervenience: learning rate gated by metabolic state ───
        grads = self._scale_grads(grads)
        updates, self.enactive_ac_state = self.enactive_ac_opt.update(
            grads, self.enactive_ac_state, self.enactive_ac_params)
        self.enactive_ac_params = optax.apply_updates(
            self.enactive_ac_params, updates)
        self.enactive_td_ema = 0.98 * self.enactive_td_ema + 0.02 * float(aux['td'])
        self.enactive_loss_ema = 0.98 * self.enactive_loss_ema + 0.02 * float(loss)
        # ── Thermodynamic cost: online AC learning step ──────────────────────
        self.cog_metabolism.charge(self.cog_metabolism.learning_cost())

    def _update_policy_online(self, viability: float, reward: float) -> None:
        if self.prev_policy_latent is None or self.prev_policy_action is None:
            return
        delta_v = 0.0 if self.prev_viability is None else viability - float(self.prev_viability)
        advantage = jnp.array(reward + 2.0 * delta_v + 0.25 * viability,
                              dtype=jnp.float32)
        latent = jnp.array(self.prev_policy_latent, dtype=jnp.float32)
        action = jnp.array(self.prev_policy_action, dtype=jnp.float32)
        loss, grads = jax.value_and_grad(gaussian_policy_online_loss)(
            self.policy.to_params(), latent, action, advantage, self.config)
        # ── Metabolic supervenience: learning rate gated by metabolic state ───
        grads = self._scale_grads(grads)
        updates, self.policy_opt_state = self.policy_opt.update(
            grads, self.policy_opt_state, self.policy.to_params())
        new_params = optax.apply_updates(self.policy.to_params(), updates)
        self.policy.from_params(new_params)
        self.policy_loss_ema = 0.98 * self.policy_loss_ema + 0.02 * float(loss)
        # ── Thermodynamic cost: policy gradient step ─────────────────────────
        self.cog_metabolism.charge(self.cog_metabolism.learning_cost())

    def _update_sensorimotor_model(self, S_full: jnp.ndarray) -> None:
        if self.prev_S_full is None or self.prev_action is None:
            return
        S_prev = jnp.array(self.prev_S_full, dtype=jnp.float32)
        action_prev = jnp.array(self.prev_action, dtype=jnp.float32)
        loss_aux, grads = jax.value_and_grad(sensorimotor_loss, has_aux=True)(
            self.sensorimotor_params, S_prev, action_prev, S_full)
        loss, aux = loss_aux
        # ── Metabolic supervenience: learning rate gated by metabolic state ───
        grads = self._scale_grads(grads)
        updates, self.sensorimotor_opt_state = self.sensorimotor_opt.update(
            grads, self.sensorimotor_opt_state, self.sensorimotor_params)
        self.sensorimotor_params = optax.apply_updates(
            self.sensorimotor_params, updates)
        self.sensorimotor_mse_ema = (
            0.98 * self.sensorimotor_mse_ema + 0.02 * float(aux['mse']))
        # ── Thermodynamic cost: sensorimotor learning step ───────────────────
        self.cog_metabolism.charge(self.cog_metabolism.learning_cost())

    def _update_wm_online(self, S_full: jnp.ndarray) -> None:
        if self.prev_wm_train is None:
            return
        x_prev, h_f_prev, h_m_prev, h_s_prev, t_prev = self.prev_wm_train
        (loss, aux), grads = jax.value_and_grad(wm_online_loss, has_aux=True)(
            self.wm.to_params(),
            jnp.array(x_prev, dtype=jnp.float32),
            jnp.array(h_f_prev, dtype=jnp.float32),
            jnp.array(h_m_prev, dtype=jnp.float32),
            jnp.array(h_s_prev, dtype=jnp.float32),
            int(t_prev),
            S_full,
            self.config)
        # ── Metabolic supervenience: learning rate gated by metabolic state ───
        grads = self._scale_grads(grads)
        updates, self.wm_online_state = self.wm_online_opt.update(
            grads, self.wm_online_state, self.wm.to_params())
        self.wm.from_params(optax.apply_updates(self.wm.to_params(), updates))
        self.wm_online_mse_ema = (
            0.98 * self.wm_online_mse_ema + 0.02 * float(aux['mse']))
        # ── Thermodynamic cost: world model online learning step ─────────────
        self.cog_metabolism.charge(self.cog_metabolism.learning_cost())

    # ── Main step ────────────────────────────────────────────────────────────

    def step(self, S0: np.ndarray, action: np.ndarray,
             reward: float = 0.0, rng: Optional[jax.random.KeyArray] = None,
             external_field: Optional[SigmaFieldGeometric] = None,
             pump_field: bool = True) -> Tuple[np.ndarray, dict]:
        if rng is None:
            rng = jax.random.PRNGKey(self._step)
        rng, key_wm, key_pol, key_info = random.split(rng, 4)
        cog = self.config.cognition
        if external_field is not None:
            self.sigma_field = external_field
            self.metastability_field.field = external_field

        # ── 1. Encode observation ──────────────────────────────────────────
        d_E = self.config.d_E
        obs_np  = np.array(S0[:d_E] if len(S0) >= d_E
                           else np.pad(S0, (0, d_E - len(S0))))
        obs_jnp = jnp.array(obs_np, dtype=jnp.float32)
        viability, organism_obs = self._viability_from_obs(obs_jnp)
        viability_features = self._viability_features(obs_jnp)

        # ── Supervenience quality gates (derived from body state cached by
        #    self_maintain() before this call; always available this step) ──
        _info_quality = self.info_supervenience.compute_quality(
            self._current_si or {}, self._current_body_energy)

        self._adapt_viability_actor(viability, reward)
        self._update_enactive_actor_critic(viability, reward, viability_features)
        self._update_policy_online(viability, reward)

        # Assemble full state vector
        S_full = self.ss.assemble(obs_jnp, self.deter_state,
                                  self.stoch_state[:self.config.d_I])
        peer_summary, world_summary, consequence_risk = self._update_auxiliary_context(
            S_full, organism_obs, viability, reward)
        self._update_sensorimotor_model(S_full)
        self._update_wm_online(S_full)
        if self.prev_prediction is None:
            wm_pred_mse = 0.0
        else:
            wm_pred_mse = float(jnp.mean((jnp.array(self.prev_prediction) - S_full) ** 2))

        # ── 2. Project obs → n_slots features ─────────────────────
        features = self._obs_to_features(obs_jnp)    # (1, n_slots, slot_dim)

        # ── 3. Slot attention → ObjectBus ──────────────────────────────────
        slots_init = self.prev_slots if self.prev_slots is not None else None
        slots, attn_weights = self.slot_attn(features, key_wm, slots_init=slots_init)
        slots_2d = slots[0]                             # (n_slots, slot_dim)
        self.prev_slots = slots
        mask_np  = np.ones(cog.n_slots, dtype=np.float32)

        # ── Metabolic supervenience: narrow attentional spotlight when starved ─
        # Hungry agents deploy fewer slots — a smaller "spotlight of attention".
        # The lowest-norm (least activated) slots are zeroed beyond the budget.
        _n_active_slots = self.metabolic_super.attention_n_active(
            self._current_body_energy, cog.n_slots)
        if _n_active_slots < cog.n_slots:
            _slot_norms = jnp.linalg.norm(slots_2d, axis=-1)          # (n_slots,)
            _sorted_idx = jnp.argsort(_slot_norms)                    # ascending
            _active_mask = jnp.zeros(cog.n_slots).at[
                _sorted_idx[cog.n_slots - _n_active_slots:]].set(1.0)
            slots_2d = slots_2d * _active_mask[:, None]
            mask_np  = np.array(_active_mask)
        # ── Thermodynamic cost: slot attention iterations ────────────────────
        self.cog_metabolism.charge(self.cog_metabolism.attention_cost(
            _n_active_slots, self.slot_attn.iters, cog.slot_dim))

        # Derive pseudo-positions from attention weights and scale to field space
        slot_positions = jnp.zeros((cog.n_slots, 3))
        # Spread slots evenly across field space (deterministic seed positions)
        for i in range(cog.n_slots):
            slot_positions = slot_positions.at[i].set(jnp.array([
                float(i % cog.object_world_size),
                float((i // int(math.sqrt(cog.n_slots))) % cog.object_world_size),
                float(cog.world_depth // 2),
            ]))
        slot_energies = jnp.linalg.norm(slots_2d, axis=-1)    # (n_slots,)

        # ── 4. Pump slots into sigma field PDE ───────────────────
        self.pending_slot_positions = slot_positions
        self.pending_slot_energies = slot_energies
        if pump_field:
            self.sigma_field.step(
                agent_positions=slot_positions,
                agent_energies=slot_energies,
                dt=0.05,
                D=cog.field_diffusion,
                decay=cog.field_decay_rate,
                pump_gain=cog.field_pump_gain,
            )

        # ── 5. Causal learning + do-intervention ──────────────────────────
        slots_np = np.array(slots_2d)
        slots_np, interv_idx = self.causal_learner.maybe_intervene(
            slots_np, self._np_rng)
        self.causal_learner.update(slots_np)
        cb = self.causal_learner.to_bus(jnp.array(slots_np))

        # ── 6. World model step ────────────────────────────────────────────
        t_enc = get_time_encoding(
            jnp.array([float(self._step) * self.config.dt]),
            jnp.array([10., 50., 200., 1000.]),
            cog.time_embed_dim)
        field_ctx = jnp.zeros(cog.spatial_attn_out)
        try:
            field_patch = self.sigma_field.sample_patch(slot_positions[0], patch_size=4)
            field_ctx   = field_patch[:cog.spatial_attn_out]
        except Exception as exc:
            self._record_soft_failure('field_context', exc)

        # ── Concept context injection ────────────────────────────────────────
        concept_ctx_jnp = jnp.zeros(cog.spatial_attn_out)
        if len(self.memory.episodic) >= 4:
            try:
                _raw_ctx = self.memory.retrieve_context(np.array(S_full))
                _raw_np  = np.array(_raw_ctx, dtype=np.float32)
                _cdim    = min(len(_raw_np), self.W_concept_to_ctx.shape[1])
                _pad     = max(0, self.W_concept_to_ctx.shape[1] - _cdim)
                _vec     = np.pad(_raw_np[:_cdim], (0, _pad))
                concept_ctx_jnp = jnp.tanh(
                    self.W_concept_to_ctx @ jnp.array(_vec, dtype=jnp.float32))
            except Exception as exc:
                self._record_soft_failure('memory_context', exc)

        # ── Broker feedback injection → GRU ─────────────────────────────────
        broker_ctx_jnp = jnp.zeros(cog.spatial_attn_out)
        if self.prev_broker_context is not None:
            try:
                _fb_np  = np.array(self.prev_broker_context, dtype=np.float32)
                _fbdim  = min(len(_fb_np), self.W_broker_feedback.shape[1])
                _fbpad  = max(0, self.W_broker_feedback.shape[1] - _fbdim)
                _fbvec  = np.pad(_fb_np[:_fbdim], (0, _fbpad))
                broker_ctx_jnp = jnp.tanh(
                    self.W_broker_feedback @ jnp.array(_fbvec, dtype=jnp.float32))
            except Exception as exc:
                self._record_soft_failure('broker_feedback', exc)

        # Compose enriched field context: raw field + concept memory + broker feedback
        field_ctx = field_ctx + 0.30 * concept_ctx_jnp + 0.20 * broker_ctx_jnp

        # ── Field supervenience: field is a PRECONDITION for GRU computation ──
        # Neural gain is computed from the sigma field BEFORE the GRU runs, so
        # the field shapes what the GRU computes — not just how much it expresses.
        # This is the constitutive relation: fix the field, fix the computation.
        _body_pos_approx = np.array(obs_jnp[:3])
        _neural_gain = self.field_supervenience.compute_neural_gain(
            self.sigma_field, _body_pos_approx,
            self._current_genome_fidelity)

        # (a) Pre-gate x_wm: field topology shapes representational content
        x_wm_base = jnp.concatenate([S_full, t_enc, field_ctx])
        x_wm = x_wm_base * float(_neural_gain)

        h_f_prev, h_m_prev, h_s_prev = self.h_fast, self.h_medium, self.h_slow
        S_next, h_f2, h_m2, h_s2, kl, gate_ent = self.wm.step(
            x_wm, h_f_prev, h_m_prev, h_s_prev,
            self._step, key_wm)
        S_next = jnp.clip(S_next, -5.0, 5.0)
        sm_pred = sensorimotor_predict(
            self.sensorimotor_params, S_full, jnp.array(action, dtype=jnp.float32))
        S_next = jnp.clip(0.7 * S_next + 0.3 * sm_pred, -5.0, 5.0)
        wm_mse = wm_pred_mse

        # (b) Constitutive mixing: h = gain * h_gru + (1 − gain) * h_field
        # h_field = 0 (vacuum attractor) — a dissipated field drives h toward
        # zero, not toward a scaled version of what GRU computed independently.
        # This means the hidden state cannot exist without the field substrate.
        self.h_fast   = h_f2 * float(_neural_gain)
        self.h_medium = h_m2 * float(_neural_gain)
        self.h_slow   = h_s2 * float(_neural_gain)

        # Thermodynamic supervenience: drop timescale layers below energy threshold.
        # This is a hard structural truncation — the slow/medium temporal contexts
        # literally do not update, not merely scale.  Their carry-forward state
        # collapses to zero, erasing temporal integration at that timescale.
        if self._thermo_n_timescale_layers < 3:
            self.h_slow   = jnp.zeros_like(self.h_slow)
        if self._thermo_n_timescale_layers < 2:
            self.h_medium = jnp.zeros_like(self.h_medium)

        # ── Thermodynamic cost: active GRU layers only ────────────────────────
        self.cog_metabolism.charge(
            self.cog_metabolism.gru_cost(
                self.config.latent_dim, n_layers=self._thermo_n_timescale_layers))

        # Update deter state from world model output
        E_next, D_next, I_next = self.ss.decompose(S_next)
        self.deter_state = D_next[:cog.deter_dim]
        self.stoch_state = jnp.pad(I_next, (0, max(0, cog.stoch_dim - I_next.shape[-1])))[:cog.stoch_dim]

        # ── 7. Anderson DEQ — joint equilibrium over [deter ‖ ws_partial] ───
        gain      = self.metastability_field.contraction_gain
        _deter_dim = cog.deter_dim
        _ws_half   = cog.workspace_dim // 2

        def _deq_f_joint(z: jnp.ndarray,
                         ctx_deter: jnp.ndarray,
                         ctx_ws: jnp.ndarray) -> jnp.ndarray:
            deter_z = z[:_deter_dim]
            ws_z    = z[_deter_dim:]
            new_deter = jnp.tanh(deter_z + 0.10 * gain * (ctx_deter - deter_z))
            new_ws    = jnp.tanh(ws_z    + 0.05        * (ctx_ws    - ws_z))
            return jnp.concatenate([new_deter, new_ws])

        _ws_ctx = self.workspace_state[:_ws_half]
        _z0     = jnp.concatenate([self.equilibrium_state, _ws_ctx])
        try:
            z_joint, _ = anderson_solver(
                _deq_f_joint, _z0,
                (self.deter_state[:_deter_dim], _ws_ctx),
                self._thermo_max_fp_iter, cog.fp_tol,
                cog.anderson_memory, cog.anderson_ridge,
                cog.anderson_damping)
            z_star    = z_joint[:_deter_dim]
            ws_z_star = z_joint[_deter_dim:]
        except Exception as exc:
            self._record_soft_failure('anderson_deq', exc)
            z_star    = self.deter_state[:_deter_dim]
            ws_z_star = _ws_ctx

        deq_res = float(jnp.linalg.norm(z_star - self.equilibrium_state))
        self.stability.record_deq_residual(deq_res)
        self.equilibrium_state = z_star
        # Blend equilibrium-resolved workspace partial back into workspace state
        self.workspace_state = self.workspace_state.at[:_ws_half].set(
            0.7 * self.workspace_state[:_ws_half] + 0.3 * ws_z_star)
        # ── Thermodynamic cost: Anderson DEQ iterations ──────────────────────
        self.cog_metabolism.charge(
            self.cog_metabolism.deq_cost(cog.deter_dim, cog.max_fp_iter))

        # ── z_star → symbolic layer ────────────────────────────────────────
        try:
            z_concept = jnp.tanh(
                self.W_z_to_concept @ z_star[:self.W_z_to_concept.shape[1]])
            z_concept_np = np.array(z_concept, dtype=np.float32)
            self.symbolic_sys.structure = (
                0.80 * self.symbolic_sys.structure + 0.20 * z_concept_np)
            _snorm = np.linalg.norm(self.symbolic_sys.structure)
            if _snorm > 1e-8:
                self.symbolic_sys.structure /= _snorm
        except Exception as exc:
            self._record_soft_failure('symbolic_projection', exc)

        # ── 8. Metastability field update ──────────────────────────────────
        workspace_activation = jnp.concatenate([
            self.deter_state[:cog.workspace_dim // 2],
            self.stoch_state[:cog.workspace_dim // 2],
        ])
        meta_stats = self.metastability_field.update(
            workspace_activation, self._np_rng)

        # ── 9. Dynamical stability (Lyapunov QR) ──────────────────────────
        if self._step % cog.lyapunov_renorm_steps == 0:
            def _f_lyap(z):
                return jnp.tanh(z + 0.1 * (self.equilibrium_state - z))
            exps = self.stability.update_lyapunov(_f_lyap, self.deter_state[:cog.deter_dim])
            self.stability.classify_phase(exps)
        self.stability.update_sparsity(self.deter_state)
        stab_bus = self.stability.to_bus()

        # ── 10. Compositional HRR binding (with cleanup)   ───────
        sym_bus  = self.symbolic_sys.bind_objects(slots_np, mask_np)
        sym_vec  = np.array(self.symbolic_sys.structure)
        sym_jnp  = jnp.array(
            sym_vec[:cog.hrr_dim] if len(sym_vec) >= cog.hrr_dim
            else np.pad(sym_vec, (0, cog.hrr_dim - len(sym_vec))))
        # ── Informational supervenience: symbolic layer supervenes on substrate ─
        # HRR binding quality degrades when the neural substrate is compromised.
        # This means starved/injured agents lose the ability to form coherent
        # compositional symbols, not just to act on them.
        sym_jnp = self.info_supervenience.apply_symbolic_attenuation(
            sym_jnp, _info_quality['symbolic'])
        language_token = self.language.encode(sym_jnp)
        language_action = jnp.tanh(
            self.language.action_bias(language_token, MAX_MOTORS))
        language_confidence = self.language.transition_confidence()

        # ── 11. Global workspace broadcast (non-zero weights)   ────
        pooled_obj  = jnp.mean(slots_2d, axis=0)
        pooled_caus = jnp.mean(cb.nodes, axis=0)
        obj_ws   = jnp.tanh(self.W_obj_to_ws   @ pooled_obj)
        caus_ws  = jnp.tanh(self.W_caus_to_ws  @ pooled_caus)
        sym_ws   = jnp.tanh(self.W_sym_to_ws   @ sym_jnp)
        candidates = jnp.stack([obj_ws, caus_ws, sym_ws], axis=0)
        ws_broadcast = candidates[jnp.argmax(jnp.linalg.norm(candidates, axis=-1))]
        ws_salience  = jax.nn.softmax(ws_broadcast)
        # ── Informational supervenience: workspace EMA collapses with substrate ─
        # Before the EMA update runs, scale workspace_state by workspace quality.
        # A dissipated neural substrate doesn't carry forward a stale coherent
        # workspace — the EMA itself collapses, not just the output of this step.
        # This is constitutive: the workspace cannot persist without its basis.
        _ws_q = float(_info_quality['workspace'])
        self.workspace_state = self.workspace_state * _ws_q
        self.workspace_state, ws_out, ws_focus, ws_entropy = update_global_workspace(
            S_full, self.workspace_state, self.workspace_params,
            self.affect_state, self.config)

        # Blend WTA broadcast with workspace EMA
        ws_final = 0.5 * ws_broadcast + 0.5 * ws_out
        # ── Informational supervenience: workspace degrades with neural substrate
        # When the world model substrate is compromised (degraded SI or low energy),
        # the workspace state becomes noisy — higher cognition literally loses
        # coherence because its physical basis is impaired.
        ws_final = self.info_supervenience.apply_workspace_noise(
            ws_final, _info_quality['workspace'], key_info)
        # ── Thermodynamic cost: workspace broadcast + affect ────────────────
        self.cog_metabolism.charge(
            self.cog_metabolism.workspace_cost(cog.workspace_dim))
        identity_drift = self.identity_mon.update(np.array(S_full), np.array(self.last_S))
        stage_metrics = {
            'age': organism_obs['age'],
            'viability': viability,
            'identity_drift': identity_drift,
        }
        dev_stage = self.dev_gate.check(stage_metrics)
        dev_stage_name = self.dev_gate.name()

        # ── 12. Affect ─────────────────────────────────────────────────────
        pred_err  = float(jnp.mean(jnp.abs(self.deter_state - self.equilibrium_state)))
        homeo_dev = float(jnp.mean(jnp.abs(obs_jnp - 0.35)))
        self_stability = self.meta_self.update(
            1.0 / (1.0 + identity_drift + pred_err + consequence_risk))
        drives    = self.drive_system.update(
            organism_obs, reward,
            self.novelty.score(np.array(S_full)), wm_mse)

        if self.config.use_affect:
            _, _, _, new_aff = compute_affect(
                S_full, pred_err, homeo_dev,
                self.affect_params, self.affect_state,
                drives, self.config)
            self.affect_state = new_aff

        # ── 12.5 ToolRequestBroker — decide mode and build next-step context ─
        survival_pressure = float(jnp.clip(1.0 - viability, 0.0, 1.0))
        broker_mode = self.broker.decide(
            ws_final, drives,
            viability, float(ws_entropy), pred_err, survival_pressure)

        _broker_context_next: Optional[np.ndarray] = None
        if broker_mode == ToolRequestBroker.MODE_IMAGINE:
            # Run internal rollout; imagined terminal state feeds GRU next step
            try:
                _S_imagined = self.broker.imagine(
                    S_full, self.wm.to_params(), self.config, rng)
                _broker_context_next = np.array(_S_imagined, dtype=np.float32)
            except Exception as exc:
                self._record_soft_failure('broker_imagine', exc)
                _broker_context_next = np.array(S_full, dtype=np.float32)
        elif broker_mode == ToolRequestBroker.MODE_QUERY:
            # Concept retrieval already done above; store raw context for injection
            try:
                _broker_context_next = np.array(
                    self.memory.retrieve_context(np.array(S_full)), dtype=np.float32)
            except Exception as exc:
                self._record_soft_failure('broker_query', exc)
                _broker_context_next = None
        elif broker_mode == ToolRequestBroker.MODE_DEFER:
            # Feed current ws_final back as next-step context (stronger recurrence)
            _broker_context_next = np.array(ws_final, dtype=np.float32)
        # ACT: no injection needed; prev_broker_context cleared

        # ── 13. Policy: sample action from latent ─────────────────────────
        wdim = cog.workspace_dim
        latent = jnp.pad(ws_final, (0, max(0, self.config.latent_dim - wdim))
                         )[:self.config.latent_dim]
        action_jnp, log_prob, entropy = GaussianPolicy.sample_and_log_prob(
            latent, key_pol, self.policy.to_params())
        # ── Thermodynamic cost: policy forward pass ──────────────────────────
        self.cog_metabolism.charge(self.cog_metabolism.policy_cost())
        reflex_action = self._viability_reflex(obs_jnp)
        learned_action = jnp.tanh(self.viability_actor_W @ viability_features)
        enactive_action = enactive_ac_mean(self.enactive_ac_params, viability_features)
        planner_action = self.planner.plan(S_full, drives, world_summary)
        social_action = self.social.action_prior(peer_summary, viability_features)
        objective_weights = self.meta_hypernet.forward(drives, ws_final)
        # ── Metabolic supervenience: narrow memory retrieval when starved ─────
        _mem_k = self.metabolic_super.memory_retrieval_k(
            self._current_body_energy, k_base=8)
        memory_action = self.memory.retrieve_action_prior(np.array(S_full), k=_mem_k)
        self.cog_metabolism.charge(self.cog_metabolism.memory_retrieve_cost())
        energy_pressure = jnp.clip(1.0 - organism_obs['energy'], 0.0, 1.0)
        policy_suppression = self.config.cognition.low_viability_policy_suppression * (
            0.5 * survival_pressure + 0.5 * energy_pressure)
        policy_weight = jnp.maximum(0.05, 0.35 - policy_suppression)
        reflex_weight = 0.30 + 0.30 * jnp.maximum(survival_pressure, energy_pressure)
        stage_scale = jnp.array([0.75, 0.9, 1.0, 1.1], dtype=jnp.float32)[dev_stage]
        exploration_scale = jnp.array([0.55, 0.75, 0.95, 1.0], dtype=jnp.float32)[dev_stage]
        policy_component = exploration_scale * (0.20 + objective_weights[0]) * policy_weight * action_jnp
        reflex_component = stage_scale * (0.20 + objective_weights[1]) * reflex_weight * reflex_action
        enactive_component = (0.10 + 0.35 * objective_weights[2]) * enactive_action
        memory_component = (self.config.cognition.enactive_memory_gain
                            * (0.50 + objective_weights[3]) * memory_action)
        planner_component = (0.08 + 0.35 * objective_weights[4]) * planner_action
        action_mix = (
            policy_component
            + reflex_component
            + 0.15 * learned_action
            + enactive_component
            + memory_component
            + planner_component
            + 0.08 * social_action
            + 0.04 * language_confidence * language_action)

        # ── Broker mode gates motor output ────────────────────────────────────
        _broker_motor_scale = {
            ToolRequestBroker.MODE_ACT:     1.00,
            ToolRequestBroker.MODE_IMAGINE: 0.00,
            ToolRequestBroker.MODE_QUERY:   1.00,
            ToolRequestBroker.MODE_DEFER:   0.10,
        }.get(broker_mode, 1.0)
        if _broker_motor_scale < 1.0:
            # Preserve homeostatic reflex even in suppressed modes
            action_mix = (
                _broker_motor_scale * action_mix
                + (1.0 - _broker_motor_scale) * reflex_weight * reflex_action)

        action_out = np.array(jnp.clip(action_mix, -3.0, 3.0))[:MAX_MOTORS]

        # ── 14. Free energy computation ────────────────────────────────────
        fe_terms = self.free_energy.compute(
            prediction_error  = pred_err,
            deter             = self.deter_state,
            equilibrium       = self.equilibrium_state,
            entropy_composite = float(ws_entropy),
            causal_adj        = cb.adjacency,
            topo_charge       = self.sigma_field.total_charge(),
            sparsity          = self.stability.sparsity_ema,
        )

        # ── 15. Hereditary replication ─────────────────────────────────────
        self.hereditary.replicate(
            current_structure=np.array(self.symbolic_sys.structure),
            free_energy=fe_terms['total'])
        self.hereditary.inject(self.symbolic_sys)

        # ── 16. Memory ─────────────────────────────────────────────────────
        S_arr  = np.array(S_full)
        S_next_arr = np.array(S_next)
        # ── Metabolic supervenience: consolidation quality gated by biosyn ───
        # Hungry agents (low energy / low biosynthetic budget) consolidate fewer
        # memories or skip consolidation entirely.  This gates the metabolically
        # expensive process of transferring episodic → semantic memory.
        _consol_cycles = self.metabolic_super.consolidation_cycles(
            self._current_body_energy, self._current_biosynthetic)
        self.memory._metabolic_consolidation_cycles = _consol_cycles
        if _consol_cycles > 0:
            self.cog_metabolism.charge(
                self.cog_metabolism.memory_consolidate_cost() * _consol_cycles)
        self.memory.add(S_arr, S_next_arr, reward,
                        prediction_error=pred_err,
                        action=action_out,
                        affect_state=np.array(self.affect_state))
        # ── Thermodynamic cost: episodic write ───────────────────────────────
        self.cog_metabolism.charge(self.cog_metabolism.memory_add_cost())
        self.last_S = S_full

        # ── Misc cognition updates ─────────────────────────────────────────
        self.self_model.update(S_arr)
        if self._step % self.config.cognition.developmental_memory_interval == 0:
            self.autobio.record(self._step, {
                'stage': dev_stage_name,
                'age': organism_obs['age'],
                'viability': viability,
                'energy': organism_obs['energy'],
                'membrane': organism_obs['membrane'],
                'identity_drift': identity_drift,
                'inventory': organism_obs['inventory'],
            })
        self.competence_ema = (0.98 * self.competence_ema + 0.02 * float(reward))
        self.survival_ema = 0.99 * self.survival_ema + 0.01 * viability
        self.prev_viability = viability
        self.prev_viability_features = np.array(viability_features)
        self.prev_action = np.array(action_out)
        self.prev_S_full = np.array(S_full)
        self.prev_prediction = np.array(S_next)
        self.prev_wm_train = (
            np.array(x_wm),
            np.array(h_f_prev),
            np.array(h_m_prev),
            np.array(h_s_prev),
            int(self._step),
        )
        self.prev_policy_latent = np.array(latent)
        self.prev_policy_action = np.array(action_out)
        # ── Broker state carry-forward ─────────────────────────────────────
        self.prev_ws_final      = np.array(ws_final)
        self.prev_broker_context = _broker_context_next

        self._step += 1

        metrics = {
            # World model
            'wm_mse':        wm_mse,
            'kl':            float(kl),
            'gate_entropy':  float(gate_ent),
            # Stability
            'lambda_max':    stab_bus['lambda_max'],
            'phase':         stab_bus['phase'],
            'sparsity':      stab_bus['sparsity'],
            'convergence_r': stab_bus['convergence_r'],
            # Field
            'phi_eoc':       meta_stats['phi_eoc'],
            'r_kuramoto':    meta_stats['r_kura'],
            'tau_soc':       meta_stats['tau_soc'],
            'hopf_mu':       meta_stats['hopf_mu'],
            'field_phase':   meta_stats['phase'],
            # Broker
            **self.broker.snapshot(),
            'field_action':  meta_stats['action'],
            'contraction_gain': meta_stats['contraction_gain'],
            'topo_charge':   self.sigma_field.total_charge(),
            **getattr(self.sigma_field, 'last_stability', {}),
            'soft_failure_count': int(sum(self.soft_failures.values())),
            # Symbolic
            'hrr_n_bound':   sym_bus.n_bound,
            'hrr_quality':   sym_bus.retrieval_q,
            'heredity_gen':  self.hereditary._generation,
            # Causal
            'causal_density': float(np.mean(self.causal_learner.C > 0)),
            'interv_idx':    interv_idx,
            # Free energy
            'free_energy':   fe_terms['total'],
            'F_prediction':  fe_terms['prediction'],
            'F_homeostasis': fe_terms['homeostasis'],
            'F_structural':  fe_terms['structural'],
            # Affect
            'valence':       float(jnp.mean(self.affect_state)),
            'arousal':       float(jnp.linalg.norm(self.affect_state)),
            # Meta
            'viability':      viability,
            'survival_ema':   self.survival_ema,
            'energy':         organism_obs['energy'],
            'health':         organism_obs['health'],
            'membrane':       organism_obs['membrane'],
            'hazard_prox':    organism_obs['hazard_prox'],
            'resource_prox':  1.0 - organism_obs['resource_dist'],
            'inventory':      organism_obs['inventory'],
            'age':            organism_obs['age'],
            'dev_stage':      dev_stage,
            'dev_stage_name': dev_stage_name,
            'identity_drift': identity_drift,
            'autobio_events': len(self.autobio._log),
            'viability_actor_norm': float(jnp.linalg.norm(self.viability_actor_W)),
            'enactive_td_ema': self.enactive_td_ema,
            'enactive_loss_ema': self.enactive_loss_ema,
            'enactive_action_norm': float(jnp.linalg.norm(enactive_action)),
            'planner_action_norm': float(jnp.linalg.norm(planner_action)),
            'social_action_norm': float(jnp.linalg.norm(social_action)),
            'language_token': int(language_token),
            'language_confidence': language_confidence,
            'objective_policy_w': float(objective_weights[0]),
            'objective_reflex_w': float(objective_weights[1]),
            'objective_planner_w': float(objective_weights[4]),
            'peer_need': peer_summary.get('peer_need', 0.0),
            'peer_count': peer_summary.get('peer_count', 0),
            'tom_error': peer_summary.get('tom_error', 0.0),
            'consequence_risk': consequence_risk,
            'self_stability': self_stability,
            'policy_loss_ema': self.policy_loss_ema,
            'sensorimotor_mse_ema': self.sensorimotor_mse_ema,
            'wm_online_mse_ema': self.wm_online_mse_ema,
            'memory_action_norm': float(jnp.linalg.norm(memory_action)),
            'death_count':    organism_obs.get('death_count', 0),
            'competence_ema': self.competence_ema,
            'workspace_focus': float(ws_focus),
            'reservoir_T':   self.sigma_field.reservoir.T,
            'step':          self._step,
            # ── Supervenience metrics ──────────────────────────────────────
            'cog_cost_ema':       self.cog_metabolism.total_cost_ema,
            'field_neural_gain':  self.field_supervenience._gain_ema,
            'info_q_neural':      _info_quality.get('neural',    1.0),
            'info_q_workspace':   _info_quality.get('workspace', 1.0),
            'info_q_symbolic':    _info_quality.get('symbolic',  1.0),
            'metabolic_lr_mod':   self._current_lr_mod,
            'attn_n_active':      _n_active_slots,
            'consol_cycles':      self.memory._metabolic_consolidation_cycles
                                  if hasattr(self.memory, '_metabolic_consolidation_cycles')
                                  else 1,
            # Thermodynamic supervenience: iteration budgets set as preconditions
            # by self_maintain(); zero means the field is dead, not just weak.
            'thermo_max_fp_iter':        self._thermo_max_fp_iter,
            'thermo_n_timescale_layers': self._thermo_n_timescale_layers,
        }
        self.last_metrics = metrics
        # ── Physical supervenience: action is mediated by the body's physical state
        # Three components combine multiplicatively:
        #   cog_gate   — structural integrity of the neural substrate (SI: policy × wm)
        #   energy_gate — immediate motor power available (body.energy proxy)
        #   viab_gate   — viability substrate integrity (SI: viability module)
        # This implements the constraint that neural activation → physical force
        # passes through the body's physical condition, not just cognitive capacity.
        if self._current_si is not None:
            _cog_gate  = float(np.sqrt(max(0.0,
                self._current_si.get('policy',      1.0) *
                self._current_si.get('world_model', 1.0))))
            # Energy-dependent motor actuation: at zero energy only 15% output
            _energy_gate = float(np.clip(
                0.15 + 0.85 * self._current_body_energy, 0.05, 1.0))
            # Viability module integrity gates action routing fidelity
            _viab_gate = float(np.clip(
                self._current_si.get('viability', 1.0) ** 0.5, 0.05, 1.0))
            _phys_gate = _cog_gate * _energy_gate * _viab_gate
            action_out = action_out * _phys_gate
        return action_out, metrics

    # ── Agent as locus of causation ──────────────────────────────────────────
    # The agent reaches into the world and maintains its own existence.
    # The outer loop provides a physics substrate; the agent drives it —
    # not the other way around.

    def self_maintain(self, world: 'World3D', body: 'AgentBodyPhys',
                      all_bodies: List['AgentBodyPhys'],
                      prev_action: np.ndarray) -> Tuple[bool, np.ndarray, dict]:
        """
        One self-maintenance cycle.

        Sequence (all initiated by the agent):
          1. Agent assembles its own sensory picture from the world.
          2. Agent pumps field to maintain its genome; pays energy.
          3. Structural integrity is exposed to the cognitive step.
          4. Agent runs its cognitive cycle and decides action.
          5. Agent applies resulting force to its own body (self-directed).
          6. Agent self-assesses its viability and signals if it has died.

        Returns (alive, action_out, metrics).
        """
        # 1. Sensory self-assembly ────────────────────────────────────────────
        rich        = build_rich_body(body, efference=prev_action)
        q_scalar    = body.last_q
        field_patch = world.field.sample_patch(jnp.array(body.pos), patch_size=4)
        field_grad  = world.field.field_gradient(jnp.array(body.pos))
        topo_stab   = float(abs(world.field.total_charge()))
        attn_ctx    = world.affordance_context(body)
        obs = np.array(observe_full_vector(
            rich, body.energy, body.health,
            body.inventory / max(1, world.n_resources),
            field_patch=field_patch, q_scalar=q_scalar,
            field_grad=field_grad, topo_stability=topo_stab,
            attn_context=attn_ctx))
        self._peer_observations = []
        for peer_body in all_bodies:
            if peer_body is body:
                continue
            peer_rich = build_rich_body(peer_body)
            peer_patch = world.field.sample_patch(jnp.array(peer_body.pos), patch_size=4)
            peer_grad = world.field.field_gradient(jnp.array(peer_body.pos))
            peer_ctx = world.affordance_context(peer_body)
            self._peer_observations.append(np.array(observe_full_vector(
                peer_rich, peer_body.energy, peer_body.health,
                peer_body.inventory / max(1, world.n_resources),
                field_patch=peer_patch,
                q_scalar=peer_body.last_q,
                field_grad=peer_grad,
                topo_stability=topo_stab,
                attn_context=peer_ctx)))

        # 2. Maintain genome in sigma field ───────────────────────────────────
        gf_cost = self.genome_field_iface.write_to_field(self.genome, body, world.field)
        body.energy = max(0.0, body.energy - gf_cost)
        body.genome_field_fidelity = self.genome_field_iface.genome_fidelity(
            self.genome, body, world.field)

        # 3. Expose structural integrity for cognitive gating ─────────────────
        self._current_si = body.structural_integrity

        # 3b. Cache body metabolic state for supervenience gates inside step() ─
        # step() uses these to gate attention breadth, learning rates, memory
        # retrieval k, symbolic attenuation, and workspace noise — all without
        # needing a reference to body.  Must be set before calling step().
        self._current_body_energy     = float(body.energy)
        self._current_biosynthetic    = float(body.biosynthetic_budget)
        self._current_genome_fidelity = float(body.genome_field_fidelity)
        self._current_lr_mod = self.metabolic_super.learning_rate_scale(
            self._current_body_energy, self._current_biosynthetic)

        # 4. Cognitive cycle ──────────────────────────────────────────────────
        # ── Thermodynamic supervenience: energy budget is a PRECONDITION ──────
        # Available energy determines how many iterations and timescale layers
        # are allowed to execute.  Starvation truncates computation before it
        # runs — not as a post-hoc scaling applied to its output.
        #
        # Thresholds (calibrated so normal operation uses full budgets):
        #   energy ≥ 0.65 → all 3 timescale layers + full Anderson iters
        #   energy ≥ 0.35 → fast + medium layers; Anderson at 60% capacity
        #   energy <  0.35 → fast layer only; Anderson at minimum viable iters
        _energy_frac = float(np.clip(body.energy, 0.0, 1.0))
        cog = self.config.cognition
        self._thermo_max_fp_iter = max(1, int(round(cog.max_fp_iter * _energy_frac)))
        self._thermo_n_timescale_layers = (
            3 if _energy_frac >= 0.65 else
            2 if _energy_frac >= 0.35 else
            1
        )

        action_out, metrics = self.step(
            obs, prev_action, reward=body.last_reward,
            external_field=world.field, pump_field=False)

        # 4b. Flush accumulated cognitive metabolic cost → body energy ────────
        # This is the thermodynamic closure: every cognitive operation debited
        # to cog_metabolism is now settled against the body's energy account.
        # An agent that cannot sustain the cognitive bill literally collapses.
        cog_cost = self.cog_metabolism.flush()
        body.energy = max(0.0, body.energy - cog_cost)
        metrics['cog_metabolic_cost'] = round(cog_cost, 7)

        # 5. Self-directed force application ──────────────────────────────────
        force = np.clip(action_out[:3], -5.0, 5.0)
        _, physics_dead = world.step_body_only(force, body)

        # 6. Self-assessment: organism is the judge of its own death ──────────
        mean_si  = float(np.mean(list(body.structural_integrity.values())))
        gf_fid   = body.genome_field_fidelity
        # Genome fidelity < 2%: hereditary information lost — cannot reproduce
        thresh_scale = (
            cog.juvenile_death_threshold_scale
            if body.age < cog.juvenile_age else 1.0)
        structural_min = cog.death_structural_min * thresh_scale
        genome_min = cog.death_genome_fidelity_min * thresh_scale
        organism_dead = (
            physics_dead or mean_si < structural_min or gf_fid < genome_min)

        metrics['structural_integrity_mean'] = round(mean_si, 4)
        metrics['genome_field_fidelity']     = round(gf_fid, 4)
        metrics['biosynthetic_budget']       = round(float(body.biosynthetic_budget), 4)
        metrics['death_structural_min']      = round(structural_min, 4)
        metrics['death_genome_fidelity_min'] = round(genome_min, 4)

        return not organism_dead, action_out, metrics

    def snapshot(self) -> dict:
        snap = {
            'step': self._step,
            **self.hereditary.snapshot(),
            **self.free_energy.couplings.snapshot(),
            **self.sigma_field.reservoir.snapshot(),
            **self.field_supervenience.snapshot(),
            **self.cog_metabolism.snapshot(),
            'n_episodic':     len(self.memory.episodic),
            'n_semantic':     len(self.memory.semantic),
            'n_concepts':     self.memory.concept_reg.n_concepts,
            'survival_ema':   self.survival_ema,
            'viability_actor_norm': float(jnp.linalg.norm(self.viability_actor_W)),
            'enactive_td_ema': self.enactive_td_ema,
            'enactive_loss_ema': self.enactive_loss_ema,
            'policy_loss_ema': self.policy_loss_ema,
            'sensorimotor_mse_ema': self.sensorimotor_mse_ema,
            'wm_online_mse_ema': self.wm_online_mse_ema,
            'dev_stage': self.dev_gate.stage,
            'dev_stage_name': self.dev_gate.name(),
            'autobio_events': len(self.autobio._log),
            'developmental_transitions': list(self.dev_gate.transitions),
            'identity_drift': self.identity_mon.drift_ema,
            # Supervenience state
            'metabolic_lr_mod':        round(self._current_lr_mod, 4),
            'body_energy_cached':      round(self._current_body_energy, 4),
            'genome_fidelity_cached':  round(self._current_genome_fidelity, 4),
        }
        return snap

    def spawn_offspring(self, rng: jax.random.KeyArray,
                        self_idx: int,
                        mutation_sigma: Optional[float] = None,
                        other_parent: Optional['TopogenesisAgent'] = None):
        sigma = (self.config.cognition.offspring_mutation_sigma
                 if mutation_sigma is None else mutation_sigma)
        child = TopogenesisAgent(
            self.config, rng, num_agents=1, self_idx=self_idx)
        rng_np = np.random.default_rng(int(jax.random.randint(rng, (), 0, 1_000_000)))

        # ── Genome: recombine if second parent available, then mutate ────────
        if other_parent is not None:
            base_genome = Genome.recombine(self.genome, other_parent.genome, rng_np)
        else:
            base_genome = self.genome
        child.genome = base_genome.mutate(rng_np, sigma=sigma)
        _dev_rng, _ = jax.random.split(rng)
        child.dev_decoder = DevelopmentalDecoder(_dev_rng, child.genome)

        # ── Genome-guided developmental decode → phenotype perturbations ─────
        parent_params = {
            'policy':       self.policy.to_params(),
            'wm':           self.wm.to_params(),
            'A_params':     self.A_params,
            'W_pol_proj':   self.W_pol_proj,
            'W_rel_proj':   self.W_rel_proj,
            'W_obj_to_ws':  self.W_obj_to_ws,
            'W_caus_to_ws': self.W_caus_to_ws,
            'W_sym_to_ws':  self.W_sym_to_ws,
            'W_obs_to_feat':self.W_obs_to_feat,
            'workspace':    self.workspace_params,
            'goal_net':     self.goal_net_params,
            'concept_enc_W':self.concept_enc_W,
            'vq_codebook':  self.vq_codebook,
            'entity_attn':  self.entity_attn.to_params(),
            'relational_net':self.relational_net.to_params(),
            'enactive_ac':  self.enactive_ac_params,
            'sensorimotor': self.sensorimotor_params,
            'viability_actor_W': self.viability_actor_W,
            # Causal loop projections — heritable so evolution can tune the loop
            'W_concept_to_ctx':  self.W_concept_to_ctx,
            'W_broker_feedback': self.W_broker_feedback,
            'W_z_to_concept':    self.W_z_to_concept,
            'broker_W':          self.broker.W,
        }
        perturbed = child.dev_decoder.decode_perturbation(
            child.genome, parent_params, rng_np)

        child.policy.from_params(perturbed['policy'])
        child.wm.from_params(perturbed['wm'])
        child.wm_online_state = child.wm_online_opt.init(child.wm.to_params())
        child.A_params        = perturbed['A_params']
        child.W_pol_proj      = perturbed['W_pol_proj']
        child.W_rel_proj      = perturbed['W_rel_proj']
        child.W_obj_to_ws     = perturbed['W_obj_to_ws']
        child.W_caus_to_ws    = perturbed['W_caus_to_ws']
        child.W_sym_to_ws     = perturbed['W_sym_to_ws']
        child.W_obs_to_feat   = perturbed['W_obs_to_feat']
        child.workspace_params= perturbed['workspace']
        child.goal_net_params = perturbed['goal_net']
        child.concept_enc_W   = perturbed['concept_enc_W']
        child.vq_codebook     = perturbed['vq_codebook']
        child.entity_attn.from_params(perturbed['entity_attn'])
        child.relational_net.from_params(perturbed['relational_net'])
        child.enactive_ac_params  = perturbed['enactive_ac']
        child.enactive_ac_state   = child.enactive_ac_opt.init(child.enactive_ac_params)
        child.sensorimotor_params = perturbed['sensorimotor']
        child.sensorimotor_opt_state = child.sensorimotor_opt.init(child.sensorimotor_params)
        child.viability_actor_W   = perturbed['viability_actor_W']
        # Causal loop projections — mutated from parent, subject to selection
        child.W_concept_to_ctx  = perturbed['W_concept_to_ctx']
        child.W_broker_feedback = perturbed['W_broker_feedback']
        child.W_z_to_concept    = perturbed['W_z_to_concept']
        child.broker.W          = perturbed['broker_W']

        # ── Hereditary channel: parent HRR population seeds child ────────────
        child.hereditary.population = (
            self.hereditary.population
            + rng_np.standard_normal(self.hereditary.population.shape).astype(np.float32)
            * sigma)
        norms = np.linalg.norm(child.hereditary.population, axis=1, keepdims=True) + 1e-8
        child.hereditary.population = child.hereditary.population / norms

        child.autobio.record(0, {
            'event': 'birth',
            'parent_step': self._step,
            'parent_survival_ema': self.survival_ema,
            'mutation_sigma': sigma,
            'genome_modules': list(child.genome.modules.keys()),
        })
        return child

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(description='topogenesis_engine')
    parser.add_argument('--steps',            type=int,   default=1000)
    parser.add_argument('--agents',           type=int,   default=2)
    parser.add_argument('--world_size',       type=int,   default=32)
    parser.add_argument('--seed',             type=int,   default=0)
    parser.add_argument('--log_every',        type=int,   default=50)
    parser.add_argument('--checkpoint_every', type=int,   default=0,
                        help='Save checkpoint every N steps. 0=disabled.')
    parser.add_argument('--checkpoint_path',  type=str,   default='ckpt',
                        help='Checkpoint filename prefix.')
    args = parser.parse_args(argv)

    print(f"[topogenesis] Initialising  steps={args.steps}  agents={args.agents}"
          f"  world={args.world_size}")

    config = TopogenesisConfig()
    rng    = jax.random.PRNGKey(args.seed)

    # Create world + agents
    world = World3D(
        size=(args.world_size, args.world_size, args.world_size),
        n_resources=config.cognition.n_resources,
        n_hazards=config.cognition.n_hazards,
        membrane_repair_rate=config.cognition.membrane_repair_rate,
        membrane_decay_rate=config.cognition.membrane_decay_rate,
        ground_resource_frac=config.cognition.ground_resource_frac,
        max_spawn_height=config.cognition.max_spawn_height,
        ground_locomotion_gain=config.cognition.ground_locomotion_gain,
        interaction_radius=config.cognition.interaction_radius,
        safe_spawn_radius=config.cognition.safe_spawn_radius,
        starter_resource_patch=config.cognition.starter_resource_patch,
        energy_decay=config.cognition.energy_decay,
        resource_energy_gain=config.cognition.resource_energy_gain,
        resource_repair_gain=config.cognition.resource_repair_gain,
        force_metabolic_cost=config.cognition.force_metabolic_cost,
        resource_regen_interval=config.cognition.resource_regen_interval,
        resource_regen_count=config.cognition.resource_regen_count,
        starter_regen_count=config.cognition.starter_regen_count)

    bodies = [AgentBodyPhys(start_pos=(args.world_size//2 + i*2,
                                       args.world_size//2, 1))
              for i in range(args.agents)]

    rng, *agent_rngs = random.split(rng, args.agents + 1)
    agents = [TopogenesisAgent(config, jax.random.PRNGKey(args.seed + i),
                               num_agents=args.agents, self_idx=i)
              for i in range(args.agents)]
    births = 0

    def maybe_reproduce(agent, body, idx, field):
        nonlocal births, rng
        cog = config.cognition
        # ── Spatial density gate: carrying capacity emerges from physics ──────
        # Max packing: one agent per (interaction_radius)^3 volume unit.
        world_volume = world.size[0] * world.size[1] * world.size[2]
        unit_vol = max(1.0, cog.interaction_radius ** 3)
        physical_capacity = world_volume / unit_vol
        if len(agents) >= physical_capacity:
            return None
        if body.repro_cooldown > 0:
            body.repro_cooldown -= 1
            return None
        mature = body.age >= cog.reproduction_min_age
        viable = (body.energy >= cog.reproduction_energy
                  and body.membrane_integrity >= cog.reproduction_membrane
                  and body.inventory >= cog.reproduction_inventory)
        if not (mature and viable):
            return None

        # Reproduction only permitted where topological charge is locally stable
        # (|Q| < 2.0) — prevents reproduction in topologically chaotic regions.
        z_idx = int(np.clip(round(float(body.pos[2])), 0, world.size[2] - 1))
        local_q = abs(field.topological_charge_at(z_idx))
        if local_q > 2.0:
            return None
        child_pos = body.pos.copy()
        offset = np.array([
            1.5 if births % 2 == 0 else -1.5,
            1.5 if (births // 2) % 2 == 0 else -1.5,
            0.0,
        ], dtype=np.float32)
        child_pos = np.clip(child_pos + offset, [0, 0, 1], [s - 1e-3 for s in world.size])
        rng, child_rng = random.split(rng)
        child_idx = len(agents) + births
        child_agent = agent.spawn_offspring(
            child_rng, child_idx, cog.offspring_mutation_sigma)
        child_body = AgentBodyPhys(start_pos=tuple(child_pos),
                                   n_joints=child_agent.genome.n_joints,
                                   n_motors=child_agent.genome.n_motors,
                                   n_tactile=child_agent.genome.n_tactile)
        child_body.energy = max(0.35, body.energy * 0.45)
        child_body.health = body.health
        child_body.membrane_integrity = max(0.85, body.membrane_integrity * 0.98)
        child_body.generation = body.generation + 1
        child_body.parent_id = body.lineage_id
        child_body.lineage_id = int(world.rng.integers(0, 1_000_000))
        body.energy = max(0.05, body.energy - cog.reproduction_energy_cost)
        body.inventory = max(0, body.inventory - cog.reproduction_inventory_cost)
        body.repro_cooldown = cog.reproduction_cooldown
        births += 1
        # Imprint child genome into sigma field at birth position.
        agent.genome_field_iface.write_offspring_genome(
            child_agent.genome, child_body, field)
        print(
            f"[topogenesis] Birth parent={idx} child={child_idx} "
            f"generation={child_body.generation} lineage={child_body.lineage_id}"
        )
        return child_agent, child_body

    # Synthetic observation from body + field
    def make_obs(body, world, agent_idx, last_action):
        rich         = build_rich_body(body, efference=last_action)
        q_scalar     = body.last_q
        field_patch  = world.field.sample_patch(
            jnp.array(body.pos), patch_size=4)
        field_grad   = world.field.field_gradient(jnp.array(body.pos))
        topo_stab    = float(abs(world.field.total_charge()))
        attn_ctx     = world.affordance_context(body)
        return np.array(observe_full_vector(
            rich, body.energy, body.health,
            body.inventory / max(1, world.n_resources),
            field_patch=field_patch,
            q_scalar=q_scalar,
            field_grad=field_grad,
            topo_stability=topo_stab,
            attn_context=attn_ctx))

    action_bufs = [np.zeros(MAX_MOTORS) for _ in range(args.agents)]
    reward_hist = [[] for _ in range(args.agents)]
    metric_hist = [[] for _ in range(args.agents)]

    # ── Serialization helpers ────────────────────────────────────────────────
    def save_checkpoint(path: str):
        """Persist population state for later resumption."""
        import pickle
        state = {
            'step': step if 'step' in dir() else 0,
            'births': births,
            'rng': np.array(rng),
            'bodies': [
                {
                    'pos': b.pos.tolist(), 'vel': b.vel.tolist(),
                    'energy': b.energy, 'health': b.health,
                    'membrane_integrity': b.membrane_integrity,
                    'repair_budget': b.repair_budget,
                    'inventory': b.inventory, 'age': b.age,
                    'death_count': b.death_count, 'generation': b.generation,
                    'lineage_id': b.lineage_id,
                    'parent_id': b.parent_id,
                    'repro_cooldown': b.repro_cooldown,
                    't': b.t,
                    'n_joints': b.n_joints,
                    'n_motors': b.n_motors,
                    'n_tactile': b.n_tactile,
                } for b in bodies
            ],
            'agent_snapshots': [a.snapshot() for a in agents],
            'agent_genomes': [a.genome.to_dict() for a in agents],
            'agent_steps': [a._step for a in agents],
            'population_size': len(agents),
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f'[topogenesis] Checkpoint saved → {path}  pop={len(agents)}  births={births}')

    # ── Inter-agent material interaction ────────────────────────────────────
    def inter_agent_material_step(bodies_list: list, world_ref):
        """
        Agents within contact radius exchange material:
        - If one has high inventory and neighbour is energy-depleted: transfer resource unit
        - Collision damage if both moving fast toward each other
        This runs after all individual body steps.
        """
        n = len(bodies_list)
        if n < 2:
            return
        for i in range(n):
            bi = bodies_list[i]
            for j in range(i + 1, n):
                bj = bodies_list[j]
                dist = float(np.linalg.norm(bi.pos - bj.pos))
                if dist > world_ref.interaction_radius * 1.5:
                    continue
                # Resource transfer: donor has surplus, receiver is depleted
                if bi.inventory >= 3 and bj.energy < 0.25:
                    bi.inventory -= 1
                    bj.energy = min(1.0, bj.energy + world_ref.resource_energy_gain * 0.5)
                    bj.membrane_integrity = min(1.0,
                        bj.membrane_integrity + 0.06)  # autopoietic synthesis from received material
                elif bj.inventory >= 3 and bi.energy < 0.25:
                    bj.inventory -= 1
                    bi.energy = min(1.0, bi.energy + world_ref.resource_energy_gain * 0.5)
                    bi.membrane_integrity = min(1.0,
                        bi.membrane_integrity + 0.06)
                # Collision: relative velocity damage
                rel_vel = float(np.linalg.norm(bi.vel - bj.vel))
                if rel_vel > 4.0 and dist < world_ref.interaction_radius * 0.6:
                    dmg = 0.005 * (rel_vel - 4.0)
                    bi.health = max(0.0, bi.health - dmg)
                    bj.health = max(0.0, bj.health - dmg)

    print("[topogenesis] Starting main loop …")
    t_start = time.time()

    for step in range(args.steps):
        # World field advances using all body positions
        world.advance_field(bodies)

        current_n = len(agents)
        cognitive_positions = []
        cognitive_energies = []
        pending_births = []
        dead_indices = []

        # Each agent self-maintains: it drives its own physics, not the loop.
        for i in range(current_n):
            agent, body = agents[i], bodies[i]
            alive, action_out, metrics = agent.self_maintain(
                world, body, bodies, action_bufs[i])
            action_bufs[i] = action_out
            reward_hist[i].append(body.last_reward)
            metric_hist[i].append(dict(metrics))
            if hasattr(agent, 'pending_slot_positions'):
                cognitive_positions.append(agent.pending_slot_positions)
                cognitive_energies.append(agent.pending_slot_energies)

            if not alive:
                dead_indices.append(i)
                print(
                    f'[topogenesis] Death  agent={i}  gen={body.generation}'
                    f'  lineage={body.lineage_id}  age={body.age}'
                    f'  si={metrics.get("structural_integrity_mean", 0):.3f}'
                    f'  gf={metrics.get("genome_field_fidelity", 0):.3f}'
                    f'  pop_before={len(agents)}'
                )
            else:
                child = maybe_reproduce(agent, body, i, world.field)
                if child is not None:
                    pending_births.append(child)

        if cognitive_positions:
            world.field.step(
                agent_positions=jnp.concatenate(cognitive_positions, axis=0),
                agent_energies=jnp.concatenate(cognitive_energies, axis=0),
                dt=0.05,
                D=config.cognition.field_diffusion,
                decay=config.cognition.field_decay_rate,
                pump_gain=config.cognition.field_pump_gain,
            )

        # ── Inter-agent material exchange ────────────────────────────────────
        inter_agent_material_step(bodies, world)

        # ── Population pruning: remove dead agents (reverse order) ───────────
        for i in sorted(dead_indices, reverse=True):
            agents.pop(i)
            bodies.pop(i)
            action_bufs.pop(i)
            reward_hist.pop(i)
            metric_hist.pop(i)

        for child_agent, child_body in pending_births:
            agents.append(child_agent)
            bodies.append(child_body)
            action_bufs.append(np.zeros(MAX_MOTORS))
            reward_hist.append([])
            metric_hist.append([])

        # ── Periodic checkpoint ──────────────────────────────────────────────
        if args.checkpoint_every > 0 and step > 0 and step % args.checkpoint_every == 0:
            save_checkpoint(f'{args.checkpoint_path}.step{step}.pkl')

        if step % args.log_every == 0:
            elapsed = time.time() - t_start
            if not agents:
                print(f'  step={step:5d}  EXTINCTION  t={elapsed:.1f}s')
                continue
            m       = agents[0].last_metrics
            print(
                f"  step={step:5d}  "
                f"F={m.get('free_energy', 0):.4f}  "
                f"V={m.get('viability', 0):.3f}  "
                f"E={m.get('energy', 0):.3f}  "
                f"H={m.get('health', 0):.3f}  "
                f"M={m.get('membrane', 0):.3f}  "
                f"D={m.get('death_count', 0)}  "
                f"Dev={m.get('dev_stage_name', '?')}  "
                f"Pop={len(agents)}  "
                f"Births={births}  "
                f"Inv={m.get('inventory', 0):.2f}  "
                f"Rsrc={m.get('resource_prox', 0):.2f}  "
                f"Hz={m.get('hazard_prox', 0):.2f}  "
                f"lambda_max={m.get('lambda_max', 0):.4f}  "
                f"phi_eoc={m.get('phi_eoc', 0):.3f}  "
                f"r_kura={m.get('r_kuramoto', 0):.3f}  "
                f"tau_soc={m.get('tau_soc', 0):.2f}  "
                f"hrr_q={m.get('hrr_quality', 0):.3f}  "
                f"T={m.get('reservoir_T', 1):.3f}  "
                f"Q={m.get('topo_charge', 0):.3f}  "
                f"field={m.get('field_phase', '?')}  "
                f"t={elapsed:.1f}s"
            )

    print("[topogenesis] Done.")
    if args.checkpoint_every > 0:
        save_checkpoint(f'{args.checkpoint_path}.final.pkl')
    if not agents:
        print("[topogenesis] Population extinct.")
        return [], world
    for i, hist in enumerate(metric_hist):
        if not hist:
            continue
        viability = np.array([m.get('viability', 0.0) for m in hist], dtype=np.float32)
        energy = np.array([m.get('energy', 0.0) for m in hist], dtype=np.float32)
        membrane = np.array([m.get('membrane', 0.0) for m in hist], dtype=np.float32)
        final = hist[-1]
        summary = {
            'agent': i,
            'steps': len(hist),
            'deaths': int(final.get('death_count', 0)),
            'viability_final': round(float(viability[-1]), 4),
            'viability_min': round(float(np.min(viability)), 4),
            'viability_mean': round(float(np.mean(viability)), 4),
            'energy_final': round(float(energy[-1]), 4),
            'energy_min': round(float(np.min(energy)), 4),
            'membrane_final': round(float(membrane[-1]), 4),
            'membrane_min': round(float(np.min(membrane)), 4),
            'inventory_final': round(float(final.get('inventory', 0.0)), 4),
            'dev_stage_final': final.get('dev_stage_name', '?'),
            'identity_drift_final': round(float(final.get('identity_drift', 0.0)), 6),
            'autobio_events': int(final.get('autobio_events', 0)),
            'population_final': len(agents),
            'births_total': births,
        }
        print("[topogenesis] Summary " + json.dumps(summary, sort_keys=True))
    return agents, world

if __name__ == '__main__':
    main()

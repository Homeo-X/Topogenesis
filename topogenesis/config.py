"""Engine configuration dataclasses and functional ablations.

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

from topogenesis.constants import ATTN_DIM, BODY_VEC_LEN, FIELD_OBS_DIM

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
    use_need_pressure: bool = True
    use_reflex: bool = True
    use_memory: bool = True
    use_world_model: bool = True
    use_future_simulation: bool = True
    use_communication: bool = True
    use_social_model: bool = True
    use_field_coupling: bool = True
    record_functional_roles: bool = True
    affect:    AffectConfig   = field(default_factory=AffectConfig)
    memory:    MemoryConfig   = field(default_factory=MemoryConfig)
    cognition: CognitiveConfig = field(default_factory=CognitiveConfig)

ABLATION_FLAGS: Dict[str, str] = {
    "affect": "use_affect",
    "needs": "use_need_pressure",
    "reflex": "use_reflex",
    "memory": "use_memory",
    "world_model": "use_world_model",
    "imagination": "use_future_simulation",
    "communication": "use_communication",
    "social": "use_social_model",
    "field": "use_field_coupling",
}

def apply_ablations(config: TopogenesisConfig,
                    ablations: Optional[List[str]] = None) -> TopogenesisConfig:
    """Return a config with named functional subsystems disabled."""
    updates = {}
    for name in ablations or []:
        key = name.strip().lower()
        if not key:
            continue
        if key not in ABLATION_FLAGS:
            allowed = ", ".join(sorted(ABLATION_FLAGS))
            raise ValueError(f"Unknown ablation '{name}'. Allowed: {allowed}")
        updates[ABLATION_FLAGS[key]] = False
    return replace(config, **updates) if updates else config

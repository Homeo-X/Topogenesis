"""The integrated Topogenesis agent.

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
    BODY_VEC_LEN,
    FIELD_GRAD_IDX,
    GENOME_DIM,
    MAX_JOINTS,
    MAX_MOTORS,
    MAX_TACTILE,
)
from topogenesis.config import ABLATION_FLAGS, TopogenesisConfig
from topogenesis.fields.sigma import SigmaFieldGeometric
from topogenesis.cognition.networks import (
    AutobiographicalMemory,
    CausalLearner,
    CompositionalSymbolicSystem,
    DriveSystem,
    DynamicalStabilityMonitor,
    EmergentMetastabilityField,
    EntityAttention,
    FieldSupervenience,
    FreeEnergyFunctional,
    GaussianPolicy,
    GoalManager,
    HierarchicalGRU,
    IdentityStabilityMonitor,
    ImaginationPlanner,
    InformationalSupervenience,
    IrreversibleConsequenceTracker,
    LanguageModule,
    MetaObjectiveHypernetwork,
    MetaStableSelfMonitor,
    MetabolicSupervenience,
    NoveltyDetector,
    PersistentWorldModel,
    RelationalReasoningNet,
    SelfModel,
    SlotAttention,
    SocialEngine,
    SparseModularMemory,
    StateSpace,
    TheoryOfMind,
    anderson_deq_joint,
    autoregressive_rollout,
    compute_affect,
    enactive_ac_loss,
    enactive_ac_mean,
    gaussian_policy_online_loss,
    get_time_encoding,
    init_A_params,
    init_affect_params,
    init_critic_params,
    init_enactive_ac_params,
    init_goal_net_params,
    init_sensorimotor_params,
    init_workspace_params,
    sensorimotor_loss,
    sensorimotor_predict,
    update_global_workspace,
    wm_online_loss,
    xavier,
)
from topogenesis.body.body_state import (
    CognitiveMetabolism,
    build_rich_body,
    observe_full_vector,
)
from topogenesis.evolution.genome import (
    DevelopmentalDecoder,
    DevelopmentalGate,
    Genome,
    GenomeFieldInterface,
    HereditaryChannel,
)

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

    def __init__(self, config: 'TopogenesisConfig', rng: jax.Array) -> None:
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
                config: 'TopogenesisConfig', rng: jax.Array) -> jnp.ndarray:
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

      13. Motor action synthesis
      14. Free energy computation + Lagrange multiplier update
      15. Hereditary replication on symbolic structure
      16. Memory store (SparseModularMemory)
    """

    def __init__(self, config: TopogenesisConfig, rng: jax.Array,
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
            self.affect_state  = jnp.zeros(config.affect.valence_dim)

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
        self.npc_affect          = NpcAffectField()
        self.npc_social_memory   = SocialMemory(max_events=1024)
        self.npc_minds           = {
            'world': OtherMindModel(agent_id='world', trust=0.5, respect=0.5),
        }
        self.last_npc_state: dict = {}
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


    def _sensory_stage(self, S0: np.ndarray, reward: float) -> dict:
        """Normalize observation input and update sensory-adjacent learners."""
        d_E = self.config.d_E
        obs_np = np.array(S0[:d_E] if len(S0) >= d_E
                          else np.pad(S0, (0, d_E - len(S0))), dtype=np.float32)
        obs_jnp = jnp.asarray(obs_np, dtype=jnp.float32)
        viability, organism_obs = self._viability_from_obs(obs_jnp)
        viability_features = self._viability_features(obs_jnp)

        info_quality = self.info_supervenience.compute_quality(
            self._current_si or {}, self._current_body_energy)

        self._adapt_viability_actor(viability, reward)
        self._update_enactive_actor_critic(viability, reward, viability_features)
        self._update_policy_online(viability, reward)

        S_full = self.ss.assemble(obs_jnp, self.deter_state,
                                  self.stoch_state[:self.config.d_I])
        peer_summary, world_summary, consequence_risk = self._update_auxiliary_context(
            S_full, organism_obs, viability, reward)
        self._update_sensorimotor_model(S_full)
        self._update_wm_online(S_full)
        if self.prev_prediction is None:
            wm_pred_mse = 0.0
        else:
            wm_pred_mse = float(jnp.mean(
                (jnp.asarray(self.prev_prediction, dtype=jnp.float32) - S_full) ** 2))

        return {
            'obs_jnp': obs_jnp,
            'viability': viability,
            'organism_obs': organism_obs,
            'viability_features': viability_features,
            'info_quality': info_quality,
            'S_full': S_full,
            'peer_summary': peer_summary,
            'world_summary': world_summary,
            'consequence_risk': consequence_risk,
            'wm_pred_mse': wm_pred_mse,
        }

    def _attention_stage(self, obs_jnp: jnp.ndarray,
                         key_wm: jax.Array,
                         pump_field: bool) -> dict:
        """Project observations into object slots and update field/causal buses."""
        cog = self.config.cognition
        features = self._obs_to_features(obs_jnp)
        slots_init = self.prev_slots if self.prev_slots is not None else None
        slots, _attn_weights = self.slot_attn(features, key_wm, slots_init=slots_init)
        slots_2d = slots[0]
        self.prev_slots = slots
        mask_np = np.ones(cog.n_slots, dtype=np.float32)

        n_active_slots = self.metabolic_super.attention_n_active(
            self._current_body_energy, cog.n_slots)
        if n_active_slots < cog.n_slots:
            slot_norms = jnp.linalg.norm(slots_2d, axis=-1)
            sorted_idx = jnp.argsort(slot_norms)
            active_mask = jnp.zeros(cog.n_slots).at[
                sorted_idx[cog.n_slots - n_active_slots:]].set(1.0)
            slots_2d = slots_2d * active_mask[:, None]
            mask_np = np.array(active_mask)

        self.cog_metabolism.charge(self.cog_metabolism.attention_cost(
            n_active_slots, self.slot_attn.iters, cog.slot_dim))

        coords = []
        grid_w = max(1, int(math.sqrt(cog.n_slots)))
        for i in range(cog.n_slots):
            coords.append([
                float(i % cog.object_world_size),
                float((i // grid_w) % cog.object_world_size),
                float(cog.world_depth // 2),
            ])
        slot_positions = jnp.asarray(coords, dtype=jnp.float32)
        slot_energies = jnp.linalg.norm(slots_2d, axis=-1)

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

        slots_np = np.array(slots_2d, dtype=np.float32)
        slots_np, interv_idx = self.causal_learner.maybe_intervene(
            slots_np, self._np_rng)
        self.causal_learner.update(slots_np)
        cb = self.causal_learner.to_bus(jnp.asarray(slots_np, dtype=jnp.float32))

        return {
            'slots_2d': slots_2d,
            'slots_np': slots_np,
            'mask_np': mask_np,
            'slot_positions': slot_positions,
            'interv_idx': interv_idx,
            'cb': cb,
            'n_active_slots': n_active_slots,
        }

    def _world_model_stage(self, S_full: jnp.ndarray, action: np.ndarray,
                           slot_positions: jnp.ndarray,
                           key_wm: jax.Array,
                           wm_pred_mse: float) -> dict:
        """Run field-conditioned predictive dynamics and update hidden state."""
        cog = self.config.cognition
        t_enc = get_time_encoding(
            jnp.asarray([float(self._step) * self.config.dt], dtype=jnp.float32),
            jnp.asarray([10., 50., 200., 1000.], dtype=jnp.float32),
            cog.time_embed_dim)
        field_ctx = jnp.zeros(cog.spatial_attn_out)
        if self.config.use_field_coupling:
            try:
                field_patch = self.sigma_field.sample_patch(slot_positions[0], patch_size=4)
                field_ctx = field_patch[:cog.spatial_attn_out]
            except Exception as exc:
                self._record_soft_failure('field_context', exc)

        concept_ctx_jnp = jnp.zeros(cog.spatial_attn_out)
        if len(self.memory.episodic) >= 4:
            try:
                raw_ctx = self.memory.retrieve_context(np.array(S_full))
                raw_np = np.array(raw_ctx, dtype=np.float32)
                cdim = min(len(raw_np), self.W_concept_to_ctx.shape[1])
                vec = np.pad(raw_np[:cdim], (0, max(0, self.W_concept_to_ctx.shape[1] - cdim)))
                concept_ctx_jnp = jnp.tanh(
                    self.W_concept_to_ctx @ jnp.asarray(vec, dtype=jnp.float32))
            except Exception as exc:
                self._record_soft_failure('memory_context', exc)

        broker_ctx_jnp = jnp.zeros(cog.spatial_attn_out)
        if self.prev_broker_context is not None:
            try:
                fb_np = np.array(self.prev_broker_context, dtype=np.float32)
                fbdim = min(len(fb_np), self.W_broker_feedback.shape[1])
                fbvec = np.pad(fb_np[:fbdim], (0, max(0, self.W_broker_feedback.shape[1] - fbdim)))
                broker_ctx_jnp = jnp.tanh(
                    self.W_broker_feedback @ jnp.asarray(fbvec, dtype=jnp.float32))
            except Exception as exc:
                self._record_soft_failure('broker_feedback', exc)

        field_ctx = field_ctx + 0.30 * concept_ctx_jnp + 0.20 * broker_ctx_jnp
        body_pos_approx = np.array(jax.device_get(S_full[:3]), dtype=np.float32)
        neural_gain = (
            self.field_supervenience.compute_neural_gain(
                self.sigma_field, body_pos_approx, self._current_genome_fidelity)
            if self.config.use_field_coupling else 1.0)

        x_wm_base = jnp.concatenate([S_full, t_enc, field_ctx])
        x_wm = x_wm_base * float(neural_gain)
        h_f_prev, h_m_prev, h_s_prev = self.h_fast, self.h_medium, self.h_slow
        if self.config.use_world_model:
            S_next, h_f2, h_m2, h_s2, kl, gate_ent = self.wm.step(
                x_wm, h_f_prev, h_m_prev, h_s_prev, self._step, key_wm)
            S_next = jnp.clip(S_next, -5.0, 5.0)
            sm_pred = sensorimotor_predict(
                self.sensorimotor_params, S_full, jnp.asarray(action, dtype=jnp.float32))
            S_next = jnp.clip(0.7 * S_next + 0.3 * sm_pred, -5.0, 5.0)
        else:
            S_next = S_full
            h_f2 = h_f_prev
            h_m2 = h_m_prev
            h_s2 = h_s_prev
            kl = jnp.array(0.0, dtype=jnp.float32)
            gate_ent = jnp.array(0.0, dtype=jnp.float32)

        self.h_fast = h_f2 * float(neural_gain)
        self.h_medium = h_m2 * float(neural_gain)
        self.h_slow = h_s2 * float(neural_gain)
        if self._thermo_n_timescale_layers < 3:
            self.h_slow = jnp.zeros_like(self.h_slow)
        if self._thermo_n_timescale_layers < 2:
            self.h_medium = jnp.zeros_like(self.h_medium)

        if self.config.use_world_model:
            self.cog_metabolism.charge(self.cog_metabolism.gru_cost(
                self.config.latent_dim, n_layers=self._thermo_n_timescale_layers))

        E_next, D_next, I_next = self.ss.decompose(S_next)
        self.deter_state = D_next[:cog.deter_dim]
        self.stoch_state = jnp.pad(
            I_next, (0, max(0, cog.stoch_dim - I_next.shape[-1])))[:cog.stoch_dim]

        return {
            'S_next': S_next,
            'wm_mse': wm_pred_mse,
            'kl': kl,
            'gate_ent': gate_ent,
            'x_wm': x_wm,
            'h_f_prev': h_f_prev,
            'h_m_prev': h_m_prev,
            'h_s_prev': h_s_prev,
            'neural_gain': float(neural_gain),
        }

    def _motor_stage(self, ws_final: jnp.ndarray, drives: jnp.ndarray,
                     key_pol: jax.Array, S_full: jnp.ndarray,
                     obs_jnp: jnp.ndarray, viability_features: jnp.ndarray,
                     organism_obs: dict, survival_pressure: float,
                     dev_stage: int, world_summary: dict, peer_summary: dict,
                     language_token: int, language_confidence: float,
                     broker_mode: int) -> dict:
        """Compose policy, reflex, memory, social, symbolic, and broker motor priors."""
        cog = self.config.cognition
        wdim = cog.workspace_dim
        latent = jnp.pad(ws_final, (0, max(0, self.config.latent_dim - wdim)))[:self.config.latent_dim]
        action_jnp, log_prob, entropy = GaussianPolicy.sample_and_log_prob(
            latent, key_pol, self.policy.to_params())
        self.cog_metabolism.charge(self.cog_metabolism.policy_cost())

        reflex_action = (
            self._viability_reflex(obs_jnp)
            if self.config.use_reflex else jnp.zeros(MAX_MOTORS, dtype=jnp.float32))
        learned_action = jnp.tanh(self.viability_actor_W @ viability_features)
        enactive_action = enactive_ac_mean(self.enactive_ac_params, viability_features)
        planner_action = self.planner.plan(S_full, drives, world_summary)
        social_action = (
            self.social.action_prior(peer_summary, viability_features)
            if self.config.use_social_model else jnp.zeros(MAX_MOTORS, dtype=jnp.float32))
        objective_weights = self.meta_hypernet.forward(drives, ws_final)
        mem_k = self.metabolic_super.memory_retrieval_k(
            self._current_body_energy, k_base=8)
        if self.config.use_memory:
            memory_action = self.memory.retrieve_action_prior(np.array(S_full), k=mem_k)
            self.cog_metabolism.charge(self.cog_metabolism.memory_retrieve_cost())
        else:
            memory_action = jnp.zeros(MAX_MOTORS, dtype=jnp.float32)

        language_action = jnp.tanh(
            self.language.action_bias(language_token, MAX_MOTORS))
        energy_pressure = jnp.clip(1.0 - organism_obs['energy'], 0.0, 1.0)
        policy_suppression = self.config.cognition.low_viability_policy_suppression * (
            0.5 * survival_pressure + 0.5 * energy_pressure)
        policy_weight = jnp.maximum(0.05, 0.35 - policy_suppression)
        reflex_weight = 0.30 + 0.30 * jnp.maximum(survival_pressure, energy_pressure)
        stage_scale = jnp.asarray([0.75, 0.9, 1.0, 1.1], dtype=jnp.float32)[dev_stage]
        exploration_scale = jnp.asarray([0.55, 0.75, 0.95, 1.0], dtype=jnp.float32)[dev_stage]

        action_mix = (
            exploration_scale * (0.20 + objective_weights[0]) * policy_weight * action_jnp
            + stage_scale * (0.20 + objective_weights[1]) * reflex_weight * reflex_action
            + 0.15 * learned_action
            + (0.10 + 0.35 * objective_weights[2]) * enactive_action
            + (self.config.cognition.enactive_memory_gain
               * (0.50 + objective_weights[3]) * memory_action)
            + (0.08 + 0.35 * objective_weights[4]) * planner_action
            + 0.08 * social_action
            + 0.04 * language_confidence * language_action)

        broker_motor_scale = {
            ToolRequestBroker.MODE_ACT: 1.00,
            ToolRequestBroker.MODE_IMAGINE: 0.00,
            ToolRequestBroker.MODE_QUERY: 1.00,
            ToolRequestBroker.MODE_DEFER: 0.10,
        }.get(broker_mode, 1.0)
        if broker_motor_scale < 1.0:
            action_mix = (broker_motor_scale * action_mix
                          + (1.0 - broker_motor_scale) * reflex_weight * reflex_action)

        return {
            'latent': latent,
            'action_mix': action_mix,
            'log_prob': log_prob,
            'entropy': entropy,
            'reflex_weight': reflex_weight,
            'reflex_action': reflex_action,
            'enactive_action': enactive_action,
            'planner_action': planner_action,
            'social_action': social_action,
            'memory_action': memory_action,
            'objective_weights': objective_weights,
        }

    def _update_npc_cognition(self, *, organism_obs: dict, viability: float,
                              pred_err: float, wm_mse: float,
                              identity_drift: float, consequence_risk: float,
                              reward: float, peer_summary: dict) -> dict:
        """Map engine viability metrics into pressure-driven cognition."""
        health = float(organism_obs.get('health', 1.0))
        membrane = float(organism_obs.get('membrane', 1.0))
        body_integrity = float(np.clip(0.5 * health + 0.5 * membrane, 0.0, 1.0))
        prediction_coherence = float(1.0 / (
            1.0 + max(0.0, pred_err) + max(0.0, wm_mse)
            + max(0.0, identity_drift)))
        memory_integrity = float(1.0 / (
            1.0 + max(0.0, self.sensorimotor_mse_ema)
            + max(0.0, self.wm_online_mse_ema)))
        peer_count = int(peer_summary.get('peer_count', 0))
        peer_need = float(peer_summary.get('peer_need', 0.0))
        social_stability = float(np.clip(
            0.55 + 0.20 * min(1, peer_count) - 0.35 * peer_need, 0.0, 1.0))
        attachment_integrity = float(np.clip(
            0.50 + 0.35 * self.survival_ema + 0.15 * max(-1.0, min(1.0, reward)),
            0.0, 1.0))
        environmental_safety = float(np.clip(
            1.0 - max(
                float(organism_obs.get('hazard_prox', 0.0)),
                float(consequence_risk)),
            0.0, 1.0))

        viability_state = ViabilityState(
            energy=float(organism_obs.get('energy', self._current_body_energy)),
            bodily_integrity=body_integrity,
            memory_integrity=memory_integrity,
            prediction_coherence=prediction_coherence,
            social_stability=social_stability,
            attachment_integrity=attachment_integrity,
            environmental_safety=environmental_safety,
        )
        need_pressure = (
            NeedPressure.from_viability(viability_state)
            if self.config.use_need_pressure else
            NeedPressure(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        if self.config.use_affect:
            self.npc_affect.update(
                prediction_error=1.0 - prediction_coherence,
                uncertainty=max(need_pressure.epistemic, need_pressure.safety),
                threat=need_pressure.safety,
                attachment_delta=attachment_integrity - self.npc_affect.attachment_security,
                social_support=social_stability,
                control_feedback=viability - need_pressure.total,
            )

        mind = self.npc_minds.setdefault(
            'world', OtherMindModel(agent_id='world', trust=0.5, respect=0.5))
        intent = CommunicationIntent(
            target_agent='world',
            intended_effect='stabilize_prediction',
            belief_to_modify='local_viability_state',
            confidence=prediction_coherence,
            emotional_weight=self.npc_affect.threat_salience,
            social_risk=need_pressure.safety,
            urgency=max(need_pressure.metabolic, need_pressure.repair),
        )
        if self.config.use_communication:
            interpretation = interpret_intent(intent, mind, self.npc_affect)
            mind.update_belief(
                intent.belief_to_modify,
                interpretation.accepted_confidence,
                source_trust=mind.trust)
            self.npc_social_memory.remember(
                agent_id='world',
                kind='self_maintenance',
                salience=max(need_pressure.total, interpretation.accepted_confidence),
                valence=viability - need_pressure.total,
                claim=need_pressure.dominant,
            )
            accepted_confidence = interpretation.accepted_confidence
            suspicion_delta = interpretation.suspicion_delta
        else:
            accepted_confidence = 0.0
            suspicion_delta = 0.0
        future_action = (
            'seek_food' if need_pressure.metabolic >= need_pressure.epistemic
            else 'verify')
        future = (
            simulate_future(
                action=future_action,
                needs=need_pressure,
                affect=self.npc_affect,
                listener=mind,
                intent=intent,
            )
            if self.config.use_future_simulation else
            simulate_future(
                action='observe',
                needs=NeedPressure(0, 0, 0, 0, 0, 0, 0),
                affect=self.npc_affect))
        modulators = need_pressure.cognitive_modulators()
        state = {
            'viability': viability_state,
            'needs': need_pressure,
            'affect': self.npc_affect,
            'intent': intent,
            'future': future,
            'modulators': modulators,
            'accepted_confidence': accepted_confidence,
            'suspicion_delta': suspicion_delta,
        }
        self.last_npc_state = state
        return state

    def step(self, S0: np.ndarray, action: np.ndarray,
             reward: float = 0.0, rng: Optional[jax.Array] = None,
             external_field: Optional[SigmaFieldGeometric] = None,
             pump_field: bool = True) -> Tuple[np.ndarray, dict]:
        if rng is None:
            rng = jax.random.PRNGKey(self._step)
        rng, key_wm, key_pol, key_info = random.split(rng, 4)
        cog = self.config.cognition
        if external_field is not None:
            self.sigma_field = external_field
            self.metastability_field.field = external_field


        sensory = self._sensory_stage(S0, reward)
        obs_jnp = sensory['obs_jnp']
        viability = sensory['viability']
        organism_obs = sensory['organism_obs']
        viability_features = sensory['viability_features']
        _info_quality = sensory['info_quality']
        S_full = sensory['S_full']
        peer_summary = sensory['peer_summary']
        world_summary = sensory['world_summary']
        consequence_risk = sensory['consequence_risk']

        attention = self._attention_stage(
            obs_jnp, key_wm, pump_field and self.config.use_field_coupling)
        slots_2d = attention['slots_2d']
        slots_np = attention['slots_np']
        mask_np = attention['mask_np']
        slot_positions = attention['slot_positions']
        interv_idx = attention['interv_idx']
        cb = attention['cb']
        _n_active_slots = attention['n_active_slots']

        world_model = self._world_model_stage(
            S_full, action, slot_positions, key_wm, sensory['wm_pred_mse'])
        S_next = world_model['S_next']
        wm_mse = world_model['wm_mse']
        kl = world_model['kl']
        gate_ent = world_model['gate_ent']
        x_wm = world_model['x_wm']
        h_f_prev = world_model['h_f_prev']
        h_m_prev = world_model['h_m_prev']
        h_s_prev = world_model['h_s_prev']
        field_neural_gain_current = world_model['neural_gain']

        # ── 7. Anderson DEQ — joint equilibrium over [deter ‖ ws_partial] ───
        gain      = self.metastability_field.contraction_gain
        _deter_dim = cog.deter_dim
        _ws_half   = cog.workspace_dim // 2

        _ws_ctx = self.workspace_state[:_ws_half]
        _z0     = jnp.concatenate([self.equilibrium_state, _ws_ctx])
        try:
            z_joint, _ = anderson_deq_joint(
                _z0, self.deter_state[:_deter_dim], _ws_ctx, gain,
                _deter_dim, self._thermo_max_fp_iter,
                cog.anderson_memory, cog.fp_tol,
                cog.anderson_ridge, cog.anderson_damping)
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

        npc_state = self._update_npc_cognition(
            organism_obs=organism_obs,
            viability=viability,
            pred_err=pred_err,
            wm_mse=wm_mse,
            identity_drift=identity_drift,
            consequence_risk=consequence_risk,
            reward=reward,
            peer_summary=peer_summary,
        )

        # ── 12.5 ToolRequestBroker — decide mode and build next-step context ─
        survival_pressure = float(jnp.clip(1.0 - viability, 0.0, 1.0))
        broker_mode = self.broker.decide(
            ws_final, drives,
            viability, float(ws_entropy), pred_err, survival_pressure)

        _broker_context_next: Optional[np.ndarray] = None
        if (broker_mode == ToolRequestBroker.MODE_IMAGINE
                and self.config.use_future_simulation
                and self.config.use_world_model):
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


        motor = self._motor_stage(
            ws_final, drives, key_pol, S_full, obs_jnp, viability_features,
            organism_obs, survival_pressure, dev_stage, world_summary,
            peer_summary, language_token, language_confidence, broker_mode)
        latent = motor['latent']
        action_mix = motor['action_mix']
        reflex_weight = motor['reflex_weight']
        enactive_action = motor['enactive_action']
        planner_action = motor['planner_action']
        social_action = motor['social_action']
        memory_action = motor['memory_action']
        objective_weights = motor['objective_weights']
        npc_motor_gate = float(np.clip(
            0.35 + 0.65 * motor['objective_weights'][1]
            * npc_state['modulators']['risk_tolerance'],
            0.15, 1.0))
        action_mix = (
            npc_motor_gate * action_mix
            + (1.0 - npc_motor_gate) * reflex_weight * motor['reflex_action'])
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
        _consol_cycles = (
            self.metabolic_super.consolidation_cycles(
                self._current_body_energy, self._current_biosynthetic)
            if self.config.use_memory else 0)
        self.memory._metabolic_consolidation_cycles = _consol_cycles
        if _consol_cycles > 0:
            self.cog_metabolism.charge(
                self.cog_metabolism.memory_consolidate_cost() * _consol_cycles)
        if self.config.use_memory:
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
            'npc_affect_stability': self.npc_affect.stability,
            'npc_affect_safety': self.npc_affect.safety,
            'npc_threat_salience': self.npc_affect.threat_salience,
            'npc_need_total': npc_state['needs'].total,
            'npc_need_dominant': npc_state['needs'].dominant,
            'npc_future_action': npc_state['future'].action,
            'npc_future_value': npc_state['future'].value,
            'npc_motor_gate': npc_motor_gate,
            'npc_social_memory_events': len(self.npc_social_memory.events),
            'npc_message_confidence': npc_state['accepted_confidence'],
            'npc_message_suspicion': npc_state['suspicion_delta'],
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
        if self.config.record_functional_roles:
            functional_roles = {
                'viability_pressure': float(1.0 - viability),
                'need_pressure': float(npc_state['needs'].total),
                'affect_stability': float(self.npc_affect.stability),
                'reflex_norm': float(jnp.linalg.norm(motor['reflex_action'])),
                'memory_prior_norm': float(jnp.linalg.norm(memory_action)),
                'world_model_error': float(wm_mse),
                'future_value': float(npc_state['future'].value),
                'communication_confidence': float(npc_state['accepted_confidence']),
                'social_prior_norm': float(jnp.linalg.norm(social_action)),
                'field_neural_gain': float(field_neural_gain_current),
                'final_action_norm': float(jnp.linalg.norm(jnp.asarray(action_out))),
            }
            metrics['functional_roles'] = functional_roles
            for role_name, role_value in functional_roles.items():
                metrics[f'role_{role_name}'] = role_value
            metrics['ablations_active'] = ",".join([
                name for name, attr in ABLATION_FLAGS.items()
                if not getattr(self.config, attr)
            ])
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
        if self.config.use_field_coupling:
            gf_cost = self.genome_field_iface.write_to_field(self.genome, body, world.field)
            body.energy = max(0.0, body.energy - gf_cost)
            body.genome_field_fidelity = self.genome_field_iface.genome_fidelity(
                self.genome, body, world.field)
        else:
            gf_cost = 0.0
            body.genome_field_fidelity = 1.0

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

    def spawn_offspring(self, rng: jax.Array,
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

"""Neural and memory primitives: world model, policy, workspace, symbolic systems.

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

from topogenesis.constants import (
    ATTN_DIM,
    BODY_VEC_LEN,
    FIELD_OBS_START,
    MAX_MOTORS,
    sigmoid,
)
from topogenesis.config import TopogenesisConfig
from topogenesis.fields.sigma import SigmaFieldGeometric

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

def xavier(rng: jax.Array,
           shape: Tuple[int, ...],
           scale: float = 1.0) -> jnp.ndarray:
    """Xavier/Glorot uniform initialisation."""
    fan_in  = shape[-1] if len(shape) >= 2 else shape[0]
    fan_out = shape[-2] if len(shape) >= 2 else shape[0]
    limit   = scale * math.sqrt(6.0 / (fan_in + fan_out + 1e-8))
    return random.uniform(rng, shape, minval=-limit, maxval=limit)

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
                               rng: jax.Array) -> jnp.ndarray:
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

class EntityAttention:
    def __init__(self, config: TopogenesisConfig, rng: jax.Array) -> None:
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

    def __init__(self, rng: jax.Array, input_dim: int,
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
    def __init__(self, rng: jax.Array, state_dim: int,
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

def init_A_params(rng: jax.Array, state_dim: int,
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

def init_affect_params(rng: jax.Array,
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

class GaussianPolicy:
    _KEYS = ('W1', 'b1', 'mean_W', 'mean_b', 'logstd_W', 'logstd_b')

    def __init__(self, rng: jax.Array, latent_dim: int,
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

def init_critic_params(rng: jax.Array, state_dim: int,
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

def init_sensorimotor_params(rng: jax.Array,
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

def init_enactive_ac_params(rng: jax.Array,
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

    def __init__(self, rng: jax.Array, concept_dim: int,
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

def init_goal_net_params(rng: jax.Array, E_dim: int,
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

def init_workspace_params(rng: jax.Array, state_dim: int,
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
                 rng: jax.Array) -> None:
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

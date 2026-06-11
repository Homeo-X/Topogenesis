"""Genome, heredity, development, and genome-field interface.

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
    GENOME_DEL_PROB,
    GENOME_DIM,
    GENOME_DUP_PROB,
    GENOME_DUP_SEGMENT,
    GENOME_FIELD_MAINT_COST,
    GENOME_FIELD_RADIUS,
    GENOME_FIELD_STRENGTH,
    GENOME_LOCI_PER_MODULE,
    GENOME_MAX_MODULE_DIM,
    MAX_JOINTS,
    MAX_MOTORS,
    MAX_TACTILE,
)
from topogenesis.cognition.networks import CompositionalSymbolicSystem

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

    def __init__(self, rng: jax.Array, genome: Genome):
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

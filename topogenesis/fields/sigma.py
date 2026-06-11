"""S² sigma field: PDE kernel, thermodynamic reservoir, geometry.

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

def pad_agent_arrays(positions: jnp.ndarray, energies: jnp.ndarray,
                     min_capacity: int = 8) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Pad (N,3)/(N,) agent arrays to the next power-of-two capacity.

    The jitted PDE kernel recompiles for every distinct input shape, so a
    population that changes size every birth/death forces one XLA compile per
    N and exhausts memory on long runs. Padding bounds the compiled
    signatures to {8, 16, 32, ...}. Padding rows carry zero energy, whose
    pump contribution clip(0, 0, 2) * pump_gain is exactly zero.
    """
    n = int(positions.shape[0])
    cap = max(min_capacity, 1 << (max(1, n) - 1).bit_length())
    if n == cap:
        return positions, energies
    pad = cap - n
    pos_pad = jnp.concatenate(
        [positions, jnp.zeros((pad, 3), dtype=positions.dtype)], axis=0)
    eng_pad = jnp.concatenate(
        [energies, jnp.zeros((pad,), dtype=energies.dtype)], axis=0)
    return pos_pad, eng_pad

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
        agent_positions, agent_energies = pad_agent_arrays(
            agent_positions, agent_energies)
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

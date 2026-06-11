"""Reference population run loop and CLI entry point.

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

from topogenesis.constants import MAX_MOTORS
from topogenesis.config import ABLATION_FLAGS, TopogenesisConfig, apply_ablations
from topogenesis.body.body_state import (
    AgentBodyPhys,
    build_rich_body,
    observe_full_vector,
)
from topogenesis.world.world3d import World3D
from topogenesis.cognition.agent import TopogenesisAgent

def main(argv=None):
    parser = argparse.ArgumentParser(description='topogenesis_engine')
    parser.add_argument('--steps',            type=int,   default=1000)
    parser.add_argument('--agents',           type=int,   default=2)
    parser.add_argument('--world_size',       type=int,   default=32)
    parser.add_argument('--seed',             type=int,   default=0)
    parser.add_argument('--log_every',        type=int,   default=50)
    parser.add_argument('--max_population',   type=int,   default=64,
                        help='Hard cap on live agents, combined with the '
                             'physical-capacity gate. Bounds memory on long runs.')
    parser.add_argument('--checkpoint_every', type=int,   default=0,
                        help='Save checkpoint every N steps. 0=disabled.')
    parser.add_argument('--checkpoint_path',  type=str,   default='ckpt',
                        help='Checkpoint filename prefix.')
    parser.add_argument('--ablate', action='append', choices=sorted(ABLATION_FLAGS),
                        default=[],
                        help='Disable a functional subsystem; repeatable.')
    args = parser.parse_args(argv)

    print(f"[topogenesis] Initialising  steps={args.steps}  agents={args.agents}"
          f"  world={args.world_size}")

    config = apply_ablations(TopogenesisConfig(), args.ablate)
    if args.ablate:
        print(f"[topogenesis] Ablations active: {', '.join(args.ablate)}")
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
        if len(agents) >= min(physical_capacity, args.max_population):
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
        # Unique, monotonic uid: seeds the child's RNG and must never collide,
        # unlike a population index that shifts as agents die.
        child_idx = args.agents + births
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
            f"[topogenesis] Birth parent={idx} child_uid={child_idx} "
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
    def save_checkpoint(path: str, step_num: int = 0):
        """Persist population state for later resumption."""
        import pickle
        state = {
            'step': step_num,
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
            # Several parents can pass the gate in one step; re-check the cap
            # here so the population never overshoots it.
            if len(agents) >= args.max_population:
                break
            agents.append(child_agent)
            bodies.append(child_body)
            action_bufs.append(np.zeros(MAX_MOTORS))
            reward_hist.append([])
            metric_hist.append([])

        # ── Periodic checkpoint ──────────────────────────────────────────────
        if args.checkpoint_every > 0 and step > 0 and step % args.checkpoint_every == 0:
            save_checkpoint(f'{args.checkpoint_path}.step{step}.pkl', step)

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
        save_checkpoint(f'{args.checkpoint_path}.final.pkl', args.steps)
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

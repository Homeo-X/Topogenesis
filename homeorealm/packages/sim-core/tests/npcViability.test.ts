import { describe, it, expect } from 'vitest';
import { computeViability, decayNeeds } from '../src/npc/needs.js';
import { initializeWorld } from '../src/simulation/initializeWorld.js';
import { runSimulation } from '../src/simulation/runSimulation.js';
import { getSeasonalModifiers } from '../src/time.js';

describe('NPC viability', () => {
  it('computeViability returns value between 0 and 1', () => {
    const { world } = initializeWorld(42, 5);
    for (const npc of Object.values(world.npcs)) {
      const v = computeViability(npc);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });

  it('needs decay over time', () => {
    const { world } = initializeWorld(42, 5);
    const npc = Object.values(world.npcs)[0]!;
    const summer = getSeasonalModifiers('summer');
    const before = { ...npc.needs };
    const after = decayNeeds(npc.needs, summer, true);
    expect(after.hunger).toBeLessThan(before.hunger);
    expect(after.rest).toBeLessThan(before.rest);
  });

  it('low health reduces viability', () => {
    const { world } = initializeWorld(42, 5);
    const npc = Object.values(world.npcs)[0]!;
    const highHealth = { ...npc, health: 1.0, needs: { hunger: 0.9, rest: 0.9, safety: 0.9, belonging: 0.9, purpose: 0.9, autonomy: 0.9, esteem: 0.9 } };
    const lowHealth = { ...highHealth, health: 0.2 };
    expect(computeViability(highHealth)).toBeGreaterThan(computeViability(lowHealth));
  });

  it('NPCs change viability over 10 days', () => {
    const { world, eventStore } = initializeWorld(42, 10);
    const initialViabilities = Object.values(world.npcs).map(n => computeViability(n));
    const result = runSimulation(world, eventStore, 10);
    const finalViabilities = Object.values(result.finalWorld.npcs)
      .filter(n => n.isAlive)
      .map(n => computeViability(n));

    // Viabilities should have changed (not all identical)
    const allSame = finalViabilities.every(v => v === initialViabilities[0]);
    expect(allSame).toBe(false);
  });

  it('low food causes hunger crisis events', () => {
    const { world, eventStore } = initializeWorld(42, 20);
    // Drain food
    const settId = Object.keys(world.settlements)[0]!;
    world.settlements[settId]!.resources.food = 0;
    for (const hh of Object.values(world.households)) {
      world.households[hh.id]!.foodStores = 0;
    }

    runSimulation(world, eventStore, 5);
    const hungerEvents = eventStore.events.filter(e => e.type === 'npc_hunger_crisis');
    expect(hungerEvents.length).toBeGreaterThan(0);
  });
});

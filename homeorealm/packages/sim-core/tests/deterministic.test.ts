import { describe, it, expect } from 'vitest';
import { createRng } from '../src/rng.js';
import { initializeWorld } from '../src/simulation/initializeWorld.js';
import { runSimulation } from '../src/simulation/runSimulation.js';
import { createEventStore } from '../src/events.js';

describe('Deterministic simulation', () => {
  it('produces the same world state from the same seed', () => {
    const seed = 42;
    const { world: w1, eventStore: e1 } = initializeWorld(seed, 20);
    const { world: w2, eventStore: e2 } = initializeWorld(seed, 20);

    const r1 = runSimulation(w1, e1, 5);
    const r2 = runSimulation(w2, e2, 5);

    expect(r1.finalWorld.day).toBe(5);
    expect(r2.finalWorld.day).toBe(5);

    // Same NPC count should survive
    const alive1 = Object.values(r1.finalWorld.npcs).filter(n => n.isAlive).length;
    const alive2 = Object.values(r2.finalWorld.npcs).filter(n => n.isAlive).length;
    expect(alive1).toBe(alive2);

    // Resource values should be identical
    const set1 = Object.values(r1.finalWorld.settlements)[0]!;
    const set2 = Object.values(r2.finalWorld.settlements)[0]!;
    expect(set1.resources.food.toFixed(4)).toBe(set2.resources.food.toFixed(4));
    expect(r1.finalWorld.environment.weather).toBe(r2.finalWorld.environment.weather);
    expect(r1.finalWorld.environment.physics.airTemperatureC.toFixed(4)).toBe(r2.finalWorld.environment.physics.airTemperatureC.toFixed(4));
    expect(r1.finalWorld.environment.chemistry.soilPH.toFixed(4)).toBe(r2.finalWorld.environment.chemistry.soilPH.toFixed(4));
  });

  it('different seeds produce different outcomes', () => {
    const { world: w1, eventStore: e1 } = initializeWorld(1, 20);
    const { world: w2, eventStore: e2 } = initializeWorld(999, 20);

    const r1 = runSimulation(w1, e1, 10);
    const r2 = runSimulation(w2, e2, 10);

    const s1 = Object.values(r1.finalWorld.settlements)[0]!;
    const s2 = Object.values(r2.finalWorld.settlements)[0]!;
    // Resources should differ (seeded differently)
    expect(s1.resources.food).not.toBe(s2.resources.food);
  });

  it('can run 30 days with 20 NPCs without crashing', () => {
    const { world, eventStore } = initializeWorld(123, 20);
    const result = runSimulation(world, eventStore, 30);
    expect(result.finalWorld.day).toBe(30);
    const alive = Object.values(result.finalWorld.npcs).filter(n => n.isAlive).length;
    expect(alive).toBeGreaterThan(0);
    expect(result.finalWorld.environment.physics.soilMoisture).toBeGreaterThanOrEqual(0);
    expect(result.finalWorld.environment.chemistry.corrosion).toBeGreaterThanOrEqual(0);
  });

  it('seeded RNG produces consistent values', () => {
    const rng = createRng(100);
    const seq1 = [rng.next(), rng.next(), rng.next()];
    const rng2 = createRng(100);
    const seq2 = [rng2.next(), rng2.next(), rng2.next()];
    expect(seq1).toEqual(seq2);
  });
});

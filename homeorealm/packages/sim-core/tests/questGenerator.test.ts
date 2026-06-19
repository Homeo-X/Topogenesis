import { describe, it, expect } from 'vitest';
import { generateEmergentQuests } from '../src/quests/questGenerator.js';
import { initializeWorld } from '../src/simulation/initializeWorld.js';
import { runSimulation } from '../src/simulation/runSimulation.js';
import { createRng } from '../src/rng.js';

describe('Quest generator', () => {
  it('generates food quests when food is critically low', () => {
    const { world } = initializeWorld(42, 20);
    const settId = Object.keys(world.settlements)[0]!;
    world.settlements[settId]!.resources.food = 0.5; // critical
    const rng = createRng(42);
    const quests = generateEmergentQuests(world, settId, rng);
    const foodQuests = quests.filter(q => q.tags.includes('food'));
    expect(foodQuests.length).toBeGreaterThan(0);
  });

  it('generates security quests when security is low', () => {
    const { world } = initializeWorld(42, 20);
    const settId = Object.keys(world.settlements)[0]!;
    world.settlements[settId]!.resources.security = 0.1;
    const rng = createRng(42);
    const quests = generateEmergentQuests(world, settId, rng);
    const securityQuests = quests.filter(q => q.tags.includes('security'));
    expect(securityQuests.length).toBeGreaterThan(0);
  });

  it('does not generate duplicate quests for same cause', () => {
    const { world } = initializeWorld(42, 20);
    const settId = Object.keys(world.settlements)[0]!;
    world.settlements[settId]!.resources.food = 0.5;
    const rng = createRng(42);

    const q1 = generateEmergentQuests(world, settId, rng);
    // Add those quests to world
    for (const q of q1) { world.quests[q.id] = q; }

    const q2 = generateEmergentQuests(world, settId, rng);
    // Should not duplicate food shortage quest
    const foodCauses1 = q1.filter(q => q.cause.includes('food')).map(q => q.cause);
    const foodCauses2 = q2.filter(q => q.cause.includes('food')).map(q => q.cause);
    for (const cause of foodCauses2) {
      expect(foodCauses1).not.toContain(cause);
    }
  });

  it('quests have valid structure', () => {
    const { world } = initializeWorld(42, 20);
    const settId = Object.keys(world.settlements)[0]!;
    world.settlements[settId]!.resources.food = 1;
    world.settlements[settId]!.resources.security = 0.15;
    const rng = createRng(42);
    const quests = generateEmergentQuests(world, settId, rng);
    for (const q of quests) {
      expect(q.id).toBeTruthy();
      expect(q.title).toBeTruthy();
      expect(q.description).toBeTruthy();
      expect(q.urgency).toBeGreaterThanOrEqual(0);
      expect(q.urgency).toBeLessThanOrEqual(1);
      expect(q.objectives.length).toBeGreaterThan(0);
      expect(q.rewards.length).toBeGreaterThan(0);
    }
  });

  it('running simulation under shortage conditions creates quests', () => {
    const { world, eventStore } = initializeWorld(77, 20);
    const settId = Object.keys(world.settlements)[0]!;
    world.settlements[settId]!.resources.food = 2; // low food
    const result = runSimulation(world, eventStore, 5);
    const activeQuests = Object.values(result.finalWorld.quests).filter(q => q.isActive);
    expect(activeQuests.length).toBeGreaterThan(0);
  });
});

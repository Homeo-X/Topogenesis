import { describe, it, expect } from 'vitest';
import { applyDailyConsumption, applyResourceDelta, getSettlementPressures } from '../src/settlement/economy.js';
import { createInitialResources } from '../src/settlement/resources.js';
import { getSeasonalModifiers } from '../src/time.js';
import { initializeWorld } from '../src/simulation/initializeWorld.js';
import { runSimulation } from '../src/simulation/runSimulation.js';

describe('Settlement economy', () => {
  it('food decreases after daily consumption with population', () => {
    const resources = createInitialResources('crown_valley');
    const seasonal = getSeasonalModifiers('summer');
    const before = resources.food;
    const { resources: after } = applyDailyConsumption(resources, 20, seasonal);
    expect(after.food).toBeLessThan(before);
  });

  it('food shortage is detected and reported', () => {
    const resources = { ...createInitialResources('crown_valley'), food: 1 };
    const seasonal = getSeasonalModifiers('winter');
    const { shortages } = applyDailyConsumption(resources, 20, seasonal);
    expect(shortages).toContain('food');
  });

  it('resource deltas apply correctly', () => {
    const resources = createInitialResources('crown_valley');
    const updated = applyResourceDelta(resources, { food: 10, wood: -5 });
    expect(updated.food).toBe(resources.food + 10);
    expect(updated.wood).toBe(resources.wood - 5);
  });

  it('resources cannot go below 0', () => {
    const resources = { ...createInitialResources('crown_valley'), food: 2 };
    const updated = applyResourceDelta(resources, { food: -100 });
    expect(updated.food).toBe(0);
  });

  it('getSettlementPressures returns critical_food_shortage when food is near zero', () => {
    const resources = { ...createInitialResources('crown_valley'), food: 0.5 };
    const pressures = getSettlementPressures(resources, 20);
    expect(pressures).toContain('critical_food_shortage');
  });

  it('NPC jobs affect settlement resources over time', () => {
    const { world, eventStore } = initializeWorld(42, 20);
    const settId = Object.keys(world.settlements)[0]!;
    const initialFood = world.settlements[settId]!.resources.food;
    const result = runSimulation(world, eventStore, 10);
    const finalFood = result.finalWorld.settlements[settId]!.resources.food;
    // Food should have changed (either consumed or produced)
    expect(finalFood).not.toBe(initialFood);
  });
});

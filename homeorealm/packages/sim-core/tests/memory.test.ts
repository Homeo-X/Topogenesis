import { describe, it, expect } from 'vitest';
import { createMemory, decayMemories, retrieveMemories } from '../src/npc/memory.js';
import { resetIdCounter } from '../src/ids.js';

describe('Memory model', () => {
  beforeEach(() => resetIdCounter());

  it('creates a memory with correct fields', () => {
    const mem = createMemory('npc1', 5, 'gift', 0.8, 0.7, ['npc1', 'npc2'], ['social', 'gift'], 'Received a gift');
    expect(mem.npcId).toBe('npc1');
    expect(mem.salience).toBe(0.8);
    expect(mem.decay).toBe(1.0);
  });

  it('high-salience memories decay slower than low-salience', () => {
    const highSalience = createMemory('npc1', 0, 'event', 0.9, 0.5, [], ['food'], 'High salience');
    const lowSalience = createMemory('npc1', 0, 'event', 0.1, 0.5, [], ['food'], 'Low salience');

    const afterHigh = decayMemories([highSalience], 20);
    const afterLow = decayMemories([lowSalience], 20);

    const highDecay = afterHigh.length > 0 ? afterHigh[0]!.decay : 0;
    const lowDecay = afterLow.length > 0 ? afterLow[0]!.decay : 0;

    expect(highDecay).toBeGreaterThanOrEqual(lowDecay);
  });

  it('fully decayed memories are removed', () => {
    const m = createMemory('npc1', 0, 'minor', 0.05, 0.0, [], [], 'trivial');
    // Simulate heavy decay
    const decayed = { ...m, decay: 0.04 };
    const result = decayMemories([decayed], 100);
    expect(result.length).toBe(0);
  });

  it('retrieval ranks salient memories higher', () => {
    const high = createMemory('npc1', 10, 'gift', 0.9, 0.8, ['npc2'], ['social', 'gift'], 'High importance gift');
    const low = createMemory('npc1', 10, 'chat', 0.2, 0.1, [], ['social'], 'Quick chat');

    const results = retrieveMemories([high, low], { currentDay: 15, tags: ['social'] }, 2);
    expect(results[0]!.id).toBe(high.id);
  });

  it('retrieval boosts memories matching participant tags', () => {
    const relevant = createMemory('npc1', 5, 'help', 0.5, 0.6, ['npc2'], ['help'], 'Helped npc2');
    const irrelevant = createMemory('npc1', 5, 'idle', 0.5, 0.0, [], ['work'], 'Worked alone');

    const results = retrieveMemories([relevant, irrelevant], { currentDay: 10, participants: ['npc2'] }, 2);
    expect(results[0]!.id).toBe(relevant.id);
  });
});

/** NPC memory model: episodic traces, decay, and retrieval scoring. */

import { nextId } from '../ids.js';
import type { MemoryTrace } from '../types.js';

const MAX_MEMORIES = 50;
const BASE_DECAY_PER_DAY = 0.02;
const SALIENCE_DECAY_PROTECTION = 0.5; // high salience reduces decay rate

/** Create a new memory trace. */
export function createMemory(
  npcId: string,
  day: number,
  eventType: string,
  salience: number,
  emotionalValence: number,
  participants: string[],
  tags: string[],
  summary: string,
): MemoryTrace {
  return {
    id: nextId('mem'),
    npcId,
    day,
    eventType,
    salience: Math.max(0, Math.min(1, salience)),
    emotionalValence: Math.max(-1, Math.min(1, emotionalValence)),
    participants,
    tags,
    summary,
    decay: 1.0,
  };
}

/** Apply daily decay to a list of memories. Returns surviving memories. */
export function decayMemories(memories: MemoryTrace[], currentDay: number): MemoryTrace[] {
  const updated = memories.map(m => {
    const daysSince = currentDay - m.day;
    const decayRate = BASE_DECAY_PER_DAY * (1 - m.salience * SALIENCE_DECAY_PROTECTION);
    const newDecay = Math.max(0, m.decay - decayRate * Math.max(1, daysSince * 0.05));
    return { ...m, decay: newDecay };
  });
  // Remove fully decayed and trim to max
  return updated
    .filter(m => m.decay > 0.05)
    .sort((a, b) => b.salience * b.decay - a.salience * a.decay)
    .slice(0, MAX_MEMORIES);
}

export type RetrievalContext = {
  currentDay: number;
  tags?: string[];
  participants?: string[];
  emotionalTone?: 'positive' | 'negative' | 'any';
};

/** Retrieve and rank memories by recency, salience, emotional valence, and tag relevance. */
export function retrieveMemories(memories: MemoryTrace[], ctx: RetrievalContext, limit = 5): MemoryTrace[] {
  const scored = memories.map(m => {
    const recency = 1 / (1 + (ctx.currentDay - m.day) * 0.1);
    const salienceScore = m.salience * m.decay;
    const tagMatch = ctx.tags
      ? ctx.tags.filter(t => m.tags.includes(t)).length / Math.max(1, ctx.tags.length)
      : 0.5;
    const participantMatch = ctx.participants
      ? (ctx.participants.some(p => m.participants.includes(p)) ? 0.3 : 0)
      : 0;
    const emotionBonus =
      ctx.emotionalTone === 'positive' ? Math.max(0, m.emotionalValence) * 0.2
      : ctx.emotionalTone === 'negative' ? Math.max(0, -m.emotionalValence) * 0.2
      : 0;
    return { memory: m, score: recency * 0.2 + salienceScore * 0.4 + tagMatch * 0.2 + participantMatch + emotionBonus };
  });
  return scored
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map(s => s.memory);
}

/** Check if NPC has relevant memories about a target (for relationship context). */
export function hasNegativeMemoriesAbout(memories: MemoryTrace[], targetId: string): boolean {
  return memories.some(m => m.participants.includes(targetId) && m.emotionalValence < -0.3 && m.decay > 0.3);
}

export function hasPositiveMemoriesAbout(memories: MemoryTrace[], targetId: string): boolean {
  return memories.some(m => m.participants.includes(targetId) && m.emotionalValence > 0.3 && m.decay > 0.3);
}

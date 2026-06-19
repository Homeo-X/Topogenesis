/** NPC frame distinction learning — the Homeo-X Topogenesis core. */

import { nextId } from '../ids.js';
import type { NPCState, FrameDistinction, NPCFrameState } from '../types.js';
import type { MemoryTrace } from '../types.js';
import { computeViability } from './needs.js';

const MIN_EVENTS_FOR_DISTINCTION = 3;
const MIN_CONFIDENCE_TO_RETAIN = 0.3;
const CONFIDENCE_GAIN_PER_REINFORCEMENT = 0.15;
const MAX_DISTINCTIONS = 20;

export type DistinctionCandidate = {
  label: string;
  domain: FrameDistinction['domain'];
  sharedTags: string[];
  avgViabilityEffect: number;
  eventCount: number;
  participantIds: string[];
};

/**
 * Analyze recent memories for patterns that warrant a new frame distinction.
 * Returns a new distinction if the pattern meets the retention threshold.
 */
export function proposeDistinction(
  npc: NPCState,
  recentMemories: MemoryTrace[],
  currentDay: number,
): FrameDistinction | null {
  if (recentMemories.length < MIN_EVENTS_FOR_DISTINCTION) return null;

  // Find clusters of memories with shared tags and similar emotional valence
  const tagCounts: Record<string, { count: number; totalValence: number; eventIds: string[] }> = {};
  for (const mem of recentMemories) {
    for (const tag of mem.tags) {
      if (!tagCounts[tag]) tagCounts[tag] = { count: 0, totalValence: 0, eventIds: [] };
      tagCounts[tag].count++;
      tagCounts[tag].totalValence += mem.emotionalValence;
      tagCounts[tag].eventIds.push(mem.id);
    }
  }

  // Find the most repeated tag cluster
  const repeatedTags = Object.entries(tagCounts)
    .filter(([, v]) => v.count >= MIN_EVENTS_FOR_DISTINCTION)
    .sort(([, a], [, b]) => b.count - a.count);

  if (repeatedTags.length === 0) return null;

  const [dominantTag, data] = repeatedTags[0]!;
  const avgValence = data.totalValence / data.count;
  const confidence = Math.min(1, data.count * CONFIDENCE_GAIN_PER_REINFORCEMENT);

  if (confidence < MIN_CONFIDENCE_TO_RETAIN) return null;

  // Check if we already have a similar distinction
  const existing = npc.frame.distinctions.find(d => d.domain === tagToD(dominantTag) && d.label.includes(dominantTag));
  if (existing) {
    return null; // Will reinforce instead
  }

  // Check viability benefit — only retain if positive or protective
  const viabilityGain = avgValence > 0 ? 'beneficial' : 'harmful';
  const label = avgValence > 0
    ? `${dominantTag} is reliably ${viabilityGain} for survival`
    : `${dominantTag} is a recurrent threat or loss`;

  return {
    id: nextId('dist'),
    label,
    domain: tagToD(dominantTag),
    confidence,
    utilityWeight: Math.abs(avgValence) * 0.5,
    learnedFromEventIds: data.eventIds,
    createdOnDay: currentDay,
    lastReinforcedDay: currentDay,
  };
}

/** Reinforce an existing distinction with new supporting evidence. */
export function reinforceDistinction(
  distinction: FrameDistinction,
  newEventId: string,
  currentDay: number,
): FrameDistinction {
  return {
    ...distinction,
    confidence: Math.min(1, distinction.confidence + CONFIDENCE_GAIN_PER_REINFORCEMENT),
    utilityWeight: Math.min(1, distinction.utilityWeight + 0.05),
    learnedFromEventIds: [...distinction.learnedFromEventIds, newEventId],
    lastReinforcedDay: currentDay,
  };
}

/** Integrate a new or reinforced distinction into NPC frame state. */
export function extendFrame(frame: NPCFrameState, newDistinction: FrameDistinction): NPCFrameState {
  const existing = frame.distinctions.find(d =>
    d.domain === newDistinction.domain && d.label === newDistinction.label,
  );

  let distinctions: FrameDistinction[];
  if (existing) {
    distinctions = frame.distinctions.map(d => d.id === existing.id ? { ...d, ...newDistinction, id: d.id } : d);
  } else {
    distinctions = [...frame.distinctions, newDistinction].slice(-MAX_DISTINCTIONS);
  }

  // Update attention bias based on distinction domains
  const attentionBias = { ...frame.attentionBias };
  for (const dist of distinctions) {
    attentionBias[dist.domain] = (attentionBias[dist.domain] ?? 0) + dist.utilityWeight * 0.1;
  }

  return { ...frame, distinctions, attentionBias };
}

function tagToD(tag: string): FrameDistinction['domain'] {
  const MAP: Record<string, FrameDistinction['domain']> = {
    food: 'food', hunger: 'food', harvest: 'food', farming: 'food',
    safety: 'safety', combat: 'combat', danger: 'safety', threat: 'safety',
    social: 'social', belonging: 'social', festival: 'ritual', ritual: 'ritual',
    craft: 'craft', crafting: 'craft', tools: 'craft',
    trade: 'trade', merchant: 'trade', coin: 'trade',
    politics: 'politics', governance: 'politics',
    ecology: 'ecology', gathering: 'ecology',
    memory: 'memory', hearthwell: 'memory',
  };
  return MAP[tag] ?? 'social';
}

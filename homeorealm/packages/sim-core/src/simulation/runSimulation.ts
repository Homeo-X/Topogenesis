/** Run simulation for N days, returning the final world state and event log. */

import type { WorldState } from '../types.js';
import type { EventStore } from '../events.js';
import { tickDay } from './tick.js';

export type SimulationResult = {
  finalWorld: WorldState;
  eventStore: EventStore;
  dayLogs: DayLog[];
};

export type DayLog = {
  day: number;
  questsGenerated: number;
  npcActions: { npcId: string; action: string; motivation: string }[];
  shortages: string[];
};

/** Run N ticks (days). */
export function runSimulation(
  initialWorld: WorldState,
  eventStore: EventStore,
  days: number,
): SimulationResult {
  let world = initialWorld;
  const dayLogs: DayLog[] = [];

  for (let d = 0; d < days; d++) {
    const prevEventCount = eventStore.events.length;
    world = tickDay(world, eventStore);

    const newEvents = eventStore.events.slice(prevEventCount);
    const questsGenerated = newEvents.filter(e => e.type === 'quest_generated').length;
    const shortages = newEvents
      .filter(e => e.type === 'settlement_shortage')
      .map(e => String(e.payload['resource']));

    dayLogs.push({ day: world.day, questsGenerated, npcActions: [], shortages });
  }

  return { finalWorld: world, eventStore, dayLogs };
}

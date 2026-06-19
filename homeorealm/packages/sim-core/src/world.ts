/** World state management: snapshot and replay utilities. */

import type { WorldState } from './types.js';
import type { EventStore } from './events.js';

export type WorldSnapshot = {
  world: WorldState;
  events: ReturnType<EventStore['exportSnapshot']>;
  createdAt: string;
};

export function exportWorldSnapshot(world: WorldState, eventStore: EventStore): WorldSnapshot {
  return { world, events: eventStore.exportSnapshot(), createdAt: new Date().toISOString() };
}

export function importWorldSnapshot(snapshot: WorldSnapshot, eventStore: EventStore): WorldState {
  eventStore.importSnapshot(snapshot.events);
  return snapshot.world;
}

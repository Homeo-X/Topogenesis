/** In-memory world state store with optional JSON snapshot persistence. */

import { writeFileSync, readFileSync, existsSync } from 'fs';
import type { WorldState } from '@homeorealm/sim-core';
import { createEventStore, type EventStore } from '@homeorealm/sim-core';
import { initializeWorld } from '@homeorealm/sim-core';
import { ensureEnvironment } from '@homeorealm/sim-core';

export type AppState = {
  world: WorldState;
  eventStore: EventStore;
};

let _state: AppState | null = null;

export function getState(): AppState {
  if (!_state) {
    const { world, eventStore } = initializeWorld(42, 25);
    _state = { world, eventStore };
  }
  return _state;
}

export function setState(world: WorldState, eventStore: EventStore): void {
  _state = { world, eventStore };
}

export function resetState(seed: number, npcCount = 25): AppState {
  const { world, eventStore } = initializeWorld(seed, npcCount);
  _state = { world, eventStore };
  return _state;
}

export function saveSnapshot(path: string): void {
  const state = getState();
  const snapshot = {
    world: state.world,
    events: state.eventStore.exportSnapshot(),
    savedAt: new Date().toISOString(),
  };
  writeFileSync(path, JSON.stringify(snapshot, null, 2));
}

export function loadSnapshot(path: string): boolean {
  if (!existsSync(path)) return false;
  try {
    const raw = JSON.parse(readFileSync(path, 'utf-8'));
    const eventStore = createEventStore();
    eventStore.importSnapshot(raw.events);
    const world = raw.world as WorldState;
    _state = {
      world: { ...world, environment: ensureEnvironment(world.environment, world.seed, Object.values(world.settlements)[0]?.regionId) },
      eventStore,
    };
    return true;
  } catch {
    return false;
  }
}

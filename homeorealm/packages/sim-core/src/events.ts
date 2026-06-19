/** Append-only event ledger for HomeoRealm simulation. */

import { nextId } from './ids.js';

export type WorldEvent = {
  id: string;
  day: number;
  tick: number;
  type: string;
  actorId?: string;
  targetIds?: string[];
  settlementId?: string;
  payload: Record<string, unknown>;
  tags: string[];
  salience: number; // 0–1
};

export type EventStore = {
  events: WorldEvent[];
  append(event: Omit<WorldEvent, 'id'>): WorldEvent;
  query(filter: EventFilter): WorldEvent[];
  recentEvents(n: number): WorldEvent[];
  eventsForActor(actorId: string, limit?: number): WorldEvent[];
  eventsForSettlement(settlementId: string, limit?: number): WorldEvent[];
  highSalienceEvents(threshold?: number, limit?: number): WorldEvent[];
  exportSnapshot(): WorldEvent[];
  importSnapshot(events: WorldEvent[]): void;
};

export type EventFilter = {
  actorId?: string;
  settlementId?: string;
  type?: string;
  tags?: string[];
  minSalience?: number;
  fromDay?: number;
  toDay?: number;
  limit?: number;
};

export function createEventStore(): EventStore {
  const events: WorldEvent[] = [];

  const store: EventStore = {
    events,
    append(event) {
      const full: WorldEvent = { id: nextId('ev'), ...event };
      events.push(full);
      return full;
    },
    query(filter) {
      let result = events;
      if (filter.actorId) result = result.filter(e => e.actorId === filter.actorId || (e.targetIds ?? []).includes(filter.actorId!));
      if (filter.settlementId) result = result.filter(e => e.settlementId === filter.settlementId);
      if (filter.type) result = result.filter(e => e.type === filter.type);
      if (filter.tags) result = result.filter(e => filter.tags!.every(t => e.tags.includes(t)));
      if (filter.minSalience !== undefined) result = result.filter(e => e.salience >= filter.minSalience!);
      if (filter.fromDay !== undefined) result = result.filter(e => e.day >= filter.fromDay!);
      if (filter.toDay !== undefined) result = result.filter(e => e.day <= filter.toDay!);
      if (filter.limit) result = result.slice(-filter.limit);
      return result;
    },
    recentEvents(n) { return events.slice(-n); },
    eventsForActor(actorId, limit = 20) {
      return events.filter(e => e.actorId === actorId || (e.targetIds ?? []).includes(actorId)).slice(-limit);
    },
    eventsForSettlement(settlementId, limit = 50) {
      return events.filter(e => e.settlementId === settlementId).slice(-limit);
    },
    highSalienceEvents(threshold = 0.7, limit = 20) {
      return events.filter(e => e.salience >= threshold).slice(-limit);
    },
    exportSnapshot() { return [...events]; },
    importSnapshot(imported) {
      events.length = 0;
      events.push(...imported);
    },
  };
  return store;
}

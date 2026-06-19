import { describe, it, expect } from 'vitest';
import { createEventStore } from '../src/events.js';
import { resetIdCounter } from '../src/ids.js';

describe('Event ledger', () => {
  beforeEach(() => resetIdCounter());

  it('appends events and assigns IDs', () => {
    const store = createEventStore();
    const ev = store.append({ day: 1, tick: 0, type: 'test_event', payload: { x: 1 }, tags: ['test'], salience: 0.5 });
    expect(ev.id).toBeTruthy();
    expect(store.events.length).toBe(1);
  });

  it('events are ordered by insertion', () => {
    const store = createEventStore();
    for (let i = 1; i <= 5; i++) {
      store.append({ day: i, tick: 0, type: 'tick', payload: {}, tags: [], salience: 0.1 });
    }
    expect(store.events.map(e => e.day)).toEqual([1, 2, 3, 4, 5]);
  });

  it('eventsForActor filters correctly', () => {
    const store = createEventStore();
    store.append({ day: 1, tick: 0, type: 'action', actorId: 'npc1', payload: {}, tags: [], salience: 0.3 });
    store.append({ day: 1, tick: 0, type: 'action', actorId: 'npc2', payload: {}, tags: [], salience: 0.3 });
    const npc1Events = store.eventsForActor('npc1');
    expect(npc1Events.length).toBe(1);
    expect(npc1Events[0]!.actorId).toBe('npc1');
  });

  it('highSalienceEvents filters by threshold', () => {
    const store = createEventStore();
    store.append({ day: 1, tick: 0, type: 'minor', payload: {}, tags: [], salience: 0.2 });
    store.append({ day: 1, tick: 0, type: 'major', payload: {}, tags: [], salience: 0.9 });
    const high = store.highSalienceEvents(0.7);
    expect(high.length).toBe(1);
    expect(high[0]!.type).toBe('major');
  });

  it('snapshot export and import is idempotent', () => {
    const store = createEventStore();
    store.append({ day: 1, tick: 0, type: 'event', payload: {}, tags: [], salience: 0.5 });
    const snapshot = store.exportSnapshot();
    const store2 = createEventStore();
    store2.importSnapshot(snapshot);
    expect(store2.events.length).toBe(store.events.length);
    expect(store2.events[0]!.id).toBe(store.events[0]!.id);
  });
});

import type { FastifyInstance } from 'fastify';
import { getState, resetState, setState } from '../persistence/store.js';
import { getWorldSummary } from '@homeorealm/sim-core';
import { tickDay } from '@homeorealm/sim-core';
import { runSimulation } from '@homeorealm/sim-core';

export async function worldRoutes(app: FastifyInstance): Promise<void> {
  app.get('/api/world', async (_req, reply) => {
    const { world } = getState();
    return reply.send(getWorldSummary(world));
  });

  app.get('/api/world/full', async (_req, reply) => {
    const { world } = getState();
    return reply.send(world);
  });

  app.post('/api/world/tick', async (_req, reply) => {
    const { world, eventStore } = getState();
    const newWorld = tickDay(world, eventStore);
    setState(newWorld, eventStore);
    return reply.send({ day: newWorld.day, summary: getWorldSummary(newWorld) });
  });

  app.post<{ Body: { days: number } }>('/api/world/run', {
    schema: { body: { type: 'object', properties: { days: { type: 'number', minimum: 1, maximum: 365 } }, required: ['days'] } },
  }, async (req, reply) => {
    const { days } = req.body;
    const { world, eventStore } = getState();
    const result = runSimulation(world, eventStore, days);
    setState(result.finalWorld, result.eventStore);
    return reply.send({ day: result.finalWorld.day, summary: getWorldSummary(result.finalWorld) });
  });

  app.post<{ Body: { seed: number; npcCount?: number } }>('/api/world/reset', {
    schema: { body: { type: 'object', properties: { seed: { type: 'number' }, npcCount: { type: 'number' } }, required: ['seed'] } },
  }, async (req, reply) => {
    const { seed, npcCount = 25 } = req.body;
    const newState = resetState(seed, npcCount);
    return reply.send({ message: `World reset with seed ${seed}`, day: newState.world.day });
  });

  app.get('/api/world/events', async (req, reply) => {
    const { eventStore } = getState();
    const limit = parseInt(String((req.query as any).limit ?? '50'));
    return reply.send(eventStore.recentEvents(Math.min(limit, 200)));
  });

  app.get('/api/world/events/high-salience', async (_req, reply) => {
    const { eventStore } = getState();
    return reply.send(eventStore.highSalienceEvents(0.7, 30));
  });
}

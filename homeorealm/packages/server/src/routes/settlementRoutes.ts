import type { FastifyInstance } from 'fastify';
import { getState } from '../persistence/store.js';
import { getSettlementPressures } from '@homeorealm/sim-core';

export async function settlementRoutes(app: FastifyInstance): Promise<void> {
  app.get('/api/settlements', async (_req, reply) => {
    const { world } = getState();
    return reply.send(Object.values(world.settlements));
  });

  app.get<{ Params: { id: string } }>('/api/settlements/:id', async (req, reply) => {
    const { world } = getState();
    const s = world.settlements[req.params.id];
    if (!s) return reply.status(404).send({ error: 'Settlement not found' });
    const pressures = getSettlementPressures(s.resources, s.population);
    return reply.send({ ...s, pressures });
  });

  app.get<{ Params: { id: string } }>('/api/settlements/:id/households', async (req, reply) => {
    const { world } = getState();
    const s = world.settlements[req.params.id];
    if (!s) return reply.status(404).send({ error: 'Settlement not found' });
    const hhs = s.householdIds.map(id => world.households[id]).filter(Boolean);
    return reply.send(hhs);
  });

  app.get('/api/households', async (_req, reply) => {
    const { world } = getState();
    return reply.send(Object.values(world.households));
  });

  app.get<{ Params: { id: string } }>('/api/households/:id', async (req, reply) => {
    const { world } = getState();
    const hh = world.households[req.params.id];
    if (!hh) return reply.status(404).send({ error: 'Household not found' });
    const members = hh.memberIds.map(id => world.npcs[id]).filter(Boolean).map(n => ({ id: n!.id, name: n!.name, jobId: n!.jobId }));
    return reply.send({ ...hh, members });
  });

  app.get('/api/dungeons', async (_req, reply) => {
    const { world } = getState();
    return reply.send(world.dungeonRooms);
  });
}

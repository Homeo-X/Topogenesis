import type { FastifyInstance } from 'fastify';
import { getState } from '../persistence/store.js';
import { computeViability, getLiveNPCs, getBottomViabilityNPCs, getTopViabilityNPCs } from '@homeorealm/sim-core';

export async function npcRoutes(app: FastifyInstance): Promise<void> {
  app.get('/api/npcs', async (req, reply) => {
    const { world } = getState();
    const live = getLiveNPCs(world);
    const q = (req.query as any);
    let filtered = live;
    if (q.job) filtered = filtered.filter(n => n.jobId === q.job);
    if (q.household) filtered = filtered.filter(n => n.householdId === q.household);
    if (q.settlement) filtered = filtered.filter(n => n.settlementId === q.settlement);
    const npcs = filtered.map(n => ({
      id: n.id, name: n.name, people: n.people, jobId: n.jobId,
      householdId: n.householdId, settlementId: n.settlementId,
      viability: computeViability(n), health: n.health, wealth: n.wealth,
      currentAction: n.currentIntention?.action ?? 'idle',
    }));
    return reply.send(npcs);
  });

  app.get<{ Params: { id: string } }>('/api/npcs/:id', async (req, reply) => {
    const { world } = getState();
    const npc = world.npcs[req.params.id];
    if (!npc) return reply.status(404).send({ error: 'NPC not found' });
    return reply.send({ ...npc, viability: computeViability(npc) });
  });

  app.get<{ Params: { id: string } }>('/api/npcs/:id/relationships', async (req, reply) => {
    const { world } = getState();
    const npc = world.npcs[req.params.id];
    if (!npc) return reply.status(404).send({ error: 'NPC not found' });
    const rels = Object.values(npc.relationships).map(r => ({
      ...r,
      targetName: world.npcs[r.targetId]?.name ?? 'Unknown',
    }));
    return reply.send(rels);
  });

  app.get<{ Params: { id: string } }>('/api/npcs/:id/memories', async (req, reply) => {
    const { world } = getState();
    const npc = world.npcs[req.params.id];
    if (!npc) return reply.status(404).send({ error: 'NPC not found' });
    return reply.send(npc.memories.sort((a, b) => b.salience * b.decay - a.salience * a.decay));
  });

  app.get('/api/npcs/ranking/top', async (_req, reply) => {
    const { world } = getState();
    return reply.send(getTopViabilityNPCs(world, 10).map(n => ({ id: n.id, name: n.name, viability: computeViability(n), jobId: n.jobId })));
  });

  app.get('/api/npcs/ranking/bottom', async (_req, reply) => {
    const { world } = getState();
    return reply.send(getBottomViabilityNPCs(world, 10).map(n => ({ id: n.id, name: n.name, viability: computeViability(n), jobId: n.jobId })));
  });
}

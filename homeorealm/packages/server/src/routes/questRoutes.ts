import type { FastifyInstance } from 'fastify';
import { getState } from '../persistence/store.js';
import { getActiveQuests, sortQuestsByPriority } from '@homeorealm/sim-core';

export async function questRoutes(app: FastifyInstance): Promise<void> {
  app.get('/api/quests', async (_req, reply) => {
    const { world } = getState();
    const active = sortQuestsByPriority(getActiveQuests(world));
    return reply.send(active);
  });

  app.get('/api/quests/all', async (_req, reply) => {
    const { world } = getState();
    return reply.send(Object.values(world.quests));
  });

  app.get<{ Params: { id: string } }>('/api/quests/:id', async (req, reply) => {
    const { world } = getState();
    const quest = world.quests[req.params.id];
    if (!quest) return reply.status(404).send({ error: 'Quest not found' });
    return reply.send(quest);
  });
}

import type { FastifyInstance } from 'fastify';
import { applyPlayerAction, ensurePlayer, type PlayerActionInput } from '@homeorealm/sim-core';
import { getState, setState } from '../persistence/store.js';

export async function playerRoutes(app: FastifyInstance): Promise<void> {
  app.get('/api/player', async (_req, reply) => {
    const { world } = getState();
    return reply.send(ensurePlayer(world));
  });

  app.post<{ Body: PlayerActionInput }>('/api/player/action', {
    schema: {
      body: {
        type: 'object',
        properties: {
          type: {
            type: 'string',
            enum: ['accept_quest', 'complete_quest', 'aid_settlement', 'travel', 'gather', 'rest', 'train', 'trade', 'talk', 'delve_dungeon'],
          },
          questId: { type: 'string' },
          settlementId: { type: 'string' },
          location: { type: 'string', enum: ['town', 'market', 'wilds', 'dungeon', 'home'] },
          npcId: { type: 'string' },
        },
        required: ['type'],
      },
    },
  }, async (req, reply) => {
    const { world, eventStore } = getState();
    try {
      const result = applyPlayerAction(world, eventStore, req.body);
      setState(result.world, eventStore);
      return reply.send({ player: result.player, message: result.message });
    } catch (err) {
      return reply.status(400).send({ error: err instanceof Error ? err.message : String(err) });
    }
  });
}

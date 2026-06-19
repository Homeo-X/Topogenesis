import Fastify from 'fastify';
import cors from '@fastify/cors';
import { worldRoutes } from './routes/worldRoutes.js';
import { npcRoutes } from './routes/npcRoutes.js';
import { questRoutes } from './routes/questRoutes.js';
import { settlementRoutes } from './routes/settlementRoutes.js';
import { simulationRoutes } from './routes/simulationRoutes.js';

export async function buildApp(): Promise<ReturnType<typeof Fastify>> {
  const app = Fastify({ logger: false });

  await app.register(cors, { origin: true });

  await app.register(worldRoutes);
  await app.register(npcRoutes);
  await app.register(questRoutes);
  await app.register(settlementRoutes);
  await app.register(simulationRoutes);

  app.get('/api/health', async (_req, reply) => {
    return reply.send({ status: 'ok', service: 'HomeoRealm API' });
  });

  return app;
}

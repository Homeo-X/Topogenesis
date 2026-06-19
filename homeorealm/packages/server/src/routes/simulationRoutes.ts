import type { FastifyInstance } from 'fastify';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { loadContentPack, validateContentPack } from '@homeorealm/sim-core';

const __dirname = dirname(fileURLToPath(import.meta.url));
const CONTENT_DIR = join(__dirname, '../../../content');

export async function simulationRoutes(app: FastifyInstance): Promise<void> {
  app.get('/api/lore/content', async (_req, reply) => {
    try {
      const pack = loadContentPack(CONTENT_DIR);
      return reply.send(pack);
    } catch (err) {
      return reply.status(500).send({ error: String(err) });
    }
  });

  app.get('/api/lore/regions', async (_req, reply) => {
    const { regions } = loadContentPack(CONTENT_DIR);
    return reply.send(regions);
  });

  app.get('/api/lore/factions', async (_req, reply) => {
    const { factions } = loadContentPack(CONTENT_DIR);
    return reply.send(factions);
  });

  app.get('/api/lore/peoples', async (_req, reply) => {
    const { peoples } = loadContentPack(CONTENT_DIR);
    return reply.send(peoples);
  });

  app.get('/api/lore/assets', async (_req, reply) => {
    const { assetManifest } = loadContentPack(CONTENT_DIR);
    return reply.send(assetManifest);
  });

  app.get('/api/lore/validate', async (_req, reply) => {
    const result = validateContentPack(CONTENT_DIR);
    return reply.send(result);
  });
}

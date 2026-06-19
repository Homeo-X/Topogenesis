import { describe, it, expect, beforeAll } from 'vitest';
import { buildApp } from '../src/app.js';
import { resetState } from '../src/persistence/store.js';
import type { FastifyInstance } from 'fastify';

let app: FastifyInstance;

beforeAll(async () => {
  resetState(42, 20);
  app = await buildApp();
  await app.ready();
});

describe('API endpoints', () => {
  it('GET /api/health returns ok', async () => {
    const res = await app.inject({ method: 'GET', url: '/api/health' });
    expect(res.statusCode).toBe(200);
    expect(JSON.parse(res.body).status).toBe('ok');
  });

  it('GET /api/world returns world summary', async () => {
    const res = await app.inject({ method: 'GET', url: '/api/world' });
    expect(res.statusCode).toBe(200);
    const body = JSON.parse(res.body);
    expect(body.day).toBeDefined();
    expect(body.settlements).toBeDefined();
    expect(body.totalNPCs).toBeGreaterThan(0);
  });

  it('GET /api/npcs returns NPC list', async () => {
    const res = await app.inject({ method: 'GET', url: '/api/npcs' });
    expect(res.statusCode).toBe(200);
    const body = JSON.parse(res.body);
    expect(Array.isArray(body)).toBe(true);
    expect(body.length).toBeGreaterThan(0);
    expect(body[0].name).toBeDefined();
  });

  it('GET /api/npcs/:id returns a specific NPC', async () => {
    const listRes = await app.inject({ method: 'GET', url: '/api/npcs' });
    const list = JSON.parse(listRes.body);
    const firstId = list[0].id;
    const res = await app.inject({ method: 'GET', url: `/api/npcs/${firstId}` });
    expect(res.statusCode).toBe(200);
    const npc = JSON.parse(res.body);
    expect(npc.id).toBe(firstId);
    expect(npc.needs).toBeDefined();
    expect(npc.memories).toBeDefined();
  });

  it('GET /api/npcs/:id returns 404 for unknown NPC', async () => {
    const res = await app.inject({ method: 'GET', url: '/api/npcs/nonexistent_id' });
    expect(res.statusCode).toBe(404);
  });

  it('GET /api/settlements returns settlements', async () => {
    const res = await app.inject({ method: 'GET', url: '/api/settlements' });
    expect(res.statusCode).toBe(200);
    const body = JSON.parse(res.body);
    expect(Array.isArray(body)).toBe(true);
    expect(body.length).toBeGreaterThan(0);
    expect(body[0].resources).toBeDefined();
  });

  it('GET /api/quests returns quest list', async () => {
    const res = await app.inject({ method: 'GET', url: '/api/quests' });
    expect(res.statusCode).toBe(200);
    const body = JSON.parse(res.body);
    expect(Array.isArray(body)).toBe(true);
  });

  it('POST /api/world/tick advances the day', async () => {
    const before = await app.inject({ method: 'GET', url: '/api/world' });
    const beforeDay = JSON.parse(before.body).day;

    const res = await app.inject({ method: 'POST', url: '/api/world/tick' });
    expect(res.statusCode).toBe(200);
    const afterDay = JSON.parse(res.body).day;
    expect(afterDay).toBe(beforeDay + 1);
  });

  it('POST /api/world/run runs multiple days', async () => {
    resetState(42, 20);
    const res = await app.inject({ method: 'POST', url: '/api/world/run', payload: { days: 5 } });
    expect(res.statusCode).toBe(200);
    expect(JSON.parse(res.body).day).toBe(5);
  });

  it('POST /api/world/reset resets world with given seed', async () => {
    const res = await app.inject({ method: 'POST', url: '/api/world/reset', payload: { seed: 777 } });
    expect(res.statusCode).toBe(200);
    expect(JSON.parse(res.body).day).toBe(0);
  });

  it('GET /api/lore/regions returns regions', async () => {
    const res = await app.inject({ method: 'GET', url: '/api/lore/regions' });
    expect(res.statusCode).toBe(200);
    const body = JSON.parse(res.body);
    expect(Array.isArray(body)).toBe(true);
    expect(body.length).toBeGreaterThan(0);
    expect(body[0].id).toBeDefined();
  });

  it('GET /api/lore/validate returns valid result', async () => {
    const res = await app.inject({ method: 'GET', url: '/api/lore/validate' });
    expect(res.statusCode).toBe(200);
    const body = JSON.parse(res.body);
    expect(body.valid).toBe(true);
    expect(body.errors).toEqual([]);
  });

  it('GET /api/households returns household list', async () => {
    resetState(42, 20);
    const res = await app.inject({ method: 'GET', url: '/api/households' });
    expect(res.statusCode).toBe(200);
    const body = JSON.parse(res.body);
    expect(Array.isArray(body)).toBe(true);
    expect(body.length).toBeGreaterThan(0);
  });
});

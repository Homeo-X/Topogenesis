/**
 * Self-contained local API — runs the sim-core simulation entirely in the browser.
 * Mirrors the same interface as api.ts so all components work unchanged.
 */

import {
  initializeWorld,
  tickDay,
  computeViability,
  ensureEnvironment,
  ensurePlayer,
  applyPlayerAction,
  getActiveQuests,
  getTopViabilityNPCs as coreTopViability,
  getBottomViabilityNPCs as coreBottomViability,
} from '@homeorealm/sim-core';
import type { WorldState, NPCState, EventStore } from '@homeorealm/sim-core';
import type {
  WorldSummary, WorldEvent, NPCSummary, NPCFull, Relationship, Memory,
  FrameDistinction, Settlement, Household, DungeonRoom, Quest,
  Region, Faction, People, Asset, PlayerState, PlayerActionInput, PlayerActionResponse,
} from './api.js';

// ─── Persistent state ──────────────────────────────────────────────────────

const WORLD_KEY = 'homeorealm_world';
const EVENTS_KEY = 'homeorealm_events';
const DEFAULT_SEED = 42;
const DEFAULT_NPC_COUNT = 24;

let world: WorldState = null!;
let eventStore: EventStore = null!;

function loadOrInit(): void {
  // Try to restore from localStorage
  const savedWorld = localStorage.getItem(WORLD_KEY);
  const savedEvents = localStorage.getItem(EVENTS_KEY);
  ({ world, eventStore } = initializeWorld(DEFAULT_SEED, DEFAULT_NPC_COUNT));

  if (savedWorld) {
    try { world = JSON.parse(savedWorld) as WorldState; } catch { /* ignore */ }
  }
  world = { ...world, environment: ensureEnvironment(world.environment, world.seed, Object.values(world.settlements)[0]?.regionId) };
  if (savedEvents) {
    try { eventStore.importSnapshot(JSON.parse(savedEvents)); } catch { /* ignore */ }
  }
}

function persist(): void {
  localStorage.setItem(WORLD_KEY, JSON.stringify(world));
  localStorage.setItem(EVENTS_KEY, JSON.stringify(eventStore.exportSnapshot()));
}

// Lazy init on first use
function ensureInit(): void {
  if (!world) loadOrInit();
}

// ─── Mapping helpers ───────────────────────────────────────────────────────

function npcToSummary(npc: NPCState): NPCSummary {
  return {
    id: npc.id,
    name: npc.name,
    people: npc.people,
    jobId: npc.jobId,
    householdId: npc.householdId,
    settlementId: npc.settlementId,
    viability: computeViability(npc),
    health: npc.health,
    wealth: npc.wealth,
    currentAction: npc.currentIntention?.action ?? 'idle',
  };
}

function npcToFull(npc: NPCState): NPCFull {
  return {
    ...npcToSummary(npc),
    needs: npc.needs as unknown as Record<string, number>,
    affect: npc.affect as unknown as Record<string, number>,
    skills: npc.skills,
    relationships: npc.relationships as unknown as Record<string, unknown>,
    memories: npc.memories.map(m => ({
      id: m.id, day: m.day, eventType: m.eventType,
      salience: m.salience, emotionalValence: m.emotionalValence,
      participants: m.participants, tags: m.tags, summary: m.summary, decay: m.decay,
    })),
    frame: {
      distinctions: npc.frame.distinctions.map(d => ({
        id: d.id, label: d.label, domain: d.domain,
        confidence: d.confidence, utilityWeight: d.utilityWeight, createdOnDay: d.createdOnDay,
      } satisfies FrameDistinction)),
      attentionBias: npc.frame.attentionBias,
      tabooTags: npc.frame.tabooTags,
      valuedTags: npc.frame.valuedTags,
    },
    currentIntention: npc.currentIntention
      ? { action: npc.currentIntention.action, motivation: npc.currentIntention.motivation, expectedViabilityGain: npc.currentIntention.expectedViabilityGain }
      : undefined,
  };
}

function makeWorldSummary(): WorldSummary {
  const liveNpcs = Object.values(world.npcs).filter(n => n.isAlive);
  const avgViab = liveNpcs.length > 0
    ? liveNpcs.reduce((s, n) => s + computeViability(n), 0) / liveNpcs.length
    : 0;
  return {
    day: world.day,
    settlements: Object.values(world.settlements).map(s => ({
      id: s.id, name: s.name, population: s.population, resources: s.resources,
    })),
    totalNPCs: liveNpcs.length,
    avgViability: avgViab.toFixed(3),
    activeQuests: getActiveQuests(world).length,
    dungeonRooms: world.dungeonRooms.length,
    environment: ensureEnvironment(world.environment, world.seed, Object.values(world.settlements)[0]?.regionId),
  };
}

// ─── Static lore content (bundled with app) ────────────────────────────────

const REGIONS: Region[] = [
  { id: 'ashenveil', name: 'Ashenveil', description: 'Mist-shrouded highland forests.', climate: 'temperate', mainResources: ['wood', 'ore', 'herbs'] },
  { id: 'sunreach', name: 'Sunreach Plains', description: 'Fertile lowland plains.', climate: 'warm', mainResources: ['food', 'cloth', 'coin'] },
  { id: 'ironmoor', name: 'Ironmoor Wastes', description: 'Arid badlands rich in minerals.', climate: 'arid', mainResources: ['ore', 'tools', 'stone'] },
];

const FACTIONS: Faction[] = [
  { id: 'hearthcourt', name: 'The Hearthcourt', description: 'Keepers of hearth flame and communal memory.', function: 'cultural preservation' },
  { id: 'wayfarers', name: 'Wayfarers Compact', description: 'Merchants and explorers binding distant settlements.', function: 'trade and transit' },
  { id: 'ironweavers', name: 'Ironweavers Guild', description: 'Smiths and engineers who maintain civilization\'s tools.', function: 'production' },
];

const PEOPLES: People[] = [
  { id: 'valari', name: 'Valari', region: 'ashenveil', description: 'Ancestral memory-keepers with deep oral traditions.', assetTheme: 'twilight purple', culturalTraits: ['communal', 'ritual-minded', 'long-memory'] },
  { id: 'threnosi', name: 'Threnosi', region: 'sunreach', description: 'Seafarers and coastal traders.', assetTheme: 'sea green', culturalTraits: ['independent', 'adaptable', 'pragmatic'] },
  { id: 'aurath', name: 'Aurath', region: 'ironmoor', description: 'Desert forgers and precision craftspeople.', assetTheme: 'amber gold', culturalTraits: ['disciplined', 'proud', 'merit-focused'] },
];

const ASSETS: Asset[] = [
  { id: 'hearthflame', category: 'artifact', name: 'Hearthflame Shard', description: 'A fragment of the original settlement fire.', gameplayUse: 'Raises heartwellStability by 0.2 when placed.', visualKeywords: ['glowing', 'amber', 'warmth'], productionPriority: 'high' },
  { id: 'memorystone', category: 'artifact', name: 'Memory Stone', description: 'Crystallised emotional record of a past event.', gameplayUse: 'Teaches an NPC a FrameDistinction instantly.', visualKeywords: ['crystalline', 'iridescent', 'heavy'], productionPriority: 'medium' },
];

// ─── Local API surface ─────────────────────────────────────────────────────

export const api = {
  getWorld(): Promise<WorldSummary> {
    ensureInit();
    return Promise.resolve(makeWorldSummary());
  },

  getPlayer(): Promise<PlayerState> {
    ensureInit();
    const player = ensurePlayer(world);
    if (!world.player) {
      world = { ...world, player };
      persist();
    }
    return Promise.resolve(player);
  },

  playerAction(action: PlayerActionInput): Promise<PlayerActionResponse> {
    ensureInit();
    const result = applyPlayerAction(world, eventStore, action);
    world = result.world;
    persist();
    return Promise.resolve({ player: result.player, message: result.message });
  },

  tick(): Promise<{ day: number; summary: WorldSummary }> {
    ensureInit();
    world = tickDay(world, eventStore);
    persist();
    return Promise.resolve({ day: world.day, summary: makeWorldSummary() });
  },

  runDays(days: number): Promise<{ day: number; summary: WorldSummary }> {
    ensureInit();
    for (let i = 0; i < days; i++) {
      world = tickDay(world, eventStore);
    }
    persist();
    return Promise.resolve({ day: world.day, summary: makeWorldSummary() });
  },

  resetWorld(seed: number, npcCount = DEFAULT_NPC_COUNT): Promise<{ message: string; day: number }> {
    ({ world, eventStore } = initializeWorld(seed, npcCount));
    persist();
    return Promise.resolve({ message: 'World reset', day: world.day });
  },

  getEvents(limit = 50): Promise<WorldEvent[]> {
    ensureInit();
    return Promise.resolve(
      eventStore.recentEvents(limit).map(e => ({
        id: e.id, day: e.day, tick: e.tick, type: e.type,
        actorId: e.actorId, settlementId: e.settlementId,
        payload: e.payload as Record<string, unknown>,
        tags: e.tags, salience: e.salience,
      }))
    );
  },

  getHighSalienceEvents(): Promise<WorldEvent[]> {
    ensureInit();
    return Promise.resolve(
      eventStore.highSalienceEvents(0.6, 20).map(e => ({
        id: e.id, day: e.day, tick: e.tick, type: e.type,
        actorId: e.actorId, settlementId: e.settlementId,
        payload: e.payload as Record<string, unknown>,
        tags: e.tags, salience: e.salience,
      }))
    );
  },

  getNPCs(filters?: { job?: string; household?: string }): Promise<NPCSummary[]> {
    ensureInit();
    let npcs = Object.values(world.npcs).filter(n => n.isAlive);
    if (filters?.job) npcs = npcs.filter(n => n.jobId === filters.job);
    if (filters?.household) npcs = npcs.filter(n => n.householdId === filters.household);
    return Promise.resolve(npcs.map(npcToSummary));
  },

  getNPC(id: string): Promise<NPCFull> {
    ensureInit();
    const npc = world.npcs[id];
    if (!npc) return Promise.reject(new Error(`NPC ${id} not found`));
    return Promise.resolve(npcToFull(npc));
  },

  getNPCRelationships(id: string): Promise<Relationship[]> {
    ensureInit();
    const npc = world.npcs[id];
    if (!npc) return Promise.reject(new Error(`NPC ${id} not found`));
    return Promise.resolve(
      Object.values(npc.relationships).map(r => ({
        targetId: r.targetId,
        targetName: world.npcs[r.targetId]?.name ?? r.targetId,
        familiarity: r.familiarity,
        affection: r.affection,
        trust: r.trust,
        respect: r.respect,
        conflict: r.conflict,
        kinshipType: r.kinshipType,
      }))
    );
  },

  getNPCMemories(id: string): Promise<Memory[]> {
    ensureInit();
    const npc = world.npcs[id];
    if (!npc) return Promise.reject(new Error(`NPC ${id} not found`));
    return Promise.resolve(
      npc.memories.map(m => ({
        id: m.id, day: m.day, eventType: m.eventType,
        salience: m.salience, emotionalValence: m.emotionalValence,
        participants: m.participants, tags: m.tags, summary: m.summary, decay: m.decay,
      }))
    );
  },

  getTopViabilityNPCs(): Promise<NPCSummary[]> {
    ensureInit();
    return Promise.resolve(coreTopViability(world, 10).map(npcToSummary));
  },

  getBottomViabilityNPCs(): Promise<NPCSummary[]> {
    ensureInit();
    return Promise.resolve(coreBottomViability(world, 10).map(npcToSummary));
  },

  getSettlements(): Promise<Settlement[]> {
    ensureInit();
    return Promise.resolve(
      Object.values(world.settlements).map(s => ({
        id: s.id, name: s.name, regionId: s.regionId, population: s.population,
        resources: s.resources,
      }))
    );
  },

  getSettlement(id: string): Promise<Settlement> {
    ensureInit();
    const s = world.settlements[id];
    if (!s) return Promise.reject(new Error(`Settlement ${id} not found`));
    return Promise.resolve({ id: s.id, name: s.name, regionId: s.regionId, population: s.population, resources: s.resources });
  },

  getHouseholds(): Promise<Household[]> {
    ensureInit();
    return Promise.resolve(
      Object.values(world.households).map(h => ({
        id: h.id, name: h.name, memberIds: h.memberIds,
        foodStores: h.foodStores, wealth: h.wealth,
        homeQuality: h.homeQuality, reputation: h.reputation,
        members: h.memberIds.map(mid => {
          const n = world.npcs[mid];
          return n ? { id: n.id, name: n.name, jobId: n.jobId } : { id: mid, name: mid };
        }),
      }))
    );
  },

  getHousehold(id: string): Promise<Household> {
    ensureInit();
    const h = world.households[id];
    if (!h) return Promise.reject(new Error(`Household ${id} not found`));
    return Promise.resolve({
      id: h.id, name: h.name, memberIds: h.memberIds,
      foodStores: h.foodStores, wealth: h.wealth,
      homeQuality: h.homeQuality, reputation: h.reputation,
      members: h.memberIds.map(mid => {
        const n = world.npcs[mid];
        return n ? { id: n.id, name: n.name, jobId: n.jobId } : { id: mid, name: mid };
      }),
    });
  },

  getDungeons(): Promise<DungeonRoom[]> {
    ensureInit();
    return Promise.resolve(
      world.dungeonRooms.map(d => ({
        id: d.id, settlementId: d.settlementId, type: d.type,
        pressureSource: d.pressureSource, description: d.description,
        difficulty: d.difficulty, tags: d.tags,
      }))
    );
  },

  getQuests(): Promise<Quest[]> {
    ensureInit();
    return Promise.resolve(
      Object.values(world.quests).map(q => ({
        id: q.id, title: q.title, description: q.description,
        cause: q.cause, urgency: q.urgency, difficulty: q.difficulty,
        isActive: q.isActive,
        objectives: q.objectives.map(o => ({ description: o.description, completed: o.completed, type: o.type })),
        tags: q.tags, generatedOnDay: q.generatedOnDay, issuerNpcId: q.issuerNpcId,
      }))
    );
  },

  getRegions: () => Promise.resolve(REGIONS),
  getFactions: () => Promise.resolve(FACTIONS),
  getPeoples: () => Promise.resolve(PEOPLES),
  getAssets: () => Promise.resolve(ASSETS),
};

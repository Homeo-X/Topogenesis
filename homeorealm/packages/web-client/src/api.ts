/** HomeoRealm API client */

const BASE = '/api';

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`API error ${res.status}: ${path}`);
  return res.json() as Promise<T>;
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`API error ${res.status}: ${path}`);
  return res.json() as Promise<T>;
}

export const api = {
  getWorld: () => get<WorldSummary>('/world'),
  getFullWorld: () => get<any>('/world/full'),
  getPlayer: () => get<PlayerState>('/player'),
  playerAction: (action: PlayerActionInput) => post<PlayerActionResponse>('/player/action', action),
  tick: () => post<{ day: number; summary: WorldSummary }>('/world/tick'),
  runDays: (days: number) => post<{ day: number; summary: WorldSummary }>('/world/run', { days }),
  resetWorld: (seed: number, npcCount?: number) => post<{ message: string; day: number }>('/world/reset', { seed, npcCount }),
  getEvents: (limit?: number) => get<WorldEvent[]>(`/world/events${limit ? `?limit=${limit}` : ''}`),
  getHighSalienceEvents: () => get<WorldEvent[]>('/world/events/high-salience'),

  getNPCs: (filters?: { job?: string; household?: string }) => {
    const q = new URLSearchParams(filters as any).toString();
    return get<NPCSummary[]>(`/npcs${q ? `?${q}` : ''}`);
  },
  getNPC: (id: string) => get<NPCFull>(`/npcs/${id}`),
  getNPCRelationships: (id: string) => get<Relationship[]>(`/npcs/${id}/relationships`),
  getNPCMemories: (id: string) => get<Memory[]>(`/npcs/${id}/memories`),
  getTopViabilityNPCs: () => get<NPCSummary[]>('/npcs/ranking/top'),
  getBottomViabilityNPCs: () => get<NPCSummary[]>('/npcs/ranking/bottom'),

  getSettlements: () => get<Settlement[]>('/settlements'),
  getSettlement: (id: string) => get<Settlement>(`/settlements/${id}`),
  getHouseholds: () => get<Household[]>('/households'),
  getHousehold: (id: string) => get<Household>(`/households/${id}`),
  getDungeons: () => get<DungeonRoom[]>('/dungeons'),

  getQuests: () => get<Quest[]>('/quests'),

  getRegions: () => get<Region[]>('/lore/regions'),
  getFactions: () => get<Faction[]>('/lore/factions'),
  getPeoples: () => get<People[]>('/lore/peoples'),
  getAssets: () => get<Asset[]>('/lore/assets'),
};

// ─── API types ──────────────────────────────────────────────────────────────

export type WorldSummary = {
  day: number;
  settlements: { id: string; name: string; population: number; resources: Resources }[];
  totalNPCs: number;
  avgViability: string;
  activeQuests: number;
  dungeonRooms: number;
  environment: EnvironmentState;
};

export type EnvironmentalPhysics = {
  airTemperatureC: number;
  humidity: number;
  rainfallMm: number;
  windSpeedMps: number;
  windDirectionDeg: number;
  airPressureKpa: number;
  soilMoisture: number;
  groundwater: number;
  evaporationRate: number;
  turbulence: number;
};

export type EnvironmentalChemistry = {
  oxygenRatio: number;
  carbonDioxidePpm: number;
  soilPH: number;
  mineralSaturation: number;
  dissolvedIron: number;
  organicMatter: number;
  fermentation: number;
  corrosion: number;
};

export type EnvironmentState = {
  regionId: string;
  season: string;
  weather: 'clear' | 'mist' | 'rain' | 'storm' | 'dry_heat' | 'frost';
  physics: EnvironmentalPhysics;
  chemistry: EnvironmentalChemistry;
  signals: string[];
};

export type PlayerActionInput = {
  type:
    | 'accept_quest'
    | 'complete_quest'
    | 'aid_settlement'
    | 'travel'
    | 'gather'
    | 'rest'
    | 'train'
    | 'trade'
    | 'talk'
    | 'delve_dungeon';
  questId?: string;
  settlementId?: string;
  location?: PlayerLocation;
  npcId?: string;
};

export type PlayerLocation = 'town' | 'market' | 'wilds' | 'dungeon' | 'home';

export type PlayerQuestRecord = {
  questId: string;
  status: 'accepted' | 'completed';
  acceptedOnDay: number;
  completedOnDay?: number;
};

export type PlayerState = {
  id: string;
  name: string;
  settlementId: string;
  location: PlayerLocation;
  health: number;
  stamina: number;
  level: number;
  experience: number;
  skills: Record<string, number>;
  wealth: number;
  reputation: Record<string, number>;
  inventory: { itemId: string; quantity: number }[];
  questLog: PlayerQuestRecord[];
  actionLog: string[];
};

export type PlayerActionResponse = {
  player: PlayerState;
  message: string;
};

export type Resources = {
  food: number; wood: number; ore: number; cloth: number;
  medicine: number; tools: number; coin: number;
  heartwellStability: number; publicMorale: number; security: number;
};

export type WorldEvent = {
  id: string; day: number; tick: number; type: string;
  actorId?: string; settlementId?: string; payload: Record<string, unknown>;
  tags: string[]; salience: number;
};

export type NPCSummary = {
  id: string; name: string; people: string; jobId?: string;
  householdId?: string; settlementId: string;
  viability: number; health: number; wealth: number; currentAction: string;
};

export type NPCFull = NPCSummary & {
  needs: Record<string, number>;
  affect: Record<string, number>;
  skills: Record<string, number>;
  relationships: Record<string, unknown>;
  memories: Memory[];
  frame: { distinctions: FrameDistinction[]; attentionBias: Record<string, number>; tabooTags: string[]; valuedTags: string[] };
  currentIntention?: { action: string; motivation: string; expectedViabilityGain: number };
};

export type Relationship = {
  targetId: string; targetName: string; familiarity: number; affection: number;
  trust: number; respect: number; conflict: number; kinshipType: string;
};

export type Memory = {
  id: string; day: number; eventType: string; salience: number;
  emotionalValence: number; participants: string[]; tags: string[]; summary: string; decay: number;
};

export type FrameDistinction = {
  id: string; label: string; domain: string;
  confidence: number; utilityWeight: number; createdOnDay: number;
};

export type Settlement = {
  id: string; name: string; regionId: string; population: number;
  resources: Resources; pressures?: string[];
};

export type Household = {
  id: string; name: string; memberIds: string[]; foodStores: number;
  wealth: number; homeQuality: number; reputation: number;
  members?: { id: string; name: string; jobId?: string }[];
};

export type DungeonRoom = {
  id: string; settlementId: string; type: string; pressureSource: string;
  description: string; difficulty: number; tags: string[];
};

export type Quest = {
  id: string; title: string; description: string; cause: string;
  urgency: number; difficulty: number; isActive: boolean;
  objectives: { description: string; completed: boolean; type: string }[];
  tags: string[]; generatedOnDay: number; issuerNpcId?: string;
  acceptedByPlayerId?: string; completedByPlayerId?: string;
};

export type Region = { id: string; name: string; description: string; climate: string; mainResources: string[]; faction?: string };
export type Faction = { id: string; name: string; description: string; function: string; allegiances?: string[] };
export type People = { id: string; name: string; region: string; description: string; assetTheme: string; culturalTraits: string[] };
export type Asset = {
  id: string;
  category: string;
  name: string;
  region?: string;
  faction?: string;
  description: string;
  gameplayUse: string;
  visualKeywords: string[];
  productionPriority: string;
  sourceCollection?: string;
  sourceRepository?: string;
  sourceUrl?: string;
  sourceLicense?: string;
  localAssetUrl?: string;
};

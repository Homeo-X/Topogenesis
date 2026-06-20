/** Core shared types for HomeoRealm Online simulation. */

// ─── NPC Core ──────────────────────────────────────────────────────────────

export type NPCNeeds = {
  hunger: number;     // 0 = starving, 1 = satisfied
  rest: number;       // 0 = exhausted, 1 = rested
  safety: number;     // 0 = endangered, 1 = secure
  belonging: number;  // 0 = isolated, 1 = connected
  purpose: number;    // 0 = purposeless, 1 = fulfilled
  autonomy: number;   // 0 = constrained, 1 = free
  esteem: number;     // 0 = scorned, 1 = respected
};

export type NPCAffect = {
  valence: number;     // -1 = very negative, 1 = very positive
  arousal: number;     // 0 = calm, 1 = excited/agitated
  stress: number;      // 0 = relaxed, 1 = severely stressed
  trustBaseline: number; // default trust extended to new contacts
};

export type RelationshipState = {
  targetId: string;
  familiarity: number;   // 0–1 how well known
  affection: number;     // -1–1 warmth/hostility
  trust: number;         // 0–1 reliability expectation
  respect: number;       // 0–1 competence/standing
  conflict: number;      // 0–1 active tension
  kinshipType: KinshipType;
  romanceInterest: number; // 0–1
  obligationStrength: number; // 0–1 duty/debt
  lastInteractionDay: number;
};

export type KinshipType = 'stranger' | 'acquaintance' | 'friend' | 'rival' | 'enemy' | 'kin' | 'spouse' | 'parent' | 'child' | 'mentor' | 'apprentice' | 'ally';

export type MemoryTrace = {
  id: string;
  npcId: string;
  day: number;
  eventType: string;
  salience: number;    // 0–1 importance
  emotionalValence: number; // -1–1
  participants: string[];
  tags: string[];
  summary: string;
  decay: number;       // 0–1, reduces over time
};

export type FrameDistinction = {
  id: string;
  label: string;
  domain: 'food' | 'safety' | 'social' | 'craft' | 'trade' | 'ritual' | 'politics' | 'ecology' | 'combat' | 'memory';
  confidence: number;      // 0–1
  utilityWeight: number;   // how much this changes action scoring
  learnedFromEventIds: string[];
  createdOnDay: number;
  lastReinforcedDay: number;
};

export type NPCFrameState = {
  distinctions: FrameDistinction[];
  attentionBias: Record<string, number>;  // tag → weight modifier
  tabooTags: string[];
  valuedTags: string[];
};

export type Obligation = {
  id: string;
  type: 'debt' | 'promise' | 'duty' | 'contract' | 'ritual';
  targetId?: string;
  description: string;
  urgency: number;
  dueDay?: number;
  fulfilled: boolean;
};

export type ItemStack = {
  itemId: string;
  quantity: number;
  quality: number; // 0–1
};

export type Intention = {
  action: ActionType;
  targetId?: string;
  targetResource?: string;
  motivation: string;
  expectedViabilityGain: number;
};

export type ActionType =
  | 'eat' | 'cook' | 'seek_food' | 'gather_resource'
  | 'work_job' | 'rest' | 'sleep'
  | 'social_visit' | 'help_household'
  | 'train_skill' | 'craft_item'
  | 'seek_healer' | 'respond_to_danger'
  | 'attend_festival' | 'generate_request'
  | 'migrate' | 'change_role' | 'idle';

// ─── NPC State ─────────────────────────────────────────────────────────────

export type NPCState = {
  id: string;
  name: string;
  ageDays: number;
  people: string;
  householdId?: string;
  settlementId: string;
  roleIds: string[];
  skills: Record<string, number>;
  traits: Record<string, number>;
  needs: NPCNeeds;
  affect: NPCAffect;
  relationships: Record<string, RelationshipState>;
  memories: MemoryTrace[];
  obligations: Obligation[];
  inventory: ItemStack[];
  wealth: number;
  health: number;
  reputation: Record<string, number>; // settlementId → score
  frame: NPCFrameState;
  currentIntention?: Intention;
  isAlive: boolean;
  birthDay: number;
  jobId?: string;
};

// ─── Household ─────────────────────────────────────────────────────────────

export type HouseholdState = {
  id: string;
  name: string;
  memberIds: string[];
  settlementId: string;
  sharedInventory: ItemStack[];
  foodStores: number;    // normalized 0–10 (10 = abundant)
  wealth: number;
  homeQuality: number;   // 0–1
  reputation: number;    // 0–1 in settlement
  inheritanceRules: string;
  lineage: LineageRecord[];
};

export type LineageRecord = {
  npcId: string;
  name: string;
  birthDay: number;
  deathDay?: number;
  role: string;
};

// ─── Settlement ────────────────────────────────────────────────────────────

export type SettlementResources = {
  food: number;
  wood: number;
  ore: number;
  cloth: number;
  medicine: number;
  tools: number;
  coin: number;
  heartwellStability: number; // 0–1
  publicMorale: number;       // 0–1
  security: number;           // 0–1
};

export type SettlementState = {
  id: string;
  name: string;
  regionId: string;
  factionId?: string;
  population: number;
  resources: SettlementResources;
  npcIds: string[];
  householdIds: string[];
  day: number;
};

// Physics and chemistry model for the playable environment. Values are
// normalized unless a physical unit is explicitly included in the name.
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

// ─── Quests ────────────────────────────────────────────────────────────────

export type QuestObjective = {
  id: string;
  description: string;
  type: 'deliver' | 'gather' | 'escort' | 'craft' | 'heal' | 'mediate' | 'explore' | 'guard' | 'investigate';
  quantity?: number;
  resourceId?: string;
  targetNpcId?: string;
  locationId?: string;
  completed: boolean;
};

export type QuestReward = {
  type: 'coin' | 'item' | 'reputation' | 'relationship' | 'skill';
  value: number;
  targetId?: string;
};

export type QuestConsequence = {
  type: 'resource_change' | 'npc_viability_change' | 'relationship_change' | 'settlement_change';
  targetId?: string;
  magnitude: number;
  description: string;
};

export type EmergentQuest = {
  id: string;
  title: string;
  description: string;
  issuerNpcId?: string;
  settlementId: string;
  cause: string;
  urgency: number;    // 0–1
  difficulty: number; // 0–1
  objectives: QuestObjective[];
  rewards: QuestReward[];
  consequences: QuestConsequence[];
  expiresOnDay?: number;
  tags: string[];
  generatedOnDay: number;
  isActive: boolean;
  acceptedByPlayerId?: string;
  completedByPlayerId?: string;
};

export type PlayerInventoryItem = {
  itemId: string;
  quantity: number;
};

export type PlayerQuestStatus = 'accepted' | 'completed';

export type PlayerQuestRecord = {
  questId: string;
  status: PlayerQuestStatus;
  acceptedOnDay: number;
  completedOnDay?: number;
};

export type PlayerState = {
  id: string;
  name: string;
  settlementId: string;
  location: 'town' | 'market' | 'wilds' | 'dungeon' | 'home';
  health: number;
  stamina: number;
  level: number;
  experience: number;
  skills: Record<string, number>;
  wealth: number;
  reputation: Record<string, number>;
  inventory: PlayerInventoryItem[];
  questLog: PlayerQuestRecord[];
  actionLog: string[];
};

// ─── World State ───────────────────────────────────────────────────────────

export type WorldState = {
  seed: number;
  day: number;
  tick: number;
  settlements: Record<string, SettlementState>;
  npcs: Record<string, NPCState>;
  households: Record<string, HouseholdState>;
  quests: Record<string, EmergentQuest>;
  dungeonRooms: DungeonRoom[];
  environment: EnvironmentState;
  player?: PlayerState;
};

// ─── Dungeon ───────────────────────────────────────────────────────────────

export type DungeonRoom = {
  id: string;
  settlementId: string;
  type: 'hunger' | 'oath' | 'boundary' | 'kin' | 'knowledge' | 'hearthwell';
  pressureSource: string;
  description: string;
  difficulty: number;
  tags: string[];
  generatedOnDay: number;
  linkedQuestId?: string;
};

export type QuestDomain =
  | "craft"
  | "mind"
  | "body"
  | "order"
  | "social"
  | "courage"
  | "recovery"
  | "learning"
  | "planning"
  | "creation"
  | "exploration";

export type QuestMode =
  | "push"
  | "steady"
  | "recover"
  | "clarity"
  | "courage"
  | "boss"
  | "maintenance"
  | "silent"
  | "rebellion";

export type OutcomeType =
  | "COMPLETED_FULL"
  | "EXCEEDED_OBJECTIVE"
  | "COMPLETED_CREATIVELY"
  | "COMPLETED_WITH_COST"
  | "COMPLETED_LOW_QUALITY"
  | "REPLACED_WITH_EQUIVALENT"
  | "COMPLETED_PARTIAL"
  | "FAILED_ATTEMPTED"
  | "FAILED_BLOCKED"
  | "FAILED_AVOIDED"
  | "SKIPPED_CONSCIOUSLY";

export type EvidenceType = "self_report" | "reflection" | "artifact" | "timer" | "location";

/** Timer capture for evidence: planned vs actually-measured effort. */
export interface TimerEvidence {
  startedAt: number;
  endedAt: number;
  plannedMinutes: number;
  actualMinutes: number;
}

/** One concrete piece of evidence attached to an outcome. */
export interface EvidenceItem {
  id: string;
  kind: EvidenceType;
  /** Free text: the reflection, self-report, or artifact description. */
  note?: string;
  /** Link / file reference for artifact evidence. */
  artifactUri?: string;
  /** Present when kind === "timer". */
  timer?: TimerEvidence;
  /** Present when kind === "location": the discovered-grid cell id (never raw coordinates). */
  cellId?: string;
  /** Per-item confidence (0..1) derived from the evidence kind. */
  confidence: number;
  recordedAt: number;
}

export type StatName =
  | "Vitality"
  | "Insight"
  | "Discipline"
  | "Courage"
  | "Order"
  | "Craft"
  | "Bond"
  | "Recovery";

export interface BodyComposition {
  recordedAt: number;
  /** All fields optional — a scan may not report every metric. Stored as-is;
   *  display shows everything, but quest logic reads only capability signals. */
  weightKg?: number;
  fatMassKg?: number;
  muscleMassKg?: number;
  boneMassKg?: number;
  proteinMassKg?: number;
  waterWeightKg?: number;
  skeletalMuscleKg?: number;
  bmi?: number;
  bodyFatPct?: number;
  visceralFatGrade?: number;
  basalMetabolicRate?: number;
  smi?: number; // skeletal muscle index kg/m^2
  bodyAge?: number;
  /** Segmental muscle % of standard, used for L/R balance (capability signal). */
  segmentLeftArmPct?: number;
  segmentRightArmPct?: number;
  segmentTrunkPct?: number;
  segmentLeftLegPct?: number;
  segmentRightLegPct?: number;
  note?: string;
}

export interface HostState {
  energy: number;
  focus: number;
  stress: number;
  mood: string;
  timeAvailableMinutes: number;
  recoveryNeed: number;
  challengeReadiness: number;
  /** Readiness for people-facing quests right now (0..1). */
  socialReadiness: number;
  /** Readiness for open-ended creative work right now (0..1). */
  creativeReadiness: number;
  /** How the body itself feels today (0..1): sore/ill low, strong high. */
  bodyStatus: number;
  /** Where this check-in came from. Manual now; sensor integrations later. */
  source: "manual" | "sensor";
  desiredMode: QuestMode;
  scannedAt: number;
}

export interface Goal {
  id: string;
  text: string;
  domain: QuestDomain;
  priority: number;
  status: "active" | "paused" | "done" | "archived";
  arc: {
    title: string;
    currentPhase: string;
    bottleneck: string;
    weeklyFocus: string;
    milestones: Array<{
      id: string;
      label: string;
      objective: string;
      status: "locked" | "active" | "done";
    }>;
    nextActions: string[];
  };
}

export interface HostProfile {
  goals: Goal[];
  values: string[];
  constraints: string[];
  preferences: {
    narrativeStyle: "system_dark" | "quiet" | "heroic";
    proofMode: "trust" | "proof";
    intensityMode: "gentle" | "balanced" | "intense";
    notificationConsent: boolean;
    physicalQuests: boolean;
    socialQuests: boolean;
    outdoorQuests: boolean;
    soundEnabled: boolean;
    /** Master gate for director-issued surprise quests (emergency/hidden/boss). */
    surpriseQuests: boolean;
    /** Fine-grained gates under the master. */
    hiddenQuests: boolean;
    bossQuests: boolean;
    /** Emergency quest framing: direct uses urgent wording, calm uses soft wording. */
    emergencyWording: "direct" | "calm";
    /** Highest quest risk tier (0-4) the user consents to be offered. */
    maxRiskTier: number;
    /** Android: deliver System notifications on a high-importance (alarm-like) channel. */
    alarmStyleNotifications: boolean;
    /** Opt-in: read position on demand (opening the World map, completing a
     *  quest) for territory + evidence. Never continuous, never background. */
    locationQuests: boolean;
    hapticsEnabled: boolean;
    reduceMotion: boolean;
    nightMode: boolean;
  };
  difficultyCalibration: Record<string, number>;
  knownPatterns: string[];
  resistancePatterns: string[];
}

export type QuestRank = "F" | "E" | "D" | "C" | "B" | "A" | "S";

export type QuestType =
  | "micro"
  | "stabilizer"
  | "clarifier"
  | "builder"
  | "practice"
  | "social"
  | "exploration"
  | "boss"
  | "maintenance"
  | "rebellion";

export type DifficultyBand = "easier" | "standard" | "harder";

export interface QuestScoreBreakdown {
  goalRelevance: number;
  stateFit: number;
  timeFit: number;
  difficultyFit: number;
  safety: number;
  growthValue: number;
  novelty: number;
  evidenceClarity: number;
}

export interface CeremonyEvent {
  kind: "level" | "mastery" | "skill" | "discovery";
  title: string;
  lines: string[];
  at: number;
}

export interface Quest {
  id: string;
  /** One visible line of council reasoning: why the System ranks/frames this offer. */
  councilNote?: string;
  /** Council-adjusted score used for final ranking (acceptanceScore x advisor weights). */
  councilScore?: number;
  candidateId: string;
  goalId: string;
  title: string;
  rank: QuestRank;
  domain: QuestDomain;
  mode: QuestMode;
  questType: QuestType;
  difficultyBand: DifficultyBand;
  objective: string;
  activityPlan: {
    intent: string;
    /** Short, honest framing of why this quest matters — what it builds or
     *  protects. Non-coercive: never frames stakes as loss/punishment. */
    stakes: string;
    steps: Array<{
      id: string;
      label: string;
      minutes: number;
      instruction: string;
      output: string;
    }>;
    successCriteria: string[];
    fallback: string;
    antiAvoidanceRule: string;
  };
  proofRequired: string;
  timeLimitMinutes: number;
  riskTier: 0 | 1 | 2 | 3 | 4;
  rewards: {
    xp: number;
    stats: Partial<Record<StatName, number>>;
  };
  systemMessage: string;
  acceptanceScore: number;
  scoreBreakdown: QuestScoreBreakdown;
  generatorSource: "rules" | "llm_refined";
  createdAt: number;
  status: "offered" | "accepted" | "completed" | "rejected";
}

export interface SkillDef {
  id: string;
  label: string;
  stat: StatName;
  /** Stat value at which this skill unlocks. */
  threshold: number;
  /** Prerequisite skill id that must be unlocked first, or null for tier-1. */
  requires: string | null;
  description: string;
  /** Whether this skill actually changes engine behavior (vs. an earned marker). */
  functional: boolean;
}

export interface ProgressionState {
  level: number;
  xp: number;
  xpToNext: number;
  stats: Record<StatName, number>;
  classPath: {
    chosen: string | null;
    inferred: string | null;
    progress: Record<string, number>;
  };
  titles: string[];
  unlockedSkills: string[];
  streak: {
    current: number;
    best: number;
    shields: number;
    lastQuestDate: string | null;
  };
}

export interface WorldState {
  chapter: number;
  season: number;
  currentRegion: string;
  /** 0..1 progress toward unlocking the next region in the current domain track. */
  regionProgress: number;
  threatLevel: number;
  unlockedLocations: string[];
  companions: string[];
  activeMysteries: string[];
  /** Monotonic count of resolved quests; drives chapter advancement reliably. */
  questsResolved: number;
  log: string[];
  /** Fog-of-war territory: grid cell ids the host has physically visited.
   *  Derived facts only — raw coordinates are never stored. */
  discoveredCells: string[];
}

export interface QuestOutcome {
  type: OutcomeType;
  /** Primary evidence kind (kept for compatibility; first item of evidence[]). */
  evidenceType: EvidenceType;
  note: string;
  completedAt: number;
  /** Full evidence record: every item attached to this outcome. */
  evidence?: EvidenceItem[];
  /** Aggregate verification confidence (0..1) across all evidence items. */
  verificationConfidence?: number;
}

/** LLM-proposed reading of a free-text outcome note. Never applied directly:
 *  the user confirms before any reclassification takes effect. */
export interface OutcomeSuggestion {
  /** What the model thinks actually happened, if it differs from the user's pick. */
  suggestedOutcome: OutcomeType;
  /** Whether that differs from what the user originally selected. */
  differsFromSelected: boolean;
  /** Short named blocker/pattern, always shown (the inform-only signal). */
  blocker: string;
  /** One-line rationale, shown with the suggestion. */
  reasoning: string;
}

export interface LlmConfig {
  enabled: boolean;
  provider: "backend" | "ollama";
  endpoint: string;
  model: string;
}

export type SpecialQuestKind = "hidden" | "boss" | "emergency";

export interface DirectorEvent {
  kind: SpecialQuestKind;
  reason: string;
  triggeredAt: number;
}

export interface SystemSignals {
  /** Consecutive avoid/fail count per domain, used to surface hidden quests. */
  domainAvoidance: Record<string, number>;
  /** Rolling count of recent low-quality outcomes, used for emergency detection. */
  failureStreak: number;
  /** Rolling count of recent strong completions, used for boss readiness. */
  momentum: number;
  /** Timestamp of last boss offer, for cooldown. */
  lastBossOfferAt: number | null;
  /** Timestamp of last emergency quest, for cooldown. */
  lastEmergencyAt: number | null;
  /** Domains already surfaced as a hidden quest, so the System does not repeat. */
  resolvedHiddenDomains: string[];
  /** Most recent director decision, for UI/telemetry. */
  lastEvent: DirectorEvent | null;
  /** True when the most recent location sample landed in never-visited territory. */
  currentCellNovel: boolean;
}

export interface Assumption {
  id: string;
  label: string;
  value: string;
  confidence: number;
  source: string[];
  protected: boolean;
  status: "active" | "rejected";
  evidenceCount: number;
  lastUpdated: number;
}

export interface AppState {
  profile: HostProfile;
  host: HostState;
  progression: ProgressionState;
  world: WorldState;
  llm: LlmConfig;
  selectedGoalId: string | null;
  goalSelectionReason: string;
  questOffers: Quest[];
  /** All quests currently in progress (cap enforced by the engine). */
  activeQuests: Quest[];
  /** DERIVED alias: always activeQuests[0] ?? null. Maintained by the engine
   *  for backward compatibility — never set directly. */
  activeQuest: Quest | null;
  questHistory: Array<Quest & { outcome?: QuestOutcome }>;
  assumptions: Assumption[];
  adaptationLog: string[];
  systemSignals: SystemSignals;
  lastSystemMessage: string;
  /** One-shot pin: when true, generateQuestOffers respects the current
   *  selectedGoalId instead of auto-reselecting, then clears the pin. Set by
   *  explicit user targeting (e.g. the activity browser). */
  goalPinned?: boolean;
  /** A pending full-screen ceremony (level-up / domain mastery / skill unlock).
   *  Set by the engine on the qualifying outcome; cleared by the UI on dismiss. */
  pendingCeremony?: CeremonyEvent | null;
  hasOnboarded: boolean;
  bodyHistory: BodyComposition[];
}

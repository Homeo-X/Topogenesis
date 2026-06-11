import {
  AppState,
  CeremonyEvent,
  EvidenceItem,
  TimerEvidence,
  Assumption,
  DifficultyBand,
  EvidenceType,
  Goal,
  HostProfile,
  HostState,
  OutcomeType,
  ProgressionState,
  Quest,
  QuestDomain,
  QuestMode,
  QuestRank,
  QuestScoreBreakdown,
  QuestType,
  SpecialQuestKind,
  StatName,
  SystemSignals,
  WorldState
} from "./types";
import { voiceOpener } from "./voice";
import { computeUnlockedSkills, newlyUnlocked } from "./skills";
import { deriveBodySignals } from "./body";
import { deliberate } from "./council";
import { EffortTier, domainLevelFromPoints, renderAction, selectArchetype, tierXpMultiplier, unlockedArchetypes } from "./questArchetypes";

const stats: StatName[] = [
  "Vitality",
  "Insight",
  "Discipline",
  "Courage",
  "Order",
  "Craft",
  "Bond",
  "Recovery"
];

export const domainToStat: Record<QuestDomain, StatName> = {
  craft: "Craft",
  mind: "Insight",
  body: "Vitality",
  order: "Order",
  social: "Bond",
  courage: "Courage",
  recovery: "Recovery",
  learning: "Insight",
  planning: "Order",
  creation: "Craft",
  exploration: "Vitality"
};

const outcomeQuality: Record<OutcomeType, number> = {
  EXCEEDED_OBJECTIVE: 1.25,
  COMPLETED_CREATIVELY: 1.08,
  COMPLETED_FULL: 1,
  REPLACED_WITH_EQUIVALENT: 0.82,
  COMPLETED_WITH_COST: 0.72,
  COMPLETED_LOW_QUALITY: 0.62,
  COMPLETED_PARTIAL: 0.55,
  FAILED_ATTEMPTED: 0.15,
  FAILED_BLOCKED: 0.1,
  FAILED_AVOIDED: 0,
  SKIPPED_CONSCIOUSLY: 0
};

const evidenceConfidence: Record<EvidenceType, number> = {
  self_report: 0.58,
  reflection: 0.7,
  artifact: 0.86,
  timer: 0.75,
  // GPS-backed completion cell: strong, hard-to-fake, but coarse (cell-level).
  location: 0.8
};

const scoreWeights: Record<keyof QuestScoreBreakdown, number> = {
  goalRelevance: 0.18,
  stateFit: 0.18,
  timeFit: 0.12,
  difficultyFit: 0.12,
  safety: 0.14,
  growthValue: 0.1,
  novelty: 0.08,
  evidenceClarity: 0.08
};

export const domains: Array<{ id: QuestDomain; label: string }> = [
  { id: "craft", label: "Craft" },
  { id: "mind", label: "Mind" },
  { id: "body", label: "Body" },
  { id: "order", label: "Order" },
  { id: "social", label: "Social" },
  { id: "courage", label: "Courage" },
  { id: "recovery", label: "Recovery" },
  { id: "learning", label: "Learning" },
  { id: "planning", label: "Planning" },
  { id: "creation", label: "Creation" },
  { id: "exploration", label: "Exploration" }
];

export const questModes: Array<{ id: QuestMode; label: string }> = [
  { id: "steady", label: "Steady" },
  { id: "clarity", label: "Clarity" },
  { id: "recover", label: "Recover" },
  { id: "courage", label: "Courage" },
  { id: "push", label: "Push" },
  { id: "maintenance", label: "Maintain" },
  { id: "silent", label: "Silent" },
  { id: "rebellion", label: "Rebellion" },
  { id: "boss", label: "Boss" }
];

export function makeInitialState(): AppState {
  return {
    profile: {
      goals: [],
      values: [],
      constraints: ["No shame language", "Ask before high-risk quests"],
      preferences: {
        narrativeStyle: "system_dark",
        proofMode: "trust",
        intensityMode: "balanced",
        notificationConsent: false,
        physicalQuests: false,
        socialQuests: false,
        outdoorQuests: false,
      soundEnabled: true,
      surpriseQuests: true,
      hiddenQuests: true,
      bossQuests: true,
      emergencyWording: "direct" as const,
      maxRiskTier: 4,
      alarmStyleNotifications: false,
      locationQuests: false,
        hapticsEnabled: false,
        reduceMotion: false,
        nightMode: false
      },
      difficultyCalibration: {
        time: 0.5,
        energy: 0.5,
        focus: 0.5,
        emotional: 0.5,
        social: 0.5,
        physical: 0.5,
        activation: 0.5
      },
      knownPatterns: [],
      resistancePatterns: []
    },
    host: emptyHostState(),
    progression: emptyProgression(),
    world: emptyWorld(),
    llm: {
      enabled: false,
      provider: "backend",
      endpoint: "http://localhost:8787",
      model: "llama3.2"
    },
    selectedGoalId: null,
    goalSelectionReason: "No goal selected yet.",
    questOffers: [],
    activeQuests: [],
    activeQuest: null,
    questHistory: [],
    assumptions: [],
    adaptationLog: [],
    systemSignals: emptySystemSignals(),
    lastSystemMessage: "System ready. Add a goal to get your first quest.",
    hasOnboarded: false,
    bodyHistory: []
  };
}

export function emptySystemSignals(): SystemSignals {
  return {
    domainAvoidance: {},
    failureStreak: 0,
    momentum: 0,
    lastBossOfferAt: null,
    lastEmergencyAt: null,
    resolvedHiddenDomains: [],
    lastEvent: null,
    currentCellNovel: false
  };
}

export function emptyHostState(): HostState {
  return {
    energy: 0.55,
    focus: 0.55,
    stress: 0.35,
    mood: "unknown",
    timeAvailableMinutes: 20,
    recoveryNeed: 0.35,
    challengeReadiness: 0.5,
    socialReadiness: 0.5,
    creativeReadiness: 0.5,
    bodyStatus: 0.5,
    source: "manual" as const,
    desiredMode: "steady",
    scannedAt: Date.now()
  };
}

function emptyProgression(): ProgressionState {
  return {
    level: 1,
    xp: 0,
    xpToNext: 100,
    stats: Object.fromEntries(stats.map((stat) => [stat, 0])) as Record<StatName, number>,
    classPath: { chosen: null, inferred: "Initiate", progress: {} },
    titles: [],
    unlockedSkills: [],
    streak: { current: 0, best: 0, shields: 0, lastQuestDate: null }
  };
}

function emptyWorld(): WorldState {
  return {
    chapter: 1,
    season: 1,
    currentRegion: "The Outer Archive",
    regionProgress: 0,
    threatLevel: 0.2,
    unlockedLocations: ["The Outer Archive"],
    companions: [],
    activeMysteries: [],
    questsResolved: 0,
    discoveredCells: [],
    log: ["The System is ready and waiting for a goal."]
  };
}

function buildGoalArc(goal: string, domain: QuestDomain): Goal["arc"] {
  const domainLabel = domains.find((item) => item.id === domain)?.label ?? "Growth";
  const isBuild = domain === "craft" || domain === "creation";
  const isMind = domain === "mind" || goal.toLowerCase().includes("mind") || goal.toLowerCase().includes("mood");
  const isLearning = domain === "learning";

  const milestones = isBuild
    ? [
        ["scope", "Plan the smallest version", "Decide the smallest version worth shipping."],
        ["prototype", "Build a rough version", "Make a rough version that works."],
        ["test", "Try it for real", "Use it, see what breaks, and note the fixes."],
        ["ship", "Share it", "Package, publish, or share what you made."]
      ]
    : isLearning
      ? [
          ["map", "Get the lay of the land", "List the main ideas and pick where to start."],
          ["practice", "Practice from memory", "Turn what you read into memory and examples."],
          ["apply", "Use it once", "Use the skill on a real task."],
          ["teach", "Explain it", "Teach or sum up the skill in your own words."]
        ]
      : isMind
        ? [
            ["observe", "Notice what's going on", "Name the patterns without judging them."],
            ["regulate", "Find what steadies you", "Find actions that reliably calm things down."],
            ["protect", "Set up your surroundings", "Cut down on triggers and add support."],
            ["integrate", "Make it a habit", "Turn what steadies you into a routine."]
          ]
        : [
            ["define", "Decide what 'done' looks like", "Get clear on what progress means."],
            ["first", "Take the first step", "Do one visible action."],
            ["repeat", "Make it repeatable", "Make the action easier to do again."],
            ["expand", "Do a little more", "Take on more once it's clearly working."]
          ];

  return {
    title: `${domainLabel} Arc`,
    currentPhase: milestones[0][1],
    bottleneck: isMind ? "staying steady" : isBuild ? "knowing what to build next" : "getting started",
    weeklyFocus: milestones[0][2],
    milestones: milestones.map(([id, label, objective], index) => ({
      id,
      label,
      objective,
      status: index === 0 ? "active" : "locked"
    })),
    nextActions: [
      `Write the smallest visible version of: ${goal}`,
      "Run a quick scan, then pick from the quests it suggests.",
      "Log what actually happened, even if you only got part way."
    ]
  };
}

export function addGoal(state: AppState, text: string, domain: QuestDomain): AppState {
  const goal = {
    id: makeId("goal"),
    text: text.trim(),
    domain,
    priority: 1,
    status: "active" as const,
    arc: buildGoalArc(text.trim(), domain)
  };
  return autoSelectGoal({
    ...ensureStateShape(state),
    profile: { ...state.profile, goals: [goal, ...state.profile.goals] },
    lastSystemMessage: `${voiceOpener("goal_registered", state)} I can suggest quests for you now.`
  });
}

const SYSTEM_GOAL_BANK: Array<{ text: string; domain: QuestDomain; needs?: keyof HostProfile["preferences"]; fit: (state: AppState) => number }> = [
  { text: "Drink water, breathe for two minutes, and write the next safe step", domain: "recovery", fit: (state) => state.host.recoveryNeed + state.host.stress * 0.4 },
  { text: "Choose one next action and write what done means today", domain: "planning", fit: (state) => 1 - state.host.focus },
  { text: "Create one visible file, note, sketch, or app change", domain: "craft", fit: (state) => state.host.energy * 0.5 + state.host.focus * 0.5 },
  { text: "Make one rough paragraph, sketch, audio note, or idea draft", domain: "creation", fit: (state) => state.host.focus * 0.35 + state.host.challengeReadiness * 0.45 },
  { text: "Calm your mind with one short walk, breath set, or journal note", domain: "mind", fit: (state) => state.host.stress * 0.45 + state.host.recoveryNeed * 0.35 },
  { text: "Study one small topic and write three things you remember", domain: "learning", fit: (state) => state.host.focus * 0.5 + state.host.energy * 0.25 },
  { text: "Clear one visible pile, app inbox, desk corner, or downloads folder", domain: "order", fit: (state) => 0.35 + (1 - state.host.focus) * 0.35 },
  { text: "Do one small thing you have been avoiding for two minutes", domain: "courage", fit: (state) => state.host.challengeReadiness * 0.55 + state.systemSignals.failureStreak * 0.04 },
  { text: "Visit one nearby safe place or take one short outdoor route", domain: "exploration", needs: "outdoorQuests", fit: (state) => state.host.energy * 0.35 + state.host.timeAvailableMinutes / 120 },
  { text: "Do one gentle movement set: walk, stretch, or bodyweight reps", domain: "body", needs: "physicalQuests", fit: (state) => state.host.energy * 0.45 + state.host.recoveryNeed * 0.15 },
  { text: "Send one kind, clear message or prepare the first sentence", domain: "social", needs: "socialQuests", fit: (state) => state.host.energy * 0.25 + state.host.focus * 0.25 }
];

export function generateSystemGoal(state: AppState): AppState {
  const nextState = ensureStateShape(state);
  const existing = new Set(nextState.profile.goals.map((goal) => goal.text.toLowerCase()));
  const viable = SYSTEM_GOAL_BANK
    .filter((candidate) => !candidate.needs || nextState.profile.preferences[candidate.needs])
    .filter((candidate) => !existing.has(candidate.text.toLowerCase()));
  const pool = viable.length > 0 ? viable : SYSTEM_GOAL_BANK.filter((candidate) => !candidate.needs || nextState.profile.preferences[candidate.needs]);
  const scored = pool
    .map((candidate) => ({ candidate, score: candidate.fit(nextState) + Math.random() * 0.18 }))
    .sort((a, b) => b.score - a.score);
  const selected = scored[0]?.candidate ?? SYSTEM_GOAL_BANK[0];
  const created = addGoal(nextState, selected.text, selected.domain);
  return {
    ...created,
    lastSystemMessage: `${voiceOpener("goal_registered", nextState)} Goal added. I can suggest quests for you now.`
  };
}

export function updateHost(state: AppState, host: HostState): AppState {
  return autoSelectGoal({
    ...ensureStateShape(state),
    host: { ...host, scannedAt: Date.now() },
    questOffers: [],
    lastSystemMessage: `${voiceOpener("scan_accepted", state)} I'll update the quest suggestions.`
  });
}

/** Update a goal's priority, status, or text. Pausing/finishing/archiving the
 *  currently selected goal hands selection back to the System. */
/** Territory discovery milestones that earn a ceremony (never XP: the map is
 *  its own reward, and XP stays quest-earned so the economy can't be walked). */
const DISCOVERY_MILESTONES = [1, 5, 25, 100, 250];

/** Record visited grid cells. Dedupes against known territory, flags whether
 *  the latest sample is novel ground (the Scout advisor reads this), and emits
 *  a discovery ceremony when total charted territory crosses a milestone. */
export function discoverCells(state: AppState, cellIds: string[]): AppState {
  const nextState = ensureStateShape(state);
  if (!cellIds.length) return nextState;
  const known = new Set(nextState.world.discoveredCells);
  const fresh = cellIds.filter((id, index) => !known.has(id) && cellIds.indexOf(id) === index);
  const latest = cellIds[cellIds.length - 1];
  const currentCellNovel = !known.has(latest);
  if (!fresh.length) {
    return { ...nextState, systemSignals: { ...nextState.systemSignals, currentCellNovel } };
  }
  const before = nextState.world.discoveredCells.length;
  const discoveredCells = [...nextState.world.discoveredCells, ...fresh];
  const milestone = DISCOVERY_MILESTONES.find((m) => before < m && discoveredCells.length >= m);
  const pendingCeremony = milestone
    ? {
        kind: "discovery" as const,
        title: "TERRITORY CHARTED",
        lines: [
          milestone === 1
            ? "Your first area is on the map. The world starts here."
            : `${discoveredCells.length} areas charted. The map grows because you moved.`
        ],
        at: Date.now()
      }
    : nextState.pendingCeremony ?? null;
  return {
    ...nextState,
    world: { ...nextState.world, discoveredCells },
    systemSignals: { ...nextState.systemSignals, currentCellNovel },
    pendingCeremony,
    lastSystemMessage: fresh.length === 1
      ? "New territory charted on your world map."
      : `${fresh.length} new areas charted on your world map.`
  };
}

export function updateGoal(
  state: AppState,
  goalId: string,
  patch: Partial<Pick<Goal, "priority" | "status" | "text">>
): AppState {
  const nextState = ensureStateShape(state);
  const target = nextState.profile.goals.find((g) => g.id === goalId);
  if (!target) return nextState;
  const goals = nextState.profile.goals.map((g) => (g.id === goalId ? { ...g, ...patch } : g));
  const updated = { ...nextState, profile: { ...nextState.profile, goals } };
  const deactivated = patch.status && patch.status !== "active" && nextState.selectedGoalId === goalId;
  const message =
    patch.status === "paused" ? "Goal paused. It keeps its progress and can resume anytime."
    : patch.status === "done" ? "Goal marked done. Well closed."
    : patch.status === "archived" ? "Goal archived. Out of sight, recoverable from data."
    : patch.status === "active" ? "Goal resumed."
    : "Goal updated.";
  const withMessage = { ...updated, lastSystemMessage: message };
  return deactivated ? autoSelectGoal({ ...withMessage, selectedGoalId: null }) : withMessage;
}

/** Permanently remove a goal. Selection hygiene mirrors updateGoal: deleting
 *  the focused goal hands selection back to the System. Quest history keeps
 *  its records — deleting a goal never rewrites what already happened. */
export function deleteGoal(state: AppState, goalId: string): AppState {
  const nextState = ensureStateShape(state);
  if (!nextState.profile.goals.some((g) => g.id === goalId)) return nextState;
  const goals = nextState.profile.goals.filter((g) => g.id !== goalId);
  const wasSelected = nextState.selectedGoalId === goalId;
  const updated = {
    ...nextState,
    profile: { ...nextState.profile, goals },
    lastSystemMessage: "Goal deleted. Its quest history stays in the chronicle."
  };
  return wasSelected ? autoSelectGoal({ ...updated, selectedGoalId: null }) : updated;
}

export function autoSelectGoal(state: AppState): AppState {
  const nextState = ensureStateShape(state);
  const activeGoals = nextState.profile.goals.filter((goal) => goal.status === "active");
  if (activeGoals.length === 0) {
    return {
      ...nextState,
      selectedGoalId: null,
      goalSelectionReason: "No active goals available."
    };
  }

  const mode = chooseMode(nextState.host);
  const scored = activeGoals
    .map((goal) => {
      const goalDomain = chooseDomain(goal.domain, mode);
      const fit = scoreQuestFit(nextState.host, mode, goalDomain);
      const priority = goal.priority * 0.18;
      const recoveryBonus = nextState.host.recoveryNeed > 0.65 && (goal.domain === "recovery" || goal.domain === "mind") ? 0.18 : 0;
      const focusBonus = nextState.host.focus < 0.4 && (goal.domain === "planning" || goal.domain === "mind") ? 0.14 : 0;
      const craftBonus = nextState.host.energy > 0.55 && nextState.host.focus > 0.5 && (goal.domain === "craft" || goal.domain === "creation") ? 0.12 : 0;
      // Readiness dials from check-in: each shifts its domain by at most ±0.12
      // around the neutral 0.5, so untouched dials change nothing.
      const socialBonus = goal.domain === "social" ? (nextState.host.socialReadiness - 0.5) * 0.24 : 0;
      const creativeBonus = goal.domain === "creation" ? (nextState.host.creativeReadiness - 0.5) * 0.24 : 0;
      const bodyBonus = goal.domain === "body" || goal.domain === "exploration" ? (nextState.host.bodyStatus - 0.5) * 0.2 : 0;
      const recentPenalty = recentGoalPenalty(nextState, goal.id);
      return {
        goal,
        score: fit + priority + recoveryBonus + focusBonus + craftBonus + socialBonus + creativeBonus + bodyBonus - recentPenalty,
        fit,
        reason: reasonForGoalChoice(goal.domain, mode, fit, { recoveryBonus, focusBonus, craftBonus, recentPenalty })
      };
    })
    .sort((a, b) => b.score - a.score);

  const selected = scored[0];
  return {
    ...nextState,
    selectedGoalId: selected.goal.id,
    goalSelectionReason: selected.reason
  };
}

export function generateQuest(state: AppState): AppState {
  return generateQuestOffers(state);
}

export function generateQuestOffers(state: AppState): AppState {
  // Honor an explicit one-shot goal pin (user targeted this goal, e.g. via the
  // activity browser); otherwise let the System choose what fits their state.
  const pinnedGoal = state.goalPinned
    ? state.profile.goals.find((item) => item.id === state.selectedGoalId && item.status === "active")
    : undefined;
  const selectedState = pinnedGoal
    ? { ...state, goalPinned: false, goalSelectionReason: "You chose this focus." }
    : autoSelectGoal(state);
  const goal =
    pinnedGoal ??
    selectedState.profile.goals.find((item) => item.id === selectedState.selectedGoalId && item.status === "active") ??
    selectedState.profile.goals.find((item) => item.status === "active");
  if (!goal) {
    return { ...selectedState, lastSystemMessage: "Add a goal first, then I can suggest quests." };
  }

  const candidates = buildCandidateSpecs(selectedState, goal)
    .map((spec, index) => materializeQuest(selectedState, goal, spec, index))
    .sort((a, b) => b.acceptanceScore - a.acceptanceScore);

  const unique = dedupeCandidates(candidates).slice(0, 3);

  // Event director: detect a real behavioral pattern and, if one fires,
  // surface a coherent special quest at the top of the offers. Normal
  // candidates are preserved so the user always retains choice.
  const direction = runEventDirector(selectedState, goal);
  let offers = unique;
  let signals = selectedState.systemSignals;
  let directorMessage = "";

  if (direction) {
    // Dedupe again with the special quest first so a normal candidate that
    // shares its template (e.g. a director recover quest + a normal recover
    // quest) collapses instead of showing twice. The special stays at index 0.
    offers = dedupeCandidates([direction.quest, ...unique]).slice(0, 3);
    signals = direction.signals;
    directorMessage = direction.message;
  }

  // Concurrency hygiene: never offer a template that's already running.
  const activeCandidateIds = new Set(selectedState.activeQuests.map((quest) => quest.candidateId));
  offers = offers.filter((offer) => !activeCandidateIds.has(offer.candidateId));

  // Risk consent: never offer above the user's chosen risk ceiling. If the
  // ceiling filters everything, fall back to the single lowest-risk offer so
  // generation always yields something actionable.
  const maxRisk = selectedState.profile.preferences.maxRiskTier ?? 4;
  const withinRisk = offers.filter((o) => o.riskTier <= maxRisk);
  offers = withinRisk.length ? withinRisk : [offers.slice().sort((a, b) => a.riskTier - b.riskTier)[0]];

  // Council deliberation: the multi-advisor layer re-ranks offers and attaches
  // each one's strongest "why". A director special (a detected behavioral
  // pattern) stays pinned above general deliberation — only while it survived
  // the risk filter at position 0.
  const pinHolds = Boolean(direction) && offers[0]?.candidateId === direction?.quest.candidateId;
  offers = deliberate(offers, selectedState, goal, pinHolds).offers;

  const recommended = offers[0];

  return {
    ...selectedState,
    systemSignals: signals,
    questOffers: offers,
    lastSystemMessage: directorMessage
      ? directorMessage
      : recommended
        ? `${voiceOpener("offers_ready", selectedState)} Best pick: ${recommended.title} at ${Math.round(recommended.acceptanceScore * 100)}% fit.`
        : "I couldn't put together a safe quest right now."
  };
}

const BOSS_COOLDOWN_MS = 18 * 60 * 60 * 1000;
const EMERGENCY_COOLDOWN_MS = 6 * 60 * 60 * 1000;
const HIDDEN_AVOIDANCE_THRESHOLD = 3;

interface DirectorResult {
  quest: Quest;
  signals: SystemSignals;
  message: string;
}

/**
 * Decides whether a special quest should appear, based on persisted behavioral
 * signals and current host state. Priority order is safety-first:
 * emergency (stabilize) > hidden (named avoidance) > boss (earned challenge).
 * Returns null when no pattern fires, leaving normal generation untouched.
 */
function runEventDirector(state: AppState, goal: Goal): DirectorResult | null {
  const now = Date.now();
  const host = state.host;
  const signals = state.systemSignals;
  const prefs = state.profile.preferences;
  // Consent gates: the onboarding promise. surpriseQuests is the master switch
  // for all director-issued specials; hidden/boss have fine-grained gates.
  if (!prefs.surpriseQuests) return null;
  const skills = state.progression.unlockedSkills;

  // Functional skill: Relentless makes the failure-streak emergency trigger more
  // forgiving (4 instead of 3) — the user has demonstrated durability.
  const failureTrigger = skills.includes("discipline_relentless") ? 4 : 3;
  // Functional skill: Risk Mapper surfaces hidden blockers one step earlier.
  const hiddenThreshold = skills.includes("order_riskmapper") ? 2 : HIDDEN_AVOIDANCE_THRESHOLD;

  // 1. Emergency: high instability. Gentle, opt-in via cooldown, never shaming.
  const instabilityHigh = (host.stress > 0.72 && host.energy < 0.4) || signals.failureStreak >= failureTrigger;
  const emergencyReady = !signals.lastEmergencyAt || now - signals.lastEmergencyAt > EMERGENCY_COOLDOWN_MS;
  if (instabilityHigh && emergencyReady) {
    const quest = makeSpecialQuest("emergency", state, goal);
    return {
      quest,
      signals: {
        ...signals,
        lastEmergencyAt: now,
        failureStreak: 0,
        lastEvent: { kind: "emergency", reason: "You're showing signs of being stretched thin.", triggeredAt: now }
      },
      message: state.profile.preferences.emergencyWording === "calm"
        ? "[Recovery Focus] Things look heavy right now, so I've put a gentle recovery option first. Take it, change it, or skip it."
        : "[Emergency Quest] You seem stretched thin right now. I've put a gentle recovery option first. Take it, change it, or skip it."
    };
  }

  // 2. Hidden: a domain avoided/failed repeatedly. Names the pattern, no coercion.
  const avoidedDomain = Object.entries(signals.domainAvoidance)
    .filter(([domain, count]) => count >= hiddenThreshold && !signals.resolvedHiddenDomains.includes(domain))
    .sort((a, b) => b[1] - a[1])[0];
  if (avoidedDomain && prefs.hiddenQuests) {
    const [domain] = avoidedDomain;
    const quest = makeSpecialQuest("hidden", state, goal, domain as QuestDomain);
    return {
      quest,
      signals: {
        ...signals,
        resolvedHiddenDomains: [...signals.resolvedHiddenDomains, domain],
        lastEvent: { kind: "hidden", reason: `You've skipped ${domain} quests ${avoidedDomain[1]} times.`, triggeredAt: now }
      },
      message: `[Hidden Quest] You've been steering clear of "${domain}" for a while. This quest just names it — you don't have to solve it all at once.`
    };
  }

  // 3. Boss: genuine high readiness, positive momentum, cooldown elapsed, opted in.
  const intenseAllowed = state.profile.preferences.intensityMode !== "gentle";
  const readinessHigh = host.energy > 0.7 && host.focus > 0.65 && host.stress < 0.5 && host.timeAvailableMinutes >= 45;
  const momentumPositive = signals.momentum >= 2;
  const bossReady = !signals.lastBossOfferAt || now - signals.lastBossOfferAt > BOSS_COOLDOWN_MS;
  if (prefs.bossQuests && intenseAllowed && readinessHigh && momentumPositive && bossReady) {
    const quest = makeSpecialQuest("boss", state, goal);
    return {
      quest,
      signals: {
        ...signals,
        lastBossOfferAt: now,
        lastEvent: { kind: "boss", reason: "You're rested, focused, and on a roll.", triggeredAt: now }
      },
      message: "[Boss Quest] You're in a strong spot right now. Here's a bigger challenge if you want it — your choice."
    };
  }

  return null;
}

/** Builds a special quest reusing the normal materialization path for safety/reward consistency. */
function makeSpecialQuest(kind: SpecialQuestKind, state: AppState, goal: Goal, domain?: QuestDomain): Quest {
  const specByKind: Record<SpecialQuestKind, {
    mode: QuestMode;
    domain: QuestDomain;
    questType: QuestType;
    timeBias: number;
    family: number;
    difficultyBand: DifficultyBand;
  }> = {
    emergency: { mode: "recover", domain: "recovery", questType: "stabilizer", timeBias: 0.5, family: 90, difficultyBand: "easier" },
    hidden: { mode: "courage", domain: domain ?? "courage", questType: "clarifier", timeBias: 0.7, family: 91, difficultyBand: "easier" },
    boss: { mode: "boss", domain: goal.domain, questType: "boss", timeBias: 1.6, family: 92, difficultyBand: "harder" }
  };

  const quest = materializeQuest(state, goal, specByKind[kind], specByKind[kind].family);

  const overlays: Record<SpecialQuestKind, Partial<Quest>> = {
    emergency: {
      title: "Stabilize the Core",
      systemMessage: "[Emergency Quest] Preserve tomorrow's ability to continue. No identity penalty applies.",
      rank: "E"
    },
    hidden: {
      title: `Name the Shadow${domain ? `: ${domain}` : ""}`,
      systemMessage: "[Hidden Quest] Open the avoided work and write the exact obstacle. You do not have to solve it yet.",
      rank: "E"
    },
    boss: {
      title: "Breach the Gate",
      systemMessage: "[Boss Quest] A high-impact push. Offered because readiness is genuinely high. Decline freely.",
      rank: "A"
    }
  };

  return {
    ...quest,
    ...overlays[kind],
    candidateId: `special_${kind}_${Date.now()}`,
    generatorSource: "rules"
  };
}

/** Concurrent quest cap: enough to layer a habit, a push, and a recovery
 *  without diluting the commitment a quest is supposed to carry. */
export const MAX_ACTIVE_QUESTS = 3;

/** Single writer for active quests: keeps the derived activeQuest alias in
 *  lockstep so no code path can ever desynchronize the two. */
function withActiveQuests(state: AppState, activeQuests: Quest[]): AppState {
  return { ...state, activeQuests, activeQuest: activeQuests[0] ?? null };
}

export function acceptQuest(state: AppState, questId?: string, confirmedRisk = false): AppState {
  const nextState = ensureStateShape(state);
  const chosen = questId
    ? nextState.questOffers.find((quest) => quest.id === questId)
    : nextState.questOffers[0];
  if (!chosen) return nextState;
  if (nextState.activeQuests.some((quest) => quest.id === chosen.id)) return nextState;
  if (nextState.activeQuests.length >= MAX_ACTIVE_QUESTS) {
    return {
      ...nextState,
      lastSystemMessage: `Your quest log is full (${MAX_ACTIVE_QUESTS} active). Resolve one before taking another.`
    };
  }
  if (chosen.riskTier >= 3 && !confirmedRisk) {
    return {
      ...nextState,
      lastSystemMessage: "This is a high-risk quest. Confirm before you start."
    };
  }
  return {
    ...withActiveQuests(nextState, [...nextState.activeQuests, { ...chosen, status: "accepted" }]),
    // Only the accepted offer leaves the board: the rest stay available so
    // several quests can be taken from one batch.
    questOffers: nextState.questOffers.filter((quest) => quest.id !== chosen.id),
    lastSystemMessage: `${voiceOpener("quest_accepted", state)} Keep the proof simple and honest.`
  };
}

export function rejectQuest(state: AppState, questId?: string): AppState {
  const nextState = ensureStateShape(state);
  const rejected = questId
    ? nextState.questOffers.find((quest) => quest.id === questId)
    : nextState.questOffers[0];
  if (!rejected) return nextState;

  const remainingOffers = nextState.questOffers.filter((quest) => quest.id !== rejected.id);

  return {
    ...nextState,
    questHistory: [{ ...rejected, status: "rejected" as const }, ...nextState.questHistory].slice(0, 50),
    questOffers: remainingOffers,
    lastSystemMessage: remainingOffers.length > 0
      ? "Quest skipped. The other options are still here."
      : "Quest skipped. Ask for new ones whenever you're ready."
  };
}

export function submitOutcome(
  state: AppState,
  type: OutcomeType,
  evidenceType: EvidenceType,
  note: string,
  extraEvidence?: Array<{ kind: EvidenceType; note?: string; artifactUri?: string; timer?: TimerEvidence; cellId?: string }>,
  questId?: string
): AppState {
  const nextState = ensureStateShape(state);
  // Resolve the target: explicit id, or the first active quest (legacy default).
  const target = questId
    ? nextState.activeQuests.find((quest) => quest.id === questId)
    : nextState.activeQuests[0];
  if (!target) return nextState;

  // Runtime guard: an invalid outcome type (corrupted storage, future caller
  // bug) must never produce NaN XP that would persist into the save. Fall back
  // to partial-completion quality.
  const quality = outcomeQuality[type] ?? outcomeQuality.COMPLETED_PARTIAL;
  // Functional skill: Archivist rewards a demonstrated reflection habit by
  // weighting reflection evidence slightly higher (capped at 1).
  const archivist = nextState.progression.unlockedSkills.includes("insight_archivist");
  const itemConfidence = (kind: EvidenceType) => {
    const base = evidenceConfidence[kind];
    return archivist && kind === "reflection" ? Math.min(1, base + 0.1) : base;
  };
  // Evidence record: the primary item (from the chosen evidence type + note)
  // plus any extra items (timer capture, artifact link, added reflections).
  const recordedAt = Date.now();
  const evidenceItems: EvidenceItem[] = [
    { id: makeId("ev"), kind: evidenceType, note: note || undefined, confidence: itemConfidence(evidenceType), recordedAt },
    ...(extraEvidence ?? []).map((extra) => ({
      id: makeId("ev"),
      kind: extra.kind,
      note: extra.note,
      artifactUri: extra.artifactUri,
      timer: extra.timer,
      cellId: extra.cellId,
      confidence: itemConfidence(extra.kind),
      recordedAt
    }))
  ];
  // Aggregate verification: independent-evidence combination with diminishing
  // returns, capped below certainty. One item reduces exactly to its own
  // confidence, so legacy single-evidence behavior (and XP) is unchanged.
  const verificationConfidence = Math.min(
    0.98,
    1 - evidenceItems.reduce((product, item) => product * (1 - item.confidence), 1)
  );
  const confidence = verificationConfidence;
  const stability = stabilityGate(nextState, quality, confidence);
  // Functional skills: Sanctuary Maker uplifts recovery-quest XP; Abyss Walker
  // uplifts courage-quest XP (high-emotional-cost work recognized). Honest:
  // these change the reward for genuinely-completed quests in those domains.
  const unlockedSkillsNow = nextState.progression.unlockedSkills;
  const domainXpMultiplier =
    (unlockedSkillsNow.includes("recovery_sanctuary") && target.domain === "recovery") ||
    (unlockedSkillsNow.includes("courage_abyss") && target.domain === "courage")
      ? 1.25
      : 1;
  const gainedXp = Math.round(target.rewards.xp * quality * confidence * stability.learningRate * domainXpMultiplier);
  const progression = applyReward(nextState.progression, target, gainedXp, quality, type);
  const world = updateWorld(nextState.world, target, type, quality);
  const historyItem = {
    ...target,
    status: "completed" as const,
    outcome: { type, evidenceType, note: note.trim(), completedAt: recordedAt, evidence: evidenceItems, verificationConfidence }
  };
  const adaptation = adaptProfile(nextState, type, quality, confidence, stability.learningRate);
  const systemSignals = updateSignalsFromOutcome(nextState.systemSignals, target, type, quality, progression.unlockedSkills);
  const unlocked = newlyUnlocked(nextState.progression.unlockedSkills, progression.stats);
  const unlockNote = unlocked.length > 0 ? ` [Skill unlocked] ${unlocked.map((s) => s.label).join(", ")}.` : "";
  // Surface domain mastery: when this quest's domain crosses a level, tell the
  // player — the deepened archetype pool was previously invisible to them.
  const questDomain = target.domain;
  const questStat = domainToStat[questDomain];
  const levelBefore = domainLevelFromPoints(nextState.progression.stats[questStat] ?? 0);
  const levelAfter = domainLevelFromPoints(progression.stats[questStat] ?? 0);
  const newlyOpened = levelAfter > levelBefore
    ? unlockedArchetypes(questDomain, levelAfter).length - unlockedArchetypes(questDomain, levelBefore).length
    : 0;
  const masteryNote = levelAfter > levelBefore
    ? ` [${domains.find((d) => d.id === questDomain)?.label ?? questDomain} mastery ${levelAfter}]${newlyOpened > 0 ? ` ${newlyOpened} new quest types unlocked.` : ""}`
    : "";

  // Ceremony: the felt moment for real progression. Level-up outranks mastery
  // outranks skill; lines stack so one ceremony can carry all of today's wins.
  const leveledUp = progression.level > nextState.progression.level;
  const ceremonyLines: string[] = [];
  if (leveledUp) ceremonyLines.push(`You are now Level ${progression.level}.`);
  if (levelAfter > levelBefore) {
    const dLabel = domains.find((d) => d.id === questDomain)?.label ?? questDomain;
    ceremonyLines.push(`${dLabel} mastery ${levelAfter}${newlyOpened > 0 ? ` — ${newlyOpened} new quest types unlocked` : ""}.`);
  }
  if (unlocked.length > 0) ceremonyLines.push(`Skill unlocked: ${unlocked.map((s) => s.label).join(", ")}.`);
  const pendingCeremony = ceremonyLines.length
    ? {
        kind: (leveledUp ? "level" : levelAfter > levelBefore ? "mastery" : "skill") as CeremonyEvent["kind"],
        title: leveledUp ? "LEVEL UP" : levelAfter > levelBefore ? "MASTERY DEEPENS" : "SKILL UNLOCKED",
        lines: ceremonyLines,
        at: Date.now()
      }
    : null;

  return autoSelectGoal({
    ...nextState,
    profile: adaptation.profile,
    progression,
    world,
    assumptions: adaptation.assumptions,
    adaptationLog: [...adaptation.log, ...nextState.adaptationLog].slice(0, 12),
    questHistory: [historyItem, ...nextState.questHistory].slice(0, 50),
    systemSignals,
    ...(() => {
      const remaining = nextState.activeQuests.filter((quest) => quest.id !== target.id);
      return { activeQuests: remaining, activeQuest: remaining[0] ?? null };
    })(),
    questOffers: [],
    pendingCeremony,
    lastSystemMessage: `${voiceOpener("outcome_logged", nextState)} ${stability.message} ${adaptation.log[0] ?? ""}${unlockNote}${masteryNote}`.trim()
  });
}

/**
 * Folds a completed outcome into the persisted behavioral signals the event
 * director reads. Avoidance accrues per domain on avoid/block; a strong
 * completion clears that domain's avoidance and builds boss momentum.
 */
function updateSignalsFromOutcome(
  signals: SystemSignals,
  quest: Quest,
  type: OutcomeType,
  quality: number,
  skills: string[] = []
): SystemSignals {
  const domain = quest.domain;
  const avoided = type === "FAILED_AVOIDED" || type === "FAILED_BLOCKED" || type === "SKIPPED_CONSCIOUSLY";
  const strong = quality >= 0.85;
  // Functional skill: Pattern Breaker makes re-engagement decay avoidance twice
  // as fast, so an avoided domain clears more readily once the user faces it.
  const decay = skills.includes("courage_breaker") ? 2 : 1;
  // Distress = tried and struggled (overload), distinct from avoidance.
  // This keeps the emergency pathway (stabilize) separate from the hidden
  // pathway (name the avoided thing), so one cannot mask the other.
  const distress = type === "FAILED_ATTEMPTED" || type === "COMPLETED_WITH_COST" || type === "COMPLETED_LOW_QUALITY";

  const domainAvoidance = { ...signals.domainAvoidance };
  if (avoided) {
    domainAvoidance[domain] = (domainAvoidance[domain] ?? 0) + 1;
  } else if (quality > 0.5) {
    // Honest engagement with a domain relaxes its avoidance pressure.
    domainAvoidance[domain] = Math.max(0, (domainAvoidance[domain] ?? 0) - decay);
  }

  // If the user re-engaged a previously resolved hidden domain, allow it to
  // surface again later should avoidance rebuild.
  const resolvedHiddenDomains = quality > 0.5
    ? signals.resolvedHiddenDomains.filter((item) => item !== domain || (domainAvoidance[domain] ?? 0) > 0)
    : signals.resolvedHiddenDomains;

  return {
    ...signals,
    domainAvoidance,
    resolvedHiddenDomains,
    failureStreak: distress ? signals.failureStreak + 1 : quality >= 0.55 ? Math.max(0, signals.failureStreak - 1) : signals.failureStreak,
    momentum: strong ? signals.momentum + 1 : quality < 0.4 ? 0 : signals.momentum
  };
}

export function updateAssumption(
  state: AppState,
  id: string,
  action: "confirm" | "protect" | "reject"
): AppState {
  const nextState = ensureStateShape(state);
  return {
    ...nextState,
    assumptions: nextState.assumptions.map((assumption) => {
      if (assumption.id !== id) return assumption;
      if (action === "reject") return { ...assumption, status: "rejected", confidence: 0.05, protected: false, lastUpdated: Date.now() };
      if (action === "protect") return { ...assumption, protected: !assumption.protected, confidence: Math.max(assumption.confidence, 0.72), lastUpdated: Date.now() };
      return {
        ...assumption,
        confidence: clamp(assumption.confidence + 0.15, 0, 1),
        evidenceCount: assumption.evidenceCount + 1,
        lastUpdated: Date.now()
      };
    }),
    lastSystemMessage: `${voiceOpener("assumption_updated", state)} You can always see what I assume about you.`
  };
}

function buildCandidateSpecs(state: AppState, goal: Goal) {
  const baseMode = chooseMode(state.host);
  const goalDomain = chooseDomain(goal.domain, baseMode);
  const calibration = state.profile.difficultyCalibration.activation;
  // Level nudges the same calibration toward harder bands (level 1 = no nudge).
  // This only changes which bands are OFFERED; scoreCandidate's difficultyFit
  // still ranks them against the user's calibration, so a too-hard quest can't
  // win — the adaptive cap holds.
  const levelBias = levelBandBias(state.progression.level);
  const preferredBand: DifficultyBand = calibration < 0.42 - levelBias ? "easier" : calibration > 0.68 - levelBias ? "harder" : "standard";
  const alternates: DifficultyBand[] = preferredBand === "easier"
    ? ["easier", "standard"]
    : preferredBand === "harder"
      ? ["harder", "standard"]
      : ["standard", "easier", "harder"];
  const specs: Array<{ mode: QuestMode; domain: QuestDomain; questType: QuestType; timeBias: number; family: number; difficultyBand: DifficultyBand }> = [
    { mode: baseMode, domain: goalDomain, questType: typeFor(baseMode, goalDomain), timeBias: 1, family: 0, difficultyBand: alternates[0] },
    { mode: "clarity", domain: "planning", questType: "clarifier", timeBias: 0.75, family: 1, difficultyBand: "standard" },
    { mode: "steady", domain: goal.domain, questType: typeFor("steady", goal.domain), timeBias: 1, family: 2, difficultyBand: alternates[1] ?? "standard" },
    { mode: "recover", domain: "recovery", questType: "stabilizer", timeBias: 0.5, family: 3, difficultyBand: "easier" },
    { mode: "maintenance", domain: goal.domain === "body" ? "body" : "order", questType: "maintenance", timeBias: 0.6, family: 4, difficultyBand: "easier" },
    { mode: "push", domain: goal.domain, questType: typeFor("push", goal.domain), timeBias: 1.35, family: 5, difficultyBand: "harder" },
    { mode: "silent", domain: goal.domain, questType: "micro", timeBias: 0.45, family: 6, difficultyBand: "easier" },
    { mode: "rebellion", domain: goal.domain, questType: "rebellion", timeBias: 0.8, family: 7, difficultyBand: alternates[0] },
    { mode: "boss", domain: goal.domain, questType: "boss", timeBias: 1.6, family: 8, difficultyBand: "harder" },
    { mode: baseMode, domain: goalDomain, questType: typeFor(baseMode, goalDomain), timeBias: preferredBand === "harder" ? 1.3 : 0.55, family: 9, difficultyBand: preferredBand === "harder" ? "harder" : "easier" },
    { mode: "steady", domain: "exploration", questType: "exploration", timeBias: 1.1, family: 10, difficultyBand: alternates[0] },
    { mode: "silent", domain: "exploration", questType: "exploration", timeBias: 0.65, family: 11, difficultyBand: "easier" }
  ];

  return specs.filter((spec) => {
    if (spec.mode === "boss" && (state.host.challengeReadiness < 0.72 || state.host.stress > 0.62)) return false;
    if (spec.mode === "push" && state.host.energy < 0.38) return false;
    if (spec.domain === "exploration") {
      const explicitlyRequested = goal.domain === "exploration";
      const consented = state.profile.preferences.outdoorQuests;
      if (!explicitlyRequested && !consented) return false;
      if (state.host.timeAvailableMinutes < 15) return false;
      if (state.host.energy < 0.28 && !explicitlyRequested) return false;
    }
    return true;
  }).concat(skillCandidateSpecs(state));
}

/**
 * Functional skills that surface extra candidate quests when unlocked and
 * relevant. These add OPTIONS to the offer pool (scored alongside the rest);
 * they never force a quest, preserving the agency principle.
 */
function skillCandidateSpecs(state: AppState): Array<{ mode: QuestMode; domain: QuestDomain; questType: QuestType; timeBias: number; family: number; difficultyBand: DifficultyBand }> {
  const skills = state.progression.unlockedSkills;
  const extra: Array<{ mode: QuestMode; domain: QuestDomain; questType: QuestType; timeBias: number; family: number; difficultyBand: DifficultyBand }> = [];

  // Shadow Namer: when avoidance exists anywhere, surface a courage option.
  const anyAvoidance = Object.values(state.systemSignals.domainAvoidance).some((c) => c >= 1);
  if (skills.includes("courage_namer") && anyAvoidance) {
    extra.push({ mode: "courage", domain: "courage", questType: "clarifier", timeBias: 0.7, family: 20, difficultyBand: "easier" });
  }

  // Connector: surface a social option, but only if the user has not disabled
  // social quests — the skill never overrides a consent preference.
  if (skills.includes("bond_connector") && state.profile.preferences.socialQuests) {
    extra.push({ mode: "steady", domain: "social", questType: typeFor("steady", "social"), timeBias: 0.8, family: 21, difficultyBand: "easier" });
  }

  // Prototyper: surface an extra build/craft option.
  if (skills.includes("craft_prototyper")) {
    extra.push({ mode: "steady", domain: "craft", questType: "builder", timeBias: 1, family: 22, difficultyBand: "standard" });
  }

  // Rhythm: surface a daily-maintenance option to support consistency.
  if (skills.includes("discipline_rhythm")) {
    extra.push({ mode: "maintenance", domain: "order", questType: "maintenance", timeBias: 0.6, family: 23, difficultyBand: "easier" });
  }

  // Initiate: surface a body/physical option, only if physical quests are enabled.
  if (skills.includes("vitality_initiate") && state.profile.preferences.physicalQuests) {
    extra.push({ mode: "steady", domain: "body", questType: typeFor("steady", "body"), timeBias: 0.7, family: 24, difficultyBand: "easier" });
  }

  // Body-scan signal: a meaningful left/right imbalance surfaces a corrective
  // MOVEMENT option (behavior), only if physical quests are enabled. This is a
  // capability quest — never a weight/fat target. Composition change, if any,
  // is a downstream consequence of doing the movement, not the quest's goal.
  const body = deriveBodySignals(state);
  if (body.imbalanceDetected && state.profile.preferences.physicalQuests) {
    extra.push({ mode: "steady", domain: "body", questType: typeFor("steady", "body"), timeBias: 0.65, family: 25, difficultyBand: "easier" });
  }

  return extra;
}

function materializeQuest(
  state: AppState,
  goal: Goal,
  spec: { mode: QuestMode; domain: QuestDomain; questType: QuestType; timeBias: number; family: number; difficultyBand: DifficultyBand },
  index: number
): Quest {
  const time = chooseTimeLimit(state.host, spec.mode, bandedTimeBias(spec.timeBias, spec.difficultyBand));
  const scoreBreakdown = scoreCandidate(state, goal, spec.mode, spec.domain, spec.questType, spec.difficultyBand, time, index);
  const acceptanceScore = weightedScore(scoreBreakdown);
  const rank = rankFor(acceptanceScore, spec.mode, spec.questType);
  const stat = domainToStat[spec.domain];
  const baseXp = 16 + acceptanceScore * 48 + (rank === "S" ? 40 : rank === "A" ? 28 : rank === "B" ? 18 : 0);
  // Variable XP by effort: harder bands draw larger archetypes, worth more.
  const effortTier: EffortTier =
    spec.difficultyBand === "easier" ? "small" : spec.difficultyBand === "harder" ? "large" : "medium";
  const xp = Math.round(baseXp * levelRewardFactor(state.progression.level) * tierXpMultiplier(effortTier));

  return {
    id: makeId("quest"),
    candidateId: `candidate_${spec.family}_${spec.difficultyBand}_${Date.now()}`,
    goalId: goal.id,
    title: titleFor(spec.mode, spec.domain, spec.questType),
    rank,
    domain: spec.domain,
    mode: spec.mode,
    questType: spec.questType,
    difficultyBand: spec.difficultyBand,
    objective: objectiveFor(goal.text, spec.mode, spec.domain, spec.questType, spec.difficultyBand, time),
    activityPlan: activityPlanFor(goal.text, spec.mode, spec.domain, spec.questType, spec.difficultyBand, time, state),
    proofRequired: proofFor(state.profile.preferences.proofMode, spec.domain, spec.questType, state.progression.unlockedSkills),
    timeLimitMinutes: time,
    riskTier: riskFor(spec.mode, spec.domain, state),
    rewards: { xp, stats: { [stat]: 1, Discipline: spec.mode === "recover" ? 0 : 1 } },
    systemMessage: messageFor(spec.mode, acceptanceScore),
    acceptanceScore,
    scoreBreakdown,
    generatorSource: "rules",
    createdAt: Date.now(),
    status: "offered"
  };
}

function scoreCandidate(
  state: AppState,
  goal: Goal,
  mode: QuestMode,
  domain: QuestDomain,
  questType: QuestType,
  difficultyBand: DifficultyBand,
  time: number,
  index: number
): QuestScoreBreakdown {
  const calibration = state.profile.difficultyCalibration;
  const goalRelevance = domain === goal.domain ? 1 : domain === "planning" || domain === "recovery" ? 0.72 : 0.62;
  const stateFit = scoreQuestFit(state.host, mode, domain);
  const timeFit = clamp(1 - Math.abs(time - state.host.timeAvailableMinutes) / Math.max(10, state.host.timeAvailableMinutes), 0, 1);
  const difficultyTarget = clamp((calibration.activation + calibration.energy + calibration.focus) / 3, 0, 1);
  const difficultyFit = clamp(1 - Math.abs(difficultyFor(mode, questType, difficultyBand) - difficultyTarget), 0, 1);
  const safety = 1 - riskFor(mode, domain, state) / 4;
  const growthValue = clamp((goal.priority * 0.25) + (questType === "boss" ? 0.55 : questType === "builder" ? 0.42 : 0.35), 0, 1);
  const novelty = noveltyFor(state, mode, domain, questType, index);
  const evidenceClarity = questType === "builder" || questType === "practice" ? 0.88 : questType === "stabilizer" ? 0.68 : 0.78;
  return { goalRelevance, stateFit, timeFit, difficultyFit, safety, growthValue, novelty, evidenceClarity };
}

function weightedScore(score: QuestScoreBreakdown) {
  return clamp(
    Object.entries(score).reduce((sum, [key, value]) => sum + value * scoreWeights[key as keyof QuestScoreBreakdown], 0),
    0,
    1
  );
}

function dedupeCandidates(candidates: Quest[]) {
  const seen = new Set<string>();
  return candidates.filter((quest) => {
    // Key on the visible template (mode + domain + questType), NOT difficultyBand.
    // titleFor/objectiveFor don't vary by band beyond a tiny prefix, so two
    // candidates that differ only in band render as duplicate offers. Candidates
    // arrive sorted by acceptanceScore, so the best-fitting band of each distinct
    // template survives and the visual duplicates (e.g. offers 1 and 3) are dropped.
    const key = `${quest.mode}_${quest.domain}_${quest.questType}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function applyReward(
  progression: ProgressionState,
  quest: Quest,
  gainedXp: number,
  quality: number,
  outcome: OutcomeType
): ProgressionState {
  let xp = progression.xp + gainedXp;
  let level = progression.level;
  let xpToNext = progression.xpToNext;
  while (xp >= xpToNext) {
    xp -= xpToNext;
    level += 1;
    xpToNext = Math.round(xpToNext * 1.22 + 20);
  }

  const nextStats = { ...progression.stats };
  Object.entries(quest.rewards.stats).forEach(([stat, amount]) => {
    const key = stat as StatName;
    nextStats[key] += Math.round((amount ?? 0) * Math.max(1, quality * 2));
  });

  const today = new Date().toISOString().slice(0, 10);
  const alreadyCountedToday = progression.streak.lastQuestDate === today;
  // Functional skill: Iron Streak lowers the bar for a day to "count," making
  // continuity harder to break. Computed from prior unlocks (earned before now).
  const ironStreak = progression.unlockedSkills.includes("discipline_streak");
  const streakWorthy = quality >= (ironStreak ? 0.45 : 0.55);
  // Functional skill: Recovery Shield grants a shield when a recovery-domain
  // quest is completed at decent quality (not only on exceed/creative outcomes).
  const recoveryShieldSkill = progression.unlockedSkills.includes("recovery_shield");
  const earnsShield =
    outcome === "EXCEEDED_OBJECTIVE" ||
    outcome === "COMPLETED_CREATIVELY" ||
    (recoveryShieldSkill && quest.domain === "recovery" && quality >= 0.55);
  // Functional skill: Bulwark raises the shield cap from 3 to 5.
  const shieldCap = progression.unlockedSkills.includes("recovery_bulwark") ? 5 : 3;
  let current = progression.streak.current;
  let shields = progression.streak.shields;

  if (streakWorthy && !alreadyCountedToday) {
    current += 1;
  } else if (!streakWorthy && !alreadyCountedToday) {
    if (shields > 0) {
      shields -= 1;
    } else {
      current = 0;
    }
  }

  if (earnsShield) {
    shields = Math.min(shieldCap, shields + 1);
  }

  const best = Math.max(progression.streak.best, current);

  return {
    ...progression,
    level,
    xp,
    xpToNext,
    stats: nextStats,
    classPath: inferClass(nextStats, progression.classPath.chosen),
    unlockedSkills: computeUnlockedSkills(nextStats),
    streak: { ...progression.streak, current, best, shields, lastQuestDate: alreadyCountedToday ? progression.streak.lastQuestDate : today }
  };
}

function inferClass(statValues: Record<StatName, number>, chosen: string | null) {
  const pairs = Object.entries(statValues).sort((a, b) => b[1] - a[1]);
  const top = pairs[0]?.[0] ?? "Insight";
  const inferred =
    top === "Craft" ? "Artificer" :
    top === "Order" ? "Strategist" :
    top === "Courage" ? "Vanguard" :
    top === "Bond" ? "Envoy" :
    top === "Recovery" ? "Guardian" :
    top === "Vitality" ? "Vanguard" :
    "Scholar";
  return { chosen, inferred, progress: Object.fromEntries(pairs.slice(0, 4)) };
}

const REGION_TRACKS: Record<string, string[]> = {
  craft: ["The Outer Archive", "The Workshop of First Fire", "The Broken Gate", "The Deployment Gate"],
  creation: ["The Outer Archive", "The Workshop of First Fire", "The Broken Gate", "The Deployment Gate"],
  mind: ["The Outer Archive", "The Hall of Echoes", "The Sage's Spire", "The Theorist's Summit"],
  learning: ["The Outer Archive", "The Hall of Echoes", "The Sage's Spire", "The Theorist's Summit"],
  planning: ["The Outer Archive", "The Cartographer's Table", "The Architect's Tower", "The Grand Design"],
  order: ["The Outer Archive", "The Cartographer's Table", "The Architect's Tower", "The Grand Design"],
  courage: ["The Outer Archive", "The Shadow Corridor", "The Threshold", "The Abyss Walk"],
  social: ["The Outer Archive", "The Meeting Hall", "The Bridge", "The Envoy's Court"],
  body: ["The Outer Archive", "The Training Yard", "The Proving Ground", "The Frontline"],
  recovery: ["The Outer Archive", "The Recovery Grove", "The Quiet Sanctum", "The Sanctuary"],
  exploration: ["The Outer Archive", "The Nearby Gate", "The Living Map", "The Horizon Road"]
};

/** A line of flavor shown when a region is first reached, by region name. */
const REGION_DESCRIPTIONS: Record<string, string> = {
  "The Workshop of First Fire": "Where rough first versions are forged. Nothing here has to be good — only real.",
  "The Broken Gate": "A threshold that yields only to shipped work. The unfinished cannot pass.",
  "The Deployment Gate": "Beyond it, what you build meets the world. Few reach it; fewer turn back.",
  "The Hall of Echoes": "Every idea returns changed. Understanding is tested by what survives the return.",
  "The Sage's Spire": "Height earned by synthesis, not accumulation. The climb narrows as it rises.",
  "The Theorist's Summit": "Where foundations become visible. The air is thin and the view is total.",
  "The Cartographer's Table": "Chaos rendered into a map. The first honest plan is drawn here.",
  "The Architect's Tower": "Structure that holds weight. Plans become load-bearing.",
  "The Grand Design": "Where many plans become one coherent arc.",
  "The Shadow Corridor": "The avoided thing is named here, in the dark, without judgment.",
  "The Threshold": "The edge of what felt impossible. Crossing it rewrites the map of yourself.",
  "The Abyss Walk": "The hardest ground. Walked slowly, it holds.",
  "The Meeting Hall": "Where one honest message begins a bond.",
  "The Bridge": "Connection sustained across distance and silence.",
  "The Envoy's Court": "Trust held at scale.",
  "The Training Yard": "Capability built rep by rep, never by force.",
  "The Proving Ground": "Where the body's quiet work becomes visible.",
  "The Frontline": "Sustained physical readiness, held over time.",
  "The Recovery Grove": "Rest treated as progress, not retreat.",
  "The Quiet Sanctum": "Stillness that protects the next move.",
  "The Sanctuary": "Recovery mastered: the ground the whole arc stands on.",
  "The Nearby Gate": "The first outside threshold: a real place entered with attention.",
  "The Living Map": "Where unfamiliar routes become usable knowledge.",
  "The Horizon Road": "The world widened by deliberate contact, not escapism."
};

/** Named companions earned by durable domain mastery (replaces generic labels). */
const DOMAIN_COMPANIONS: Record<string, string> = {
  craft: "Vire, the Maker", creation: "Vire, the Maker", mind: "Sered, the Echo-Reader",
  learning: "Sered, the Echo-Reader", planning: "Castor, the Cartographer", order: "Castor, the Cartographer",
  courage: "Nyx, who walks the dark", social: "Pell, the Envoy", body: "Bron, of the Yard",
  recovery: "Mira, of the Grove", exploration: "Aster, the Wayfinder"
};

/** Varied mystery seeds per domain (replaces the single generic line). */
const DOMAIN_MYSTERIES: Record<string, string> = {
  craft: "Something unbuilt keeps pulling at the edges of the workshop.",
  creation: "An unmade thing waits, insisting quietly on existing.",
  mind: "A thought returns unfinished, asking to be understood.",
  learning: "A concept left half-learned hums at the back of the hall.",
  planning: "An unmapped region distorts every plan drawn near it.",
  order: "A structure leans on a missing support no one has named.",
  courage: "A door stays shut. Behind it, the avoided thing waits without malice.",
  social: "A bond left untended grows quiet. The silence is not yet a wall.",
  body: "The body signals an imbalance the System has not yet addressed.",
  recovery: "Depletion gathers in the corners, asking to be tended before it spreads.",
  exploration: "A nearby place remains unknown, waiting to become part of the map."
};

/** Named chapter and season milestone events. */
function milestoneEvent(kind: "chapter" | "season", n: number): string {
  if (kind === "season") {
    const seasons = ["", "the Season of First Fire", "the Season of the Long Climb", "the Season of the Deep Work", "the Season of the Wide Map"];
    return `A new season turns: ${seasons[Math.min(n, seasons.length - 1)] ?? `Season ${n}`}.`;
  }
  const chapters = ["", "Awakening", "First Forging", "The Named Shadow", "The Held Gate", "The Long Arc", "The Coherent Design"];
  return `Chapter ${n} begins: ${chapters[Math.min(n, chapters.length - 1)] ?? `Chapter ${n}`}.`;
}

function specialKindOf(quest: Quest): SpecialQuestKind | null {
  if (quest.candidateId?.startsWith("special_hidden_")) return "hidden";
  if (quest.candidateId?.startsWith("special_boss_")) return "boss";
  if (quest.candidateId?.startsWith("special_emergency_")) return "emergency";
  return null;
}

function updateWorld(world: WorldState, quest: Quest, outcome: OutcomeType, quality: number): WorldState {
  const special = specialKindOf(quest);

  // Threat: progress lowers it, avoidance raises it; emergency completion calms more.
  const baseThreatDelta = quality > 0.5 ? -0.03 : outcome === "FAILED_AVOIDED" ? 0.04 : 0.01;
  const threatDelta = special === "emergency" && quality > 0.5 ? -0.08 : special === "boss" && quality > 0.5 ? -0.06 : baseThreatDelta;
  const threatLevel = clamp(world.threatLevel + threatDelta, 0, 1);

  // Region progress advances within the domain track on real progress.
  const track = REGION_TRACKS[quest.domain] ?? REGION_TRACKS.craft;
  const currentIndex = Math.max(0, track.indexOf(world.currentRegion) === -1 ? 0 : track.indexOf(world.currentRegion));
  const progressGain = quality > 0.5 ? (special === "boss" ? 0.5 : 0.18) : 0;
  let regionProgress = world.regionProgress + progressGain;
  let currentRegion = world.currentRegion;
  let unlockedLocations = world.unlockedLocations;
  const events: string[] = [];

  if (regionProgress >= 1 && currentIndex < track.length - 1) {
    currentRegion = track[currentIndex + 1];
    regionProgress = 0;
    if (!unlockedLocations.includes(currentRegion)) {
      unlockedLocations = [...unlockedLocations, currentRegion];
    }
    const desc = REGION_DESCRIPTIONS[currentRegion];
    events.push(`A path opened. You have reached ${currentRegion}.${desc ? ` ${desc}` : ""}`);
  } else if (regionProgress >= 1) {
    regionProgress = 1; // end of this track; stay until a new domain track is entered
  }

  // Reliable chapter advancement on a monotonic counter (every 6 resolved quests).
  const questsResolved = world.questsResolved + 1;
  const chapter = Math.floor(questsResolved / 6) + 1;
  const season = Math.floor(questsResolved / 24) + 1;
  if (chapter > world.chapter) events.push(milestoneEvent("chapter", chapter));
  if (season > world.season) events.push(milestoneEvent("season", season));

  // Mysteries: repeated avoidance seeds a mystery; resolving it (boss/hidden success) clears one.
  let activeMysteries = world.activeMysteries;
  if (outcome === "FAILED_AVOIDED") {
    const mystery = DOMAIN_MYSTERIES[quest.domain] ?? `An unresolved pull lingers around ${quest.domain}.`;
    if (!activeMysteries.includes(mystery)) activeMysteries = [mystery, ...activeMysteries].slice(0, 5);
  } else if ((special === "hidden" || special === "boss") && quality > 0.5) {
    const domainMystery = DOMAIN_MYSTERIES[quest.domain];
    activeMysteries = activeMysteries.filter((item) => item !== domainMystery && !item.includes(quest.domain));
  }

  // Companions: durable strength in a domain earns a NAMED companion (once).
  let companions = world.companions;
  if (special === "boss" && quality >= 0.85) {
    const companion = DOMAIN_COMPANIONS[quest.domain] ?? `Ally of ${quest.domain}`;
    if (!companions.includes(companion)) companions = [...companions, companion].slice(0, 6);
  }

  // Narrative line, special-quest aware.
  const baseLine =
    special === "emergency"
      ? quality > 0.5
        ? `${quest.title}: the core held. Threat receded. Tomorrow is preserved.`
        : `${quest.title}: stabilization attempted. No identity penalty applied.`
      : special === "hidden"
        ? quality > 0.5
          ? `${quest.title}: the shadow was named. Its grip on ${quest.domain} weakened.`
          : `${quest.title}: the shadow remains unnamed. It waits without judgment.`
        : special === "boss"
          ? quality > 0.5
            ? `${quest.title}: the gate was breached. ${quest.domain} mastery rose sharply.`
            : `${quest.title}: the gate held this time. The attempt itself was data.`
          : quality > 0
            ? `${quest.title} resolved as ${outcome.toLowerCase().replace(/_/g, " ")}. ${quest.domain} resonance increased.`
            : `${quest.title} produced calibration data. No identity penalty applied.`;

  const log = [baseLine, ...events, ...world.log].slice(0, 16);

  return {
    ...world,
    chapter,
    season,
    currentRegion,
    regionProgress,
    threatLevel,
    unlockedLocations,
    companions,
    activeMysteries,
    questsResolved,
    log
  };
}

function stabilityGate(state: AppState, quality: number, confidence: number) {
  const failureStreak = state.questHistory.slice(0, 3).filter((quest) => {
    const type = quest.outcome?.type;
    return type === "FAILED_AVOIDED" || type === "FAILED_BLOCKED";
  }).length;
  const instability = state.host.stress * 0.35 + state.host.recoveryNeed * 0.25 + failureStreak * 0.12;
  const learningRate = clamp(confidence * (1 - instability), 0.15, 1);
  if (quality <= 0) {
    return {
      learningRate,
      message: "Logged as one data point, not a verdict on you. I adjusted things only a little."
    };
  }
  return {
    learningRate,
    message: learningRate < 0.45
      ? "Quest done. Keeping rewards modest since you're a bit stretched right now."
      : "Quest done. Got it — your progress is updated."
  };
}

function adaptProfile(state: AppState, type: OutcomeType, quality: number, confidence: number, learningRate: number) {
  const quest = state.activeQuest;
  if (!quest) return { profile: state.profile, assumptions: state.assumptions, log: [] };

  const delta = (quality - 0.55) * 0.16 * learningRate;
  const profile: HostProfile = {
    ...state.profile,
    difficultyCalibration: {
      ...state.profile.difficultyCalibration,
      activation: clamp(state.profile.difficultyCalibration.activation + delta, 0.15, 0.95),
      energy: clamp(state.profile.difficultyCalibration.energy + (quest.mode === "push" ? delta : delta * 0.45), 0.15, 0.95),
      focus: clamp(state.profile.difficultyCalibration.focus + (quest.mode === "clarity" ? delta : delta * 0.45), 0.15, 0.95),
      emotional: clamp(state.profile.difficultyCalibration.emotional + (quest.domain === "mind" || quest.domain === "recovery" ? delta : delta * 0.35), 0.15, 0.95),
      social: clamp(state.profile.difficultyCalibration.social + (quest.domain === "social" ? delta : 0), 0.15, 0.95),
      physical: clamp(state.profile.difficultyCalibration.physical + (quest.domain === "body" ? delta : 0), 0.15, 0.95),
      time: clamp(state.profile.difficultyCalibration.time + (quest.timeLimitMinutes > state.host.timeAvailableMinutes ? -0.04 : delta * 0.4), 0.15, 0.95)
    },
    knownPatterns: updatePatternList(
      state.profile.knownPatterns,
      quality >= 0.55 ? `${quest.questType} quests can work for ${quest.domain}` : ""
    ),
    resistancePatterns: updatePatternList(
      state.profile.resistancePatterns,
      type === "FAILED_BLOCKED" || type === "FAILED_AVOIDED" || type === "COMPLETED_WITH_COST"
        ? `${quest.domain} quests have been hard lately`
        : ""
    )
  };

  const assumptions = upsertAssumptions(state.assumptions, [
    {
      label: "Quest type that's working",
      value: quality >= 0.55 ? `${quest.questType} / ${quest.mode}` : `Less of ${quest.questType} / ${quest.mode}`,
      confidence: clamp(confidence * learningRate, 0.1, 0.86),
      source: [`outcome:${type}`, `quest:${quest.id}`]
    },
    {
      label: "How quests are landing",
      value: type === "FAILED_BLOCKED" ? "blocked" : type === "FAILED_AVOIDED" ? "being avoided" : type === "COMPLETED_WITH_COST" ? "got done, but it cost you" : "manageable",
      confidence: type === "COMPLETED_FULL" || type === "EXCEEDED_OBJECTIVE" ? 0.35 : clamp(confidence * 0.8, 0.1, 0.8),
      source: [`outcome:${type}`]
    }
  ]);

  const log = [
    quality >= 0.55
      ? `Leaning a bit more toward ${quest.questType} quests for ${quest.domain}.`
      : `Easing off a little after ${type.toLowerCase().replace(/_/g, " ")}.`
  ];
  return { profile, assumptions, log };
}

function upsertAssumptions(
  assumptions: Assumption[],
  updates: Array<{ label: string; value: string; confidence: number; source: string[] }>
) {
  const now = Date.now();
  let next = [...assumptions];
  updates.forEach((update) => {
    const index = next.findIndex((assumption) => assumption.label === update.label && assumption.status === "active");
    if (index < 0) {
      next = [
        {
          id: makeId("assumption"),
          label: update.label,
          value: update.value,
          confidence: update.confidence,
          source: update.source,
          protected: false,
          status: "active",
          evidenceCount: 1,
          lastUpdated: now
        },
        ...next
      ];
      return;
    }

    const existing = next[index];
    if (existing.protected && update.confidence < existing.confidence) return;
    next[index] = {
      ...existing,
      value: update.value,
      confidence: clamp(existing.confidence * 0.65 + update.confidence * 0.35, 0, 1),
      source: Array.from(new Set([...update.source, ...existing.source])).slice(0, 6),
      evidenceCount: existing.evidenceCount + 1,
      lastUpdated: now
    };
  });
  return next.filter((assumption) => assumption.status === "active").slice(0, 12);
}

function chooseMode(host: HostState): QuestMode {
  if (host.recoveryNeed > 0.68 || host.energy < 0.28) return "recover";
  if (host.focus < 0.35) return "clarity";
  if (host.challengeReadiness > 0.78 && host.stress < 0.55) return "push";
  return host.desiredMode;
}

function chooseDomain(goalDomain: QuestDomain, mode: QuestMode): QuestDomain {
  if (mode === "recover") return "recovery";
  if (mode === "maintenance") return goalDomain === "body" ? "body" : "order";
  if (mode === "clarity") return "planning";
  return goalDomain;
}

function chooseTimeLimit(host: HostState, mode: QuestMode, bias = 1) {
  const cap = Math.max(5, host.timeAvailableMinutes);
  const base =
    mode === "recover" ? 10 :
    mode === "clarity" ? 15 :
    mode === "push" ? 35 :
    mode === "boss" ? 45 :
    mode === "silent" ? 8 :
    22;
  return Math.max(5, Math.min(cap, Math.round(base * bias)));
}

function scoreQuestFit(host: HostState, mode: QuestMode, domain: QuestDomain) {
  const energyFit = mode === "recover" ? 1 - host.energy * 0.4 : host.energy;
  const focusFit = mode === "clarity" ? 1 - host.focus * 0.3 : host.focus;
  const stressFit = 1 - host.stress;
  const recoveryFit = domain === "recovery" ? host.recoveryNeed : 1 - host.recoveryNeed * 0.4;
  return clamp(energyFit * 0.25 + focusFit * 0.25 + stressFit * 0.2 + recoveryFit * 0.15 + host.challengeReadiness * 0.15, 0, 1);
}

function recentGoalPenalty(state: AppState, goalId: string) {
  const recent = state.questHistory.slice(0, 3).filter((quest) => quest.goalId === goalId).length;
  return recent * 0.08;
}

function noveltyFor(state: AppState, mode: QuestMode, domain: QuestDomain, questType: QuestType, index: number) {
  const recent = state.questHistory.slice(0, 6);
  const repeats = recent.filter((quest) => quest.mode === mode || quest.domain === domain || quest.questType === questType).length;
  return clamp(0.88 - repeats * 0.11 + index * 0.015, 0.25, 1);
}

function bandedTimeBias(timeBias: number, difficultyBand: DifficultyBand) {
  if (difficultyBand === "easier") return timeBias * 0.72;
  if (difficultyBand === "harder") return timeBias * 1.22;
  return timeBias;
}

/**
 * Modest level scaling for quest XP. Level 1 is a no-op (factor 1.0); each level
 * adds 8%, capped at +1.6 (≈2.6× by level 21). Linear growth only partly offsets
 * the geometric xpToNext curve, so higher levels still take real work but each
 * quest visibly scales with how far you've come.
 */
function levelRewardFactor(level: number) {
  return 1 + Math.min(1.6, Math.max(0, level - 1) * 0.08);
}

/**
 * Upward difficulty-band nudge as level rises. Level 1 is a no-op; each level
 * adds 0.012, capped at 0.12 (about one band notch). Lowering both band
 * thresholds makes harder quests appear sooner — but only as OPTIONS; the
 * difficultyFit score still gates them against the user's real calibration.
 */
function levelBandBias(level: number) {
  return Math.min(0.12, Math.max(0, level - 1) * 0.012);
}

function difficultyFor(mode: QuestMode, questType: QuestType, difficultyBand: DifficultyBand = "standard") {
  const base =
    mode === "boss" || questType === "boss" ? 0.9 :
    mode === "push" ? 0.75 :
    mode === "rebellion" ? 0.62 :
    mode === "recover" || questType === "stabilizer" ? 0.28 :
    mode === "silent" || questType === "micro" ? 0.22 :
    0.5;
  const shift = difficultyBand === "easier" ? -0.16 : difficultyBand === "harder" ? 0.16 : 0;
  return clamp(base + shift, 0.08, 0.95);
}

function typeFor(mode: QuestMode, domain: QuestDomain): QuestType {
  if (mode === "boss") return "boss";
  if (mode === "recover") return "stabilizer";
  if (mode === "clarity") return "clarifier";
  if (mode === "maintenance") return "maintenance";
  if (mode === "rebellion") return "rebellion";
  if (domain === "exploration") return "exploration";
  if (mode === "silent") return "micro";
  if (domain === "craft" || domain === "creation") return "builder";
  if (domain === "learning") return "practice";
  if (domain === "social") return "social";
  return "micro";
}

function rankFor(score: number, mode: QuestMode, questType: QuestType): QuestRank {
  if (questType === "boss" && score > 0.82) return "S";
  if (mode === "boss") return "A";
  if (score >= 0.86) return "A";
  if (score >= 0.76) return "B";
  if (score >= 0.64) return "C";
  if (score >= 0.5) return "D";
  if (score >= 0.36) return "E";
  return "F";
}

function reasonForGoalChoice(
  domain: QuestDomain,
  mode: QuestMode,
  fit: number,
  bonuses: { recoveryBonus: number; focusBonus: number; craftBonus: number; recentPenalty: number }
) {
  const domainLabel = domains.find((item) => item.id === domain)?.label ?? domain;
  const parts = [`${domainLabel} fits a ${mode} approach right now, about ${Math.round(fit * 100)}%.`];
  if (bonuses.recoveryBonus > 0) parts.push("You seem low on energy, so a steadying goal makes sense.");
  if (bonuses.focusBonus > 0) parts.push("Focus is low, so clearing things up first helps.");
  if (bonuses.craftBonus > 0) parts.push("Energy and focus are high enough to build something.");
  if (bonuses.recentPenalty > 0) parts.push("Goals you just worked on were bumped down a little.");
  return parts.join(" ");
}

function titleFor(mode: QuestMode, domain: QuestDomain, questType: QuestType) {
  const domainLabel = domains.find((item) => item.id === domain)?.label ?? "Quest";
  const prefix =
    questType === "rebellion" ? "Break the Avoidance" :
    mode === "silent" ? "Quiet Win" :
    mode === "recover" ? "Recharge" :
    mode === "clarity" ? "Get Clear" :
    mode === "push" ? "Push Yourself" :
    mode === "boss" ? "Big Challenge" :
    mode === "maintenance" ? "Quick Upkeep" :
    "Focused Task";
  return `${prefix}: ${domainLabel}`;
}

function objectiveFor(goal: string, mode: QuestMode, domain: QuestDomain, questType: QuestType, difficultyBand: DifficultyBand, minutes: number) {
  const difficultyPrefix =
    difficultyBand === "easier" ? "Easy version: " :
    difficultyBand === "harder" ? "Go further: " :
    "";
  if (domain === "order") return `${difficultyPrefix}Pick one messy or unresolved place, spend ${minutes} minutes reducing it, and leave a clear before/after result.`;
  if (questType === "rebellion") return `Name one avoided action around "${goal}", then attempt the smallest opposite action for two minutes.`;
  if (mode === "silent") return `${difficultyPrefix}Make a private ${minutes}-minute move on "${goal}" with no announcement and no performance pressure.`;
  if (mode === "recover") return `${difficultyPrefix}Spend ${minutes} minutes protecting recovery while keeping connection to: ${goal}`;
  if (mode === "clarity") return `${difficultyPrefix}Define the next visible action for: ${goal}. End with one concrete artifact.`;
  if (domain === "creation" || domain === "craft") return `${difficultyPrefix}Build, revise, or publish one small artifact tied to: ${goal}.`;
  if (domain === "learning") return `${difficultyPrefix}Study one narrow piece of ${goal} and write a short recall note.`;
  if (domain === "exploration") return `${difficultyPrefix}Visit one safe reachable place or outdoor route tied to ${goal}, then record one observation.`;
  return `${difficultyPrefix}Advance ${goal} with one action small enough to finish in ${minutes} minutes.`;
}

/**
 * Produces a short, honest "why this matters" clause — the emotional/moral
 * framing the genre calls for, adapted for a wellbeing app: stakes are framed
 * as what the quest BUILDS or PROTECTS, never as loss, threat, or failure cost.
 * Draws on the user's real resistance/known patterns when present, so the
 * framing is grounded in their actual history rather than generic drama.
 */
function stakesFor(mode: QuestMode, domain: QuestDomain, questType: QuestType, state: AppState): string {
  const resistance = state.profile.resistancePatterns[0];
  const known = state.profile.knownPatterns[0];

  if (mode === "recover" || domain === "recovery") {
    return "This keeps you going. Resting now is what lets you keep at the bigger goal later.";
  }
  if (domain === "order") {
    return "Tidying one visible spot clears your head right away and makes the next step easier to start.";
  }
  if (questType === "rebellion" || mode === "courage" || domain === "courage") {
    return resistance
      ? `This is where real growth happens: facing "${resistance}" in a small, safe way.`
      : "This is where real growth happens: meeting what you've been avoiding in a small, safe way.";
  }
  if (mode === "clarity" || domain === "planning") {
    return "Getting clear now saves you the effort you'd spend every time you try later.";
  }
  if (domain === "craft" || domain === "creation") {
    return "Having something real, even if it's rough, opens up what you can do next.";
  }
  if (domain === "learning") {
    return "Understanding one piece makes the next piece easier to pick up.";
  }
  if (domain === "exploration") {
    return "Getting out to one new place gives you fresh energy and a clearer sense of where you are.";
  }
  if (domain === "social") {
    return "Small honest moments are what keep a relationship strong.";
  }
  if (domain === "mind") {
    return "Feeling steady isn't a detour from your goal — it's how you reach it.";
  }
  return known
    ? `Small steps add up — and "${known}" is a strength you can lean on here.`
    : "Small steps, repeated, are what the bigger goal is actually made of.";
}

/**
 * Compositional generation layer. Rather than one fixed instruction per
 * domain/type, the execute step is accented with a method × constraint pair
 * drawn from interchangeable fragments, rotated deterministically by quest
 * count + domain. This expands the effective quest space combinatorially
 * (methods × constraints × the existing templates) without the unverifiable
 * risk of LLM authoring. Deterministic: same state -> same quest (testable).
 *
 * This is the verifiable form of "true generation." The LLM-authored open-ended
 * version remains backend-gated and is layered on top via refineArcWithLlm /
 * generateQuestWithLlm when the backend is live.
 */
const METHOD_FRAGMENTS: Record<string, string[]> = {
  craft: ["Build the smallest working piece first.", "Make a rough version you can throw away.", "Start from the part you understand least."],
  creation: ["Sketch before refining.", "Make one concrete artifact, however small.", "Begin with the messiest draft."],
  mind: ["Name the state before acting on it.", "Do one regulating action, fully.", "Separate the feeling from the next step."],
  learning: ["Study one narrow piece, then recall it from memory.", "Explain it aloud in your own words.", "Find the one example that makes it click."],
  planning: ["List three options, pick the smallest startable one.", "Write the obstacle, then the next visible action.", "Map only the next step, not the whole path."],
  order: ["Clear one item completely before the next.", "Sort by what unblocks the most.", "Focus on the one task that frees up the most."],
  courage: ["Do the smallest opposite of avoidance.", "Name the avoided thing in blunt words first.", "Take the first small, safe step toward it."],
  social: ["Draft one honest, bounded message.", "Prepare the first sentence, then send.", "Make one low-pressure move."],
  exploration: ["Visit one reachable place with a clear exit plan.", "Take a safe alternate route and note what changed.", "Step outside and observe one real detail before returning."],
  body: ["Move at a sustainable pace, not a punishing one.", "Do the corrective movement slowly and fully.", "Stop while it still feels repeatable."],
  recovery: ["Do one recovery action without multitasking.", "Remove one source of stimulation.", "Rest in a way that protects the next move."]
};

const CONSTRAINT_FRAGMENTS = [
  "Stop the moment real evidence exists.",
  "Keep it small enough to start immediately.",
  "Do not optimize; only make it real.",
  "Protect the next action over finishing perfectly.",
  "If it stalls for two minutes, shrink it."
];

type ActivityTemplate = {
  intent: string;
  prepare: string;
  execute: string;
  verify: string;
  close: string;
  output: string;
  prepareOutput?: string;
  verifyOutput?: string;
  closeOutput?: string;
  successCriteria: string[];
  fallback: string;
  antiAvoidanceRule: string;
};

function composeAccent(domain: QuestDomain, rotation: number): { method: string; constraint: string } {
  const methods = METHOD_FRAGMENTS[domain] ?? METHOD_FRAGMENTS.craft;
  const method = methods[rotation % methods.length];
  const constraint = CONSTRAINT_FRAGMENTS[(rotation + domain.length) % CONSTRAINT_FRAGMENTS.length];
  return { method, constraint };
}

function activityPlanFor(
  goal: string,
  mode: QuestMode,
  domain: QuestDomain,
  questType: QuestType,
  difficultyBand: DifficultyBand,
  minutes: number,
  state: AppState
): Quest["activityPlan"] {
  const safeMinutes = Math.max(5, minutes);
  const prep = Math.max(1, Math.round(safeMinutes * 0.15));
  const execute = Math.max(2, Math.round(safeMinutes * 0.55));
  const verify = Math.max(1, Math.round(safeMinutes * 0.2));
  const close = Math.max(1, safeMinutes - prep - execute - verify);
  const activity = activityTemplate(goal, mode, domain, questType, difficultyBand, state);

  // Compositional accent: rotate a method + constraint by history count + domain
  // so repeated quests in a domain don't read identically. Deterministic.
  const rotation = state.questHistory.length + domain.length;
  const accent = composeAccent(domain, rotation);
  const shouldAccent = !["order", "exploration"].includes(domain) && questType !== "rebellion";

  // Per-domain archetype: the concrete, actionable spine of the quest. The
  // domain's level (derived from its stat points) gates which archetypes are
  // unlocked; lower-level ones stay available forever as repeatable habits.
  const preferredTier: EffortTier =
    difficultyBand === "easier" ? "small" : difficultyBand === "harder" ? "large" : "medium";
  const domainPoints = state.progression.stats[domainToStat[domain]] ?? 0;
  const domainLevel = domainLevelFromPoints(domainPoints);
  const archetype = selectArchetype(domain, rotation, preferredTier, domainLevel);
  const archetypeAction = renderAction(archetype);
  const accentedExecute = shouldAccent ? `${archetypeAction} ${accent.constraint}` : archetypeAction;

  // Functional skill: Decomposer splits Execute into two smaller, clearer steps.
  const decomposer = state.progression.unlockedSkills.includes("order_decomposer");
  const executeSteps = decomposer
    ? [
        {
          id: "execute_a",
          label: "Do it · Part 1",
          minutes: Math.max(1, Math.round(execute / 2)),
          instruction: `First half: ${archetypeAction}`,
          output: "Partial output from the first half."
        },
        {
          id: "execute_b",
          label: "Do it · Part 2",
          minutes: Math.max(1, execute - Math.round(execute / 2)),
          instruction: shouldAccent ? `Second half: continue until done. ${accent.constraint}` : `Second half: finish the remaining visible work and stop when ${activity.output.toLowerCase()}`,
          output: activity.output
        }
      ]
    : [
        {
          id: "execute",
          label: "Do it",
          minutes: execute,
          instruction: accentedExecute,
          output: activity.output
        }
      ];

  return {
    intent: activity.intent,
    stakes: `${stakesFor(mode, domain, questType, state)} ${archetype.builds}`,
    steps: [
      {
        id: "prepare",
        label: "Prepare",
        minutes: prep,
        instruction: activity.prepare,
        output: activity.prepareOutput ?? "One written starting point."
      },
      ...executeSteps,
      {
        id: "verify",
        label: "Verify",
        minutes: verify,
        instruction: activity.verify,
        output: activity.verifyOutput ?? "A clear yes/partial/no completion check."
      },
      {
        id: "close",
        label: "Close",
        minutes: close,
        instruction: activity.close,
        output: activity.closeOutput ?? "One next-action note."
      }
    ],
    successCriteria: activity.successCriteria,
    fallback: activity.fallback,
    antiAvoidanceRule: activity.antiAvoidanceRule
  };
}

function activityTemplate(goal: string, mode: QuestMode, domain: QuestDomain, questType: QuestType, difficultyBand: DifficultyBand, state: AppState): ActivityTemplate {
  const lower = goal.toLowerCase();
  const mindOrMood = domain === "mind" || lower.includes("mind") || lower.includes("mood") || lower.includes("calm") || lower.includes("stable");
  const frictionHint = state.profile.resistancePatterns[0] ? ` What's been hard: ${state.profile.resistancePatterns[0]}.` : "";
  const bandInstruction =
    difficultyBand === "easier"
      ? "Use the smallest real version; stop as soon as evidence exists."
      : difficultyBand === "harder"
        ? "Add one deliberate stretch: make the output clearer, testable, or shareable."
        : "Keep the action proportional to today."

  if (domain === "order") {
    return {
      intent: `Make one specific area easier to use for "${goal}".`,
      prepare: `Pick exactly one target: desk corner, bed, bag, dishes, floor pile, app inbox, downloads folder, or unread message list. Take a quick before look. ${bandInstruction}`,
      execute: "Work only on that target. Throw away trash, put obvious items back, move uncertain items into one temporary pile, and stop when the area is visibly easier to use.",
      verify: "Look at the same target again. Decide: clear, better, or still blocked.",
      close: "Write the next exact cleanup action if anything remains.",
      output: "One chosen area is visibly clearer or one blocker is named.",
      prepareOutput: "One selected target, written in plain words.",
      verifyOutput: "A clear/better/blocked status for that target.",
      closeOutput: "One exact next cleanup action or 'done'.",
      successCriteria: ["Only one target was chosen.", "Something visible changed.", "Remaining mess has a named next action."],
      fallback: "If the target feels too large, remove only five items or clear only one phone screen/folder.",
      antiAvoidanceRule: "Do not reorganize the whole system. Make one place easier to use."
    };
  }

  if (questType === "rebellion") {
    return {
      intent: `Make one avoided action around "${goal}" smaller by attempting it for two minutes.${frictionHint}`,
      prepare: `Write the avoided action as a simple sentence: "I am avoiding ___." ${bandInstruction}`,
      execute: "Do the smallest opposite action for two minutes: open the file, write one bad sentence, send one draft, move one item, or ask one question.",
      verify: "Mark what happened: started, partially done, blocked, or avoided.",
      close: "Write the next smallest opposite action.",
      output: "One two-minute opposite action attempted.",
      prepareOutput: "One sentence naming the avoided action.",
      verifyOutput: "Started, partial, blocked, or avoided.",
      closeOutput: "One next two-minute action.",
      successCriteria: ["Avoided action was named.", "One opposite action was attempted.", "The next tiny action is clear."],
      fallback: "If action still feels too large, only open the place where the action would happen and stop.",
      antiAvoidanceRule: "Do not solve the whole task. Only make avoidance smaller."
    };
  }

  if (mode === "recover" || domain === "recovery") {
    return {
      intent: `Lower stress enough to continue later without forcing "${goal}".`,
      prepare: `Choose exactly one recovery action: drink water, take 10 slow breaths, stretch neck/shoulders, sit quietly, or clean one tiny surface. Put the phone face down if possible. ${bandInstruction}`,
      execute: "Do only the chosen recovery action. Do not combine it with scrolling, planning, or another task.",
      verify: "Rate energy, stress, and mood from 1-5 after the action.",
      close: "Write one next safe step you could do later, even if you do not do it now.",
      output: "One recovery action completed plus one later step.",
      prepareOutput: "One selected recovery action.",
      verifyOutput: "Energy/stress/mood ratings from 1-5.",
      closeOutput: "One later safe step.",
      successCriteria: ["One recovery action was chosen.", "It was done without multitasking.", "A later step was written."],
      fallback: "If that is too much, only drink water and write: 'Next safe step: ___'.",
      antiAvoidanceRule: "Stop after the recovery action. Do not turn rest into another productivity task."
    };
  }

  if (mode === "clarity" || domain === "planning") {
    return {
      intent: `Turn "${goal}" into one action you can actually start today.`,
      prepare: `Write three lines: Goal, blocker, and today done means. Example blocker: "I do not know the first step." ${bandInstruction}`,
      execute: "List three possible next actions. Pick the smallest one that can be started now, such as open file, write outline, send question, clear prerequisite, or make checklist.",
      verify: "Check the chosen action: can it start in under two minutes with your current tools?",
      close: "Write the chosen action as a command beginning with a verb.",
      output: "One startable next-action command.",
      prepareOutput: "Goal, blocker, and today-done lines.",
      verifyOutput: "Yes/no: startable in under two minutes.",
      closeOutput: "One verb-first action command.",
      successCriteria: ["Blocker is written.", "Three possible actions were listed.", "One action can start today."],
      fallback: "If still unclear, write three unknowns and choose the easiest one to answer.",
      antiAvoidanceRule: "Planning is finished when one startable action exists."
    };
  }

  if (domain === "exploration" || questType === "exploration") {
    return {
      intent: `Visit or inspect one real place safely, then record what you learned.`,
      prepare: `Choose one safe option: walk to a nearby shop, lobby, park path, street corner, building entrance, or public place. Check battery and weather. Choose a return point before leaving. ${bandInstruction}`,
      execute: "Go to the chosen place or route. Look for three things: one useful detail, one possible future activity, and one reason you might avoid this place.",
      verify: "Confirm: you stayed safe, returned or reached the planned stop point, and recorded one observation.",
      close: "Write whether this place is worth revisiting: yes, no, or maybe.",
      output: "One place/route note with useful detail, future activity, and avoidance reason.",
      prepareOutput: "Place/route name plus return point.",
      verifyOutput: "Safe return/stop confirmation plus one observation.",
      closeOutput: "Revisit: yes, no, or maybe.",
      successCriteria: ["One real place or route was chosen.", "The return point was respected.", "At least one observation was written."],
      fallback: "If going out is unsafe, stand at the door/window for two minutes and write one place to visit later.",
      antiAvoidanceRule: "Return while it still feels repeatable. Do not turn exploration into exhaustion."
    };
  }

  if (mindOrMood) {
    return {
      intent: `Make your current mental state easier to work with before touching "${goal}".`,
      prepare: `Write three short labels: feeling, likely trigger, and what would help. Examples: tired, worried, overstimulated, hungry, unclear. ${bandInstruction}`,
      execute: "Choose one calming action: 10 slow breaths, two-minute walk, short journal note, desk reset, wash face, or put one worry into a list.",
      verify: "Rate how steady you feel before and after from 1-5. Write one thing that changed, even if small.",
      close: "Choose one rule to protect the next hour, such as no extra tabs, no big decisions, or one task only.",
      output: "State labels, one calming action, and one next-hour rule.",
      prepareOutput: "Feeling, trigger, and helpful need.",
      verifyOutput: "Before/after steadiness rating plus one change.",
      closeOutput: "One next-hour protection rule.",
      successCriteria: ["State was named without judgment.", "One calming action was completed.", "One protection rule was chosen."],
      fallback: "If the mood is intense, breathe, hydrate, and postpone major decisions for 20 minutes.",
      antiAvoidanceRule: "Do not analyze your whole life. Name the state and do one calming action."
    };
  }

  if (domain === "craft" || domain === "creation") {
    return {
      intent: `Create one visible artifact for "${goal}".`,
      prepare: `Choose one artifact type: file, paragraph, sketch, button, screen, checklist, commit, diagram, or draft. Open the place where it will be made. ${bandInstruction}`,
      execute: "Make the smallest visible version. It can be ugly: one paragraph, one rough sketch, one working button, one TODO list, or one saved file.",
      verify: "Confirm the artifact can be shown, saved, copied, opened, or described in one sentence.",
      close: "Write the next build step as one verb-first sentence.",
      output: "One visible artifact plus one next build step.",
      prepareOutput: "Artifact type and where it will be made.",
      verifyOutput: "Shown/saved/copied/opened/described status.",
      closeOutput: "One next build step.",
      successCriteria: ["An artifact exists.", "It can be shown or described.", "The next build step is written."],
      fallback: "If making stalls, create only the filename, outline, sketch, or TODO list.",
      antiAvoidanceRule: "Do not polish. Make the rough version exist."
    };
  }

  if (domain === "learning") {
    return {
      intent: `Learn one small piece of "${goal}" well enough to recall it.`,
      prepare: `Choose exactly one learning item: one page, one video section, one term, one example, one error message, or one function. ${bandInstruction}`,
      execute: "Study only that item. Close or look away, then write three bullets from memory.",
      verify: "Explain the item in one plain sentence. If you cannot, write the exact confusing part.",
      close: "Write the next learning item to study later.",
      output: "Three memory bullets, one plain explanation or confusion, and one next item.",
      prepareOutput: "One selected learning item.",
      verifyOutput: "One plain explanation or one exact confusion.",
      closeOutput: "One next learning item.",
      successCriteria: ["Only one item was studied.", "Recall was attempted from memory.", "A next item is written."],
      fallback: "If it is too hard, write three vocabulary words and one question.",
      antiAvoidanceRule: "Do not keep rereading. Recall from memory, even badly."
    };
  }

  if (domain === "social") {
    return {
      intent: `Make one respectful social move connected to "${goal}".`,
      prepare: `Choose one person or group and one low-pressure action: draft message, send check-in, ask one question, thank someone, or prepare the first sentence. ${bandInstruction}`,
      execute: "Write or send one bounded message. Keep it specific and optional, for example: 'Can I ask one quick question?' or 'No need to reply fast.'",
      verify: "Confirm the action respected both sides: no pressure, no oversharing, no demand for instant response.",
      close: "Write whether a follow-up is needed: none, wait, or follow up later.",
      output: "One draft/sent message plus follow-up status.",
      prepareOutput: "Person/group plus chosen action.",
      verifyOutput: "Consent check: respectful yes/no.",
      closeOutput: "Follow-up status: none, wait, or later.",
      successCriteria: ["One person/group was chosen.", "One bounded message/action was made or drafted.", "Follow-up status is clear."],
      fallback: "If sending is too much, write the draft only and stop.",
      antiAvoidanceRule: "Do not judge the whole relationship from one message."
    };
  }

  if (domain === "body") {
    return {
      intent: `Do one safe physical action that supports "${goal}".`,
      prepare: `Choose one movement: five-minute walk, gentle stretch, 10 squats, 10 wall pushups, shoulder mobility, or tidy while standing. Check pain and space first. ${bandInstruction}`,
      execute: "Do the chosen movement at an easy pace. Stop if pain, dizziness, or unsafe conditions appear.",
      verify: "Rate effort from 1-5 and write whether the movement felt safe.",
      close: "Write the next repeatable movement or rest choice.",
      output: "One completed movement set plus effort/safety rating.",
      prepareOutput: "One selected movement and safety check.",
      verifyOutput: "Effort 1-5 plus safe/not safe.",
      closeOutput: "Next movement or rest choice.",
      successCriteria: ["One movement was chosen.", "It was done safely.", "Effort and safety were recorded."],
      fallback: "If movement is not safe, do only a posture reset or breathing for one minute.",
      antiAvoidanceRule: "Do not chase intensity. The win is a safe repeatable movement."
    };
  }

  return {
    intent: `Complete one small visible action for "${goal}".`,
    prepare: `Choose one concrete action you can see or count: open, write, clean, move, send, save, read, walk, stretch, or decide. ${bandInstruction}`,
    execute: "Do the chosen action until one visible result exists or one specific blocker appears.",
    verify: "Mark the result: done, partial, or blocked. If blocked, name the blocker in one sentence.",
    close: "Write the next exact action based on what happened.",
    output: "One visible result or one named blocker plus next action.",
    prepareOutput: "One chosen concrete action.",
    verifyOutput: "Done/partial/blocked plus blocker if any.",
    closeOutput: "One next exact action.",
    successCriteria: ["One concrete action was chosen.", "A visible result or blocker exists.", "The next action is written."],
    fallback: "If blocked, shrink to a two-minute action such as open, list, ask, move, or save.",
    antiAvoidanceRule: "If it stays vague, rewrite it as a physical verb plus object."
  };
}

function proofFor(mode: "trust" | "proof", domain: QuestDomain, questType: QuestType, skills: string[] = []) {
  const isBuild = questType === "builder" || domain === "craft" || domain === "creation";
  // Functional skill: Integrator nudges demonstrated builders toward artifact
  // evidence even in trust mode (their work tends to produce something concrete).
  if (isBuild && skills.includes("craft_integrator")) return "Attach or describe the artifact produced.";
  if (mode === "trust") return "A short completion note is enough.";
  if (isBuild) return "Attach or describe the artifact produced.";
  if (domain === "exploration" || questType === "exploration") return "Submit a place note, route note, timer note, or photo-free observation.";
  return "Submit a reflection, timer note, or observable completion detail.";
}

function riskFor(mode: QuestMode, domain: QuestDomain, state: AppState): 0 | 1 | 2 | 3 | 4 {
  if (domain === "social" && !state.profile.preferences.socialQuests) return 3;
  if (domain === "body" && !state.profile.preferences.physicalQuests) return 3;
  if (domain === "exploration") return state.profile.preferences.outdoorQuests ? 2 : 3;
  if (mode === "boss") return 3;
  if (mode === "recover" || mode === "silent") return 0;
  return 1;
}

function messageFor(mode: QuestMode, score: number) {
  const fit = Math.round(score * 100);
  if (mode === "recover") return `Resting counts as progress. This one fits about ${fit}%.`;
  if (mode === "clarity") return `Things feel unclear right now. This one fits about ${fit}%. Make the next step smaller.`;
  if (mode === "rebellion") return `This is something you've been putting off. This one fits about ${fit}%. Start with one small piece.`;
  return `This quest fits about ${fit}%. It's your call.`;
}

function updatePatternList(patterns: string[], value: string) {
  if (!value) return patterns;
  return [value, ...patterns.filter((pattern) => pattern !== value)].slice(0, 8);
}

export function ensureStateShape(state: AppState): AppState {
  const defaults = makeInitialState();
  return {
    ...defaults,
    ...state,
    profile: {
      ...defaults.profile,
      ...state.profile,
      preferences: { ...defaults.profile.preferences, ...state.profile?.preferences },
      difficultyCalibration: {
        ...defaults.profile.difficultyCalibration,
        ...state.profile?.difficultyCalibration
      },
      goals: (state.profile?.goals ?? []).map((goal) => ({
        ...goal,
        arc: goal.arc ?? buildGoalArc(goal.text, goal.domain)
      })),
      knownPatterns: state.profile?.knownPatterns ?? [],
      resistancePatterns: state.profile?.resistancePatterns ?? []
    },
    llm: state.llm ?? defaults.llm,
    host: { ...defaults.host, ...state.host },
    progression: {
      ...defaults.progression,
      ...state.progression,
      // Heal corrupted saves: non-finite xp/level (e.g. from a past NaN bug)
      // must never survive a load, or they poison every later calculation.
      xp: Number.isFinite(state.progression?.xp) ? state.progression.xp : defaults.progression.xp,
      level: Number.isFinite(state.progression?.level) ? state.progression.level : defaults.progression.level,
      xpToNext: Number.isFinite(state.progression?.xpToNext) ? state.progression.xpToNext : defaults.progression.xpToNext,
      unlockedSkills:
        state.progression?.unlockedSkills ??
        (state.progression?.stats ? computeUnlockedSkills(state.progression.stats) : []),
      streak: { ...defaults.progression.streak, ...state.progression?.streak }
    },
    world: {
      ...emptyWorld(),
      ...state.world,
      unlockedLocations: state.world?.unlockedLocations ?? emptyWorld().unlockedLocations,
      companions: state.world?.companions ?? [],
      activeMysteries: state.world?.activeMysteries ?? [],
      log: state.world?.log ?? emptyWorld().log,
      discoveredCells: state.world?.discoveredCells ?? []
    },
    selectedGoalId: state.selectedGoalId ?? state.profile?.goals?.find((goal) => goal.status === "active")?.id ?? null,
    goalSelectionReason: state.goalSelectionReason ?? "Selection restored from saved state.",
    questOffers: (state.questOffers ?? []).map(normalizeQuest).filter(Boolean) as Quest[],
    // Migration: old saves carried a single activeQuest; fold it into the
    // array. The alias is then re-derived so the invariant always holds.
    ...(() => {
      const source = state.activeQuests ?? (state.activeQuest ? [state.activeQuest] : []);
      const activeQuests = source.map(normalizeQuest).filter(Boolean) as Quest[];
      return { activeQuests, activeQuest: activeQuests[0] ?? null };
    })(),
    questHistory: (state.questHistory ?? []).map((quest) => normalizeQuest(quest) as Quest & { outcome?: typeof quest.outcome }),
    assumptions: state.assumptions ?? [],
    adaptationLog: state.adaptationLog ?? [],
    systemSignals: {
      ...emptySystemSignals(),
      ...state.systemSignals,
      domainAvoidance: { ...state.systemSignals?.domainAvoidance },
      resolvedHiddenDomains: state.systemSignals?.resolvedHiddenDomains ?? []
    },
    // A loaded prior state means a returning user; default them to onboarded so
    // the first-run flow only ever shows to genuinely new users.
    hasOnboarded: state.hasOnboarded ?? true,
    bodyHistory: Array.isArray(state.bodyHistory) ? state.bodyHistory : []
  };
}

function normalizeQuest(quest: Quest): Quest | null {
  if (!quest) return null;
  const mode = quest.mode ?? "steady";
  const domain = quest.domain ?? "mind";
  const questType = quest.questType ?? typeFor(mode, domain);
  const scoreBreakdown = quest.scoreBreakdown ?? {
    goalRelevance: quest.acceptanceScore ?? 0.5,
    stateFit: quest.acceptanceScore ?? 0.5,
    timeFit: 0.7,
    difficultyFit: 0.7,
    safety: 0.8,
    growthValue: 0.6,
    novelty: 0.6,
    evidenceClarity: 0.7
  };
  return {
    ...quest,
    candidateId: quest.candidateId ?? quest.id,
    questType,
    difficultyBand: quest.difficultyBand ?? "standard",
    rank: quest.rank ?? "D",
    scoreBreakdown,
    generatorSource: quest.generatorSource ?? "rules"
  };
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function makeId(prefix: string) {
  return `${prefix}_${Math.random().toString(36).slice(2, 10)}`;
}

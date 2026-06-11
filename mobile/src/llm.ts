import { ensureStateShape, generateQuestOffers, makeInitialState } from "./engine";
import { AppState, Goal, LlmConfig, OutcomeSuggestion, OutcomeType, Quest } from "./types";
import { currentAccessToken } from "./sync";
import { SUPABASE_URL, isSyncConfigured } from "./config";

const DEFAULT_LLM_CONFIG: LlmConfig = makeInitialState().llm;
const LLM_TIMEOUT_MS = 15_000;

/**
 * Whether LLM refinement should run for this call. There is intentionally no
 * user-facing toggle: enrichment is automatic and silent. The only viable
 * route on a phone is the Supabase backend, which self-activates whenever a
 * backend is configured (the deployed function holds the model keys, so the
 * phone needs no setup). The local Ollama route stays dev-only and is reached
 * by explicitly setting llm.enabled in state — never through the UI. In every
 * case a missing/unreachable route falls back to the deterministic engine
 * with no visible error.
 */
function llmActive(state: AppState): boolean {
  if (state.llm.provider === "ollama") return state.llm.enabled;
  return isSyncConfigured();
}

/** Backend LLM lives at the Supabase Edge Function; derived from config so a
 *  stale persisted endpoint can never misroute the call. */
function backendFunctionUrl(): string {
  return `${SUPABASE_URL.replace(/\/$/, "")}/functions/v1/forge-llm`;
}

/**
 * Calls the Supabase Edge Function (forge-llm) for any backend LLM task. The
 * function URL is the configured llm.endpoint; auth is the Supabase access
 * token. A `task` discriminator selects quest/arc/outcome server-side. Throws on
 * any non-OK response so the caller's silent deterministic fallback engages.
 */
async function callBackendFunction(state: AppState, task: "quest" | "arc" | "outcome", payload: unknown): Promise<unknown> {
  const endpoint = backendFunctionUrl();
  const token = await currentAccessToken();
  if (!token) throw new Error("Not signed in: backend LLM requires authentication.");
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), LLM_TIMEOUT_MS);
  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`
      },
      body: JSON.stringify({ task, payload }),
      signal: controller.signal
    });
    if (!response.ok) throw new Error(`Backend HTTP ${response.status}`);
    return response.json();
  } finally {
    clearTimeout(timer);
  }
}

/** Shared Ollama chat call: sends a system prompt + user payload, returns parsed JSON. */
async function callOllama(state: AppState, systemPrompt: string, userPayload: unknown): Promise<unknown> {
  const endpoint = state.llm.endpoint.replace(/\/$/, "");
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), LLM_TIMEOUT_MS);
  try {
    const response = await fetch(`${endpoint}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: state.llm.model,
        stream: false,
        format: "json",
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: JSON.stringify(userPayload) }
        ]
      }),
      signal: controller.signal
    });
    if (!response.ok) throw new Error(`Ollama HTTP ${response.status}`);
    const payload = await response.json();
    return JSON.parse(extractJson(String(payload?.message?.content ?? "")));
  } finally {
    clearTimeout(timer);
  }
}

type LlmQuestPatch = Pick<Quest, "title" | "objective" | "activityPlan" | "proofRequired" | "systemMessage">;

const VALID_OUTCOMES: OutcomeType[] = [
  "COMPLETED_FULL",
  "EXCEEDED_OBJECTIVE",
  "COMPLETED_CREATIVELY",
  "COMPLETED_WITH_COST",
  "COMPLETED_LOW_QUALITY",
  "REPLACED_WITH_EQUIVALENT",
  "COMPLETED_PARTIAL",
  "FAILED_ATTEMPTED",
  "FAILED_BLOCKED",
  "FAILED_AVOIDED",
  "SKIPPED_CONSCIOUSLY"
];

export function withDefaultLlmConfig(state: AppState): AppState {
  return ensureStateShape({
    ...state,
    llm: state.llm ?? DEFAULT_LLM_CONFIG,
    selectedGoalId: state.selectedGoalId ?? state.profile.goals.find((goal) => goal.status === "active")?.id ?? null,
    goalSelectionReason: state.goalSelectionReason ?? "Selection restored from saved state.",
    profile: {
      ...state.profile,
      goals: state.profile.goals.map((goal) => ({
        ...goal,
        arc: goal.arc ?? {
          title: "Recovery Arc",
          currentPhase: "Rebuild Structure",
          bottleneck: "legacy saved goal",
          weeklyFocus: "Regenerate this goal for a richer arc.",
          milestones: [
            { id: "define", label: "Define the Win", objective: "Clarify what progress looks like.", status: "active" },
            { id: "first", label: "First Movement", objective: "Complete one visible action.", status: "locked" },
            { id: "repeat", label: "Repeatable Loop", objective: "Make the action easier to repeat.", status: "locked" }
          ],
          nextActions: ["Generate a quest from this goal.", "Submit evidence after one attempt."]
        }
      }))
    }
  });
}

export async function generateQuestWithLlm(state: AppState): Promise<AppState> {
  const normalized = withDefaultLlmConfig(state);
  const baseState = generateQuestOffers(normalized);
  const baseQuest = baseState.questOffers[0];

  if (!baseQuest || !llmActive(normalized)) return baseState;

  try {
    const patch = await requestQuestPatch(baseState, baseQuest);
    // Refinement is invisible: swap richer text into the top offer but keep the
    // engine's system message so the user never sees LLM/route internals.
    const refinedQuest: Quest = {
      ...baseQuest,
      ...patch,
      generatorSource: "llm_refined"
    };
    return {
      ...baseState,
      questOffers: [refinedQuest, ...baseState.questOffers.slice(1)]
    };
  } catch {
    // Any failure silently keeps the deterministic offers — no error surfaced.
    return baseState;
  }
}

async function requestQuestPatch(state: AppState, quest: Quest): Promise<LlmQuestPatch> {
  if (state.llm.provider === "backend") return requestBackendQuestPatch(state, quest);
  return requestOllamaQuestPatch(state, quest);
}

async function requestBackendQuestPatch(state: AppState, quest: Quest): Promise<LlmQuestPatch> {
  const payload = await callBackendFunction(state, "quest", buildQuestRequest(state, quest));
  return validatePatch(payload, quest);
}

async function requestOllamaQuestPatch(state: AppState, quest: Quest): Promise<LlmQuestPatch> {
  const raw = await callOllama(
    state,
    "You are Forge System OS, a consent-based quest planner. Return strict JSON only. " +
      "Never shame the user. Never give medical, legal, or financial directives. " +
      "Make the quest executable through specific, small activity steps.",
    buildQuestRequest(state, quest)
  );
  return validatePatch(raw, quest);
}

function buildQuestRequest(state: AppState, quest: Quest) {
  const selectedGoal =
    state.profile.goals.find((item) => item.id === state.selectedGoalId && item.status === "active") ??
    state.profile.goals.find((item) => item.id === quest.goalId) ??
    state.profile.goals.find((item) => item.status === "active");

  return {
    task:
      "Refine this base quest. Keep the same risk tier, time limit, domain, mode, and reward. " +
      "Return JSON with title, objective, activityPlan, proofRequired, and systemMessage only.",
    model: state.llm.model,
    goal: selectedGoal?.text,
    goalSelectionReason: state.goalSelectionReason,
    host: state.host,
    baseQuest: {
      title: quest.title,
      objective: quest.objective,
      domain: quest.domain,
      mode: quest.mode,
      timeLimitMinutes: quest.timeLimitMinutes,
      riskTier: quest.riskTier,
      activityPlan: quest.activityPlan,
      proofRequired: quest.proofRequired
    },
    requiredShape: {
      title: "short quest title",
      objective: "one concrete objective",
      activityPlan: {
        intent: "why this activity serves the goal",
        stakes: "one honest sentence on why this matters — what it builds or protects, never framed as loss or punishment",
        steps: [
          { id: "prepare", label: "Prepare", minutes: 3, instruction: "specific action", output: "observable output" },
          { id: "execute", label: "Execute", minutes: 10, instruction: "specific action", output: "observable output" },
          { id: "verify", label: "Verify", minutes: 4, instruction: "specific check", output: "completion signal" },
          { id: "close", label: "Close", minutes: 3, instruction: "specific closure action", output: "next step" }
        ],
        successCriteria: ["criterion one", "criterion two", "criterion three"],
        fallback: "smaller safe alternative if blocked",
        antiAvoidanceRule: "one short rule"
      },
      proofRequired: "what evidence to submit",
      systemMessage: "brief System-style message"
    }
  };
}

function extractJson(text: string) {
  const trimmed = text.trim();
  if (trimmed.startsWith("{")) return trimmed;
  const start = trimmed.indexOf("{");
  const end = trimmed.lastIndexOf("}");
  if (start < 0 || end < start) throw new Error("LLM did not return JSON");
  return trimmed.slice(start, end + 1);
}

function validatePatch(value: unknown, fallback: Quest): LlmQuestPatch {
  if (typeof value !== "object" || value === null) throw new Error("LLM did not return an object");
  const patch = value as Record<string, unknown>;
  const activityPlan = patch.activityPlan as Record<string, unknown> | undefined;
  const steps = Array.isArray(activityPlan?.steps) ? (activityPlan!.steps as unknown[]) : undefined;
  if (!patch.title || !patch.objective || !activityPlan || !steps || steps.length < 3) {
    throw new Error("LLM JSON shape invalid");
  }

  const successCriteria = Array.isArray(activityPlan.successCriteria)
    ? (activityPlan.successCriteria as unknown[])
    : fallback.activityPlan.successCriteria;

  return {
    title: String(patch.title).slice(0, 80),
    objective: String(patch.objective).slice(0, 240),
    proofRequired: String(patch.proofRequired || fallback.proofRequired).slice(0, 180),
    systemMessage: String(patch.systemMessage || fallback.systemMessage).slice(0, 180),
    activityPlan: {
      intent: String(activityPlan.intent || fallback.activityPlan.intent).slice(0, 220),
      stakes: String(activityPlan.stakes || fallback.activityPlan.stakes).slice(0, 220),
      steps: steps.slice(0, 5).map((step, index) => {
        const s = (typeof step === "object" && step !== null ? step : {}) as Record<string, unknown>;
        return {
          id: String(s.id || `step_${index + 1}`).slice(0, 24),
          label: String(s.label || `Step ${index + 1}`).slice(0, 32),
          minutes: Math.max(1, Math.min(90, Number(s.minutes) || fallback.activityPlan.steps[index]?.minutes || 5)),
          instruction: String(s.instruction || fallback.activityPlan.steps[index]?.instruction || "Take one concrete action.").slice(0, 240),
          output: String(s.output || fallback.activityPlan.steps[index]?.output || "Observable output.").slice(0, 160)
        };
      }),
      successCriteria: successCriteria.slice(0, 5).map((item) => String(item).slice(0, 120)),
      fallback: String(activityPlan.fallback || fallback.activityPlan.fallback).slice(0, 220),
      antiAvoidanceRule: String(activityPlan.antiAvoidanceRule || fallback.activityPlan.antiAvoidanceRule).slice(0, 180)
    }
  };
}

// ============================================================================
// Goal -> Arc refinement (inform/refine).
// The engine always produces a valid arc; if the LLM is enabled and reachable,
// it proposes richer text. The engine still owns milestone IDs and structure;
// the model only fills titles/objectives/phase/bottleneck/focus/next actions.
// Any failure returns the engine arc unchanged.
// ============================================================================

type ArcPatch = Goal["arc"];

export async function refineArcWithLlm(state: AppState, goal: Goal): Promise<Goal["arc"]> {
  const normalized = withDefaultLlmConfig(state);
  if (!llmActive(normalized)) return goal.arc;
  try {
    const patch = await requestArcPatch(normalized, goal);
    return mergeArc(goal.arc, patch);
  } catch {
    return goal.arc;
  }
}

async function requestArcPatch(state: AppState, goal: Goal): Promise<Partial<ArcPatch>> {
  const body = {
    task:
      "Refine this goal arc. Keep the same number of milestones and the same milestone ids and order. " +
      "Make titles, objectives, phase, bottleneck, weeklyFocus, and nextActions specific to the goal. " +
      "Return JSON only with: title, currentPhase, bottleneck, weeklyFocus, milestones[{id,label,objective}], nextActions[].",
    model: state.llm.model,
    goal: goal.text,
    domain: goal.domain,
    baseArc: goal.arc
  };
  if (state.llm.provider === "backend") {
    return (await callBackendFunction(state, "arc", body)) as Partial<ArcPatch>;
  }
  return (await callOllama(
    state,
    "You are Forge System OS, a consent-based planner. Return strict JSON only. Keep milestone ids and count identical to the input.",
    body
  )) as Partial<ArcPatch>;
}

/** Merge LLM text onto the engine arc. Engine owns ids/structure/status; the
 *  model can only replace text on the milestones that already exist. */
function mergeArc(base: ArcPatch, patch: Partial<ArcPatch>): ArcPatch {
  const patchById = new Map((patch.milestones ?? []).map((m) => [String(m.id), m]));
  return {
    title: clampText(patch.title, base.title, 60),
    currentPhase: clampText(patch.currentPhase, base.currentPhase, 60),
    bottleneck: clampText(patch.bottleneck, base.bottleneck, 60),
    weeklyFocus: clampText(patch.weeklyFocus, base.weeklyFocus, 160),
    milestones: base.milestones.map((m) => {
      const incoming = patchById.get(m.id);
      return {
        id: m.id,
        status: m.status, // engine-owned, never overwritten
        label: clampText(incoming?.label, m.label, 48),
        objective: clampText(incoming?.objective, m.objective, 160)
      };
    }),
    nextActions: Array.isArray(patch.nextActions) && patch.nextActions.length
      ? patch.nextActions.slice(0, 5).map((a) => clampText(a, "", 160)).filter(Boolean)
      : base.nextActions
  };
}

// ============================================================================
// Outcome interpretation (suggest-with-confirmation).
// Reads the user's free-text note and proposes what actually happened plus a
// named blocker. NEVER applied directly: the store surfaces the suggestion and
// only a user confirmation dispatches a real SUBMIT_OUTCOME with the new enum.
// ============================================================================

export async function interpretOutcome(
  state: AppState,
  selectedOutcome: OutcomeType,
  note: string
): Promise<OutcomeSuggestion | null> {
  const normalized = withDefaultLlmConfig(state);
  if (!llmActive(normalized) || !note.trim()) return null;
  try {
    const raw = await requestOutcomeReading(normalized, selectedOutcome, note);
    return validateOutcomeSuggestion(raw, selectedOutcome);
  } catch {
    return null;
  }
}

async function requestOutcomeReading(state: AppState, selectedOutcome: OutcomeType, note: string): Promise<unknown> {
  const body = {
    task:
      "Read the user's note about a completed quest. Decide which outcome category best fits what actually happened, " +
      "and name the main blocker or pattern in a few words. Do not shame. Return JSON only with: " +
      "suggestedOutcome (one of the allowed values), blocker (short phrase), reasoning (one sentence).",
    allowedOutcomes: VALID_OUTCOMES,
    selectedOutcome,
    note: note.trim().slice(0, 600)
  };
  if (state.llm.provider === "backend") {
    return await callBackendFunction(state, "outcome", body);
  }
  return callOllama(
    state,
    "You are Forge System OS. Return strict JSON only. Never shame the user. Classify honestly and gently.",
    body
  );
}

function validateOutcomeSuggestion(value: unknown, selectedOutcome: OutcomeType): OutcomeSuggestion | null {
  const v = value as Partial<OutcomeSuggestion>;
  const suggested = VALID_OUTCOMES.includes(v.suggestedOutcome as OutcomeType)
    ? (v.suggestedOutcome as OutcomeType)
    : selectedOutcome;
  const blocker = clampText(v.blocker, "", 80);
  const reasoning = clampText(v.reasoning, "", 200);
  // If the model added no usable signal at all, treat as no suggestion.
  if (!blocker && !reasoning && suggested === selectedOutcome) return null;
  return {
    suggestedOutcome: suggested,
    differsFromSelected: suggested !== selectedOutcome,
    blocker: blocker || "no clear blocker named",
    reasoning: reasoning || "Interpretation based on your note."
  };
}

function clampText(value: unknown, fallback: string, max: number): string {
  const s = typeof value === "string" ? value.trim() : "";
  return (s || fallback).slice(0, max);
}

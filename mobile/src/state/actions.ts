import { AppState, BodyComposition, EvidenceType, HostState, LlmConfig, OutcomeType, QuestDomain, Goal, TimerEvidence } from "../types";

/** Narrowed patch surface — prevents screens from accidentally overwriting
 *  quest state, progression, or world via PATCH. */
export type PatchableState = Partial<Pick<AppState, "lastSystemMessage" | "selectedGoalId" | "pendingCeremony" | "goalPinned">> & {
  profile?: AppState["profile"];
  llm?: LlmConfig;
};

/**
 * Every way application state can change, expressed as an intent rather than a
 * precomputed next-state. The reducer owns the engine calls; the store owns any
 * async side effects (notifications, LLM, future backend sync). Screens only
 * dispatch these.
 *
 * SYNC actions are pure: reducer handles them with no I/O.
 * ASYNC actions (generate, save-scan, accept, hydrate) are handled by the store
 * dispatcher, which performs side effects and then folds the result through the
 * reducer via an internal REPLACE action.
 */
export type Action =
  // --- pure / synchronous ---
  | { type: "ADD_GOAL"; text: string; domain: QuestDomain }
  | { type: "GENERATE_SYSTEM_GOAL" }
  | { type: "AUTO_SELECT_GOAL" }
  | { type: "REJECT_QUEST"; questId?: string }
  | { type: "SUBMIT_OUTCOME"; outcome: OutcomeType; evidence: EvidenceType; note: string; extraEvidence?: Array<{ kind: EvidenceType; note?: string; artifactUri?: string; timer?: TimerEvidence; cellId?: string }>; questId?: string }
  | { type: "UPDATE_ASSUMPTION"; id: string; action: "confirm" | "protect" | "reject" }
  | { type: "COMPLETE_ONBOARDING"; goal: { text: string; domain: QuestDomain } | null }
  | { type: "ADD_BODY_SCAN"; scan: BodyComposition }
  | { type: "DISCOVER_CELLS"; cells: string[] }
  | { type: "DELETE_GOAL"; goalId: string }
  | { type: "UPDATE_GOAL"; goalId: string; patch: Partial<Pick<Goal, "priority" | "status" | "text">> }
  | { type: "PATCH"; patch: PatchableState }
  | { type: "REPLACE"; state: AppState }
  | { type: "RESET" }
  // --- async (handled in store before reaching reducer) ---
  | { type: "SAVE_SCAN"; host: HostState }
  | { type: "GENERATE_QUEST"; goalId?: string }
  | { type: "ACCEPT_QUEST"; questId?: string; confirmedRisk?: boolean }
  | { type: "HYDRATE"; state: AppState };

export type SyncAction = Extract<
  Action,
  { type: "ADD_GOAL" | "GENERATE_SYSTEM_GOAL" | "AUTO_SELECT_GOAL" | "REJECT_QUEST" | "SUBMIT_OUTCOME" | "UPDATE_ASSUMPTION" | "COMPLETE_ONBOARDING" | "ADD_BODY_SCAN" | "DISCOVER_CELLS" | "DELETE_GOAL" | "UPDATE_GOAL" | "PATCH" | "REPLACE" | "RESET" }
>;

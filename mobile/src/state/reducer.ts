import {
  acceptQuest,
  addGoal,
  autoSelectGoal,
  ensureStateShape,
  generateSystemGoal,
  makeInitialState,
  rejectQuest,
  submitOutcome,
  updateAssumption,
  updateHost,
  deleteGoal,
  discoverCells,
  updateGoal
} from "../engine";
import { AppState } from "../types";
import { SyncAction } from "./actions";

/**
 * Pure state reducer. Every transition routes through an engine function, so
 * the engine remains the single source of truth and this layer stays trivially
 * testable: same input state + action always yields the same next state, with
 * no I/O. Async concerns (notifications, LLM, sync) live in the store, which
 * pre-resolves them and feeds results here via REPLACE/PATCH.
 */
export function reducer(state: AppState, action: SyncAction): AppState {
  switch (action.type) {
    case "ADD_GOAL":
      return addGoal(state, action.text, action.domain);

    case "GENERATE_SYSTEM_GOAL":
      return generateSystemGoal(state);

    case "AUTO_SELECT_GOAL":
      return autoSelectGoal(state);

    case "REJECT_QUEST":
      return rejectQuest(state, action.questId);

    case "DISCOVER_CELLS":
      return discoverCells(state, action.cells);
    case "DELETE_GOAL":
      return deleteGoal(state, action.goalId);
    case "UPDATE_GOAL":
      return updateGoal(state, action.goalId, action.patch);
    case "SUBMIT_OUTCOME":
      return submitOutcome(state, action.outcome, action.evidence, action.note, action.extraEvidence, action.questId);

    case "UPDATE_ASSUMPTION":
      return updateAssumption(state, action.id, action.action);

    case "COMPLETE_ONBOARDING": {
      const onboarded = { ...state, hasOnboarded: true };
      return action.goal ? addGoal(onboarded, action.goal.text, action.goal.domain) : onboarded;
    }

    case "ADD_BODY_SCAN": {
      // Newest first, capped. Stored as-is; quest logic reads only capability
      // signals via deriveBodySignals — the raw aesthetic metrics are kept for
      // display but never drive a weight/fat-target quest.
      const bodyHistory = [action.scan, ...state.bodyHistory].slice(0, 50);
      return ensureStateShape({ ...state, bodyHistory });
    }

    case "PATCH":
      // Shallow merge for screen-local state nudges (e.g. lastSystemMessage,
      // preference toggles). Kept narrow on purpose.
      return ensureStateShape({ ...state, ...action.patch });

    case "REPLACE":
      return ensureStateShape(action.state);

    case "RESET":
      return makeInitialState();

    default: {
      // Exhaustiveness guard: a new SyncAction without a case is a compile error.
      const _never: never = action;
      return _never;
    }
  }
}

/**
 * Re-exported for the store's async handlers, which need direct access to the
 * effectful engine calls (acceptQuest with confirmedRisk, updateHost) so they
 * can run side effects against the computed next-state before committing it.
 */
export const engineOps = { acceptQuest, updateHost, discoverCells };

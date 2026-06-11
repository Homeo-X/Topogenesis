import React, { createContext, useCallback, useContext, useEffect, useReducer, useRef, useState } from "react";
import { ensureStateShape, generateQuestOffers, makeInitialState } from "../engine";
import { generateQuestWithLlm, interpretOutcome, refineArcWithLlm, withDefaultLlmConfig } from "../llm";
import {
  configureNotificationCategories,
  enableDailySystemNotifications,
  schedulePostScanQuestPrompt,
  scheduleQuestWindowNotification
} from "../notifications";
import { clearAppState, loadAppState, saveAppState } from "../storage";
import {
  AuthResult,
  currentUser,
  loadRemoteState,
  saveRemoteState,
  signIn,
  signOut,
  signUp,
  syncAvailable
} from "../sync";
import { setReduceMotion } from "../motion";
import { playCue, tap } from "../feedback";
import { AppState, OutcomeSuggestion, OutcomeType } from "../types";
import { Action } from "./actions";
import { engineOps, reducer } from "./reducer";

interface StoreValue {
  state: AppState;
  /** Read the freshest committed state (safe right after an awaited dispatch). */
  getState: () => AppState;
  /** Async-capable dispatch. Sync actions update immediately; async actions
   *  (SAVE_SCAN, GENERATE_QUEST, ACCEPT_QUEST, ADD_GOAL) run side effects then commit. */
  dispatch: (action: Action) => void | Promise<void>;
  /** Query (not a mutation): asks the LLM to read an outcome note and propose a
   *  reading. Returns null if LLM is off/unreachable/uninformative. The caller
   *  decides whether to act on it — nothing changes until a SUBMIT_OUTCOME. */
  interpretOutcomeNote: (selected: OutcomeType, note: string) => Promise<OutcomeSuggestion | null>;
  loaded: boolean;
  /** Backend account. `configured` is false when no Supabase keys are set, in
   *  which case the app is fully local-only and the Account UI is hidden. */
  auth: {
    configured: boolean;
    email: string | null;
    signIn: (email: string, password: string) => Promise<AuthResult>;
    signUp: (email: string, password: string) => Promise<AuthResult>;
    signOut: () => Promise<void>;
  };
}

const StoreContext = createContext<StoreValue | null>(null);

export function StoreProvider({ children }: { children: React.ReactNode }) {
  const [state, rawDispatch] = useReducer(reducer, undefined, makeInitialState);
  const [loaded, setLoaded] = useState(false);
  // Stable ref so dispatch never needs to be recreated when state changes.
  const stateRef = useRef(state);
  useEffect(() => { stateRef.current = state; });
  // In-flight guard: prevents duplicate concurrent GENERATE_QUEST calls.
  const generatingRef = useRef(false);
  // (commit is defined below; hydration routes through commitRef so the ref is
  // fresh from the very first committed state.)
  const commitRef = useRef<(next: AppState) => void>(() => {});
  // Debounce timer for remote sync — avoids a network call on every PATCH.
  const remoteSyncTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Hydrate once on mount. Prefer remote state when signed in (survives
  // reinstall / syncs devices); fall back to local cache otherwise. Remote
  // failure silently falls through to local — sync is additive, never required.
  useEffect(() => {
    void configureNotificationCategories();
    (async () => {
      try {
        if (syncAvailable()) {
          const remote = await loadRemoteState();
          if (remote) {
            commitRef.current(remote);
            return;
          }
        }
        const saved = await loadAppState();
        if (saved) commitRef.current(ensureStateShape(withDefaultLlmConfig(saved)));
      } finally {
        setLoaded(true);
      }
    })();
  }, []);

  // Persist on every committed change: local cache always, remote debounced.
  useEffect(() => {
    setReduceMotion(state.profile.preferences.reduceMotion);
    if (loaded) {
      void saveAppState(state);
      if (remoteSyncTimer.current) clearTimeout(remoteSyncTimer.current);
      remoteSyncTimer.current = setTimeout(() => {
        void saveRemoteState(stateRef.current);
      }, 3000);
    }
  }, [loaded, state]);

  // The async-capable dispatcher: pure actions pass straight through; effectful
  // intents perform side effects against the computed next state then commit via
  // REPLACE. Uses stateRef so the callback is stable and never causes re-renders.
  // Commit helper: updates stateRef SYNCHRONOUSLY before React re-renders, so a
  // dispatch chained immediately after an awaited dispatch (e.g. ADD_GOAL then
  // GENERATE_QUEST from the activity browser) reads the fresh state instead of
  // a stale ref. Without this, chained intents silently act on old state.
  const commit = useCallback((next: AppState) => {
    stateRef.current = next;
    rawDispatch({ type: "REPLACE", state: next });
  }, []);
  commitRef.current = commit;

  const dispatch = useCallback(
    async (action: Action) => {
      const state = stateRef.current;
      switch (action.type) {
        case "SAVE_SCAN": {
          // Commit the check-in FIRST: it must never hinge on notification
          // scheduling (slow permission prompts, throwing native modules).
          const next = engineOps.updateHost(state, action.host);
          commit(next);
          if (next.profile.preferences.notificationConsent) {
            try {
              await enableDailySystemNotifications(next.host, next.profile.preferences.alarmStyleNotifications);
              const prompt = await schedulePostScanQuestPrompt(next);
              if (prompt) commit({ ...stateRef.current, lastSystemMessage: `${stateRef.current.lastSystemMessage} ${prompt}`.trim() });
            } catch {
              // Scheduling failed: the check-in is already safe; skip the note.
            }
          }
          return;
        }

        case "GENERATE_QUEST": {
          if (generatingRef.current) return;
          generatingRef.current = true;
          try {
            const base = action.goalId ? { ...state, selectedGoalId: action.goalId, goalPinned: true } : state;
            const next = await generateQuestWithLlm(base);
            const topOffer = next.questOffers[0];
            let systemMessage = next.lastSystemMessage;
            if (topOffer && next.profile.preferences.notificationConsent) {
              systemMessage = `${systemMessage} ${await scheduleQuestWindowNotification(next, topOffer)}`.trim();
            }
            commit({ ...next, lastSystemMessage: systemMessage });
          } finally {
            generatingRef.current = false;
          }
          return;
        }

        case "ACCEPT_QUEST": {
          const next = engineOps.acceptQuest(state, action.questId, action.confirmedRisk);
          // The newly accepted quest is the one matching the action id (or the
          // last appended). next.activeQuest is merely the FIRST active.
          const accepted = action.questId
            ? next.activeQuests.find((quest) => quest.id === action.questId) ?? null
            : next.activeQuests[next.activeQuests.length - 1] ?? null;
          const actuallyAccepted = accepted && !state.activeQuests.some((quest) => quest.id === accepted.id);
          if (actuallyAccepted) {
            void tap(next, "success");
            void playCue(next, "accept");
          }
          let systemMessage = next.lastSystemMessage;
          if (accepted && actuallyAccepted && next.profile.preferences.notificationConsent) {
            systemMessage = `${systemMessage} ${await scheduleQuestWindowNotification(next, accepted)}`.trim();
          }
          commit({ ...next, lastSystemMessage: systemMessage });
          return;
        }

        case "ADD_GOAL": {
          // Create the goal with its deterministic arc first (always valid),
          // then, if the LLM is enabled, refine that arc in place.
          const created = reducer(state, action);
          const newGoal = created.profile.goals[0];
          if (!newGoal) {
            commit(created);
            return;
          }
          const refinedArc = await refineArcWithLlm(created, newGoal);
          const goals = created.profile.goals.map((g) => (g.id === newGoal.id ? { ...g, arc: refinedArc } : g));
          commit({ ...created, profile: { ...created.profile, goals } });
          return;
        }

        case "RESET": {
          await clearAppState();
          commit(makeInitialState());
          return;
        }

        case "HYDRATE": {
          commit(ensureStateShape(action.state));
          return;
        }

        default: {
          // All remaining actions are pure SyncActions. Compute + commit so the
          // stateRef stays fresh for any dispatch chained in the same tick.
          commit(reducer(stateRef.current, action));
          return;
        }
      }
    },
    [] // stable — reads state via stateRef
  );

  const interpretOutcomeNote = useCallback(
    (selected: OutcomeType, note: string) => interpretOutcome(stateRef.current, selected, note),
    [] // stable — reads state via stateRef
  );

  // --- Backend account (email/password) -------------------------------------
  const [authEmail, setAuthEmail] = useState<string | null>(null);

  // Reflect any already-restored session on mount (persisted via AsyncStorage).
  useEffect(() => {
    void (async () => setAuthEmail((await currentUser())?.email ?? null))();
  }, []);

  const doSignIn = useCallback(async (email: string, password: string) => {
    const res = await signIn(email, password);
    if (res.ok) {
      setAuthEmail((await currentUser())?.email ?? null);
      // Pull this account's cloud progress; if none exists yet, seed it from local.
      const remote = await loadRemoteState();
      if (remote) commitRef.current(remote);
      else void saveRemoteState(stateRef.current);
    }
    return res;
  }, []);

  const doSignUp = useCallback(async (email: string, password: string) => {
    const res = await signUp(email, password);
    if (res.ok) {
      setAuthEmail((await currentUser())?.email ?? null);
      // If email confirmation is off, a session exists now → back up local state.
      void saveRemoteState(stateRef.current);
    }
    return res;
  }, []);

  const doSignOut = useCallback(async () => {
    await signOut();
    setAuthEmail(null);
  }, []);

  return (
    <StoreContext.Provider
      value={{
        state,
        getState: () => stateRef.current,
        dispatch,
        interpretOutcomeNote,
        loaded,
        auth: {
          configured: syncAvailable(),
          email: authEmail,
          signIn: doSignIn,
          signUp: doSignUp,
          signOut: doSignOut
        }
      }}
    >
      {children}
    </StoreContext.Provider>
  );
}

export function useStore(): StoreValue {
  const ctx = useContext(StoreContext);
  if (!ctx) throw new Error("useStore must be used within a StoreProvider.");
  return ctx;
}

/** Exposed so a non-LLM deterministic generate is reachable if ever needed. */
export const deterministicGenerate = generateQuestOffers;

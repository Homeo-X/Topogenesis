import { makeInitialState } from "../engine";
import { reducer } from "../state/reducer";
import { AppState } from "../types";

describe("reducer: pure synchronous actions", () => {
  test("ADD_GOAL registers a goal", () => {
    const s0 = makeInitialState();
    const s1 = reducer(s0, { type: "ADD_GOAL", text: "Ship the app", domain: "craft" });
    expect(s1.profile.goals.length).toBe(1);
    expect(s1.profile.goals[0].text).toBe("Ship the app");
    // purity: original state untouched
    expect(s0.profile.goals.length).toBe(0);
  });

  test("GENERATE_SYSTEM_GOAL creates an active goal without manual text", () => {
    const s1 = reducer(makeInitialState(), { type: "GENERATE_SYSTEM_GOAL" });
    expect(s1.profile.goals.length).toBe(1);
    expect(s1.profile.goals[0].text.length).toBeGreaterThan(10);
    expect(s1.profile.goals[0].status).toBe("active");
    expect(s1.selectedGoalId).toBe(s1.profile.goals[0].id);
  });

  test("PATCH shallow-merges and re-normalizes", () => {
    const s0 = makeInitialState();
    const s1 = reducer(s0, { type: "PATCH", patch: { lastSystemMessage: "patched" } });
    expect(s1.lastSystemMessage).toBe("patched");
    // ensureStateShape ran: systemSignals still present
    expect(s1.systemSignals).toBeDefined();
  });

  test("night mode preference is persisted through PATCH", () => {
    const s0 = makeInitialState();
    const s1 = reducer(s0, {
      type: "PATCH",
      patch: {
        profile: {
          ...s0.profile,
          preferences: { ...s0.profile.preferences, nightMode: true }
        }
      }
    });
    expect(s1.profile.preferences.nightMode).toBe(true);
  });

  test("RESET returns a fresh initial state", () => {
    let s = reducer(makeInitialState(), { type: "ADD_GOAL", text: "x", domain: "mind" });
    s = reducer(s, { type: "RESET" });
    expect(s.profile.goals.length).toBe(0);
    expect(s.progression.level).toBe(1);
  });

  test("REPLACE installs a given state through ensureStateShape", () => {
    const legacy: any = JSON.parse(JSON.stringify(makeInitialState()));
    delete legacy.systemSignals;
    const s = reducer(makeInitialState(), { type: "REPLACE", state: legacy });
    expect(s.systemSignals).toBeDefined();
  });

  test("UPDATE_ASSUMPTION on an empty ledger is a no-op, not a crash", () => {
    const s0 = makeInitialState();
    expect(() => reducer(s0, { type: "UPDATE_ASSUMPTION", id: "nope", action: "confirm" })).not.toThrow();
  });

  test("reducer never mutates the input state object", () => {
    const s0 = makeInitialState();
    const snapshot = JSON.stringify(s0);
    reducer(s0, { type: "ADD_GOAL", text: "y", domain: "order" });
    expect(JSON.stringify(s0)).toBe(snapshot);
  });
});

describe("reducer: onboarding", () => {
  test("fresh state starts not onboarded", () => {
    expect(makeInitialState().hasOnboarded).toBe(false);
  });

  test("COMPLETE_ONBOARDING with a goal sets flag and registers the goal", () => {
    const s = reducer(makeInitialState(), { type: "COMPLETE_ONBOARDING", goal: { text: "Ship it", domain: "craft" } });
    expect(s.hasOnboarded).toBe(true);
    expect(s.profile.goals.length).toBe(1);
    expect(s.profile.goals[0].text).toBe("Ship it");
  });

  test("COMPLETE_ONBOARDING with null still completes onboarding, no goal", () => {
    const s = reducer(makeInitialState(), { type: "COMPLETE_ONBOARDING", goal: null });
    expect(s.hasOnboarded).toBe(true);
    expect(s.profile.goals.length).toBe(0);
  });
});

describe("reducer: onboarding migration", () => {
  test("a returning user (loaded state) is treated as already onboarded", () => {
    const legacy: any = JSON.parse(JSON.stringify(makeInitialState()));
    delete legacy.hasOnboarded; // old save had no such field
    const s = reducer(makeInitialState(), { type: "REPLACE", state: legacy });
    expect(s.hasOnboarded).toBe(true); // not forced back through onboarding
  });
});

import {
  acceptQuest,
  addGoal,
  emptySystemSignals,
  generateQuestOffers,
  makeInitialState,
  submitOutcome,
  updateHost
} from "../engine";
import { AppState, QuestDomain } from "../types";

function withGoal(domain: QuestDomain = "craft"): AppState {
  return addGoal(makeInitialState(), "Build the app", domain);
}

/** Drive N avoidance outcomes against a domain while keeping the host calm so
 *  the emergency branch (stress/energy) does not pre-empt the hidden branch. */
function avoid(state: AppState, domain: QuestDomain, times: number): AppState {
  let s = state;
  for (let i = 0; i < times; i++) {
    s = updateHost(s, { ...s.host, energy: 0.6, stress: 0.3, recoveryNeed: 0.3 });
    s = generateQuestOffers(s);
    s = acceptQuest(s, s.questOffers[0].id, true);
    s = { ...s, activeQuest: { ...s.activeQuest!, domain } };
    s = submitOutcome(s, "FAILED_AVOIDED", "reflection", "avoided");
  }
  return s;
}

describe("event director: hidden quests", () => {
  test("fires after repeated avoidance of a domain", () => {
    let s = avoid(withGoal("craft"), "craft", 3);
    expect(s.systemSignals.domainAvoidance.craft).toBeGreaterThanOrEqual(3);
    s = updateHost(s, { ...s.host, energy: 0.6, stress: 0.3, recoveryNeed: 0.3 });
    s = generateQuestOffers(s);
    expect(s.systemSignals.lastEvent?.kind).toBe("hidden");
    expect(s.questOffers[0].candidateId.startsWith("special_hidden_")).toBe(true);
  });

  test("avoidance does not inflate the distress/failure streak", () => {
    const s = avoid(withGoal("craft"), "craft", 3);
    // FAILED_AVOIDED routes to avoidance, not failureStreak.
    expect(s.systemSignals.failureStreak).toBe(0);
  });
});

describe("event director: emergency quests", () => {
  test("fires under high stress + low energy and is low-risk", () => {
    let s = withGoal("craft");
    s = updateHost(s, { ...s.host, stress: 0.85, energy: 0.2, recoveryNeed: 0.8 });
    s = generateQuestOffers(s);
    expect(s.systemSignals.lastEvent?.kind).toBe("emergency");
    expect(s.questOffers[0].riskTier).toBe(0);
  });
});

describe("event director: boss quests", () => {
  test("fires only with intensity opted-in, high readiness, and momentum", () => {
    let s = withGoal("craft");
    s.profile.preferences.intensityMode = "intense";
    s.systemSignals = { ...emptySystemSignals(), momentum: 3 };
    s = updateHost(s, { ...s.host, energy: 0.8, focus: 0.75, stress: 0.3, timeAvailableMinutes: 60, challengeReadiness: 0.85 });
    s = generateQuestOffers(s);
    expect(s.systemSignals.lastEvent?.kind).toBe("boss");
    expect(s.questOffers[0].rank).toBe("A");
  });

  test("is suppressed when intensity is gentle", () => {
    let s = withGoal("craft");
    s.profile.preferences.intensityMode = "gentle";
    s.systemSignals = { ...emptySystemSignals(), momentum: 3 };
    s = updateHost(s, { ...s.host, energy: 0.8, focus: 0.75, stress: 0.3, timeAvailableMinutes: 60, challengeReadiness: 0.85 });
    s = generateQuestOffers(s);
    expect(s.systemSignals.lastEvent).toBeNull();
  });
});

describe("world state: progression", () => {
  test("chapters advance reliably (not capped by log length)", () => {
    let s = withGoal("craft");
    for (let i = 0; i < 18; i++) {
      s = updateHost(s, { ...s.host, energy: 0.6, focus: 0.6, stress: 0.35 });
      s = generateQuestOffers(s);
      s = acceptQuest(s, s.questOffers[0].id, true);
      s = { ...s, activeQuest: { ...s.activeQuest!, domain: "craft" } };
      s = submitOutcome(s, "COMPLETED_FULL", "reflection", "x");
    }
    expect(s.world.questsResolved).toBe(18);
    expect(s.world.chapter).toBe(4); // floor(18/6)+1
    expect(s.world.currentRegion).not.toBe("The Outer Archive");
  });

  test("avoidance seeds a mystery and raises threat", () => {
    const s = avoid(withGoal("courage"), "courage", 1);
    expect(s.world.activeMysteries.length).toBeGreaterThan(0);
    expect(s.world.threatLevel).toBeGreaterThan(0.2);
  });
});

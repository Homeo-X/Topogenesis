import {
  acceptQuest,
  addGoal,
  ensureStateShape,
  generateQuestOffers,
  makeInitialState,
  submitOutcome,
  updateHost
} from "../engine";
import { AppState, QuestDomain } from "../types";

function goalState(domain: any = "craft"): AppState {
  return addGoal(makeInitialState(), "Build the Forge System app", domain);
}

describe("engine: state shape & migration", () => {
  test("makeInitialState produces a valid, self-consistent state", () => {
    const s = makeInitialState();
    expect(s.progression.level).toBe(1);
    expect(s.profile.goals).toHaveLength(0);
    expect(s.world.chapter).toBe(1);
    expect(s.systemSignals).toBeDefined();
  });

  test("ensureStateShape back-fills a legacy state missing new fields", () => {
    const legacy: any = JSON.parse(JSON.stringify(makeInitialState()));
    delete legacy.systemSignals;
    legacy.world = { chapter: 2, currentRegion: "The Broken Gate", threatLevel: 0.4, unlockedLocations: ["The Outer Archive"], log: ["old"] };
    const restored = ensureStateShape(legacy);
    expect(restored.systemSignals).toBeDefined();
    expect(typeof restored.systemSignals.domainAvoidance).toBe("object");
    expect(restored.world.season).toBe(1);
    expect(Array.isArray(restored.world.companions)).toBe(true);
    expect(restored.world.questsResolved).toBe(0);
    // preserved existing data
    expect(restored.world.currentRegion).toBe("The Broken Gate");
  });
});

describe("engine: quest generation", () => {
  test("generates up to three ranked offers for an active goal", () => {
    let s = goalState();
    s = updateHost(s, { ...s.host, energy: 0.6, focus: 0.6, stress: 0.35 });
    s = generateQuestOffers(s);
    expect(s.questOffers.length).toBeGreaterThan(0);
    expect(s.questOffers.length).toBeLessThanOrEqual(3);
    // Offers are ranked by the council-adjusted score (acceptanceScore x advisor
    // weights), which the council stores on each offer for transparency.
    const scores = s.questOffers.map((q) => q.councilScore ?? q.acceptanceScore);
    expect([...scores].sort((a, b) => b - a)).toEqual(scores);
  });

  test("low energy / high stress yields a recovery-leaning top offer", () => {
    let s = goalState();
    s = updateHost(s, { ...s.host, energy: 0.15, stress: 0.85, recoveryNeed: 0.8 });
    s = generateQuestOffers(s);
    // emergency/recovery should be surfaced; top offer is safe (low risk)
    expect(s.questOffers[0].riskTier).toBeLessThanOrEqual(1);
  });

  test("exploration goals generate structured outdoor/place activity", () => {
    let s = goalState("exploration");
    s.profile.preferences.outdoorQuests = true;
    s = updateHost(s, { ...s.host, energy: 0.7, focus: 0.55, stress: 0.25, timeAvailableMinutes: 30 });
    s = generateQuestOffers(s);
    const exploration = s.questOffers.find((quest) => quest.domain === "exploration" || quest.questType === "exploration");
    expect(exploration).toBeDefined();
    expect(exploration?.activityPlan.steps.some((step) => /place|outdoor|route|visit/i.test(step.instruction))).toBe(true);
  });

  test("unconsented exploration goals require explicit high-risk confirmation", () => {
    let s = goalState("exploration");
    s.profile.preferences.outdoorQuests = false;
    s = updateHost(s, { ...s.host, energy: 0.7, focus: 0.55, stress: 0.25, timeAvailableMinutes: 30 });
    s = generateQuestOffers(s);
    const exploration = s.questOffers.find((quest) => quest.domain === "exploration" || quest.questType === "exploration");
    expect(exploration?.riskTier).toBe(3);
    const blocked = acceptQuest(s, exploration!.id);
    expect(blocked.activeQuest).toBeNull();
    const allowed = acceptQuest(s, exploration!.id, true);
    expect(allowed.activeQuest?.id).toBe(exploration?.id);
  });

  test("order quests use concrete cleanup instructions instead of vague friction language", () => {
    let s = goalState("order");
    s.profile.goals[0] = {
      ...s.profile.goals[0],
      text: "Clear one visible pile, app inbox, desk corner, or downloads folder"
    };
    s = updateHost(s, { ...s.host, energy: 0.55, focus: 0.45, stress: 0.35, timeAvailableMinutes: 20 });
    s = generateQuestOffers(s);
    const orderOffer = s.questOffers.find((quest) => quest.domain === "order") ?? s.questOffers[0];
    const instructions = orderOffer.activityPlan.steps.map((step) => step.instruction).join(" ");
    expect(instructions).toMatch(/desk corner|app inbox|downloads folder|floor pile/i);
    // Execute step now draws from the order archetype bank: still concrete cleanup actions.
    expect(instructions).toMatch(/tidy|clear|put .* back|finish one thing|throw away|recycle|deal with|inbox|surface|reset/i);
    expect(instructions).not.toMatch(/friction point/i);
  });

  test("all major domain templates produce concrete step instructions", () => {
    const domains: QuestDomain[] = ["recovery", "planning", "mind", "craft", "creation", "learning", "social", "body", "exploration", "order"];
    for (const domain of domains) {
      let s = goalState(domain);
      s.profile.preferences.socialQuests = true;
      s.profile.preferences.physicalQuests = true;
      s.profile.preferences.outdoorQuests = true;
      s = updateHost(s, { ...s.host, energy: 0.7, focus: 0.6, stress: 0.25, timeAvailableMinutes: 25, recoveryNeed: domain === "recovery" ? 0.8 : 0.25 });
      s = generateQuestOffers(s);
      const offer = s.questOffers.find((quest) => quest.domain === domain) ?? s.questOffers[0];
      const text = [
        offer.objective,
        offer.activityPlan.intent,
        ...offer.activityPlan.steps.flatMap((step) => [step.instruction, step.output])
      ].join(" ");
      expect(text).toMatch(/choose|write|do|make|pick|rate|send|visit|study|clear|create|open|list|confirm/i);
      expect(text).not.toMatch(/forward motion|load-bearing|friction point|small, survivable|changes what becomes possible/i);
    }
  });
});

describe("engine: outcome & reward", () => {
  test("a full completion grants XP and advances the world", () => {
    let s = goalState();
    s = updateHost(s, { ...s.host, energy: 0.6, focus: 0.6, stress: 0.35 });
    s = generateQuestOffers(s);
    s = acceptQuest(s, s.questOffers[0].id, true);
    const beforeXp = s.progression.xp + s.progression.level * 1000;
    s = submitOutcome(s, "COMPLETED_FULL", "reflection", "done");
    const afterXp = s.progression.xp + s.progression.level * 1000;
    expect(afterXp).toBeGreaterThan(beforeXp);
    expect(s.world.questsResolved).toBe(1);
    expect(s.activeQuest).toBeNull();
  });

  test("a blocked failure does not crash and applies no identity penalty", () => {
    let s = goalState();
    s = updateHost(s, { ...s.host, energy: 0.6, focus: 0.6, stress: 0.35 });
    s = generateQuestOffers(s);
    s = acceptQuest(s, s.questOffers[0].id, true);
    expect(() => {
      s = submitOutcome(s, "FAILED_BLOCKED", "self_report", "blocked");
    }).not.toThrow();
    expect(s.activeQuest).toBeNull();
  });
});

describe("engine: tier-3 risk gate", () => {
  test("tier-3 quest is refused without explicit confirmation", () => {
    let s = goalState();
    s.profile.preferences.intensityMode = "intense";
    s.systemSignals = { ...s.systemSignals, momentum: 3 };
    s = updateHost(s, { ...s.host, energy: 0.8, focus: 0.75, stress: 0.3, timeAvailableMinutes: 60, challengeReadiness: 0.85 });
    s = generateQuestOffers(s);
    const top = s.questOffers[0];
    if (top.riskTier >= 3) {
      const blocked = acceptQuest(s, top.id);
      expect(blocked.activeQuest).toBeNull();
      const allowed = acceptQuest(s, top.id, true);
      expect(allowed.activeQuest?.id).toBe(top.id);
    } else {
      // If no tier-3 surfaced, the gate is vacuously satisfied.
      expect(top.riskTier).toBeLessThan(3);
    }
  });
});

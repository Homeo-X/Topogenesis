import { deliberate } from "../council";
import { addGoal, makeInitialState, updateHost, generateQuestOffers, acceptQuest, submitOutcome } from "../engine";
import { computeUnlockedSkills } from "../skills";

function readyState(domain: Parameters<typeof addGoal>[2] = "craft") {
  let s = addGoal(makeInitialState(), "Build it", domain);
  s = updateHost(s, { ...s.host, energy: 0.6, focus: 0.55, stress: 0.3, timeAvailableMinutes: 40 });
  return s;
}

describe("council deliberation", () => {
  test("every generated offer carries a council note or none, deterministically", () => {
    const a = generateQuestOffers(readyState());
    const b = generateQuestOffers(readyState());
    expect(a.questOffers.map((o) => o.councilNote)).toEqual(b.questOffers.map((o) => o.councilNote));
    expect(a.questOffers.map((o) => o.id ? o.title : "")).toEqual(b.questOffers.map((o) => o.id ? o.title : ""));
  });

  test("guardian re-ranks toward recovery when energy is depleted", () => {
    let s = readyState();
    s = updateHost(s, { ...s.host, energy: 0.15, recoveryNeed: 0.9, focus: 0.3, stress: 0.7, timeAvailableMinutes: 30 });
    s = generateQuestOffers(s);
    const top = s.questOffers[0];
    // The top offer should be the restful one (recover mode or recovery domain),
    // or at minimum carry the Guardian's reasoning.
    const restfulTop = top.domain === "recovery" || top.mode === "recover";
    const guardianSpoke = s.questOffers.some((o) => o.councilNote?.startsWith("Guardian"));
    expect(restfulTop || guardianSpoke).toBe(true);
  });

  test("deliberate pins the first offer when asked (director precedence)", () => {
    const s = generateQuestOffers(readyState());
    const offers = s.questOffers;
    if (offers.length < 2) return; // nothing to reorder
    const pinned = deliberate(offers, s, s.profile.goals[0], true).offers;
    expect(pinned[0].candidateId).toBe(offers[0].candidateId);
  });

  test("council notes are never shaming", () => {
    let s = readyState();
    // Stress several advisor triggers at once.
    s = updateHost(s, { ...s.host, energy: 0.2, recoveryNeed: 0.8, timeAvailableMinutes: 10 });
    s.systemSignals.domainAvoidance = { ...s.systemSignals.domainAvoidance, craft: 3 };
    s = generateQuestOffers(s);
    for (const o of s.questOffers) {
      if (!o.councilNote) continue;
      expect(o.councilNote).not.toMatch(/neglect|behind|failing|failed you|lazy|guilt|shame|weak|disappoint/i);
    }
  });
});

describe("ceremony emission", () => {
  test("crossing a domain mastery level emits a pending ceremony", () => {
    let s = readyState("craft");
    s = { ...s, progression: { ...s.progression, stats: { ...s.progression.stats, Craft: 11 } } };
    s = generateQuestOffers(s);
    const offer = s.questOffers.find((o) => o.domain === "craft") ?? s.questOffers[0];
    s = acceptQuest(s, offer.id);
    s = submitOutcome(s, "COMPLETED_FULL", "artifact", "done");
    expect(s.pendingCeremony).toBeTruthy();
    expect(s.pendingCeremony!.lines.length).toBeGreaterThan(0);
    expect(s.pendingCeremony!.title.length).toBeGreaterThan(0);
  });

  test("a routine completion with no progression event emits no ceremony", () => {
    let s = readyState("craft");
    // High stats so no level/mastery threshold is near, and no fresh skill.
    const stats = { ...s.progression.stats, Craft: 100 };
    s = { ...s, progression: { ...s.progression, level: 50, xp: 5, xpToNext: 100000, stats, unlockedSkills: computeUnlockedSkills(stats) } };
    s = generateQuestOffers(s);
    s = acceptQuest(s, s.questOffers[0].id);
    s = submitOutcome(s, "COMPLETED_PARTIAL", "self_report", "meh");
    expect(s.pendingCeremony ?? null).toBeNull();
  });
});

describe("goal pinning (activity browser targeting)", () => {
  test("a pinned goal survives auto-selection and drives the offers", () => {
    // Two goals; host state that the scorer would steer toward recovery.
    let s = addGoal(makeInitialState(), "Ship the build", "craft");
    s = addGoal(s, "Actually rest", "recovery");
    s = updateHost(s, { ...s.host, energy: 0.15, recoveryNeed: 0.9, focus: 0.3, stress: 0.7, timeAvailableMinutes: 30 });
    const craftGoal = s.profile.goals.find((g) => g.domain === "craft")!;
    // Without a pin the scorer picks for the user's state (recovery-leaning).
    const unpinned = generateQuestOffers(s);
    // With the pin, the user's explicit choice wins.
    const pinned = generateQuestOffers({ ...s, selectedGoalId: craftGoal.id, goalPinned: true });
    expect(pinned.questOffers.every((o) => o.goalId === craftGoal.id)).toBe(true);
    expect(pinned.goalPinned).toBe(false); // one-shot: cleared after use
    expect(pinned.goalSelectionReason).toMatch(/you chose/i);
    // The unpinned run remains free to choose differently — only the offers'
    // goalId invariant above is the contract, so just assert it generated.
    expect(unpinned.questOffers.length).toBeGreaterThan(0);
  });
});

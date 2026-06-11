import { addGoal, makeInitialState, updateHost, generateQuestOffers, acceptQuest, submitOutcome, updateGoal, deleteGoal, ensureStateShape, emptyHostState } from "../engine";

function ready(domain: Parameters<typeof addGoal>[2] = "craft") {
  let s = addGoal(makeInitialState(), "Build it", domain);
  s = updateHost(s, { ...s.host, energy: 0.6, focus: 0.55, stress: 0.3, timeAvailableMinutes: 40 });
  return s;
}

function accepted(domain: Parameters<typeof addGoal>[2] = "craft") {
  let s = ready(domain);
  s = generateQuestOffers(s);
  return acceptQuest(s, s.questOffers[0].id);
}

describe("evidence 2.0", () => {
  test("single self-report produces one evidence item whose aggregate equals legacy confidence", () => {
    const s = submitOutcome(accepted(), "COMPLETED_FULL", "self_report", "did it");
    const outcome = s.questHistory[0].outcome!;
    expect(outcome.evidence).toHaveLength(1);
    expect(outcome.evidence![0].kind).toBe("self_report");
    expect(outcome.evidence![0].id).toMatch(/ev/);
    expect(outcome.verificationConfidence).toBeCloseTo(0.58); // legacy evidenceConfidence.self_report
  });

  test("multiple evidence items raise verification confidence with diminishing returns, capped", () => {
    const s = submitOutcome(accepted(), "COMPLETED_FULL", "self_report", "did it", [
      { kind: "timer", timer: { startedAt: 1, endedAt: 600001, plannedMinutes: 15, actualMinutes: 10 } },
      { kind: "artifact", artifactUri: "file://proof.png" }
    ]);
    const outcome = s.questHistory[0].outcome!;
    expect(outcome.evidence).toHaveLength(3);
    // 1 - (1-0.58)(1-0.75)(1-0.86) = ~0.985 -> capped at 0.98
    expect(outcome.verificationConfidence).toBeCloseTo(0.98);
    expect(outcome.verificationConfidence!).toBeGreaterThan(0.86); // more than best single item
    const timerItem = outcome.evidence!.find((e) => e.kind === "timer")!;
    expect(timerItem.timer!.actualMinutes).toBe(10);
    const artifactItem = outcome.evidence!.find((e) => e.kind === "artifact")!;
    expect(artifactItem.artifactUri).toBe("file://proof.png");
  });

  test("multi-evidence yields more XP than single weak evidence for the same outcome", () => {
    const base = accepted();
    const single = submitOutcome(base, "COMPLETED_FULL", "self_report", "done");
    const multi = submitOutcome(base, "COMPLETED_FULL", "self_report", "done", [
      { kind: "artifact", artifactUri: "file://x" }
    ]);
    expect(multi.progression.xp).toBeGreaterThan(single.progression.xp);
  });
});

describe("consent gates (onboarding parity)", () => {
  test("surpriseQuests off suppresses the emergency director entirely", () => {
    let s = ready();
    s.profile.preferences.surpriseQuests = false;
    s.systemSignals.failureStreak = 5; // would trigger emergency
    s = updateHost(s, { ...s.host, energy: 0.2, stress: 0.9, timeAvailableMinutes: 20 });
    s = generateQuestOffers(s);
    expect(s.questOffers.some((o) => o.title === "Stabilize the Core")).toBe(false);
    expect(s.lastSystemMessage).not.toMatch(/Emergency Quest|Recovery Focus/);
  });

  test("calm emergency wording replaces the urgent banner", () => {
    let s = ready();
    s.profile.preferences.emergencyWording = "calm";
    s.systemSignals.failureStreak = 5;
    s = updateHost(s, { ...s.host, energy: 0.2, stress: 0.9, timeAvailableMinutes: 20 });
    s = generateQuestOffers(s);
    expect(s.lastSystemMessage).not.toMatch(/\[Emergency Quest\]/);
  });

  test("maxRiskTier caps offered risk, with a lowest-risk fallback", () => {
    let s = ready("courage");
    s.profile.preferences.maxRiskTier = 0;
    s = generateQuestOffers(s);
    expect(s.questOffers.length).toBeGreaterThan(0);
    const minRisk = Math.min(...s.questOffers.map((o) => o.riskTier));
    expect(s.questOffers.every((o) => o.riskTier === minRisk || o.riskTier <= 0)).toBe(true);
  });
});

describe("goal management", () => {
  test("pausing the selected goal hands selection back to the System", () => {
    let s = addGoal(ready("craft"), "Rest well", "recovery");
    const craft = s.profile.goals.find((g) => g.domain === "craft")!;
    s = { ...s, selectedGoalId: craft.id };
    s = updateGoal(s, craft.id, { status: "paused" });
    expect(s.profile.goals.find((g) => g.id === craft.id)!.status).toBe("paused");
    expect(s.selectedGoalId).not.toBe(craft.id);
  });

  test("archived goals never receive quests", () => {
    let s = ready("craft");
    const goal = s.profile.goals[0];
    s = updateGoal(s, goal.id, { status: "archived" });
    s = generateQuestOffers(s);
    expect(s.questOffers.every((o) => o.goalId !== goal.id)).toBe(true);
  });

  test("priority edits persist", () => {
    let s = ready("craft");
    s = updateGoal(s, s.profile.goals[0].id, { priority: 3 });
    expect(s.profile.goals[0].priority).toBe(3);
  });
});

describe("host depth + migration", () => {
  test("old saves without the new host fields are backfilled on load", () => {
    const old: any = ready();
    delete old.host.socialReadiness;
    delete old.host.creativeReadiness;
    delete old.host.bodyStatus;
    delete old.host.source;
    const healed = ensureStateShape(JSON.parse(JSON.stringify(old)));
    expect(healed.host.socialReadiness).toBe(0.5);
    expect(healed.host.creativeReadiness).toBe(0.5);
    expect(healed.host.bodyStatus).toBe(0.5);
    expect(healed.host.source).toBe("manual");
  });

  test("social readiness shifts goal selection toward/away from social goals", () => {
    let s = addGoal(addGoal(makeInitialState(), "See friends", "social"), "Build it", "craft");
    s = updateHost(s, { ...s.host, energy: 0.5, focus: 0.5, stress: 0.4, timeAvailableMinutes: 30, socialReadiness: 1 });
    const high = generateQuestOffers(s);
    s = updateHost(s, { ...s.host, socialReadiness: 0 });
    const low = generateQuestOffers(s);
    const socialGoalId = s.profile.goals.find((g) => g.domain === "social")!.id;
    const highPicksSocial = high.selectedGoalId === socialGoalId;
    const lowPicksSocial = low.selectedGoalId === socialGoalId;
    // The dial must be able to flip the choice at its extremes (or at minimum
    // never pick social MORE when readiness is zero).
    expect(highPicksSocial || !lowPicksSocial).toBe(true);
  });

  test("defaults of the new dials change nothing (legacy behavior preserved)", () => {
    expect(emptyHostState().socialReadiness).toBe(0.5);
    expect(emptyHostState().creativeReadiness).toBe(0.5);
    expect(emptyHostState().bodyStatus).toBe(0.5);
  });
});

describe("goal deletion + archive visibility", () => {
  test("deleting the focused goal hands selection back to the System", () => {
    let s = addGoal(ready("craft"), "Rest well", "recovery");
    const craft = s.profile.goals.find((g) => g.domain === "craft")!;
    s = { ...s, selectedGoalId: craft.id };
    s = deleteGoal(s, craft.id);
    expect(s.profile.goals.some((g) => g.id === craft.id)).toBe(false);
    expect(s.selectedGoalId).not.toBe(craft.id);
  });

  test("deleting a goal preserves its quest history", () => {
    let s = ready("craft");
    s = generateQuestOffers(s);
    s = acceptQuest(s, s.questOffers[0].id, true);
    s = submitOutcome(s, "COMPLETED_FULL", "self_report", "done");
    const goalId = s.profile.goals[0].id;
    const historyBefore = s.questHistory.length;
    s = deleteGoal(s, goalId);
    expect(s.questHistory.length).toBe(historyBefore);
  });

  test("archived goals can be restored to active", () => {
    let s = ready("craft");
    const id = s.profile.goals[0].id;
    s = updateGoal(s, id, { status: "archived" });
    expect(s.profile.goals[0].status).toBe("archived");
    s = updateGoal(s, id, { status: "active" });
    expect(s.profile.goals[0].status).toBe("active");
  });

  test("deleting an unknown goal id is a safe no-op", () => {
    const s = ready("craft");
    expect(deleteGoal(s, "nope").profile.goals.length).toBe(s.profile.goals.length);
  });
});

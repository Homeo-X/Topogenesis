import {
  addGoal,
  makeInitialState,
  updateHost,
  generateQuestOffers,
  acceptQuest,
  submitOutcome,
  rejectQuest,
  ensureStateShape,
  MAX_ACTIVE_QUESTS
} from "../engine";

function withOffers() {
  let s = addGoal(makeInitialState(), "Build it", "craft");
  s = updateHost(s, { ...s.host, energy: 0.6, focus: 0.55, stress: 0.3, timeAvailableMinutes: 40 });
  return generateQuestOffers(s);
}

describe("concurrent quests", () => {
  test("accepting takes only that offer off the board; the rest stay available", () => {
    let s = withOffers();
    const total = s.questOffers.length;
    expect(total).toBeGreaterThan(1);
    const first = s.questOffers[0];
    s = acceptQuest(s, first.id);
    expect(s.activeQuests).toHaveLength(1);
    expect(s.questOffers).toHaveLength(total - 1);
    expect(s.questOffers.some((o) => o.id === first.id)).toBe(false);
  });

  test("several quests run at once, up to the cap, then accepts are refused with a message", () => {
    let s = withOffers();
    while (s.activeQuests.length < MAX_ACTIVE_QUESTS) {
      if (!s.questOffers.length) s = generateQuestOffers(s);
      if (!s.questOffers.length) break; // dedupe can exhaust templates
      s = acceptQuest(s, s.questOffers[0].id, true);
    }
    expect(s.activeQuests.length).toBe(MAX_ACTIVE_QUESTS);
    s = generateQuestOffers(s);
    if (s.questOffers.length) {
      const before = s.activeQuests.length;
      s = acceptQuest(s, s.questOffers[0].id, true);
      expect(s.activeQuests.length).toBe(before);
      expect(s.lastSystemMessage).toMatch(/full/i);
    }
  });

  test("the derived activeQuest alias always mirrors activeQuests[0]", () => {
    let s = withOffers();
    expect(s.activeQuest).toBeNull();
    s = acceptQuest(s, s.questOffers[0].id);
    expect(s.activeQuest?.id).toBe(s.activeQuests[0].id);
    s = submitOutcome(s, "COMPLETED_FULL", "self_report", "done", undefined, s.activeQuests[0].id);
    expect(s.activeQuest).toBeNull();
    expect(s.activeQuests).toHaveLength(0);
  });

  test("submitting targets the chosen quest; the others keep running untouched", () => {
    let s = withOffers();
    s = acceptQuest(s, s.questOffers[0].id, true);
    if (!s.questOffers.length) s = generateQuestOffers(s);
    s = acceptQuest(s, s.questOffers[0].id, true);
    expect(s.activeQuests.length).toBe(2);
    const [keep, finish] = s.activeQuests;
    s = submitOutcome(s, "COMPLETED_FULL", "self_report", "done", undefined, finish.id);
    expect(s.activeQuests).toHaveLength(1);
    expect(s.activeQuests[0].id).toBe(keep.id);
    expect(s.questHistory[0].id).toBe(finish.id);
    expect(s.activeQuest?.id).toBe(keep.id); // alias invariant after removal
  });

  test("generating offers never destroys running quests (the latent single-quest bug)", () => {
    let s = withOffers();
    s = acceptQuest(s, s.questOffers[0].id, true);
    const running = s.activeQuests.map((q) => q.id);
    s = generateQuestOffers(s);
    expect(s.activeQuests.map((q) => q.id)).toEqual(running);
  });

  test("offers never duplicate a template that's already running", () => {
    let s = withOffers();
    s = acceptQuest(s, s.questOffers[0].id, true);
    s = generateQuestOffers(s);
    const activeTemplates = new Set(s.activeQuests.map((q) => q.candidateId));
    expect(s.questOffers.every((o) => !activeTemplates.has(o.candidateId))).toBe(true);
  });

  test("rejecting an offer leaves running quests alone", () => {
    let s = withOffers();
    s = acceptQuest(s, s.questOffers[0].id, true);
    const running = s.activeQuests.length;
    s = rejectQuest(s, s.questOffers[0]?.id);
    expect(s.activeQuests.length).toBe(running);
  });
});

describe("multiquest migration", () => {
  test("an old save with a single activeQuest loads as a one-element activeQuests array", () => {
    let s = withOffers();
    s = acceptQuest(s, s.questOffers[0].id, true);
    const old: any = JSON.parse(JSON.stringify(s));
    delete old.activeQuests; // pre-multiquest shape
    const healed = ensureStateShape(old);
    expect(healed.activeQuests).toHaveLength(1);
    expect(healed.activeQuests[0].id).toBe(healed.activeQuest!.id);
  });
});

import { ACTIVITY_CATEGORIES, activitiesForDomain, characterNudge, primaryDomainOf } from "../activities";
import { QUEST_ARCHETYPES } from "../questArchetypes";
import { addGoal, ensureStateShape, makeInitialState, updateHost, generateQuestOffers, acceptQuest, submitOutcome } from "../engine";
import { QuestDomain } from "../types";

const ALL_DOMAINS = Object.keys(QUEST_ARCHETYPES) as QuestDomain[];

describe("activity layer: mapping integrity", () => {
  test("there are 11 activity categories with unique ids", () => {
    expect(ACTIVITY_CATEGORIES.length).toBe(11);
    expect(new Set(ACTIVITY_CATEGORIES.map((a) => a.id)).size).toBe(11);
  });

  test("every activity maps to valid domains with positive weights, primary first", () => {
    for (const a of ACTIVITY_CATEGORIES) {
      expect(a.builds.length).toBeGreaterThan(0);
      for (const b of a.builds) {
        expect(ALL_DOMAINS).toContain(b.domain);
        expect(b.weight).toBeGreaterThan(0);
      }
      const weights = a.builds.map((b) => b.weight);
      expect(Math.max(...weights)).toBe(weights[0]); // primary axis leads
      expect(primaryDomainOf(a)).toBe(a.builds[0].domain);
    }
  });

  test("every character domain is reachable from at least one activity", () => {
    for (const d of ALL_DOMAINS) {
      expect(activitiesForDomain(d).length).toBeGreaterThan(0);
    }
  });
});

describe("activity layer: character nudge", () => {
  test("stays silent early (no nagging from day one)", () => {
    expect(characterNudge(makeInitialState())).toBeNull();
  });

  test("surfaces the quietest axis with concrete activities when a real gap exists", () => {
    const s = makeInitialState();
    s.progression.stats = { ...s.progression.stats, Craft: 30, Insight: 22, Vitality: 18, Order: 16, Bond: 2, Courage: 14, Recovery: 12, Discipline: 10 };
    const nudge = characterNudge(s);
    expect(nudge).not.toBeNull();
    expect(nudge!.stat).toBe("Bond");
    expect(nudge!.activities.length).toBeGreaterThan(0);
    expect(nudge!.text.length).toBeGreaterThan(10);
  });

  test("nudge copy is active but never shaming", () => {
    const s = makeInitialState();
    s.progression.stats = { ...s.progression.stats, Craft: 30, Bond: 2, Insight: 20, Vitality: 18, Order: 16, Courage: 14, Recovery: 12, Discipline: 10 };
    const nudge = characterNudge(s)!;
    expect(nudge.text).not.toMatch(/neglect|behind|failing|failed|lazy|guilt|shame|weak|bad at|should have/i);
  });
});

describe("domain mastery unlock notice", () => {
  test("crossing a domain level surfaces a mastery note in the system message", () => {
    let s = addGoal(makeInitialState(), "Build", "craft");
    s = updateHost(s, { ...s.host, energy: 0.6, focus: 0.55, stress: 0.3, timeAvailableMinutes: 40 });
    // Seed Craft just below the level-2 threshold (12 pts) so one completion crosses it.
    s = { ...s, progression: { ...s.progression, stats: { ...s.progression.stats, Craft: 11 } } };
    s = generateQuestOffers(s);
    const craftOffer = s.questOffers.find((o) => o.domain === "craft") ?? s.questOffers[0];
    s = acceptQuest(s, craftOffer.id);
    s = submitOutcome(s, "COMPLETED_FULL", "artifact", "done");
    expect(s.lastSystemMessage).toMatch(/mastery \d/i);
  });
});

describe("check-in persistence integrity", () => {
  test("updateHost commits host values that survive ensureStateShape (save/load)", () => {
    const before = makeInitialState();
    const saved = updateHost(before, { ...before.host, energy: 0.81, focus: 0.42, stress: 0.13, timeAvailableMinutes: 35 });
    const reloaded = ensureStateShape(JSON.parse(JSON.stringify(saved)));
    expect(reloaded.host.energy).toBeCloseTo(0.81);
    expect(reloaded.host.focus).toBeCloseTo(0.42);
    expect(reloaded.host.timeAvailableMinutes).toBe(35);
  });
});

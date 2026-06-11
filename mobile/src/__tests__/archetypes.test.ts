import { QUEST_ARCHETYPES, selectArchetype, renderAction, tierXpMultiplier, QuestArchetype } from "../questArchetypes";
import { addGoal, makeInitialState, updateHost, generateQuestOffers } from "../engine";
import { QuestDomain } from "../types";

const ALL_DOMAINS: QuestDomain[] = ["craft", "mind", "body", "order", "social", "courage", "recovery", "learning", "planning", "creation", "exploration"];

describe("archetype library: coverage", () => {
  test("every domain has at least 5 archetypes", () => {
    for (const d of ALL_DOMAINS) {
      expect(QUEST_ARCHETYPES[d]?.length ?? 0).toBeGreaterThanOrEqual(5);
    }
  });

  test("all archetype ids are unique", () => {
    const ids = ALL_DOMAINS.flatMap((d) => QUEST_ARCHETYPES[d].map((a) => a.id));
    expect(new Set(ids).size).toBe(ids.length);
  });

  test("every archetype is time-boxed with positive minutes", () => {
    for (const d of ALL_DOMAINS) for (const a of QUEST_ARCHETYPES[d]) {
      expect(a.minutes).toBeGreaterThan(0);
    }
  });

  test("every archetype has a non-empty action and builds line", () => {
    for (const d of ALL_DOMAINS) for (const a of QUEST_ARCHETYPES[d]) {
      expect(a.action.length).toBeGreaterThan(8);
      expect(a.builds.length).toBeGreaterThan(8);
    }
  });

  test("every domain offers more than one effort tier (small floor exists)", () => {
    for (const d of ALL_DOMAINS) {
      const tiers = new Set(QUEST_ARCHETYPES[d].map((a) => a.tier));
      expect(tiers.has("small")).toBe(true);
    }
  });
});

describe("archetype library: wellbeing guardrails", () => {
  const forbidden = [
    /lose .*(kg|lb|weight|fat)/i,
    /weight loss/i,
    /fat loss/i,
    /calorie/i,
    /\bdeficit\b/i,
    /\brestrict your\b/i,
    /\bgo on a diet\b/i,
    /target weight/i,
    /\bslim down\b/i,
    /burn fat/i,
    /\bpunish\b/i,
    /\bashamed\b/i,
    /you (must|have to|should be ashamed)/i,
    /you failed/i
  ];

  test("no archetype anywhere uses restriction, weight-target, shame, or coercion language", () => {
    for (const d of ALL_DOMAINS) for (const a of QUEST_ARCHETYPES[d]) {
      const text = `${a.action} ${a.builds}`;
      for (const p of forbidden) {
        expect(text).not.toMatch(p);
      }
    }
  });

  test("body archetypes are capability/behavior framed, never aesthetic", () => {
    for (const a of QUEST_ARCHETYPES.body) {
      const text = `${a.action} ${a.builds}`.toLowerCase();
      expect(text).not.toMatch(/\bweight\b|body fat|\bthinner\b|\bslim down\b|\bscale\b|\bpounds\b|\bkg\b/);
    }
  });
});

describe("archetype selection", () => {
  test("selectArchetype is deterministic for the same inputs", () => {
    expect(selectArchetype("craft", 3).id).toBe(selectArchetype("craft", 3).id);
  });

  test("rotation produces variety within a domain", () => {
    const seen = new Set<string>();
    for (let r = 0; r < 6; r++) seen.add(selectArchetype("mind", r).id);
    expect(seen.size).toBeGreaterThanOrEqual(2);
  });

  test("preferred tier filters to that tier when available", () => {
    const small = selectArchetype("body", 0, "small");
    expect(small.tier).toBe("small");
  });

  test("renderAction substitutes the {n} placeholder", () => {
    const withN: QuestArchetype = { id: "t", action: "Move for {n} minutes.", shape: "do_a_rep", tier: "small", minutes: 15, builds: "x" };
    expect(renderAction(withN)).toBe("Move for 15 minutes.");
  });

  test("larger tiers carry a higher XP multiplier", () => {
    expect(tierXpMultiplier("large")).toBeGreaterThan(tierXpMultiplier("medium"));
    expect(tierXpMultiplier("medium")).toBeGreaterThan(tierXpMultiplier("small"));
  });
});

describe("archetype integration in generated quests", () => {
  test("generated quest execute step contains a concrete archetype action", () => {
    let s = addGoal(makeInitialState(), "Build the app", "craft");
    s = updateHost(s, { ...s.host, energy: 0.6, focus: 0.6, stress: 0.35 });
    s = generateQuestOffers(s);
    const offer = s.questOffers[0];
    const exec = offer.activityPlan.steps.find((x) => x.id === "execute" || x.id === "execute_a");
    // Craft archetypes mention building/making/shipping/fixing/prototyping.
    expect(exec?.instruction.toLowerCase()).toMatch(/build|make|ship|fix|prototype|piece|version|part/);
  });

  test("generated body quests never contain weight/aesthetic language", () => {
    let s = addGoal(makeInitialState(), "Move well", "body");
    s = { ...s, profile: { ...s.profile, preferences: { ...s.profile.preferences, physicalQuests: true } } };
    s = updateHost(s, { ...s.host, energy: 0.6, focus: 0.6, stress: 0.35 });
    s = generateQuestOffers(s);
    for (const offer of s.questOffers) {
      const text = JSON.stringify(offer.activityPlan).toLowerCase();
      expect(text).not.toMatch(/\bweight\b|fat loss|calorie|\bgo on a diet\b|\bslim down\b|\bthinner\b/);
    }
  });

  test("generation remains deterministic with the archetype layer", () => {
    function firstExec(): string {
      let s = addGoal(makeInitialState(), "Read more", "learning");
      s = updateHost(s, { ...s.host, energy: 0.6, focus: 0.6, stress: 0.35 });
      s = generateQuestOffers(s);
      const o = s.questOffers[0];
      return o.activityPlan.steps.find((x) => x.id === "execute" || x.id === "execute_a")?.instruction ?? "";
    }
    expect(firstExec()).toBe(firstExec());
  });
});

import { unlockedArchetypes, domainLevelFromPoints } from "../questArchetypes";

describe("archetype level gating", () => {
  test("level 1 unlocks only the foundational (unlockLevel<=1) archetypes", () => {
    const l1 = unlockedArchetypes("craft", 1);
    expect(l1.length).toBeGreaterThan(0);
    expect(l1.every((a) => (a.unlockLevel ?? 1) <= 1)).toBe(true);
  });

  test("higher level ADDS archetypes without removing lower ones (repeatable habits)", () => {
    const low = unlockedArchetypes("craft", 1).map((a) => a.id);
    const high = unlockedArchetypes("craft", 20).map((a) => a.id);
    // Every low-level archetype is still present at high level.
    expect(low.every((id) => high.includes(id))).toBe(true);
    // And there are strictly more at the higher level.
    expect(high.length).toBeGreaterThan(low.length);
  });

  test("selectArchetype never returns a locked archetype", () => {
    for (let r = 0; r < 30; r++) {
      const a = selectArchetype("mind", r, undefined, 2);
      expect(a.unlockLevel ?? 1).toBeLessThanOrEqual(2);
    }
  });

  test("domainLevelFromPoints is monotonic and starts at 1", () => {
    expect(domainLevelFromPoints(0)).toBe(1);
    expect(domainLevelFromPoints(300)).toBeGreaterThan(domainLevelFromPoints(30));
    expect(domainLevelFromPoints(30)).toBeGreaterThanOrEqual(domainLevelFromPoints(12));
  });

  test("expanded domains have ~30 archetypes", () => {
    for (const d of ["craft","creation","mind","learning","planning","order","body","recovery","courage","social","exploration"] as const) {
      expect(QUEST_ARCHETYPES[d].length).toBeGreaterThanOrEqual(29);
    }
  });

  test("every archetype with a subcategory has a non-empty label", () => {
    for (const d of ALL_DOMAINS) for (const a of QUEST_ARCHETYPES[d]) {
      if (a.subcategory !== undefined) expect(a.subcategory.length).toBeGreaterThan(0);
    }
  });
});

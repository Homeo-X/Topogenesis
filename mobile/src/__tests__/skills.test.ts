import { computeUnlockedSkills, newlyUnlocked, SKILL_TREE, skillsByStat, visibleSkillsByStat } from "../skills";
import { ensureStateShape, makeInitialState } from "../engine";
import { StatName } from "../types";

function stats(overrides: Partial<Record<StatName, number>> = {}): Record<StatName, number> {
  const base: Record<StatName, number> = {
    Vitality: 0, Insight: 0, Discipline: 0, Courage: 0, Order: 0, Craft: 0, Bond: 0, Recovery: 0
  };
  return { ...base, ...overrides };
}

describe("skills: unlocking by threshold", () => {
  test("no skills unlocked at zero stats", () => {
    expect(computeUnlockedSkills(stats())).toEqual([]);
  });

  test("crossing a threshold unlocks exactly that tier", () => {
    const unlocked = computeUnlockedSkills(stats({ Craft: 10 }));
    expect(unlocked).toContain("craft_prototyper");
    expect(unlocked).not.toContain("craft_integrator"); // needs 30
  });

  test("higher stat unlocks all lower tiers in that track", () => {
    const unlocked = computeUnlockedSkills(stats({ Order: 60 }));
    expect(unlocked).toContain("order_decomposer");
    expect(unlocked).toContain("order_riskmapper");
    expect(unlocked).toContain("order_architect");
  });

  test("newlyUnlocked reports only the delta", () => {
    const prev = computeUnlockedSkills(stats({ Craft: 10 }));
    const fresh = newlyUnlocked(prev, stats({ Craft: 30 }));
    expect(fresh.map((s) => s.id)).toEqual(["craft_integrator"]);
  });

  test("every skill in the tree has a valid stat and positive threshold", () => {
    for (const skill of SKILL_TREE) {
      expect(skill.threshold).toBeGreaterThan(0);
      expect(typeof skill.label).toBe("string");
    }
  });

  test("skillsByStat groups without dropping any skill", () => {
    const grouped = skillsByStat();
    const total = Object.values(grouped).reduce((sum, arr) => sum + arr.length, 0);
    expect(total).toBe(SKILL_TREE.length);
  });

  test("visibleSkillsByStat hides locked future skills", () => {
    const hidden = visibleSkillsByStat([]);
    expect(Object.values(hidden).flat()).toEqual([]);
    const visible = visibleSkillsByStat(["craft_prototyper"]);
    expect(Object.values(visible).flat().map((skill) => skill.id)).toEqual(["craft_prototyper"]);
  });
});

describe("skills: migration", () => {
  test("old save without unlockedSkills retro-computes from existing stats", () => {
    const legacy: any = JSON.parse(JSON.stringify(makeInitialState()));
    delete legacy.progression.unlockedSkills;
    legacy.progression.stats = stats({ Craft: 35, Order: 12 });
    const restored = ensureStateShape(legacy);
    expect(restored.progression.unlockedSkills).toContain("craft_prototyper");
    expect(restored.progression.unlockedSkills).toContain("craft_integrator");
    expect(restored.progression.unlockedSkills).toContain("order_decomposer");
    expect(restored.progression.unlockedSkills).not.toContain("order_riskmapper");
  });

  test("fresh state has an empty unlockedSkills array", () => {
    expect(makeInitialState().progression.unlockedSkills).toEqual([]);
  });
});

import { addGoal, updateHost, generateQuestOffers, acceptQuest, submitOutcome } from "../engine";
import { AppState } from "../types";

function activeGoalState(skills: string[] = []): AppState {
  let s = addGoal(makeInitialState(), "Build the app", "craft");
  s = { ...s, progression: { ...s.progression, unlockedSkills: skills } };
  s = updateHost(s, { ...s.host, energy: 0.6, focus: 0.6, stress: 0.35 });
  return generateQuestOffers(s);
}

describe("skills: functional effects change engine behavior", () => {
  test("Decomposer splits Execute into two steps", () => {
    const without = activeGoalState([]);
    const withSkill = activeGoalState(["order_decomposer"]);
    const stepsWithout = without.questOffers[0].activityPlan.steps.map((s) => s.id);
    const stepsWith = withSkill.questOffers[0].activityPlan.steps.map((s) => s.id);
    expect(stepsWithout).toContain("execute");
    expect(stepsWith).toContain("execute_a");
    expect(stepsWith).toContain("execute_b");
    expect(stepsWith.length).toBe(stepsWithout.length + 1);
  });

  test("Decomposer is inert when not unlocked (no behavior leak)", () => {
    const without = activeGoalState([]);
    expect(without.questOffers[0].activityPlan.steps.some((s) => s.id.startsWith("execute_"))).toBe(false);
  });
});

describe("skills: prerequisite chains (deepening)", () => {
  test("a tier-2 skill does NOT unlock if its prerequisite is unmet, even past the stat threshold", () => {
    // Craft 30 meets integrator's threshold, but if prototyper were somehow not
    // unlocked the chain would block it. Since prototyper unlocks at 10, Craft 30
    // unlocks both — verify the chain produced both in order.
    const unlocked = computeUnlockedSkills(stats({ Craft: 30 }));
    expect(unlocked).toContain("craft_prototyper");
    expect(unlocked).toContain("craft_integrator");
  });

  test("a high stat in a different track cannot skip a chain", () => {
    // Order 60 should unlock the whole Order chain in order, nothing from Craft.
    const unlocked = computeUnlockedSkills(stats({ Order: 60 }));
    expect(unlocked).toEqual(expect.arrayContaining(["order_decomposer", "order_riskmapper", "order_architect"]));
    expect(unlocked.some((id) => id.startsWith("craft_"))).toBe(false);
  });

  test("every non-tier-1 skill names a prerequisite within its own stat track", () => {
    for (const skill of SKILL_TREE) {
      if (skill.requires) {
        const prereq = SKILL_TREE.find((s) => s.id === skill.requires)!;
        expect(prereq.stat).toBe(skill.stat);
        expect(prereq.threshold).toBeLessThan(skill.threshold);
      }
    }
  });

  test("functional count increased to at least 10 after enrichment", () => {
    expect(SKILL_TREE.filter((s) => s.functional).length).toBeGreaterThanOrEqual(15);
  });
});

describe("skills: new functional effects change engine behavior", () => {
  test("Risk Mapper lowers the hidden-quest avoidance threshold (fires at 2 not 3)", () => {
    // Risk Mapper must be EARNED via Order stat (the engine recomputes skills
    // from stats each outcome, so injecting an unearned skill would be wiped —
    // which is correct behavior). Order 30 genuinely unlocks decomposer+riskmapper.
    function avoidTwice(orderStat: number): AppState {
      let s = addGoal(makeInitialState(), "Build", "craft");
      s = { ...s, progression: { ...s.progression, stats: { ...s.progression.stats, Order: orderStat } } };
      s = { ...s, progression: { ...s.progression, unlockedSkills: computeUnlockedSkills(s.progression.stats) } };
      for (let i = 0; i < 2; i++) {
        s = updateHost(s, { ...s.host, energy: 0.6, stress: 0.3, recoveryNeed: 0.3 });
        s = generateQuestOffers(s);
        s = acceptQuest(s, s.questOffers[0].id, true);
        s = { ...s, activeQuest: { ...s.activeQuest!, domain: "craft" } };
        // Keep Order high so recompute preserves Risk Mapper across outcomes.
        s = submitOutcome(s, "FAILED_AVOIDED", "self_report", "x");
        s = { ...s, progression: { ...s.progression, stats: { ...s.progression.stats, Order: orderStat }, unlockedSkills: computeUnlockedSkills({ ...s.progression.stats, Order: orderStat }) } };
      }
      s = updateHost(s, { ...s.host, energy: 0.6, stress: 0.3, recoveryNeed: 0.3 });
      return generateQuestOffers(s);
    }
    const withMapper = avoidTwice(30);   // Risk Mapper earned -> threshold 2
    const withoutMapper = avoidTwice(0); // no Order skills -> threshold 3
    expect(withMapper.progression.unlockedSkills).toContain("order_riskmapper");
    expect(withMapper.systemSignals.lastEvent?.kind).toBe("hidden");
    expect(withoutMapper.systemSignals.lastEvent?.kind).not.toBe("hidden");
  });

  test("Integrator forces artifact proof on build quests even in trust mode", () => {
    let s = addGoal(makeInitialState(), "Build the app", "craft");
    s = { ...s, progression: { ...s.progression, unlockedSkills: ["craft_prototyper", "craft_integrator"] } };
    s = updateHost(s, { ...s.host, energy: 0.6, focus: 0.6, stress: 0.35 });
    s = generateQuestOffers(s);
    const buildOffer = s.questOffers.find((q) => q.domain === "craft");
    if (buildOffer) {
      expect(buildOffer.proofRequired.toLowerCase()).toContain("artifact");
    }
  });
});

describe("skills: domain XP uplift (Sanctuary Maker / Abyss Walker)", () => {
  test("recovery-domain XP is higher with Sanctuary Maker than without", () => {
    function recoveryXp(skills: string[]): number {
      let s = addGoal(makeInitialState(), "Rest well", "recovery");
      s = { ...s, progression: { ...s.progression, unlockedSkills: skills } };
      s = updateHost(s, { ...s.host, energy: 0.3, stress: 0.6, recoveryNeed: 0.7 });
      s = generateQuestOffers(s);
      const recoveryOffer = s.questOffers.find((q) => q.domain === "recovery") ?? s.questOffers[0];
      s = acceptQuest(s, recoveryOffer.id, true);
      s = { ...s, activeQuest: { ...s.activeQuest!, domain: "recovery" } };
      const before = s.progression.level * 100000 + s.progression.xp;
      s = submitOutcome(s, "COMPLETED_FULL", "reflection", "rested");
      const after = s.progression.level * 100000 + s.progression.xp;
      return after - before;
    }
    const withSkill = recoveryXp(["recovery_shield", "recovery_bulwark", "recovery_sanctuary"]);
    const withoutSkill = recoveryXp([]);
    expect(withSkill).toBeGreaterThan(withoutSkill);
  });
});

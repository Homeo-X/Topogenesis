import { ProgressionState, SkillDef, StatName } from "./types";

/**
 * Skill tree (enriched).
 *
 * Structure (deepening): skills form prerequisite CHAINS within each stat track.
 * A skill unlocks only when BOTH conditions hold:
 *   (a) the underlying stat has crossed its threshold, AND
 *   (b) its prerequisite skill (if any) is already unlocked.
 * This makes the tree a genuine tree (tier 3 depends on tier 2 depends on tier 1)
 * rather than parallel thresholds, and it lets higher tiers gate real behavior.
 *
 * Honesty policy (functional wiring): a skill is `functional: true` ONLY when it
 * has a defensible, real engine effect — never a circular self-reported stat
 * buff. Skills whose only honest effect would require machinery that does not
 * exist yet (multi-goal arc linking, etc.) remain earned MARKERS, and that is
 * stated, not hidden. After enrichment the functional set is ~11/22.
 *
 * Functional skills and their real effects (all wired in engine.ts):
 *   order_decomposer    : splits Execute into two smaller steps
 *   order_riskmapper    : hidden-quest avoidance threshold 3 -> 2 (blockers surface earlier)
 *   recovery_shield     : recovery completion grants a shield at lower quality
 *   recovery_bulwark    : streak-shield cap 3 -> 5
 *   discipline_streak   : day "counts" at lower quality (continuity harder to break)
 *   discipline_relentless: failure-streak emergency trigger is more forgiving (4 not 3)
 *   courage_namer       : courage/clarifier candidates surface when avoidance exists
 *   courage_breaker     : resolved avoidance re-clears faster (re-engagement decays 2)
 *   insight_archivist   : reflection evidence weighted slightly higher
 *   bond_connector      : social candidates surface more readily (if not user-disabled)
 *   craft_integrator    : build quests bias toward artifact evidence
 */
export const SKILL_TREE: SkillDef[] = [
  // Craft chain
  { id: "craft_prototyper", label: "Prototyper", stat: "Craft", threshold: 10, requires: null, description: "Surfaces an extra build/craft quest option.", functional: true },
  { id: "craft_integrator", label: "Integrator", stat: "Craft", threshold: 30, requires: "craft_prototyper", description: "Build quests favor artifact evidence.", functional: true },
  { id: "craft_systemwright", label: "Systemwright", stat: "Craft", threshold: 60, requires: "craft_integrator", description: "Reality-shaping construction. (marker: needs multi-goal arc linking)", functional: false },
  // Order chain
  { id: "order_decomposer", label: "Decomposer", stat: "Order", threshold: 10, requires: null, description: "Quests favor smaller, clearer steps.", functional: true },
  { id: "order_riskmapper", label: "Risk Mapper", stat: "Order", threshold: 30, requires: "order_decomposer", description: "Hidden blockers surface earlier.", functional: true },
  { id: "order_architect", label: "Architect", stat: "Order", threshold: 60, requires: "order_riskmapper", description: "Long arcs connect across goals. (marker: needs multi-goal arc linking)", functional: false },
  // Recovery chain
  { id: "recovery_shield", label: "Recovery Shield", stat: "Recovery", threshold: 10, requires: null, description: "Recovery action protects continuity at lower quality.", functional: true },
  { id: "recovery_bulwark", label: "Bulwark", stat: "Recovery", threshold: 30, requires: "recovery_shield", description: "Holds more streak shields (cap raised to 5).", functional: true },
  { id: "recovery_sanctuary", label: "Sanctuary Maker", stat: "Recovery", threshold: 60, requires: "recovery_bulwark", description: "Recovery quests grant more XP (recovery becomes strong progression).", functional: true },
  // Discipline chain
  { id: "discipline_rhythm", label: "Rhythm", stat: "Discipline", threshold: 10, requires: null, description: "Surfaces a daily-maintenance quest option.", functional: true },
  { id: "discipline_streak", label: "Iron Streak", stat: "Discipline", threshold: 30, requires: "discipline_rhythm", description: "A day counts at lower quality; continuity harder to break.", functional: true },
  { id: "discipline_relentless", label: "Relentless", stat: "Discipline", threshold: 60, requires: "discipline_streak", description: "More forgiving instability trigger under failure pressure.", functional: true },
  // Courage chain
  { id: "courage_namer", label: "Shadow Namer", stat: "Courage", threshold: 10, requires: null, description: "Courage quests surface when avoidance exists.", functional: true },
  { id: "courage_breaker", label: "Pattern Breaker", stat: "Courage", threshold: 30, requires: "courage_namer", description: "Re-engaging an avoided domain clears it faster.", functional: true },
  { id: "courage_abyss", label: "Abyss Walker", stat: "Courage", threshold: 60, requires: "courage_breaker", description: "Courage quests grant more XP (high-cost work recognized).", functional: true },
  // Insight chain
  { id: "insight_archivist", label: "Archivist", stat: "Insight", threshold: 10, requires: null, description: "Reflection evidence weighted slightly higher.", functional: true },
  { id: "insight_sage", label: "Sage", stat: "Insight", threshold: 30, requires: "insight_archivist", description: "Deeper synthesis across domains. (marker)", functional: false },
  { id: "insight_theorist", label: "World-Theorist", stat: "Insight", threshold: 60, requires: "insight_sage", description: "Foundational understanding. (marker)", functional: false },
  // Bond chain
  { id: "bond_connector", label: "Connector", stat: "Bond", threshold: 10, requires: null, description: "Social quests surface more readily (if enabled).", functional: true },
  { id: "bond_bridge", label: "Bridgekeeper", stat: "Bond", threshold: 30, requires: "bond_connector", description: "Sustains connection across distance. (marker)", functional: false },
  // Vitality chain
  { id: "vitality_initiate", label: "Initiate", stat: "Vitality", threshold: 10, requires: null, description: "Surfaces a body/physical quest option.", functional: true },
  { id: "vitality_frontline", label: "Frontline", stat: "Vitality", threshold: 30, requires: "vitality_initiate", description: "Direct action sustained. (marker)", functional: false }
];

const BY_ID = new Map(SKILL_TREE.map((s) => [s.id, s]));

/**
 * Compute unlocked skills honoring BOTH stat thresholds AND prerequisite chains.
 * Iterates to a fixed point so a chain unlocks in order within one pass set.
 */
export function computeUnlockedSkills(stats: Record<StatName, number>): string[] {
  const unlocked = new Set<string>();
  let changed = true;
  while (changed) {
    changed = false;
    for (const skill of SKILL_TREE) {
      if (unlocked.has(skill.id)) continue;
      const statMet = (stats[skill.stat] ?? 0) >= skill.threshold;
      const prereqMet = !skill.requires || unlocked.has(skill.requires);
      if (statMet && prereqMet) {
        unlocked.add(skill.id);
        changed = true;
      }
    }
  }
  // Preserve declaration order for stable output.
  return SKILL_TREE.filter((s) => unlocked.has(s.id)).map((s) => s.id);
}

export function newlyUnlocked(prev: string[], stats: Record<StatName, number>): SkillDef[] {
  const now = computeUnlockedSkills(stats);
  const prevSet = new Set(prev);
  return SKILL_TREE.filter((s) => now.includes(s.id) && !prevSet.has(s.id));
}

export function hasSkill(progression: ProgressionState, id: string): boolean {
  return progression.unlockedSkills.includes(id);
}

/** Convenience for engine effect checks against a raw unlocked list. */
export function unlockedHas(unlocked: string[], id: string): boolean {
  return unlocked.includes(id);
}

export function skillsByStat(): Record<StatName, SkillDef[]> {
  const grouped = {} as Record<StatName, SkillDef[]>;
  for (const skill of SKILL_TREE) {
    (grouped[skill.stat] ??= []).push(skill);
  }
  return grouped;
}

export function visibleSkillsByStat(unlockedIds: string[]): Record<StatName, SkillDef[]> {
  const unlocked = new Set(unlockedIds);
  const grouped = {} as Record<StatName, SkillDef[]>;
  for (const skill of SKILL_TREE) {
    if (!unlocked.has(skill.id)) continue;
    (grouped[skill.stat] ??= []).push(skill);
  }
  return grouped;
}

export function prerequisiteLabel(id: string): string | null {
  const skill = BY_ID.get(id);
  if (!skill?.requires) return null;
  return BY_ID.get(skill.requires)?.label ?? null;
}

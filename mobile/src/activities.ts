import { AppState, QuestDomain, StatName } from "./types";

/**
 * The activity layer: concrete, browsable "what do you want to do" categories
 * that map onto the character layer ("who you're becoming" — the QuestDomains).
 *
 * The two layers form a visible causal loop:
 *   activity → character: pick a concrete activity, see which character axes
 *     it builds (the weighted mapping below).
 *   character → activity: a quiet character axis surfaces the activities that
 *     would feed it (suggestActivitiesFor / characterNudge).
 *
 * Tone rule for all generated guidance: active and direct — the System takes
 * initiative and points forward — but NEVER guilt, shame, "falling behind",
 * or neglect framing. Push toward growth, never away from failure.
 */

export interface ActivityCategory {
  id: string;
  label: string;
  /** One-line plain description of what lives here. */
  blurb: string;
  /** Weighted character-domain blend. First entry is the primary axis. */
  builds: Array<{ domain: QuestDomain; weight: number }>;
}

export const ACTIVITY_CATEGORIES: ActivityCategory[] = [
  {
    id: "fitness",
    label: "Fitness & Movement",
    blurb: "Workouts, walks, stretching, sport — anything that moves you.",
    builds: [
      { domain: "body", weight: 0.6 },
      { domain: "courage", weight: 0.2 },
      { domain: "recovery", weight: 0.2 }
    ]
  },
  {
    id: "food",
    label: "Food & Nourishment",
    blurb: "Cooking, eating well, feeding yourself as care.",
    builds: [
      { domain: "body", weight: 0.5 },
      { domain: "recovery", weight: 0.3 },
      { domain: "order", weight: 0.2 }
    ]
  },
  {
    id: "home",
    label: "Home & Order",
    blurb: "Tidying, fixing, making your space work for you.",
    builds: [
      { domain: "order", weight: 0.7 },
      { domain: "recovery", weight: 0.3 }
    ]
  },
  {
    id: "work",
    label: "Work & Craft",
    blurb: "Building, shipping, getting better at what you make.",
    builds: [
      { domain: "craft", weight: 0.6 },
      { domain: "planning", weight: 0.2 },
      { domain: "courage", weight: 0.2 }
    ]
  },
  {
    id: "money",
    label: "Money & Admin",
    blurb: "Finances, paperwork, the life admin that frees you.",
    builds: [
      { domain: "planning", weight: 0.5 },
      { domain: "order", weight: 0.3 },
      { domain: "courage", weight: 0.2 }
    ]
  },
  {
    id: "learning",
    label: "Learning & Curiosity",
    blurb: "Reading, studying, following what you want to understand.",
    builds: [
      { domain: "learning", weight: 0.7 },
      { domain: "mind", weight: 0.3 }
    ]
  },
  {
    id: "creative",
    label: "Creative Work",
    blurb: "Making things — writing, art, music, anything from nothing.",
    builds: [
      { domain: "creation", weight: 0.6 },
      { domain: "craft", weight: 0.2 },
      { domain: "courage", weight: 0.2 }
    ]
  },
  {
    id: "relationships",
    label: "Relationships",
    blurb: "Friends, family, the people who matter.",
    builds: [
      { domain: "social", weight: 0.7 },
      { domain: "courage", weight: 0.3 }
    ]
  },
  {
    id: "mindcalm",
    label: "Mind & Calm",
    blurb: "Settling, reflecting, taking care of your head.",
    builds: [
      { domain: "mind", weight: 0.7 },
      { domain: "recovery", weight: 0.3 }
    ]
  },
  {
    id: "rest",
    label: "Rest & Recovery",
    blurb: "Real rest, boundaries, refilling on purpose.",
    builds: [
      { domain: "recovery", weight: 0.7 },
      { domain: "mind", weight: 0.3 }
    ]
  },
  {
    id: "exploration",
    label: "Exploration & Outdoors",
    blurb: "New places, the outdoors, new people and experiences.",
    builds: [
      { domain: "exploration", weight: 0.4 },
      { domain: "body", weight: 0.2 },
      { domain: "courage", weight: 0.2 },
      { domain: "social", weight: 0.2 }
    ]
  }
];

/** The primary character domain an activity feeds (its first/heaviest entry). */
export function primaryDomainOf(activity: ActivityCategory): QuestDomain {
  return activity.builds[0].domain;
}

/** All activities that feed a given character domain, heaviest first. */
export function activitiesForDomain(domain: QuestDomain): ActivityCategory[] {
  return ACTIVITY_CATEGORIES
    .filter((a) => a.builds.some((b) => b.domain === domain))
    .sort((x, y) => (y.builds.find((b) => b.domain === domain)?.weight ?? 0) - (x.builds.find((b) => b.domain === domain)?.weight ?? 0));
}

const statToDomains: Record<StatName, QuestDomain[]> = {
  Craft: ["craft", "creation"],
  Insight: ["mind", "learning"],
  Vitality: ["body", "exploration"],
  Order: ["order", "planning"],
  Bond: ["social"],
  Courage: ["courage"],
  Recovery: ["recovery"],
  Discipline: ["order", "planning"]
};

/**
 * Character → activity guidance: find the quietest stat axis and the concrete
 * activities that would build it. Returns null when there's no meaningful gap
 * yet (early game) so the nudge never nags from day one.
 */
export function characterNudge(state: AppState): { stat: StatName; text: string; activities: ActivityCategory[] } | null {
  const entries = Object.entries(state.progression.stats) as Array<[StatName, number]>;
  const total = entries.reduce((sum, [, v]) => sum + v, 0);
  if (total < 12) return null; // Too early to have a meaningful pattern.
  const sorted = [...entries].sort((a, b) => a[1] - b[1]);
  const [lowStat, lowVal] = sorted[0];
  const [, highVal] = sorted[sorted.length - 1];
  if (highVal - lowVal < 6) return null; // No real gap — say nothing.
  const domains = statToDomains[lowStat] ?? [];
  const activities = domains.flatMap((d) => activitiesForDomain(d));
  const unique = activities.filter((a, i) => activities.findIndex((x) => x.id === a.id) === i).slice(0, 3);
  if (!unique.length) return null;
  const names = unique.map((a) => a.label);
  const list = names.length > 1 ? `${names.slice(0, -1).join(", ")} or ${names[names.length - 1]}` : names[0];
  // Active, forward-pointing, zero guilt: name the opportunity, not a failure.
  const text = `${lowStat} is ready to grow. ${list} would build it — pick one and I'll shape a quest.`;
  return { stat: lowStat, text, activities: unique };
}

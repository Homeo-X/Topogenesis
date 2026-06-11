import { AppState } from "./types";

/**
 * Evolving System voice (direction "b").
 *
 * The voice warms and grows more familiar as accumulated history increases, but
 * is deliberately bounded: it becomes a tool that knows you well, never a
 * companion that emotes or implies a relationship. These guardrails are
 * intentional and load-bearing for a wellbeing app:
 *   - never first-person-plural ("we"), never possessive of the user
 *   - never expresses need, longing, or that it "missed" the user
 *   - never guilt, pressure, or dependency framing
 *   - warmth shows as recognition and continuity, not affection
 * The genre trope is "the System as a character with opinions"; here that is
 * realized as continuity and earned context, not manufactured intimacy.
 */

export type FamiliarityStage = 0 | 1 | 2 | 3;

export interface VoiceContext {
  stage: FamiliarityStage;
  label: string;
}

/** Derive how much shared history exists, from durable signals only. */
export function familiarityOf(state: AppState): VoiceContext {
  const interactions = state.questHistory.length;
  const level = state.progression.level;
  const confirmedAssumptions = state.assumptions.filter((a) => a.protected || a.status === "active").length;

  // Combine signals; interactions dominate, level and confirmed knowledge nudge.
  const score = interactions + Math.max(0, level - 1) * 2 + confirmedAssumptions;

  let stage: FamiliarityStage = 0;
  if (score >= 25) stage = 3;
  else if (score >= 10) stage = 2;
  else if (score >= 3) stage = 1;

  const label = ["Distant", "Aware", "Familiar", "Attuned"][stage];
  return { stage, label };
}

/** Semantic events the voice can narrate. Engine emits these; voice phrases them. */
export type VoiceEvent =
  | "init"
  | "goal_registered"
  | "scan_accepted"
  | "offers_ready"
  | "quest_accepted"
  | "quest_rejected"
  | "outcome_logged"
  | "assumption_updated";

/**
 * Produce a stage-appropriate opening clause for an event. The engine's own
 * factual message (scores, rewards, named patterns) is appended after this by
 * the caller, so the voice colors the message without replacing its content.
 */
export function voiceOpener(event: VoiceEvent, state: AppState): string {
  const { stage } = familiarityOf(state);
  const streak = state.progression.streak.current;
  const stageVariants = OPENERS[event];
  // Clamp stage to available rows (each event defines rows for stages 0..3).
  const variants = stageVariants[Math.min(stage, stageVariants.length - 1)];
  // Deterministic rotation across the variants for this stage, keyed off how
  // much history exists. Same state -> same line (testable, no flicker on
  // re-render), but the line changes as the user progresses, so repeated
  // events stop reading identically.
  const rotation = state.questHistory.length + state.progression.level;
  const line = variants[rotation % variants.length];
  // Light, bounded personalization: acknowledge a live streak at higher stages.
  if (stage >= 2 && streak >= 3 && (event === "scan_accepted" || event === "offers_ready")) {
    return `${line} Streak at ${streak}.`;
  }
  return line;
}

// Per event: an array indexed by familiarity stage (0 Distant .. 3 Attuned),
// each containing 2-3 interchangeable variants rotated deterministically.
const OPENERS: Record<VoiceEvent, string[][]> = {
  init: [
    ["System ready.", "Ready to go."],
    ["System ready.", "Up and running."],
    ["Ready. Picking up where you left off.", "Ready. Back to it."],
    ["Ready. Plenty of history here now.", "Ready. You've done a lot already."]
  ],
  goal_registered: [
    ["Goal saved.", "Goal added."],
    ["Goal saved. A direction is forming.", "Goal added. This gives you something to aim at."],
    ["Goal saved. It fits where you're already headed.", "Goal added. It lines up with your current path."],
    ["Goal saved. It fits the bigger picture you've built.", "Goal added. It extends the path you've been on."]
  ],
  scan_accepted: [
    ["Scan saved.", "Got your scan."],
    ["Scan saved. Adjusting to how you're doing.", "Got your scan. Tuning to your current state."],
    ["Scan saved. Compared to your usual pattern.", "Got your scan. Read against what's normal for you."],
    ["Scan saved. I know your usual pattern well by now.", "Got your scan. Your pattern is familiar at this point."]
  ],
  offers_ready: [
    ["Quests ready.", "Here are some quests."],
    ["Quests ready, picked for how you're doing.", "Here are some quests, matched to your state."],
    ["Quests ready, leaning on what tends to work for you.", "Here are some quests, shaped by what usually lands."],
    ["Quests ready, tuned to what works for you.", "Here are some quests, matched to your long track record."]
  ],
  quest_accepted: [
    ["Quest accepted.", "You're on."],
    ["Quest accepted. Keep the proof honest.", "You're on. Honest evidence is enough."],
    ["Quest accepted. This one suits you.", "You're on. A good fit for you."],
    ["Quest accepted. A familiar kind of move for you.", "You're on. This fits your pattern."]
  ],
  quest_rejected: [
    ["Quest skipped.", "Skipped."],
    ["Quest skipped. Your call.", "Skipped. Your choice stands."],
    ["Quest skipped. Noted, no judgment.", "Skipped. Recorded, no penalty."],
    ["Quest skipped. A familiar choice, and a fair one.", "Skipped. In line with what you prefer."]
  ],
  outcome_logged: [
    ["Result saved.", "Logged."],
    ["Result saved. Adjusting as I go.", "Logged. Updating from this."],
    ["Result saved. It helps me read you better.", "Logged. It feeds into what I know about you."],
    ["Result saved. One more in a long history.", "Logged. Another step in a long record."]
  ],
  assumption_updated: [
    ["Notes updated.", "Updated."],
    ["Notes updated. Still open for you to review.", "Updated. You can check it anytime."],
    ["Notes updated. A sharper read of you.", "Updated. The picture is clearer."],
    ["Notes updated. The picture is detailed now.", "Updated. A well-formed read by now."]
  ]
};


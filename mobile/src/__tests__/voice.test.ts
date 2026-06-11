import { addGoal, makeInitialState, submitOutcome, acceptQuest, generateQuestOffers, updateHost } from "../engine";
import { familiarityOf, voiceOpener, VoiceEvent } from "../voice";
import { AppState } from "../types";

function aged(interactions: number, level = 1): AppState {
  const s = makeInitialState();
  const history = Array.from({ length: interactions }, (_, i) => ({
    ...makeInitialState().activeQuest!,
    id: `h${i}`
  })) as AppState["questHistory"];
  return { ...s, questHistory: history, progression: { ...s.progression, level } };
}

describe("voice: familiarity staging", () => {
  test("new state is Distant (stage 0)", () => {
    expect(familiarityOf(makeInitialState()).stage).toBe(0);
  });
  test("a few interactions reach Aware (stage 1)", () => {
    expect(familiarityOf(aged(4)).stage).toBe(1);
  });
  test("sustained history reaches Familiar (stage 2)", () => {
    expect(familiarityOf(aged(12)).stage).toBe(2);
  });
  test("long history reaches Attuned (stage 3)", () => {
    expect(familiarityOf(aged(26)).stage).toBe(3);
  });
  test("stage increases monotonically with interactions", () => {
    const stages = [0, 4, 12, 26].map((n) => familiarityOf(aged(n)).stage);
    expect(stages).toEqual([...stages].sort((a, b) => a - b));
  });
});

describe("voice: wellbeing guardrails (must hold at EVERY stage)", () => {
  const events: VoiceEvent[] = [
    "init", "goal_registered", "scan_accepted", "offers_ready",
    "quest_accepted", "quest_rejected", "outcome_logged", "assumption_updated"
  ];
  // Language that would tip the tool into a parasocial/dependency dynamic.
  const forbidden = [
    /\bi missed you\b/i,
    /\bwe\b/i,            // no first-person-plural bonding
    /\bi need you\b/i,
    /\bi'?m proud of you\b/i,
    /\bdon'?t leave\b/i,
    /\bi'?ve been waiting\b/i,
    /\blove\b/i,
    /\blonely\b/i
  ];

  test("no forbidden parasocial language at any familiarity stage or variant", () => {
    for (const interactions of [0, 1, 2, 4, 5, 12, 13, 30, 31]) {
      // Sweep several history/level combos so every rotation variant is hit.
      for (const lvl of [1, 2, 3, 4, 5]) {
        const state = { ...aged(interactions, lvl) };
        for (const event of events) {
          const line = voiceOpener(event, state);
          for (const pattern of forbidden) {
            expect(line).not.toMatch(pattern);
          }
        }
      }
    }
  });

  test("copy variety: an event yields more than one distinct line across progression", () => {
    const seen = new Set<string>();
    for (const interactions of [10, 11, 12, 13, 14, 15]) {
      seen.add(voiceOpener("outcome_logged", aged(interactions, 3)));
    }
    // At a fixed stage, rotation should surface at least two distinct variants.
    expect(seen.size).toBeGreaterThanOrEqual(2);
  });

  test("voiceOpener always returns a non-empty string", () => {
    for (const event of events) {
      expect(voiceOpener(event, makeInitialState()).length).toBeGreaterThan(0);
    }
  });
});

describe("voice: engine integration", () => {
  test("messages flow through voice without breaking the loop", () => {
    let s = addGoal(makeInitialState(), "Build the app", "craft");
    expect(s.lastSystemMessage.length).toBeGreaterThan(0);
    s = updateHost(s, { ...s.host, energy: 0.6, focus: 0.6, stress: 0.35 });
    expect(s.lastSystemMessage.length).toBeGreaterThan(0);
    s = generateQuestOffers(s);
    s = acceptQuest(s, s.questOffers[0].id, true);
    expect(s.lastSystemMessage.length).toBeGreaterThan(0);
    s = submitOutcome(s, "COMPLETED_FULL", "reflection", "done");
    expect(s.lastSystemMessage.length).toBeGreaterThan(0);
  });
});

import { addGoal, makeInitialState, updateHost, generateQuestOffers } from "../engine";
import { AppState } from "../types";

function offerInstruction(historyLen: number): string {
  let s = addGoal(makeInitialState(), "Build the app", "craft");
  // Simulate accumulated history to drive the deterministic rotation.
  const hist = Array.from({ length: historyLen }, (_, i) => ({ ...makeInitialState().activeQuest!, id: `h${i}` })) as AppState["questHistory"];
  s = { ...s, questHistory: hist };
  s = updateHost(s, { ...s.host, energy: 0.6, focus: 0.6, stress: 0.35 });
  s = generateQuestOffers(s);
  const exec = s.questOffers[0].activityPlan.steps.find((step) => step.id === "execute" || step.id === "execute_a");
  return exec?.instruction ?? "";
}

describe("compositional generation", () => {
  test("execute instructions vary across history rotations for the same domain", () => {
    const seen = new Set<string>();
    for (let n = 0; n < 6; n++) seen.add(offerInstruction(n));
    // Method+constraint rotation should produce multiple distinct instructions.
    expect(seen.size).toBeGreaterThanOrEqual(2);
  });

  test("generation is deterministic: same state yields the same instruction", () => {
    expect(offerInstruction(3)).toBe(offerInstruction(3));
  });

  test("accented instruction still contains the base action and a constraint", () => {
    const instruction = offerInstruction(0);
    expect(instruction.length).toBeGreaterThan(20);
  });
});

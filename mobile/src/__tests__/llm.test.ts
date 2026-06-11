import { addGoal, makeInitialState } from "../engine";
import { interpretOutcome, refineArcWithLlm } from "../llm";
import { AppState } from "../types";

function stateWithLlm(enabled: boolean): AppState {
  const s = makeInitialState();
  return { ...s, llm: { ...s.llm, enabled } };
}

describe("arc refinement: engine fallback", () => {
  test("returns the deterministic arc unchanged when LLM is disabled", async () => {
    const s = addGoal(stateWithLlm(false), "Build the Forge app", "craft");
    const goal = s.profile.goals[0];
    const arc = await refineArcWithLlm(s, goal);
    expect(arc).toEqual(goal.arc);
    // structure intact
    expect(arc.milestones.length).toBeGreaterThanOrEqual(3);
    expect(arc.milestones[0].status).toBe("active");
  });

  test("returns the deterministic arc when the endpoint is unreachable", async () => {
    // enabled + ollama route pointing at a dead port -> fetch throws -> fallback
    const base = addGoal(makeInitialState(), "Learn JAX deeply", "learning");
    const goal = base.profile.goals[0];
    const s: AppState = { ...base, llm: { enabled: true, provider: "ollama", endpoint: "http://127.0.0.1:1", model: "x" } };
    const arc = await refineArcWithLlm(s, goal);
    expect(arc).toEqual(goal.arc);
  });
});

describe("outcome interpretation: suggest-with-confirmation contract", () => {
  test("returns null when LLM disabled (degrades to inform-none)", async () => {
    const s = stateWithLlm(false);
    const result = await interpretOutcome(s, "COMPLETED_PARTIAL", "I got about halfway then stopped.");
    expect(result).toBeNull();
  });

  test("returns null for an empty note even if LLM is enabled", async () => {
    const s = stateWithLlm(true);
    const result = await interpretOutcome(s, "COMPLETED_FULL", "   ");
    expect(result).toBeNull();
  });

  test("returns null (not a crash) when the endpoint is unreachable", async () => {
    const s: AppState = { ...makeInitialState(), llm: { enabled: true, provider: "ollama", endpoint: "http://127.0.0.1:1", model: "x" } };
    const result = await interpretOutcome(s, "COMPLETED_PARTIAL", "kept getting distracted");
    expect(result).toBeNull();
  });
});

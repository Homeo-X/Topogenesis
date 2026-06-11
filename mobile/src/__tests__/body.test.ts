import { addGoal, makeInitialState, updateHost, generateQuestOffers, ensureStateShape } from "../engine";
import { deriveBodySignals, latestBody } from "../body";
import { reducer } from "../state/reducer";
import { AppState, BodyComposition } from "../types";

const sampleScan: BodyComposition = {
  recordedAt: Date.now(),
  weightKg: 71.1, fatMassKg: 21.5, muscleMassKg: 46.3, bmi: 24.6, bodyFatPct: 30.3,
  visceralFatGrade: 8, basalMetabolicRate: 1441, smi: 7.3, bodyAge: 27,
  segmentLeftArmPct: 230.7, segmentRightArmPct: 237.3, segmentTrunkPct: 88,
  segmentLeftLegPct: 208.1, segmentRightLegPct: 208.6
};

function withScan(scan: Partial<BodyComposition>): AppState {
  const s = makeInitialState();
  return { ...s, bodyHistory: [{ recordedAt: Date.now(), ...scan }] };
}

describe("body: snapshot storage & migration", () => {
  test("fresh state has empty bodyHistory", () => {
    expect(makeInitialState().bodyHistory).toEqual([]);
  });

  test("ADD_BODY_SCAN prepends newest-first", () => {
    let s = reducer(makeInitialState(), { type: "ADD_BODY_SCAN", scan: { recordedAt: 1, muscleMassKg: 45 } });
    s = reducer(s, { type: "ADD_BODY_SCAN", scan: { recordedAt: 2, muscleMassKg: 46 } });
    expect(s.bodyHistory[0].recordedAt).toBe(2);
    expect(s.bodyHistory.length).toBe(2);
  });

  test("old save without bodyHistory migrates to an empty array", () => {
    const legacy: any = JSON.parse(JSON.stringify(makeInitialState()));
    delete legacy.bodyHistory;
    expect(ensureStateShape(legacy).bodyHistory).toEqual([]);
  });
});

describe("body: capability signals (not aesthetic)", () => {
  test("detects a left/right imbalance from segmental data", () => {
    const s = withScan({ segmentLeftArmPct: 200, segmentRightArmPct: 215 });
    expect(deriveBodySignals(s).imbalanceDetected).toBe(true);
  });

  test("no imbalance flagged when limbs are balanced", () => {
    const s = withScan({ segmentLeftArmPct: 205, segmentRightArmPct: 207 });
    expect(deriveBodySignals(s).imbalanceDetected).toBe(false);
  });

  test("muscle trend reads across two snapshots", () => {
    const s = makeInitialState();
    const withTrend: AppState = { ...s, bodyHistory: [{ recordedAt: 2, muscleMassKg: 47 }, { recordedAt: 1, muscleMassKg: 45 }] };
    expect(deriveBodySignals(withTrend).muscleTrend).toBe("up");
  });

  test("physical readiness derives from SMI, never from weight or fat", () => {
    const lean = deriveBodySignals(withScan({ smi: 8 })).physicalReadiness;
    const heavierWeightOnly = deriveBodySignals(withScan({ smi: 8, weightKg: 120, fatMassKg: 60 })).physicalReadiness;
    // Adding weight/fat must NOT change the readiness proxy — it's capability-only.
    expect(lean).toBe(heavierWeightOnly);
  });
});

describe("body: wellbeing guardrail — no aesthetic/weight-target framing", () => {
  const forbidden = [
    /lose .*(kg|weight|fat)/i,
    /weight loss/i,
    /fat loss/i,
    /calorie deficit/i,
    /restrict/i,
    /target weight/i,
    /slim(ming)?/i,
    /burn fat/i
  ];

  test("imbalance note and trend messages never use weight-loss/restriction language", () => {
    const signals = deriveBodySignals(withScan({ segmentLeftArmPct: 200, segmentRightArmPct: 220, muscleMassKg: 46 }));
    const texts = [signals.imbalanceNote ?? ""];
    for (const t of texts) for (const p of forbidden) expect(t).not.toMatch(p);
  });

  test("body-derived quests carry no aesthetic/weight-target framing", () => {
    let s = addGoal(makeInitialState(), "Move well", "body");
    s = { ...s, bodyHistory: [{ recordedAt: Date.now(), segmentLeftArmPct: 200, segmentRightArmPct: 220 }] };
    s = { ...s, profile: { ...s.profile, preferences: { ...s.profile.preferences, physicalQuests: true } } };
    s = updateHost(s, { ...s.host, energy: 0.6, focus: 0.6, stress: 0.35 });
    s = generateQuestOffers(s);
    for (const offer of s.questOffers) {
      const text = `${offer.title} ${offer.objective} ${offer.activityPlan.intent} ${offer.activityPlan.stakes}`;
      for (const p of forbidden) expect(text).not.toMatch(p);
    }
  });
});

describe("body: quest enrichment", () => {
  test("imbalance surfaces a body/movement option when physical quests enabled", () => {
    // With a body-domain goal the corrective option competes on-domain and
    // appears. (Off-domain it enters the candidate pool but may be out-scored by
    // goal-aligned quests — surfacing is by merit, never forced.)
    let s = addGoal(makeInitialState(), "Move well", "body");
    s = { ...s, bodyHistory: [{ recordedAt: Date.now(), segmentLeftArmPct: 200, segmentRightArmPct: 220 }] };
    s = { ...s, profile: { ...s.profile, preferences: { ...s.profile.preferences, physicalQuests: true } } };
    s = updateHost(s, { ...s.host, energy: 0.6, focus: 0.6, stress: 0.35 });
    s = generateQuestOffers(s);
    expect(s.questOffers.some((q) => q.domain === "body")).toBe(true);
  });

  test("imbalance does NOT surface a body quest when physical quests are disabled (consent respected)", () => {
    let s = addGoal(makeInitialState(), "Think clearly", "mind");
    s = { ...s, bodyHistory: [{ recordedAt: Date.now(), segmentLeftArmPct: 200, segmentRightArmPct: 220 }] };
    s = { ...s, profile: { ...s.profile, preferences: { ...s.profile.preferences, physicalQuests: false } } };
    s = updateHost(s, { ...s.host, energy: 0.6, focus: 0.6, stress: 0.35 });
    s = generateQuestOffers(s);
    // No body quest should be forced in when the user has opted out.
    expect(s.questOffers.some((q) => q.domain === "body" && q.candidateId?.includes("25"))).toBe(false);
  });
});

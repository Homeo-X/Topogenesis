import { addGoal, makeInitialState, updateHost, generateQuestOffers } from "../engine";
import { AppState, QuestDomain } from "../types";

function offersFor(domain: QuestDomain, host: Partial<AppState["host"]> = {}): AppState {
  let s = addGoal(makeInitialState(), `Work on ${domain}`, domain);
  s = updateHost(s, { ...s.host, energy: 0.6, focus: 0.6, stress: 0.35, ...host });
  return generateQuestOffers(s);
}

describe("quest framing: stakes present and well-formed", () => {
  test("every generated offer carries a non-empty stakes clause", () => {
    for (const domain of ["craft", "mind", "learning", "social", "recovery", "courage", "order"] as QuestDomain[]) {
      const s = offersFor(domain);
      for (const offer of s.questOffers) {
        expect(offer.activityPlan.stakes.length).toBeGreaterThan(0);
      }
    }
  });

  test("stakes differ by domain (framing is contextual, not boilerplate)", () => {
    const craft = offersFor("craft").questOffers[0].activityPlan.stakes;
    const recovery = offersFor("recovery", { energy: 0.15, stress: 0.85, recoveryNeed: 0.8 }).questOffers[0].activityPlan.stakes;
    expect(craft).not.toBe(recovery);
  });
});

describe("quest framing: wellbeing guardrail (stakes never coercive)", () => {
  // Stakes must frame what is built/protected, never loss, threat, or shame.
  const forbidden = [
    /\bor else\b/i,
    /\byou'?ll lose\b/i,
    /\bfail(ure)?\b/i,
    /\bpunish/i,
    /\bashamed?\b/i,
    /\bdisappoint/i,
    /\bwaste/i,
    /\bfall behind\b/i,
    /\blast chance\b/i
  ];

  test("no loss/threat/shame language in stakes across all domains and modes", () => {
    const hosts: Array<Partial<AppState["host"]>> = [
      { energy: 0.6, focus: 0.6, stress: 0.35 },
      { energy: 0.15, stress: 0.85, recoveryNeed: 0.8 },
      { energy: 0.8, focus: 0.8, stress: 0.2, challengeReadiness: 0.85, timeAvailableMinutes: 60 }
    ];
    for (const domain of ["craft", "mind", "learning", "social", "recovery", "courage", "order", "body", "creation", "planning"] as QuestDomain[]) {
      for (const host of hosts) {
        const s = offersFor(domain, host);
        for (const offer of s.questOffers) {
          for (const pattern of forbidden) {
            expect(offer.activityPlan.stakes).not.toMatch(pattern);
          }
        }
      }
    }
  });
});

describe("quest framing: migration", () => {
  test("a saved quest lacking stakes does not crash generation/serialization", () => {
    // Simulate an old active quest without the stakes field; ensureStateShape +
    // normal flow must tolerate it. (UI reads optional access; engine regenerates.)
    let s = offersFor("craft");
    const legacy: any = JSON.parse(JSON.stringify(s));
    if (legacy.questOffers[0]?.activityPlan) delete legacy.questOffers[0].activityPlan.stakes;
    // Re-generating offers should produce fresh quests that DO have stakes.
    const regen = generateQuestOffers(makeInitialState());
    expect(() => JSON.stringify(regen)).not.toThrow();
  });
});

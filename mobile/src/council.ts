import { AppState, Goal, Quest } from "./types";

/**
 * The System's council: a deterministic multi-advisor deliberation layer.
 *
 * Architecture note (honest scope): this is the in-app, fully testable form of
 * the multi-agent brain. Each advisor is a specialized evaluator with one
 * concern; the arbiter aggregates their votes to re-rank offers and attaches
 * the strongest advisor's reasoning to each offer as a visible "why". When the
 * LLM backend is deployed, model-driven advisors can occupy these same seats
 * without changing the surrounding engine — the council interface is the
 * stable contract. Until then, every vote here is a deterministic function of
 * state, so the deliberation is reproducible and test-verifiable.
 *
 * Voice rule: advisor notes push toward growth, never away from failure — no
 * guilt, neglect, or "falling behind" framing, ever.
 */

export interface AdvisorVote {
  advisor: string;
  /** Multiplicative adjustment around 1.0 (0.8 = discourage, 1.2 = champion). */
  weight: number;
  /** One plain sentence of reasoning, shown to the player. */
  note: string;
}

interface Advisor {
  name: string;
  vote: (offer: Quest, state: AppState, goal: Goal) => AdvisorVote | null;
}

/** Strategist: does this offer serve the goal's current arc phase? */
const strategist: Advisor = {
  name: "Strategist",
  vote: (offer, _state, goal) => {
    if (!goal.arc) return null;
    const aligned = offer.domain === goal.domain;
    return {
      advisor: "Strategist",
      weight: aligned ? 1.15 : 0.95,
      note: aligned
        ? `This moves your main goal forward through its ${goal.arc.currentPhase} phase.`
        : "A side path — useful range, even if it isn't your main goal."
    };
  }
};

/** Guardian: protects energy and recovery; champions rest when it's needed. */
const guardian: Advisor = {
  name: "Guardian",
  vote: (offer, state) => {
    const tired = state.host.energy < 0.35 || state.host.recoveryNeed > 0.65;
    if (!tired) return null;
    const restful = offer.domain === "recovery" || offer.mode === "recover";
    return {
      advisor: "Guardian",
      weight: restful ? 1.3 : 0.75,
      note: restful
        ? "Your reserves are low — this one rebuilds them, and that counts as progress."
        : "Heavier than your current energy. Doable, but the recovery option protects tomorrow."
    };
  }
};

/** Quartermaster: does the quest actually fit the time you said you have? */
const quartermaster: Advisor = {
  name: "Quartermaster",
  vote: (offer, state) => {
    const have = state.host.timeAvailableMinutes;
    const need = offer.timeLimitMinutes;
    if (need <= have * 0.9) return null; // comfortable fit — nothing to say
    return {
      advisor: "Quartermaster",
      weight: need > have ? 0.8 : 0.92,
      note: need > have
        ? "Tighter than the time you have — a shorter option may land better today."
        : "A close fit for your window. Start promptly and it works."
    };
  }
};

/** Chronicler: variety keeper — gently favors domains you haven't visited lately. */
const chronicler: Advisor = {
  name: "Chronicler",
  vote: (offer, state) => {
    const recent = state.questHistory.slice(0, 5);
    if (recent.length < 3) return null;
    const repeats = recent.filter((q) => q.domain === offer.domain).length;
    if (repeats >= 3) {
      return {
        advisor: "Chronicler",
        weight: 0.9,
        note: "You've been living in this domain lately — fresh ground would stretch you."
      };
    }
    if (repeats === 0) {
      return {
        advisor: "Chronicler",
        weight: 1.1,
        note: "Territory you haven't touched recently — variety builds the whole character."
      };
    }
    return null;
  }
};

/** Warden: champions the courage move when avoidance has been building. */
const warden: Advisor = {
  name: "Warden",
  vote: (offer, state) => {
    const avoidance = state.systemSignals.domainAvoidance[offer.domain] ?? 0;
    if (avoidance < 2) return null;
    return {
      advisor: "Warden",
      weight: 1.2,
      note: "This is the ground you've been circling. One small step here breaks its hold."
    };
  }
};

/** Scout: champions exploration when the host is standing in uncharted ground. */
const scout: Advisor = {
  name: "Scout",
  vote: (offer, state) => {
    if (!state.systemSignals.currentCellNovel || offer.domain !== "exploration") return null;
    return {
      advisor: "Scout",
      weight: 1.18,
      note: "You're in uncharted territory right now — a natural moment to explore."
    };
  }
};

const COUNCIL: Advisor[] = [strategist, guardian, quartermaster, chronicler, warden, scout];

export interface Deliberation {
  /** Offers re-ranked by council-adjusted score (pinned first item preserved). */
  offers: Quest[];
}

/**
 * Run the council over a set of offers. `pinnedFirst` preserves the event
 * director's special quest at index 0 (a detected pattern outranks general
 * deliberation). Every offer gains a councilNote from its strongest advisor.
 */
export function deliberate(offers: Quest[], state: AppState, goal: Goal, pinnedFirst: boolean): Deliberation {
  const judged = offers.map((offer, index) => {
    const votes = COUNCIL.map((a) => a.vote(offer, state, goal)).filter((v): v is AdvisorVote => v !== null);
    const adjusted = votes.reduce((score, v) => score * v.weight, offer.acceptanceScore);
    // The visible "why": the advisor whose vote moved furthest from neutral.
    const strongest = votes.sort((a, b) => Math.abs(b.weight - 1) - Math.abs(a.weight - 1))[0];
    const councilNote = strongest ? `${strongest.advisor}: ${strongest.note}` : undefined;
    return { offer: { ...offer, councilNote, councilScore: adjusted }, adjusted, index };
  });

  const head = pinnedFirst ? judged.slice(0, 1) : [];
  const tail = (pinnedFirst ? judged.slice(1) : judged).sort((a, b) => b.adjusted - a.adjusted);
  return { offers: [...head, ...tail].map((j) => j.offer) };
}

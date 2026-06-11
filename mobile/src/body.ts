import { AppState, BodyComposition } from "./types";

/**
 * Body-composition signals for quest enrichment.
 *
 * CRITICAL DESIGN LINE: this module derives only CAPABILITY / BEHAVIOR signals
 * (segmental balance, physical readiness, recovery context, activity continuity).
 * It deliberately does NOT derive or expose aesthetic / weight-target signals
 * (no "over target", no fat-loss delta, no thinness score). Body data informs
 * what BEHAVIOR quest fits; composition change is a downstream consequence of
 * behavior, never a goal the System pushes. The history SCREEN may display all
 * raw metrics equally, but the QUEST LOGIC reads only what this module exposes.
 */

export interface BodySignals {
  hasData: boolean;
  /** True if left/right limb balance is meaningfully off (corrective movement). */
  imbalanceDetected: boolean;
  imbalanceNote: string | null;
  /** Physical readiness proxy in 0..1 from capability metrics (not weight). */
  physicalReadiness: number | null;
  /** Muscle-mass trend across the two most recent snapshots: up/flat/down/null. */
  muscleTrend: "up" | "flat" | "down" | null;
}

const IMBALANCE_THRESHOLD = 8; // percentage points between paired limbs

export function latestBody(state: AppState): BodyComposition | null {
  return state.bodyHistory.length ? state.bodyHistory[0] : null;
}

export function deriveBodySignals(state: AppState): BodySignals {
  const latest = latestBody(state);
  if (!latest) {
    return { hasData: false, imbalanceDetected: false, imbalanceNote: null, physicalReadiness: null, muscleTrend: null };
  }

  // Segmental balance (capability): compare paired limbs where both present.
  let imbalanceDetected = false;
  let imbalanceNote: string | null = null;
  const arms = pair(latest.segmentLeftArmPct, latest.segmentRightArmPct);
  const legs = pair(latest.segmentLeftLegPct, latest.segmentRightLegPct);
  if (arms && Math.abs(arms[0] - arms[1]) >= IMBALANCE_THRESHOLD) {
    imbalanceDetected = true;
    imbalanceNote = "Upper-body left/right balance is uneven — a corrective movement option is available.";
  } else if (legs && Math.abs(legs[0] - legs[1]) >= IMBALANCE_THRESHOLD) {
    imbalanceDetected = true;
    imbalanceNote = "Lower-body left/right balance is uneven — a corrective movement option is available.";
  }

  // Physical readiness proxy from capability metrics only (SMI relative to a
  // typical range, plus presence of healthy skeletal muscle). Never uses weight
  // or fat as a "readiness" input. Clamped 0..1; null if no basis.
  let physicalReadiness: number | null = null;
  if (typeof latest.smi === "number") {
    // SMI ~5.5-8.5 spans a broad typical adult range; map into 0.4..0.9 softly.
    physicalReadiness = clamp01(0.4 + ((latest.smi - 5.5) / 3) * 0.5);
  }

  // Muscle-mass trend (behavioral progress, the encouraged direction).
  let muscleTrend: BodySignals["muscleTrend"] = null;
  if (state.bodyHistory.length >= 2) {
    const a = state.bodyHistory[0].muscleMassKg;
    const b = state.bodyHistory[1].muscleMassKg;
    if (typeof a === "number" && typeof b === "number") {
      const d = a - b;
      muscleTrend = Math.abs(d) < 0.3 ? "flat" : d > 0 ? "up" : "down";
    }
  }

  return { hasData: true, imbalanceDetected, imbalanceNote, physicalReadiness, muscleTrend };
}

function pair(a?: number, b?: number): [number, number] | null {
  return typeof a === "number" && typeof b === "number" ? [a, b] : null;
}

function clamp01(x: number): number {
  return Math.max(0, Math.min(1, Number.isFinite(x) ? x : 0));
}

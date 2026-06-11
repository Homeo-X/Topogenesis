/**
 * Lightweight motion-preference flag, set by the store and read by presentational
 * components' press-animation hook. A module-level flag (not context) avoids a
 * circular import (common.tsx <-> store) and keeps the buttons prop-free.
 * Defaults to motion enabled; the store syncs it from preferences.reduceMotion.
 */
let reduceMotion = false;

export function setReduceMotion(value: boolean): void {
  reduceMotion = value;
}

export function motionReduced(): boolean {
  return reduceMotion;
}

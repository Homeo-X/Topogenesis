/** NPC emotional affect computation. */

import type { NPCAffect, NPCNeeds } from '../types.js';

/** Update affect based on needs and recent event. */
export function updateAffect(
  affect: NPCAffect,
  needs: NPCNeeds,
  eventValence: number = 0,
  eventArousal: number = 0,
): NPCAffect {
  // Valence is driven by averaged needs minus deficits
  const needsValence =
    (needs.hunger + needs.rest + needs.safety + needs.belonging + needs.purpose) / 5;
  const targetValence = needsValence * 2 - 1; // map 0-1 to -1..1
  const newValence = clamp(
    lerp(affect.valence, targetValence, 0.2) + eventValence * 0.3,
    -1, 1,
  );

  // Arousal rises with hunger/safety threats and event arousal
  const threatArousal = Math.max(0, (1 - needs.hunger) * 0.5 + (1 - needs.safety) * 0.5);
  const newArousal = clamp(lerp(affect.arousal, threatArousal, 0.15) + eventArousal * 0.2, 0, 1);

  // Stress rises from persistent low needs
  const lowNeeds = Object.values(needs).filter(v => v < 0.3).length;
  const targetStress = lowNeeds / 7;
  const newStress = clamp(lerp(affect.stress, targetStress, 0.1), 0, 1);

  return {
    valence: newValence,
    arousal: newArousal,
    stress: newStress,
    trustBaseline: affect.trustBaseline,
  };
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

/** Affect modifier on action utility. High stress reduces long-term planning; high arousal boosts reactive actions. */
export function affectActionModifier(affect: NPCAffect, actionType: string): number {
  const isReactive = ['seek_food', 'respond_to_danger', 'seek_healer'].includes(actionType);
  const isLongTerm = ['train_skill', 'craft_item', 'migrate'].includes(actionType);

  let modifier = 1.0;
  if (isReactive) modifier += affect.arousal * 0.3 + affect.stress * 0.2;
  if (isLongTerm) modifier -= affect.stress * 0.3;
  if (affect.valence < -0.5) modifier -= 0.1; // depression reduces all action
  return Math.max(0.1, modifier);
}

/** Cross-reference validation for content pack integrity. */

import type { ContentPack } from './contentLoader.js';

export type ValidationResult = {
  valid: boolean;
  errors: string[];
  warnings: string[];
};

export function crossValidateContent(pack: ContentPack): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  const regionIds = new Set(pack.regions.map(r => r.id));
  const factionIds = new Set(pack.factions.map(f => f.id));
  const peopleIds = new Set(pack.peoples.map(p => p.id));
  const jobIds = new Set(pack.jobs.map(j => j.id));
  const itemIds = new Set(pack.items.map(i => i.id));

  // People must reference valid regions
  for (const people of pack.peoples) {
    if (!regionIds.has(people.region)) {
      errors.push(`People '${people.id}' references unknown region '${people.region}'`);
    }
  }

  // Recipes must reference valid items
  for (const recipe of pack.recipes) {
    if (!itemIds.has(recipe.outputItemId)) {
      errors.push(`Recipe '${recipe.id}' produces unknown item '${recipe.outputItemId}'`);
    }
    for (const ing of recipe.ingredients) {
      if (!itemIds.has(ing.itemId)) {
        errors.push(`Recipe '${recipe.id}' requires unknown item '${ing.itemId}'`);
      }
    }
  }

  // NPC archetypes must reference valid jobs
  for (const arch of pack.npcArchetypes) {
    for (const job of arch.possibleJobs) {
      if (!jobIds.has(job)) {
        warnings.push(`NPC archetype '${arch.id}' references unregistered job '${job}'`);
      }
    }
    if (!peopleIds.has(arch.people)) {
      errors.push(`NPC archetype '${arch.id}' references unknown people '${arch.people}'`);
    }
  }

  // Asset manifest checks
  const categories = new Set(['character', 'clothing', 'building', 'environment', 'prop', 'icon', 'vfx', 'audio', 'ui']);
  const mvpCount = { character: 0, clothing: 0, building: 0, prop: 0, icon: 0, vfx: 0 };
  for (const asset of pack.assetManifest) {
    const cat = asset.category as keyof typeof mvpCount;
    if (cat in mvpCount) mvpCount[cat]++;
  }
  if (mvpCount.character < 12) warnings.push(`Asset manifest has only ${mvpCount.character} character entries (target: 12)`);
  if (mvpCount.clothing < 8) warnings.push(`Asset manifest has only ${mvpCount.clothing} clothing entries (target: 8)`);
  if (mvpCount.building < 8) warnings.push(`Asset manifest has only ${mvpCount.building} building entries (target: 8)`);

  return { valid: errors.length === 0, errors, warnings };
}

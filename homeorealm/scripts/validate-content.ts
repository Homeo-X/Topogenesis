#!/usr/bin/env tsx
/**
 * Content validation script — validates all JSON content files against schemas
 * and runs cross-validation checks.
 */
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { loadContentPack } from '../packages/sim-core/src/content/contentLoader.js';
import { crossValidateContent } from '../packages/sim-core/src/content/validators.js';

const __dir = dirname(fileURLToPath(import.meta.url));
const CONTENT_DIR = join(__dir, '../packages/content');

async function main() {
  console.log('HomeoRealm Content Validation');
  console.log('═'.repeat(50));
  console.log(`Content path: ${CONTENT_DIR}\n`);

  let pack: ReturnType<typeof loadContentPack>;
  try {
    pack = loadContentPack(CONTENT_DIR);
    console.log('✓ Schema validation passed');
    console.log(`  Regions:        ${pack.regions.length}`);
    console.log(`  Factions:       ${pack.factions.length}`);
    console.log(`  Peoples:        ${pack.peoples.length}`);
    console.log(`  Jobs:           ${pack.jobs.length}`);
    console.log(`  Items:          ${pack.items.length}`);
    console.log(`  Recipes:        ${pack.recipes.length}`);
    console.log(`  NPC Archetypes: ${pack.npcArchetypes.length}`);
    console.log(`  Quest Templates:${pack.questTemplates.length}`);
    console.log(`  Assets:         ${pack.assetManifest.length}`);
  } catch (err) {
    console.error('✗ Schema validation FAILED:', err instanceof Error ? err.message : err);
    process.exit(1);
  }

  const result = crossValidateContent(pack);
  console.log('\n── Cross-validation ──────────────────────────────');
  if (result.errors.length === 0) {
    console.log('✓ Cross-validation passed — all references are consistent');
  } else {
    console.log(`✗ Cross-validation found ${result.errors.length} error(s):`);
    for (const e of result.errors) console.log(`  • ${e}`);
    process.exit(1);
  }

  if (result.warnings.length > 0) {
    console.log(`\n⚠ Warnings (${result.warnings.length}):`);
    for (const w of result.warnings) console.log(`  • ${w}`);
  }

  console.log('\n✓ Content pack is valid and ready for use.');
}

main().catch(err => { console.error(err); process.exit(1); });

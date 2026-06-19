/** Content loader and schema validation. */

import { readFileSync } from 'fs';
import { join } from 'path';
import { z } from 'zod';

// ─── Schemas ──────────────────────────────────────────────────────────────

export const RegionSchema = z.object({
  id: z.string(),
  name: z.string(),
  description: z.string(),
  climate: z.string(),
  mainResources: z.array(z.string()),
  faction: z.string().optional(),
});

export const FactionSchema = z.object({
  id: z.string(),
  name: z.string(),
  description: z.string(),
  function: z.string(),
  allegiances: z.array(z.string()),
});

export const PeopleSchema = z.object({
  id: z.string(),
  name: z.string(),
  region: z.string(),
  description: z.string(),
  assetTheme: z.string(),
  culturalTraits: z.array(z.string()),
});

export const JobSchema = z.object({
  id: z.string(),
  name: z.string(),
  description: z.string(),
  primarySkill: z.string(),
  produces: z.array(z.string()),
  settlementNeed: z.enum(['essential', 'important', 'optional']),
});

export const ItemSchema = z.object({
  id: z.string(),
  name: z.string(),
  category: z.enum(['food', 'tool', 'material', 'clothing', 'medicine', 'artifact', 'ritual']),
  description: z.string(),
  baseValue: z.number(),
  weight: z.number().optional(),
  stackable: z.boolean().optional(),
});

export const RecipeSchema = z.object({
  id: z.string(),
  name: z.string(),
  outputItemId: z.string(),
  ingredients: z.array(z.object({ itemId: z.string(), quantity: z.number() })),
  requiredSkill: z.string(),
  minSkillLevel: z.number(),
  outputQuantity: z.number().optional(),
});

export const NPCArchetypeSchema = z.object({
  id: z.string(),
  people: z.string(),
  possibleJobs: z.array(z.string()),
  traitBias: z.record(z.number()),
  skillBias: z.record(z.number()),
  namePool: z.array(z.string()),
});

export const QuestTemplateSchema = z.object({
  id: z.string(),
  cause: z.string(),
  titleTemplate: z.string(),
  descriptionTemplate: z.string(),
  tags: z.array(z.string()),
  urgencyBase: z.number(),
  difficultyBase: z.number(),
});

export const AssetManifestEntrySchema = z.object({
  id: z.string(),
  category: z.enum(['character', 'clothing', 'building', 'environment', 'prop', 'icon', 'vfx', 'audio', 'ui']),
  name: z.string(),
  region: z.string().optional(),
  faction: z.string().optional(),
  description: z.string(),
  gameplayUse: z.string(),
  visualKeywords: z.array(z.string()),
  productionPriority: z.enum(['mvp', 'vertical-slice', 'alpha', 'later']),
});

export type Region = z.infer<typeof RegionSchema>;
export type Faction = z.infer<typeof FactionSchema>;
export type People = z.infer<typeof PeopleSchema>;
export type Job = z.infer<typeof JobSchema>;
export type Item = z.infer<typeof ItemSchema>;
export type Recipe = z.infer<typeof RecipeSchema>;
export type ContentNPCArchetype = z.infer<typeof NPCArchetypeSchema>;
export type ContentQuestTemplate = z.infer<typeof QuestTemplateSchema>;
export type AssetManifestEntry = z.infer<typeof AssetManifestEntrySchema>;

export type ContentPack = {
  regions: Region[];
  factions: Faction[];
  peoples: People[];
  jobs: Job[];
  items: Item[];
  recipes: Recipe[];
  npcArchetypes: ContentNPCArchetype[];
  questTemplates: ContentQuestTemplate[];
  assetManifest: AssetManifestEntry[];
};

function loadJSON(contentDir: string, file: string): unknown {
  const path = join(contentDir, file);
  return JSON.parse(readFileSync(path, 'utf-8'));
}

export function loadContentPack(contentDir: string): ContentPack {
  const regions = z.array(RegionSchema).parse(loadJSON(contentDir, 'regions.json'));
  const factions = z.array(FactionSchema).parse(loadJSON(contentDir, 'factions.json'));
  const peoples = z.array(PeopleSchema).parse(loadJSON(contentDir, 'peoples.json'));
  const jobs = z.array(JobSchema).parse(loadJSON(contentDir, 'jobs.json'));
  const items = z.array(ItemSchema).parse(loadJSON(contentDir, 'items.json'));
  const recipes = z.array(RecipeSchema).parse(loadJSON(contentDir, 'recipes.json'));
  const npcArchetypes = z.array(NPCArchetypeSchema).parse(loadJSON(contentDir, 'npc_archetypes.json'));
  const questTemplates = z.array(QuestTemplateSchema).parse(loadJSON(contentDir, 'quest_templates.json'));
  const assetManifest = z.array(AssetManifestEntrySchema).parse(loadJSON(contentDir, 'asset_manifest.json'));

  return { regions, factions, peoples, jobs, items, recipes, npcArchetypes, questTemplates, assetManifest };
}

export function validateContentPack(contentDir: string): { valid: boolean; errors: string[] } {
  const errors: string[] = [];
  try {
    loadContentPack(contentDir);
  } catch (err) {
    errors.push(String(err));
  }
  return { valid: errors.length === 0, errors };
}

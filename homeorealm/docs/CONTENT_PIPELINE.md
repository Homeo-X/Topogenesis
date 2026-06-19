# HomeoRealm — Content Pipeline

## Overview

All world lore lives in `content/` as JSON files validated against Zod schemas. The `sim-core` package loads and cross-validates these files at startup. Adding new content means adding entries to the appropriate JSON file and running the validation script.

## Content Directory Layout

```
content/
├── species.json      # Peoples / species (Valari, Threnosi, …)
├── cultures.json     # Cultural groups and their traits
├── classes.json      # Professions and job archetypes
├── biomes.json       # Regions and environmental zones
├── events.json       # Scheduled and random world events
├── quests.json       # Quest templates (overridden by emergent generation)
├── factions.json     # Political and social factions
├── dungeons.json     # Static dungeon templates
├── items.json        # Item / asset manifest (67 entries)
└── index.ts          # ContentPack loader (loadContentPack)
```

## Validation

Run the validation script to check all content for schema compliance and cross-references:

```bash
npx tsx scripts/validate-content.ts
```

This calls `loadContentPack(contentDir)` then `crossValidateContent(pack)` to verify:
- All required fields are present and typed correctly
- Cross-references are consistent (e.g. a faction's homeRegionId exists in biomes)
- No duplicate IDs within a category

A clean run prints a green summary. Any error line includes the filename and the failing entry ID.

## Adding a New Species / People

Edit `content/species.json`:

```json
{
  "id": "velorath",
  "name": "Velorath",
  "description": "Long-lived desert nomads who navigate by magnetic lodestones.",
  "assetTheme": "sand, amber, geometric patterns",
  "culturalTraits": ["patient", "territorial", "star-reader"],
  "startingNeeds": {
    "safety": 0.7,
    "stimulation": 0.8
  },
  "skillBonuses": {
    "navigation": 0.2,
    "survival": 0.15
  }
}
```

Then run the validation script. The new people will appear in the Lore Codex under "Peoples" and will be eligible for NPC generation in future world seeds.

## Adding a New Faction

Edit `content/factions.json`:

```json
{
  "id": "the-amber-compact",
  "name": "The Amber Compact",
  "description": "A merchant consortium that controls the resin trade across three continents.",
  "function": "economic",
  "homeRegionId": "ashwood-basin",
  "reputation": 0.6,
  "tags": ["trade", "resin", "wealth"]
}
```

The `homeRegionId` must match an `id` in `biomes.json` — the cross-validator enforces this.

## Adding a New Region / Biome

Edit `content/biomes.json`:

```json
{
  "id": "ashwood-basin",
  "name": "Ashwood Basin",
  "description": "A fertile flood plain edged by ancient ash forests.",
  "climate": "temperate",
  "mainResources": ["wood", "herbs", "food"],
  "pressureSusceptibility": {
    "drought": 0.4,
    "disease": 0.2
  }
}
```

## Adding Quest Templates

Quest templates in `content/quests.json` serve as fallback archetypes. The emergent quest generator (which creates quests from live settlement pressures) takes precedence, but templates are used when no pressure-derived quest is active for a given pressure type.

```json
{
  "id": "amber-trade-disruption",
  "title": "The Amber Routes are Closed",
  "description": "Word has come that the southern passes are held by bandits.",
  "cause": "economic",
  "urgency": 0.6,
  "difficulty": 0.5,
  "objectives": [
    { "type": "investigate", "description": "Find the source of the disruption", "completed": false },
    { "type": "resolve", "description": "Reopen at least one route", "completed": false }
  ],
  "tags": ["trade", "bandits", "travel"],
  "isActive": true
}
```

## Adding Items to the Asset Manifest

Items in `content/items.json` feed the `RelicForge` asset-generation pipeline and appear in the Lore Codex under "Assets":

```json
{
  "id": "lodestone-compass",
  "category": "tool",
  "name": "Lodestone Compass",
  "description": "A magnetic compass carved from volcanic lodestone, used by Velorath navigators.",
  "gameplayUse": "navigation, desert travel, quest item",
  "visualKeywords": ["dark stone", "amber inlay", "magnetic needle", "geometric"],
  "productionPriority": "medium"
}
```

`productionPriority` values: `"critical"`, `"high"`, `"medium"`, `"low"`. Critical items are flagged in the asset pipeline for immediate art production.

## Content Pack API

```ts
import { loadContentPack, crossValidateContent } from '@homeorealm/sim-core';

const pack = loadContentPack('./content');
const result = crossValidateContent(pack);

if (!result.ok) {
  result.errors.forEach(e => console.error(e));
  process.exit(1);
}
```

`ContentPack` type:
```ts
interface ContentPack {
  species: People[];
  cultures: Culture[];
  classes: JobClass[];
  biomes: Region[];
  events: WorldEventTemplate[];
  quests: QuestTemplate[];
  factions: Faction[];
  dungeons: DungeonTemplate[];
  items: Asset[];
}
```

## Lore Codex Integration

The web client's Lore Codex view (`/lore`) fetches content via:

- `GET /api/lore/regions` → `Region[]`
- `GET /api/lore/factions` → `Faction[]`
- `GET /api/lore/peoples` → `People[]`
- `GET /api/lore/assets` → `Asset[]`

New entries appear automatically after restarting the server — no client changes required.

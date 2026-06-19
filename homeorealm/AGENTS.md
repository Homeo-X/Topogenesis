# HomeoRealm Online — AGENTS.md

## Project Overview

HomeoRealm Online is a TypeScript MMORPG life-simulation engine featuring the Homeo-X Topogenesis NPC Civilization. The repository lives at `homeorealm/` within the Topogenesis monorepo.

## Repository Structure

```
homeorealm/
├── packages/
│   ├── sim-core/      # TypeScript simulation engine
│   ├── server/        # Fastify REST API server
│   ├── web-client/    # React + Vite dashboard
│   └── content/       # JSON game content (regions, NPCs, quests…)
├── scripts/
│   ├── validate-content.ts   # Content schema & cross-reference checks
│   └── run-demo-simulation.ts # 30-day deterministic demo run
└── AGENTS.md
```

## Development Branch

All HomeoRealm work goes on branch: `claude/homeorealm-mmorpg-build-8gjv55`

## Quick Start

```bash
cd homeorealm
npm install
npm test                  # 52 tests across sim-core, server, web-client
npm run validate:content  # content schema + cross-reference
npm run demo:sim          # 30-day demo with seed 12345
npm run dev               # starts server (3001) + client (5173)
```

## Architecture

### sim-core (`packages/sim-core/`)

The simulation engine. Deterministic — all randomness flows through a seeded `SeededRng` (mulberry32).

Key modules:
- `src/rng.ts` — SeededRng with `fork(salt)` for deterministic sub-streams
- `src/types.ts` — all shared types (WorldState, NPCState, etc.)
- `src/events.ts` — append-only EventStore
- `src/npc/` — needs decay, affect update, memory, relationships, viability scoring, action execution, frame distinction learning
- `src/settlement/` — resource economy, household management, job definitions
- `src/quests/questGenerator.ts` — emergent quest generation from settlement pressures
- `src/dungeon/dungeonTopogenesis.ts` — dungeon room generation from pressure vectors
- `src/simulation/tick.ts` — main day tick (deterministic, fully self-contained)
- `src/simulation/initializeWorld.ts` — creates `Vennholt` settlement with 20 NPCs
- `src/content/` — Zod schema validation + cross-reference checks

### server (`packages/server/`)

Fastify 4.x REST API on port 3001. State is in-memory with optional JSON snapshot persistence.

Routes:
- `GET /api/world` — world summary (day, settlements, NPC count, quests)
- `POST /api/world/tick` — advance one day
- `POST /api/world/run` — advance N days `{ days: number }`
- `POST /api/world/reset` — reinitialize `{ seed: number, npcCount?: number }`
- `GET /api/npcs` — NPC list (filterable by job/household)
- `GET /api/npcs/:id` — full NPC profile with frame distinctions
- `GET /api/npcs/:id/relationships` — relationship graph
- `GET /api/npcs/:id/memories` — memory traces
- `GET /api/settlements` — settlement economy data
- `GET /api/households` — household list
- `GET /api/dungeons` — active dungeon rooms
- `GET /api/quests` — emergent quest board
- `GET /api/lore/regions|factions|peoples|assets|validate` — content data

### web-client (`packages/web-client/`)

React 18 + Vite dashboard, proxied to server on dev. Views:
- **Dashboard** — world resources, stability bars, simulation controls
- **NPCs** — searchable list + 4-tab detail (overview/relationships/memories/frame)
- **Quests** — quest board with urgency grouping and detail modals
- **Settlement** — resource table with pressure tags
- **Households** — household grid with food shortage warnings
- **Events** — simulation controls + high-salience event log
- **Lore Codex** — regions, factions, peoples, asset manifest with filtering
- **Dungeons** — topogenesis dungeon rooms by pressure source

## Homeo-X Topogenesis Loop

Every NPC tick (once per day):
1. Decay needs (`needs.ts`)
2. Consume household food (`needs.ts`)
3. Update affect (valence/arousal/stress) (`affect.ts`)
4. Decay memories with salience protection (`memory.ts`)
5. Drift inactive relationships (`relationships.ts`)
6. Score candidate actions with need/frame/affect modifiers (`viability.ts`)
7. Select intention deterministically (`viability.ts`)
8. Execute action → emit events, update resources, create memory (`policy.ts`)
9. Apply frame distinction learning from recent memory clusters (`schedules.ts`)
10. Health crisis / death check

Frame distinctions (topogenesis): NPCs form new operational categories when 3+ similar events cluster in their memory. These distinctions modify future action scoring.

## Content Pipeline

Content lives in `packages/content/*.json`. All files are validated by Zod schemas in `sim-core/src/content/contentLoader.ts`. Cross-references (people→region, archetype→people/job, recipe→items) are checked by `validators.ts`.

Run `npm run validate:content` after any content edit.

## Adding Content

1. Edit the relevant JSON in `packages/content/`
2. Run `npm run validate:content` — it will catch schema and reference errors
3. If adding new archetypes, update `DEMO_ARCHETYPES` in `initializeWorld.ts` if needed
4. If adding new quest causes, update `PRESSURE_TO_CAUSE` in `questGenerator.ts`

## Testing

- `packages/sim-core/tests/` — 36 tests: determinism, economy, events, memory, NPC viability, quests, relationships
- `packages/server/tests/` — 13 API integration tests (Fastify inject)
- `packages/web-client/tests/` — 3 React smoke tests (jsdom + testing-library)

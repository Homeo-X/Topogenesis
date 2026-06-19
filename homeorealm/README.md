# HomeoRealm Online

**Auralis World Console** — a TypeScript MMORPG life-simulation kingdom powered by the Homeo-X Topogenesis NPC Civilization engine.

## What is this?

HomeoRealm simulates a living medieval-fantasy world where every NPC thinks, remembers, forms relationships, and makes viability-driven decisions without player intervention. Quests emerge from economic pressures, dungeons crystallise from social tension, and each NPC builds a personal **frame of distinctions** — learned heuristics that shape future decisions.

## Quick Start

```bash
npm install              # installs all workspace packages
npm run dev              # starts web client (Vite, port 5173) + server (Fastify, port 3001)
```

Open `http://localhost:5173` to view the Auralis World Console.

## Monorepo Structure

```
homeorealm/
├── packages/
│   ├── sim-core/        # Simulation engine — NPC AI, economy, quests, dungeons
│   ├── server/          # Fastify REST API
│   └── web-client/      # React 18 + Vite PWA console
├── content/             # JSON lore packs (species, cultures, factions, …)
├── scripts/             # validate-content.ts, run-demo-simulation.ts
├── docs/                # Architecture, design, and roadmap documentation
└── AGENTS.md            # Agent/CI guidance
```

## Key Commands

| Command | Description |
|---|---|
| `npm run dev` | Start all packages in watch mode |
| `npm run build` | Production build |
| `npm test` | Run all 52 tests (Vitest) |
| `npx tsx scripts/run-demo-simulation.ts` | Run a 30-day headless simulation |
| `npx tsx scripts/validate-content.ts` | Validate all content JSON files |

## Architecture Overview

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full system design.

## Game Design

See [`docs/GAME_DESIGN_IMPLEMENTATION.md`](docs/GAME_DESIGN_IMPLEMENTATION.md) for NPC AI, economy, and quest generation details.

## Content Pipeline

See [`docs/CONTENT_PIPELINE.md`](docs/CONTENT_PIPELINE.md) for adding new lore, species, factions, and assets.

## Roadmap

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for planned features and milestones.

## Tech Stack

- **Simulation**: TypeScript, pure functional tick loop, mulberry32 seeded PRNG
- **API**: Fastify 4, `@fastify/compress` (gzip/brotli), Zod validation
- **Web Client**: React 18, Vite 5, React.lazy code-splitting, PWA manifest
- **Tests**: Vitest, `@testing-library/react`

## Dev Branch

Active development: `claude/homeorealm-mmorpg-build-8gjv55`

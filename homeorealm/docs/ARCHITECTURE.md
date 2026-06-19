# HomeoRealm — Architecture

## Overview

HomeoRealm is an npm workspaces monorepo with three packages: `sim-core` (engine), `server` (API), and `web-client` (UI). The simulation runs entirely in-memory as a pure functional tick loop; all state is serialisable JSON.

```
┌────────────────────────────────────────┐
│              web-client                │  React 18 + Vite PWA
│  React.lazy chunks · PWA manifest      │
│  CSS custom properties · mobile-first  │
└───────────────────┬────────────────────┘
                    │ HTTP /api/*
┌───────────────────▼────────────────────┐
│               server                   │  Fastify 4
│  worldRoutes · npcRoutes · questRoutes  │
│  settlementRoutes · simulationRoutes    │
│  @fastify/compress (gzip/brotli)        │
└───────────────────┬────────────────────┘
                    │ TypeScript imports
┌───────────────────▼────────────────────┐
│              sim-core                  │  Pure TypeScript
│  initializeWorld · tickDay             │
│  NPC AI · Economy · Quests · Dungeons  │
│  EventStore · ContentPack              │
└────────────────────────────────────────┘
```

## Package Responsibilities

### `@homeorealm/sim-core`

The simulation engine. Exposes:

- `initializeWorld(seed, npcCount)` → `{ world: WorldState; eventStore: EventStore }`
- `tickDay(world, eventStore, opts?)` → `WorldState`
- `loadContentPack(dir)` → `ContentPack`
- `crossValidateContent(pack)` → `ValidationResult`
- All TypeScript types: `WorldState`, `NPCState`, `SettlementState`, etc.

**WorldState** is fully serialisable:
```ts
interface WorldState {
  day: number;
  seed: number;
  npcs: Record<string, NPCState>;
  settlements: Record<string, SettlementState>;
  households: Record<string, HouseholdState>;
  quests: Record<string, QuestState>;
  dungeons: Record<string, DungeonRoom>;
  events: WorldEvent[];
}
```

### `@homeorealm/server`

Fastify REST API, stateful singleton of `WorldState` + `EventStore`. All mutation goes through `tickDay`. Routes:

| Route | Method | Description |
|---|---|---|
| `/api/health` | GET | Health check |
| `/api/world` | GET | World summary |
| `/api/world/full` | GET | Full world state |
| `/api/world/tick` | POST | Advance one day |
| `/api/world/run` | POST | Advance N days |
| `/api/world/reset` | POST | Re-initialise world |
| `/api/world/events` | GET | Event log |
| `/api/world/events/high-salience` | GET | Events with salience ≥ 0.6 |
| `/api/npcs` | GET | NPC list (filterable by job, household) |
| `/api/npcs/:id` | GET | Full NPC biography |
| `/api/npcs/:id/relationships` | GET | Relationship list |
| `/api/npcs/:id/memories` | GET | Memory list |
| `/api/npcs/ranking/top` | GET | Highest viability NPCs |
| `/api/npcs/ranking/bottom` | GET | Lowest viability NPCs |
| `/api/settlements` | GET | All settlements |
| `/api/settlements/:id` | GET | Settlement detail |
| `/api/households` | GET | All households |
| `/api/households/:id` | GET | Household detail |
| `/api/dungeons` | GET | All dungeon rooms |
| `/api/quests` | GET | All quests |
| `/api/lore/regions` | GET | Lore: regions |
| `/api/lore/factions` | GET | Lore: factions |
| `/api/lore/peoples` | GET | Lore: species/peoples |
| `/api/lore/assets` | GET | Asset manifest |

### `@homeorealm/web-client`

React 18 SPA served by Vite. Each view is a `React.lazy` chunk for code splitting.

| View | Component | Route key |
|---|---|---|
| World Dashboard | `WorldDashboard` | `dashboard` |
| Residents | `NPCList` + `NPCDetail` | `npcs` |
| Quests | `QuestBoard` | `quests` |
| Settlement Economy | `SettlementEconomy` | `settlement` |
| Households | `HouseholdPanel` | `households` |
| Event Log | `SimulationControls` | `events` |
| Lore Codex | `LoreCodex` | `lore` |
| Dungeons | `DungeonView` | `dungeons` |

## Data Flow

```
User action (tick / run days)
  → React component calls api.ts
    → fetch POST /api/world/tick
      → Fastify route handler
        → tickDay(world, eventStore)
          → NPC AI loops, economy update, quest generation
          → returns new WorldState
        → server stores new state
      → response: { day, summary }
    → React re-renders dashboard
```

## Randomness

All randomness uses `mulberry32` seeded PRNG with `fork(salt)` to create per-NPC independent sub-streams. The world seed is stored in `WorldState.seed`, making simulations fully deterministic and reproducible.

## Event Sourcing

`EventStore` is append-only. Every meaningful state change produces a `WorldEvent`:

```ts
interface WorldEvent {
  id: string;
  day: number;
  tick: number;
  type: string;
  actorId?: string;
  settlementId?: string;
  payload: Record<string, unknown>;
  tags: string[];
  salience: number;  // 0–1, higher = more significant
}
```

Events are never mutated or deleted. The web client surfaces high-salience events (≥ 0.6) in the event log.

## Build Output

Vite produces 11 JS chunks (per-component lazy bundles + react-vendor):

```
react-vendor.js    ~45 KB gzip   (React + ReactDOM)
NPCProfile.js      ~1.9 KB gzip
index.js           ~2.5 KB gzip  (App + Layout)
SimulationControls ~1.1 KB gzip
... (all remaining views < 1 KB gzip each)
```

Total transfer for first load: ~55 KB gzip. Subsequent view navigations load ≤2 KB each.

## Mobile Architecture

- `useIsMobile()` hook using `window.matchMedia('(max-width: 640px)')` with live listener
- Hamburger drawer nav on mobile, horizontal nav on desktop/tablet
- NPC detail panel renders as full-screen overlay on mobile (`mobile-detail-open` CSS class)
- All interactive elements have `min-height: 44px` touch targets
- `touch-action: manipulation` on all buttons to suppress 300ms tap delay
- PWA manifest with `"display": "standalone"` and `"orientation": "any"`

# HomeoRealm — Roadmap

## Current State: MVP (v0.1)

The MVP delivers a fully functional single-settlement simulation with:

- 52 passing tests (36 engine, 13 API, 3 web-client)
- Complete Homeo-X Topogenesis NPC loop
- Web console with 8 views, code-split, PWA manifest
- Mobile-optimised UI (hamburger nav, touch targets, overlay panels)
- Deterministic seeded simulation
- Content validation pipeline
- Headless demo script (30-day run)

---

## Phase 1 — World Expansion (v0.2)

**Target**: Multiple settlements, inter-settlement trade, migration

- [ ] Add 2 additional settlement templates to content
- [ ] Trade caravans: NPCs with `merchant` job path between settlements
- [ ] Migration: NPCs below 0.25 viability may relocate to higher-opportunity settlements
- [ ] Inter-settlement diplomacy: faction alignment affects trade rates
- [ ] Map view in web console showing settlements and caravan routes
- [ ] API: `GET /api/world/map` returning settlement positions and connections

**Acceptance criteria**: Demo script shows NPC migration events; trade events logged with salience ≥ 0.5.

---

## Phase 2 — Player Presence (v0.3)

**Target**: Player character can enter and affect the world

- [ ] Player NPC type: same state vector as NPCs, controlled via API actions
- [ ] Action API: `POST /api/player/action` with intent + target
- [ ] Reputation system: player actions alter faction standing
- [ ] Player inventory: subset of asset manifest items, carried and consumed
- [ ] Quest acceptance: player can claim active quests via API
- [ ] Web console: player status panel, action queue UI

**Acceptance criteria**: Player character persists across sessions, reputation updates are reflected in NPC relationships.

---

## Phase 3 — Persistence Layer (v0.4)

**Target**: Replace in-memory state with durable storage

- [ ] PostgreSQL schema for WorldState (JSONB columns per entity type)
- [ ] Redis for hot read cache (world summary, NPC rankings)
- [ ] Append-only EventStore persisted to PostgreSQL
- [ ] World snapshots: daily checkpoint + restore
- [ ] Migration scripts with up/down
- [ ] `@homeorealm/db` package in monorepo

**Acceptance criteria**: Server restart restores world state from DB; event history survives restart.

---

## Phase 4 — MMO Sharding (v0.5)

**Target**: Multiple concurrent players across sharded world instances

- [ ] WebSocket gateway: `@homeorealm/gateway` package
- [ ] Shard coordinator: assigns players to world shards by settlement
- [ ] Cross-shard events: significant events broadcast to all shards
- [ ] Presence service: track online players per shard
- [ ] Session tokens: JWT-based auth
- [ ] Web console: real-time event feed via WebSocket subscription

**Acceptance criteria**: 50 concurrent WebSocket connections; tick latency < 100ms per shard.

---

## Phase 5 — LLM Dialogue Layer (v0.6)

**Target**: NPC speech generated from their state vector

- [ ] Dialogue prompt builder: serialises NPC needs, memories, relationships, frame distinctions → system prompt
- [ ] Claude API integration: `@homeorealm/dialogue` package
- [ ] Dialogue triggers: player proximity, quest events, high-salience moments
- [ ] Dialogue history: stored as memories with salience scoring
- [ ] Web console: chat panel appears when player is adjacent to NPC
- [ ] Rate limiting: max 3 LLM calls per NPC per day

**Acceptance criteria**: NPC speech references actual memories and current needs; dialogue memories affect future NPC decisions.

---

## Phase 6 — Combat & Danger (v0.7)

**Target**: Dungeon exploration with combat mechanics

- [ ] Combat action type: `engage`, `flee`, `negotiate`, `ambush`
- [ ] Health consequences: injury → health decay → recovery time
- [ ] Dungeon navigation: room graph traversal
- [ ] Loot system: items from asset manifest dropped on dungeon completion
- [ ] NPC death: proper handling of kin grief, inheritance, household dissolution
- [ ] Web console: dungeon map view with room graph

**Acceptance criteria**: NPCs in dungeons take realistic damage; household inheritance resolves correctly.

---

## Phase 7 — Marriage, Children & Lineage (v0.8)

**Target**: Generational simulation

- [ ] Marriage action: two NPCs with high mutual affection + trust form household bond
- [ ] Child NPC generation: genetics-weighted from parents' traits
- [ ] Inheritance: household wealth and reputation pass to children
- [ ] Lineage tree: multi-generation graph stored in NPCState
- [ ] Web console: lineage tree visualisation component
- [ ] Legacy quests: descendants can inherit unresolved ancestor quests

**Acceptance criteria**: 100-day simulation shows 3+ generations; lineage tree renders correctly.

---

## Technical Debt & Ongoing

- [ ] E2E tests with Playwright (web console smoke tests)
- [ ] GitHub Actions CI pipeline
- [ ] OpenAPI spec auto-generated from Fastify routes
- [ ] `CLAUDE.md` with Claude Code hints for common tasks
- [ ] Performance profiling: tick loop should stay < 50ms for 500 NPCs
- [ ] Bundle size budget enforcement in CI (react-vendor ≤ 50KB gzip)
- [ ] Accessibility audit (WCAG 2.1 AA)

---

## MVP Scope Exclusions (Deliberately Deferred)

The following are explicitly **out of scope** for v0.1 and will be addressed in later phases:

- Multiple settlements or continents
- Full MMO sharding or WebSockets
- Combat system beyond basic action types
- Marriage and children (data structures exist, generation deferred)
- PostgreSQL or Redis persistence
- LLM dialogue layer
- RelationshipGraph force-directed layout (static radial SVG implemented)
- Mobile app wrappers (Capacitor/React Native)

# HomeoRealm — Game Design & Implementation

## The Homeo-X Topogenesis Loop

Every NPC executes a viability-maximisation loop each game day. The loop runs in 8 phases:

```
1. Needs Assessment     → compute current need deficits
2. Affect Modulation    → emotions shift based on needs, memories, events
3. Memory Retrieval     → surface relevant memories by salience + tags
4. Relationship Weighting → weight actions by how they affect kin/allies/rivals
5. Viability Scoring    → score every candidate action
6. Action Selection     → pick highest-scoring viable action
7. State Mutation       → apply action effects to WorldState
8. Frame Learning       → update distinctions if outcome was surprising
```

### Needs Vector

Each NPC has a needs vector with values in `[0, 1]` (1 = fully satisfied):

| Need | Description |
|---|---|
| `food` | Caloric intake |
| `water` | Hydration |
| `rest` | Sleep and recovery |
| `safety` | Freedom from threat |
| `belonging` | Social connection |
| `esteem` | Respect and recognition |
| `purpose` | Meaningful contribution |
| `stimulation` | Novel experience |

Needs decay each tick. When a need falls below 0.3 it becomes a **pressure** that pushes the NPC toward satisfying actions.

### Affect Vector

Emotions are continuous variables, not states:

| Affect | Range | Description |
|---|---|---|
| `valence` | −1 … +1 | Overall positive/negative mood |
| `arousal` | 0 … 1 | Energy level |
| `tension` | 0 … 1 | Stress / anxiety |
| `hope` | 0 … 1 | Optimism about future outcomes |
| `grief` | 0 … 1 | Loss processing |

Affect colours memory encoding (high-arousal events are encoded with higher salience) and modulates action scoring (high tension penalises social actions).

### Memory Model

Memories are episodic records of past events:

```ts
interface Memory {
  id: string;
  day: number;
  eventType: string;
  salience: number;        // 0–1, affects retrieval priority
  emotionalValence: number; // −1 … +1
  participants: string[];  // NPC IDs involved
  tags: string[];
  summary: string;
  decay: number;           // 0–1, memories fade over time
}
```

Retrieval is tag-matching + salience-weighted. Memories with `decay < 0.05` are pruned. Traumatic memories (high salience + negative valence) decay slower.

### Frame Distinctions

Frame distinctions are learned heuristics — categories of experience that the NPC has found predictive or important:

```ts
interface FrameDistinction {
  id: string;
  label: string;        // e.g. "dangerous_foraging", "trustworthy_smith"
  domain: string;       // e.g. "resource", "social", "spatial"
  confidence: number;   // 0–1, increases with repeated confirmation
  utilityWeight: number; // influence on action scoring
  createdOnDay: number;
}
```

A distinction forms when the same situation type produces a surprising outcome more than twice. It then biases the NPC's attention (they notice relevant cues faster) and action scoring (they assign higher utility to approaches that have worked before).

## NPC State

Full NPC state is a flat, serialisable object:

```ts
interface NPCState {
  id: string;
  name: string;
  people: string;          // species/culture
  age: number;
  isAlive: boolean;
  jobId?: string;
  householdId?: string;
  settlementId: string;

  // Vitals
  health: number;          // 0–1
  wealth: number;          // coin units
  viability: number;       // 0–1, composite score

  // Topogenesis vectors
  needs: Record<string, number>;
  affect: Record<string, number>;
  skills: Record<string, number>;

  // Social
  relationships: Record<string, RelationshipState>;
  lineage: { motherId?: string; fatherId?: string; childIds: string[] };

  // Cognition
  memories: Memory[];
  frame: {
    distinctions: FrameDistinction[];
    attentionBias: Record<string, number>;
    tabooTags: string[];
    valuedTags: string[];
  };

  currentAction: string;
  currentIntention?: { action: string; motivation: string; expectedViabilityGain: number };
}
```

## Settlement Economy

Each settlement tracks 9 resources and 9 jobs:

**Resources**: `food`, `water`, `wood`, `stone`, `ore`, `herbs`, `cloth`, `gold`, `mana`

**Jobs**:
| Job | Produces | Needs |
|---|---|---|
| `farmer` | food | land |
| `miner` | ore, stone | tools |
| `woodcutter` | wood | axe |
| `herbalist` | herbs | knowledge |
| `weaver` | cloth | herbs, wool |
| `merchant` | gold | goods |
| `guard` | security | weapons |
| `healer` | medicine | herbs |
| `scholar` | mana | books, peace |

Resource imbalances create **pressures**. Pressures propagate: food scarcity → hunger → desperation → crime → insecurity.

**Viability Score** for a settlement: weighted average of resource sufficiency, security level, population health, and public morale. Settlements below 0.3 viability start generating distress quests.

## Quest Generation

Quests emerge from 5 pressure types. Each pressure type produces a different quest category:

| Pressure | Quest Category | Example |
|---|---|---|
| Resource scarcity | Supply | "The herbalists are desperate — bring 20 bundles from the Ashwood" |
| Social conflict | Mediation | "Elder Mira and the carpenter guild stand at an impasse" |
| Lineage events | Family | "Aiden's child is missing since the storm" |
| Anomalous world event | Investigation | "Strange lights reported near the old watchtower" |
| Reputation decay | Reputation | "The merchant guild no longer trusts Veldrath's assay" |

Quest urgency scales with pressure magnitude. Quests expire if the underlying pressure resolves (e.g., a merchant NPC trades food in and the scarcity quest becomes inactive).

## Dungeon Topogenesis

Dungeons crystallise from settlement pressure vectors. High collective tension, unresolved grief, and suppressed conflict push the world's topology into anomalous configurations — which manifest as dungeon rooms.

Each dungeon room records the pressure that created it:

```ts
interface DungeonRoom {
  id: string;
  settlementId: string;
  type: string;          // "crypt", "lair", "ruin", "cave", etc.
  pressureSource: string; // which pressure created this room
  description: string;
  difficulty: number;    // 0–1
  tags: string[];
}
```

Room type maps to pressure source:
- High `grief` → crypts and spectral ruins
- High `conflict` → bandit lairs and contested ground
- High `scarcity` → desperate forager caves
- High `instability` → unstable magical anomalies

## Viability Score Calculation

```
viability = (
  0.30 * health +
  0.25 * need_satisfaction_avg +
  0.20 * social_connectedness +
  0.15 * economic_stability +
  0.10 * affect_valence
)
```

Where:
- `need_satisfaction_avg` = mean of all needs values
- `social_connectedness` = mean affection across all relationships (0 if no relationships)
- `economic_stability` = `min(wealth / 100, 1.0)` (clamped at 1)
- `affect_valence` = `(valence + 1) / 2` (normalised to 0–1)

NPCs below 0.25 viability face severe negative events (illness, crime, family breakdown). NPCs below 0.1 may die.

## Relationship System

Relationships are bidirectional but asymmetric (NPC A's view of B may differ from B's view of A):

```ts
interface RelationshipState {
  targetId: string;
  familiarity: number;   // 0–1, grows with interaction frequency
  affection: number;     // 0–1, emotional warmth
  trust: number;         // 0–1, reliability belief
  respect: number;       // 0–1, capability belief
  conflict: number;      // 0–1, active tension
  kinshipType: string;   // "stranger", "acquaintance", "friend", "kin", "rival", "enemy"
}
```

Relationships are updated each tick based on co-location, shared events, and economic interactions.

## Household System

NPCs form households — shared economic units with pooled resources:

```ts
interface HouseholdState {
  id: string;
  name: string;
  memberIds: string[];
  foodStores: number;
  wealth: number;
  homeQuality: number;   // 0–1
  reputation: number;    // 0–1
}
```

Household viability is calculated separately from individual NPC viability. Households below 0.2 viability break apart (members seek new households or form independent ones).

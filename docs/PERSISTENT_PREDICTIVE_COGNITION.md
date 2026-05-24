# Persistent Predictive Cognition

This branch is moving toward an AGI architecture based on persistent world
models, persistent memory, and sensory confirmation.

The aim is not to make cognition happen by burning tokens. The aim is to keep a
stable model of the world, predict what sensory input should arrive next, and
update that model when observation confirms, contradicts, or extends it.

## Core Loop

```text
world model predicts expected sensory state
    ↓
sensory input arrives
    ↓
prediction is compared with observation
    ↓
confirmation / contradiction / novelty is computed
    ↓
world model updates through an evidence gate
    ↓
memory records and consolidates salient evidence
    ↓
future action uses the updated model
```

## Implemented Scaffold

The module `topogenesis.research.predictive_cognition` currently provides:

- `SensoryObservation`
- `Prediction`
- `ConfirmationSignal`
- `WorldEntity`
- `PersistentWorldModel`
- `PersistentMemory`
- `SensoryConfirmationLoop`
- `compare_prediction()`

This is intentionally small. It establishes the invariant that memory and world
state are updated through sensory evidence rather than free-form generation.

## Architectural Standard

A cognition module should eventually connect to this loop by providing one of
these functions:

- encode sensory input into `SensoryObservation`
- predict expected features from `PersistentWorldModel`
- compare prediction and observation
- update beliefs through confirmation/contradiction
- consolidate repeated evidence into semantic memory
- expose uncertainty to planning and action selection

## Hardening Requirements

Before this becomes a central runtime path, it needs:

- bounded memory growth
- finite prediction-error values
- deterministic replay under fixed sensory traces
- clear sensory schemas for body, world, social, and tool inputs
- tests for confirmation, contradiction, novelty, and missing observations
- latency gates for batch updates

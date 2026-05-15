# Refactor Stages

## Stage 1: Stable Organism

Goal: one agent can survive long runs without NaNs.

Required checks:

- engine starts from a preset
- body state remains bounded
- energy, health, membrane, and structural integrity are logged
- final summary is machine-readable

## Stage 2: Learning Organism

Goal: one agent improves survival over its lifetime.

Required checks:

- world-model prediction error is tracked
- policy/reflex/memory/planner action contributions are logged
- memory retrieval affects behavior
- ablation can disable learning

## Stage 3: Reproducing Organism

Goal: agents reproduce and offspring inherit useful traits.

Required checks:

- lineages are tracked
- parent-child genome similarity is measured
- offspring viability is measured
- reproduction can be ablated

## Stage 4: Ecological Population

Goal: multiple lineages coexist with different strategies.

Required checks:

- niche occupation is measured
- resource cycling is measured
- field modification persists
- social interaction metrics are logged

## Stage 5: Open-Ended Evolution

Goal: novelty and diversity continue over long runs.

Required checks:

- novelty rate does not collapse quickly
- genome and behavior diversity are tracked
- adaptive complexity trend is estimated
- descendants can exhibit behaviors absent in ancestors


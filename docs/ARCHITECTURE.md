# Topogenesis Architecture

Topogenesis is organized around five layers.

## 1. World Layer

Owns terrain, resources, hazards, field physics, gravity, collisions, and
world-state interfaces.

Modules:

- stable engine base: `topogenesis.world.world3d` (with field physics in
  `topogenesis.fields.sigma`)
- RPG branch: offline world/population simulator in `topogenesis.world`

## 2. Body Layer

Owns morphology, sensors, motors, energy, membrane integrity, repair,
structural integrity, and interoception.

Modules:

- `topogenesis.body.body_state` (body physics, metabolism, rich body state,
  observation vector)
- future splits: `morphology`, `membrane`, `sensors`

## 3. Cognitive Layer

Owns perception, attention, world model, memory, symbolic abstraction, affect,
drives, policy, planner, and action composition.

Modules:

- `topogenesis.cognition.agent` (the integrated agent)
- `topogenesis.cognition.pressure` (viability pressure, needs, affect,
  social-pressure primitives)
- `topogenesis.cognition.networks` (world model, policy, workspace, memory,
  and symbolic primitives)
- future splits of `networks`: `encoder`, `world_model`, `policy`,
  `workspace`, `affect`, `memory`, `symbolic`

## 4. Development And Genome Layer

Owns genotype, mutation, recombination, body-plan decoding, developmental
stages, lineage identity, and inherited priors.

Modules:

- `topogenesis.evolution.genome` (genome, heredity, development,
  genome-field interface)
- future splits: `mutation`, `reproduction`, `development`, `lineage`

## 5. Experiment Layer

Owns reproducible runs, presets, ablations, logging, checkpoints, metrics,
visualization, and pass/fail conditions.

Modules:

- `experiments.run`
- `topogenesis.run_loop` (reference population loop and CLI)
- `topogenesis.analysis.metrics`
- future: `topogenesis.analysis.plots`, `topogenesis.analysis.oee_score`

Shared configuration lives in `topogenesis.config`, and shared dimension
constants in `topogenesis.constants`. `topogenesis.engine` remains as a
compatibility facade that re-exports every public name, so existing imports
and `python -m topogenesis.engine` keep working.

## Project Rule

Topogenesis is a standalone project. Core runtime code lives under `topogenesis/`, and
experiments live under `experiments/`. Each extracted subsystem must preserve a
smoke run, expose metrics, and receive at least one direct unit test.

# Topogenesis Architecture

Topogenesis is organized around five layers.

## 1. World Layer

Owns terrain, voxels, resources, hazards, field physics, gravity, collisions,
and ecological cycles.

Target modules:

- `topogenesis.world.world3d`
- `topogenesis.world.physics`
- `topogenesis.world.resources`
- `topogenesis.world.hazards`
- `topogenesis.world.sigma_field`

## 2. Body Layer

Owns morphology, sensors, motors, energy, membrane integrity, repair,
structural integrity, and interoception.

Target modules:

- `topogenesis.body.body_state`
- `topogenesis.body.morphology`
- `topogenesis.body.metabolism`
- `topogenesis.body.membrane`
- `topogenesis.body.sensors`

## 3. Cognitive Layer

Owns perception, attention, world model, memory, symbolic abstraction, affect,
drives, policy, planner, and action composition.

Target modules:

- `topogenesis.cognition.agent`
- `topogenesis.cognition.encoder`
- `topogenesis.cognition.world_model`
- `topogenesis.cognition.policy`
- `topogenesis.cognition.workspace`
- `topogenesis.cognition.affect`
- `topogenesis.cognition.memory`
- `topogenesis.cognition.symbolic`

## 4. Development And Genome Layer

Owns genotype, mutation, recombination, body-plan decoding, developmental
stages, lineage identity, and inherited priors.

Target modules:

- `topogenesis.evolution.genome`
- `topogenesis.evolution.mutation`
- `topogenesis.evolution.reproduction`
- `topogenesis.evolution.development`
- `topogenesis.evolution.lineage`

## 5. Experiment Layer

Owns reproducible runs, presets, ablations, logging, checkpoints, metrics,
visualization, and pass/fail conditions.

Target modules:

- `experiments.run`
- `topogenesis.analysis.metrics`
- `topogenesis.analysis.plots`
- `topogenesis.analysis.oee_score`

## Project Rule

Topogenesis is a standalone project. Core runtime code lives under `topogenesis/`, and
experiments live under `experiments/`. Each extracted subsystem must preserve a
smoke run, expose metrics, and receive at least one direct unit test.

## Landed Module Layout

The reference engine has been decomposed into layered modules:

- `topogenesis.constants`, `topogenesis.config`
- `topogenesis.fields.sigma` (PDE kernel, reservoir, sigma-field geometry)
- `topogenesis.body.body_state`
- `topogenesis.world.world3d` (alongside the offline world simulator)
- `topogenesis.evolution.genome`
- `topogenesis.cognition.networks` and `topogenesis.cognition.agent`
- `topogenesis.run_loop` (reference loop and CLI)

`topogenesis.engine` remains as a compatibility facade re-exporting every
public name, so existing imports and `python -m topogenesis.engine` keep
working. The finer-grained targets listed above remain future splits.

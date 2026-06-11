# Topogenesis

Topogenesis is a standalone embodied artificial-life platform for modular,
measurable ecosystem simulation.

This `main` branch is the stable neutral base. It keeps the core Python engine,
experiments, tests, and documentation without the game-specific Godot layer,
offline RPG population simulator, or standalone NPC layer. See
[docs/BRANCHES.md](docs/BRANCHES.md) for how `main`,
`game-rpg`, and `agi-research` are separated.

Its thesis is pressure-driven cognition: needs, affect, communication, memory,
future simulation, and action are coupled to viability pressure rather than
scripted labels or pure reward optimization. See
[docs/PROJECT_THESIS.md](docs/PROJECT_THESIS.md). The functionalist standard
for judging each subsystem is defined in
[docs/FUNCTIONAL_ROLES.md](docs/FUNCTIONAL_ROLES.md).

New readers should start with [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)
for verification commands, suggested experiment presets, and contribution areas.

Topogenesis is early-stage research software. The engine runs, but the
cognitive, field, developmental, and evolutionary mechanisms should be treated
as experimental hypotheses until validated by ablations, baselines, and
long-run metrics.

The integrated reference engine lives at `topogenesis/engine.py`. The surrounding
package provides experiment presets, metrics contracts, cognition primitives,
and a target module layout so every subsystem can become separable, testable,
ablatable, and measurable.

## Platform Focus

The stable base keeps pressure-cognition compatibility inside
`topogenesis/engine.py`. The standalone `topogenesis.npc` package now belongs to
`game-rpg`, where it supports villager cognition and bridge behavior.

The offline population/world simulator has moved out of `main`. It now belongs
to `game-rpg`, where it supports the playable village and bridge layer.

## North Star

Topogenesis should become a self-maintaining artificial-life laboratory where
agents survive, learn, reproduce, mutate, build memory, alter their environment,
and evolve under measurable ecological pressure.

Specialized branches can push this foundation in different directions. The
stable base focuses on keeping the engine runnable, measurable, ablatable, and
easy to extend.

## First Stable Target

Stage 1 is a stable organism:

- one agent survives long runs without NaNs
- energy, membrane, health, and structural integrity stay measurable
- resources and hazards exert clear pressure
- cognition has explicit action contributions and metabolic costs
- checkpoints and run summaries are reproducible

## Run Presets

```bash
python -m experiments.run --experiment smoke
python -m experiments.run --experiment single_agent_survival
python -m experiments.run --experiment lifetime_learning
python -m experiments.run --experiment reproduction_basic
python -m experiments.run --experiment evolutionary_run
python -m experiments.run --experiment open_ended_ecology
```

Use `--engine-path` only when intentionally testing an alternate engine module.
Long runs accept `--max_population` (default 64) to bound memory growth from
reproduction; the weekly extended CI run uses `--max_population 32`.

## Verification

Local checks:

```bash
python -m unittest discover -s tests -p "test_*.py"
python -m experiments.run --experiment smoke
```

GitHub Actions runs unit tests with `JAX_DISABLE_JIT=0`, a short smoke
experiment, and a weekly/manual extended JIT smoke run. The manual workflow
defaults to 2,000 steps for longer stability checks without slowing every push.

## Stability Status

The current base prioritizes transparency and repeatability:

- core mechanisms are heuristic and under active hardening
- `topogenesis/engine.py` still contains the integrated reference engine
- experiment presets are intentionally explicit and reproducible
- ablations, dashboards, and baseline agents are planned next

See [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) for first steps,
[docs/PROJECT_THESIS.md](docs/PROJECT_THESIS.md) for the research thesis,
[docs/FUNCTIONAL_ROLES.md](docs/FUNCTIONAL_ROLES.md) for subsystem role
contracts and ablation standards,
[docs/KNOWN_LIMITS.md](docs/KNOWN_LIMITS.md) for the current hardening
checklist, and [docs/PRODUCTION_READINESS.md](docs/PRODUCTION_READINESS.md) for
the production-readiness gates. Stress coverage is tracked in
[docs/STRESS_VALIDATION.md](docs/STRESS_VALIDATION.md).

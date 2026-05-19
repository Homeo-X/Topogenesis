# Topogenesis

Topogenesis is a standalone embodied artificial-life research platform for modular,
measurable ecosystem simulation.

Topogenesis is early-stage research software. The engine runs, but the cognitive,
field, developmental, and evolutionary mechanisms should be treated as
experimental hypotheses until validated by ablations, baselines, and long-run
metrics.

The engine entrypoint lives inside this project at `topogenesis/engine.py`. The
surrounding package provides stable experiment presets, metrics contracts, and a
target module layout so every subsystem can become separable, testable,
ablatable, and measurable.

The `topogenesis.npc` package is the first RPG cognition layer. It models NPCs
as self-maintaining predictive agents with continuous affect fields, emergent
need pressure, compact models of other minds, communication intents, and small
future simulations. Communication is represented as an attempted intervention on
another agent's world model rather than a dialogue-tree response.

## North Star

Topogenesis should become a self-maintaining artificial-life laboratory where agents
survive, learn, reproduce, mutate, build memory, alter their environment, and
evolve under measurable ecological pressure.

## First Stable Target

Stage 1 is a stable organism:

- one agent survives long runs without NaNs
- energy, membrane, health, and structural integrity stay measurable
- resources and hazards exert clear pressure
- cognition has explicit action contributions and metabolic costs
- checkpoints and run summaries are reproducible

## RPG NPC Cognition

The NPC layer is deliberately label-light:

- affect is a continuous pressure field, not fixed emotion tags
- needs emerge from viability deficits instead of scripted counters
- social memory records reputation-shaping events
- communication intents target belief changes in other agents
- future simulation compares action consequences before acting or speaking

These primitives are not yet wired into the full engine loop. They are tested as
standalone modules first so the integrated reference engine does not become one
huge script again.

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

## Verification

Local checks:

```bash
python -m unittest discover -s tests -p "test_*.py"
python -m experiments.run --experiment smoke
```

GitHub Actions runs unit tests with `JAX_DISABLE_JIT=0`, a short smoke
experiment, and a weekly/manual extended JIT smoke run. The manual workflow
defaults to 2,000 steps for longer stability checks without slowing every push.

## Public Research Status

The current release prioritizes transparency over polish:

- core mechanisms are heuristic and under active hardening
- `topogenesis/engine.py` still contains the integrated reference engine
- experiment presets are intentionally explicit and reproducible
- ablations, dashboards, and baseline agents are planned next

See [docs/KNOWN_LIMITS.md](docs/KNOWN_LIMITS.md) for the current hardening
checklist and [docs/PRODUCTION_READINESS.md](docs/PRODUCTION_READINESS.md) for
the production-readiness gates.

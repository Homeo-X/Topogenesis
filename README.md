# Topogenesis

Topogenesis is a standalone embodied artificial-life research platform for modular,
measurable ecosystem simulation.

Its thesis is pressure-driven cognition: needs, affect, communication, and
action are coupled to viability pressure rather than scripted labels or pure
reward optimization. See [docs/PROJECT_THESIS.md](docs/PROJECT_THESIS.md).

New readers should start with [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)
for verification commands, suggested experiment presets, and contribution areas.

A first Godot 4.6 3D RPG prototype lives in [godot/](godot/). It provides a
small village, a third-person player, NPC pressure/affect state, interaction,
and a bridge point for connecting the Python Topogenesis backend.

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

These primitives now feed the reference agent loop through a narrow bridge:
engine viability metrics update NPC affect, need pressure, social memory,
communication intent, and future simulation. The resulting instability profile
is exposed in metrics and gently gates motor output toward reflex under risk.

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

See [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) for first steps,
[docs/PROJECT_THESIS.md](docs/PROJECT_THESIS.md) for the research thesis,
[docs/KNOWN_LIMITS.md](docs/KNOWN_LIMITS.md) for the current hardening
checklist, and [docs/PRODUCTION_READINESS.md](docs/PRODUCTION_READINESS.md) for
the production-readiness gates. Stress coverage is tracked in
[docs/STRESS_VALIDATION.md](docs/STRESS_VALIDATION.md).

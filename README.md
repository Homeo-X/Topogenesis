# Topogenesis

Topogenesis is a standalone embodied cognition and AGI research platform for
modular, measurable agent simulation.

This branch, `agi-research`, is intentionally research-only. The Godot RPG
prototype and game-facing bridge live on the `game-rpg` branch.

Its thesis is pressure-driven cognition: needs, affect, communication, memory,
future simulation, and action are coupled to viability pressure rather than
scripted labels or pure reward optimization. See
[docs/PROJECT_THESIS.md](docs/PROJECT_THESIS.md). The functionalist standard
for judging each subsystem is defined in
[docs/FUNCTIONAL_ROLES.md](docs/FUNCTIONAL_ROLES.md).

New readers should start with [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)
for verification commands, suggested experiment presets, and contribution areas.
This branch's deeper AGI-facing commitments are tracked in
[docs/AGI_RESEARCH_BRANCH.md](docs/AGI_RESEARCH_BRANCH.md).

Topogenesis is early-stage research software. The engine runs, but the
cognitive, field, memory, world-model, and control mechanisms should be treated
as experimental hypotheses until validated by ablations, baselines, scaling
tests, and long-run stability metrics.

The integrated reference engine lives at `topogenesis/engine.py`. The surrounding
package provides experiment presets, metrics contracts, cognition primitives,
offline load-testing support, and a target module layout so every subsystem can
become separable, testable, ablatable, scalable, and measurable.

## Research Focus

The `topogenesis.npc` package contains cognition primitives for pressure-driven
agents:

- affect as a continuous internal pressure field
- needs emerging from viability deficits
- episodic and semantic memory
- compact models of other minds
- communication intents as belief interventions
- hierarchical future simulation with cognitive cost

The `topogenesis.world` package is treated here as a headless scaling and load
testing substrate. It runs lightweight batched viability, need pressure, affect
stability, location pressure, and population-level metrics without rendering.

## North Star

Topogenesis should become a hardened cognition laboratory where agents maintain
viability, build memory, model consequences, regulate internal pressure, and
scale to larger simulations under measurable computational limits.

The AGI branch focuses on whether increasingly functional cognition can emerge
from self-maintenance, world modeling, memory, social inference, imagination,
and pressure-regulated action.

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
```

Use `--engine-path` only when intentionally testing an alternate engine module.

For fast background population simulation:

```bash
python -m topogenesis.world.offline_sim --days 30 --population 200
```

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
- ablations, dashboards, baseline agents, and scaling gates are planned next

See [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) for first steps,
[docs/PROJECT_THESIS.md](docs/PROJECT_THESIS.md) for the research thesis,
[docs/FUNCTIONAL_ROLES.md](docs/FUNCTIONAL_ROLES.md) for subsystem role
contracts and ablation standards,
[docs/KNOWN_LIMITS.md](docs/KNOWN_LIMITS.md) for the current hardening
checklist, and [docs/PRODUCTION_READINESS.md](docs/PRODUCTION_READINESS.md) for
the production-readiness gates. Stress coverage is tracked in
[docs/STRESS_VALIDATION.md](docs/STRESS_VALIDATION.md).

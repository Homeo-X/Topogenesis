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

## Public Research Status

The current release prioritizes transparency over polish:

- core mechanisms are heuristic and under active hardening
- `topogenesis/engine.py` still contains the integrated reference engine
- experiment presets are intentionally explicit and reproducible
- ablations, dashboards, and baseline agents are planned next

See [docs/KNOWN_LIMITS.md](docs/KNOWN_LIMITS.md) for the current hardening
checklist.

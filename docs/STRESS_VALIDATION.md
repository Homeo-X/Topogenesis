# Stress Validation

This project keeps stress validation repeatable through tests instead of only
manual terminal runs.

## Current Stress Coverage

- `tests/test_jit_runtime.py` verifies the JIT-oriented field kernel remains
  finite under compiled execution.
- Historical game-facing NPC stress coverage now lives on the `game-rpg` branch
  with the standalone `topogenesis.npc` layer.
- The integrated engine smoke run still exercises internal pressure-cognition
  compatibility paths because the reference agent loop updates affect-like
  pressure, need pressure, communication intent, future simulation, and motor
  gating each step.
- Previous stress checks included a sigma-field PDE soak that
  runs 256 sigma-field PDE steps with edge-clipped positions and high energies,
  then checks finite field values and near-unit normalization.

## Manual Validation Run

The latest local validation also ran:

```bash
python -m unittest discover -s tests -p "test_*.py"
JAX_DISABLE_JIT=0 python -m unittest discover -s tests -p "test_*.py"
JAX_DISABLE_JIT=0 python -m experiments.run --experiment smoke --override --steps 3 --agents 2 --world_size 16 --log_every 1
```

Observed result: all tests passed, the multi-agent engine stress completed, both
agents remained alive, and the run emitted finite summaries.

## Known Limits

The current full engine is still too slow for very long CPU stress runs on every
push. Long soak testing should run as scheduled/manual CI once profiling and
cognitive level-of-detail are in place.

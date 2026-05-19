# Stress Validation

This project keeps stress validation repeatable through tests instead of only
manual terminal runs.

## Current Stress Coverage

- `tests/test_stress_validation.py::test_npc_cognition_survives_repeated_updates`
  runs 5,000 affect, need, communication, memory, belief, and future-simulation
  updates while checking finite bounded state and bounded memory.
- `tests/test_stress_validation.py::test_sigma_field_jit_stress_remains_finite_and_normalized`
  runs 256 sigma-field PDE steps with edge-clipped positions and high energies,
  then checks finite field values and near-unit normalization.
- The engine smoke run now also exercises the NPC cognition bridge because the
  reference agent loop updates NPC affect, need pressure, social memory,
  communication intent, future simulation, and motor gating each step.

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

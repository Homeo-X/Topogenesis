# Production Readiness

Topogenesis is not yet AAA production middleware. The project is moving toward
that standard by making each subsystem bounded, testable, deterministic under
seeded runs, and explicit about failure modes.

## Current Hardening Gates

- Unit tests must pass with `python -m unittest discover -s tests -p "test_*.py"`.
- The smoke experiment must start, step, and exit without exceptions.
- JIT-sensitive kernels are covered by CI with `JAX_DISABLE_JIT=0`.
- New public primitives should sanitize non-finite inputs, clamp public scalar
  state into expected ranges, bound memory growth, and fail fast on invalid
  identifiers.

## Still Research-Grade

- The integrated engine in `topogenesis/engine.py` remains a reference engine,
  not a final modular runtime.
- The full cognitive loop still contains host-side bookkeeping around memory,
  social state, logging, and world/body interaction.
- Long-run balance, save/load compatibility, profiling budgets, and
  deterministic replay are not complete.

## AAA Direction

- Keep public APIs defensive and deterministic.
- Keep simulation loops bounded and observable.
- Add telemetry before adding more mechanisms.
- Separate low-frequency cognition from high-frequency physics.
- Use level-of-detail for large or expensive simulations.
- Require reproduction cases for every bug fixed in emergent behavior.

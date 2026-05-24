# Known Limits And Hardening Checklist

Topogenesis is public as an early research platform, not as a finished
artificial-life result. These are the current high-priority limits.

## Runtime And Architecture

- `TopogenesisAgent.step()` remains too large and stateful.
- JAX and NumPy are mixed in hot paths, limiting JIT compilation.
- Several subsystems are heuristic couplings rather than validated mechanisms.
- Recoverable subsystem failures are now counted in `soft_failure_count`, but
  the next step is replacing broad fallbacks with typed outcomes.

## Numerical Stability

- The sigma field has CFL-aware stepping and finite-value projection, but it
  still needs long-run stress tests.
- Field stability metrics are logged as:
  - `field_finite`
  - `field_max_abs`
  - `field_dissipation`
- Genome modules have hard caps, but mutation retention and complexity cost
  need population-level tracking.

## Deferred Secondary Systems

- Some broad artificial-life mechanisms still exist in the integrated reference
  engine, but they are not the current AGI-branch focus.
- Treat those systems as deferred until cognition, scalability, JAX boundaries,
  and hardening are stronger.

## Experimentation

Needed before strong scientific claims:

- null agents: random, reflex-only, memory-only
- ablation presets: no field, no memory, no affect, no world model
- long-run stability and failure-containment curves
- scaling metrics: tick latency, memory growth, active-agent budget, subsystem
  failure recovery
- dashboard/plots for energy, pressure, action contributions, prediction error,
  agent paths, and failure modes

## Success Criteria

The AGI branch should not claim AGI progress until architectural changes improve
behavior, stability, or scalability against clear baselines.

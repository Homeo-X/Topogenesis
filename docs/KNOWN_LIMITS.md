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

## Biology And Ecology

- Death thresholds are configurable and softened during juvenile life, but still
  need calibration by experiment preset.
- Reproduction currently uses hand-shaped viability gates.
- The genome-in-field mechanism is promising, but needs visualization and
  lineage-level inheritance metrics.

## Experimentation

Needed before strong scientific claims:

- null agents: random, reflex-only, memory-only
- ablation presets: no field, no memory, no affect, no reproduction
- long-run survival and extinction curves
- lineage diversity and novelty metrics
- dashboard/plots for field slices, agent paths, genome loci, and births/deaths

## Success Criteria

The platform should not claim open-ended evolution until descendants reliably
show adaptive behaviors absent in ancestors under controlled baselines.

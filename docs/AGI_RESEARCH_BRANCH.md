# AGI Research Branch

`agi-research` is the speculative cognition branch for Topogenesis. Its purpose
is not to claim solved AGI. Its purpose is to turn ambitious cognition ideas
into falsifiable role contracts, ablations, scaling gates, and measurable
hardening experiments.

Broad artificial-life expansion claims are deferred. The current branch focus
is core cognitive architecture: stable interfaces, bounded state, measurable
ablation effects, and scaling behavior.

## Core Research Standard

Every proposed cognitive mechanism should pass this chain:

```text
internal state -> functional role -> behavioral consequence -> measured outcome
```

If a subsystem does not change behavior under ablation, it is decorative until
its causal role is strengthened.

## Current Research Scaffold

The module `topogenesis.research.functionalism` defines:

- `FunctionalRoleContract`
- `EvidenceGate`
- `ScalingGate`
- `default_functionalist_ladder()`
- `default_evidence_gates()`
- `default_scaling_gates()`
- `incomplete_contracts()`

This gives the branch a concrete place to encode research commitments before
they are promoted into engine code.

The module `topogenesis.research.predictive_cognition` starts the persistent
world-model and memory architecture. Its core loop is:

```text
prediction -> sensory input -> confirmation signal -> model update -> memory
```

See `docs/PERSISTENT_PREDICTIVE_COGNITION.md`.

## Near-Term AGI Work

1. Convert each role contract into an experiment preset.
2. Add multi-seed comparisons for each evidence gate.
3. Add scaling gates for agent count, memory budget, tick latency, and failure
   containment.
4. Connect runtime sensory observations into `SensoryConfirmationLoop`.
5. Track effect sizes, not just whether a run survives.
6. Add failure-mode detectors for freezing, unbounded memory, and single-seed
   artifacts.
7. Promote only validated, branch-neutral mechanisms back to `main`.

## Research Boundary

This branch may contain speculative interfaces, but each one should be paired
with:

- at least one ablation
- at least one metric
- at least one scaling or stability limit
- a predicted behavioral difference
- a failure mode that would falsify the claim

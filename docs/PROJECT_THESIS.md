# Project Thesis

Topogenesis explores biologically grounded, pressure-driven cognition for
artificial life. Agents self-maintain under energy, bodily, social, and
predictive instability; affect emerges as continuous viability pressure; and
behavior is shaped by embodied costs, memory, field dynamics, development, and
environmental pressure.

## Core Claim

The central claim is not that Topogenesis already produces open-ended evolution
or robust autonomous intelligence. The claim is narrower:

> Cognition becomes more lifelike when needs, affect, communication, and action
> are coupled to viability pressure instead of being scripted labels or pure
> reward signals.

In this frame, an agent does not act because a designer assigned it a desire. It
acts because internal instability degrades its future continuity unless it
compensates.

## Design Commitments

- Affect is a continuous pressure field, not a list of emotion tags.
- Needs emerge from viability deficits rather than scripted counters.
- Communication is a social intervention on another agent's predictive state.
- Intelligence pays embodied costs through energy, damage, memory, and field
  stability.
- Architecture should expose pressure-management strategies through metrics and
  ablations instead of hiding them in opaque controller behavior.
- Claims must remain measurable through metrics, ablations, baselines, and
  reproducible runs.
- Every subsystem should satisfy the functional role contracts in
  [FUNCTIONAL_ROLES.md](FUNCTIONAL_ROLES.md): internal state must cause a
  behavioral consequence and a measurable outcome.

## Non-Claims

Topogenesis does not currently claim:

- proven open-ended evolution
- biological realism in the strong scientific sense
- scalable large-population cognition
- fully JAX-pure execution
- deterministic replay compatibility
- robust autonomous intelligence

Those are targets or research questions, not achievements.

## Falsifiable Milestones

The thesis becomes stronger only if future runs show measurable effects:

- Agents with affect/need coupling survive or adapt better than ablated agents.
- Communication changes listener beliefs, future behavior, and social memory in
  measurable ways.
- Future simulation improves action selection under hazard/resource pressure.
- Larger runs preserve stability instead of converging to one brittle tactic.
- Refactors reduce runtime cost without changing measured behavior.
- Long runs avoid NaNs, runaway fields, unbounded memory growth, and immediate
  extinction.

## Research Baselines

Topogenesis should be compared against:

- random agents
- reflex-only agents
- agents without affect/needs
- agents without memory
- agents without communication
- agents without field coupling
- agents with scripted discrete emotion tags
- standard RL or behavior-tree baselines where practical

## North Star

The long-term goal is a self-maintaining artificial-life laboratory where agents
survive, learn, remember, communicate, maintain internal stability, and remain
measurable under environmental pressure.

The open question is whether pressure-driven cognition can produce richer,
longer-lived, and more diverse agent behavior than scripts, reflexes, or pure
reward optimization alone.

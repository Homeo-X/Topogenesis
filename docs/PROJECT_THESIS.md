# Project Thesis

Topogenesis explores biologically grounded, pressure-driven cognition for
artificial life. Agents self-maintain under energy, bodily, social, and
predictive instability; affect emerges as continuous viability pressure;
communication is modeled as intervention on another agent's world model; and
behavior is shaped by embodied costs, memory, field dynamics, and hardening
pressure.

## Core Claim

The central claim is not that Topogenesis already produces robust autonomous
intelligence. The claim is narrower:

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
- Architecture should scale pressure-management strategies without hidden
  state leaks, runaway memory, or brittle single-agent assumptions.
- Claims must remain measurable through metrics, ablations, baselines, and
  reproducible runs.
- Every subsystem should satisfy the functional role contracts in
  [FUNCTIONAL_ROLES.md](FUNCTIONAL_ROLES.md): internal state must cause a
  behavioral consequence and a measurable outcome.

## Non-Claims

Topogenesis does not currently claim:

- biological realism in the strong scientific sense
- AAA-ready game AI middleware
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
- Larger runs preserve stability and useful behavior instead of collapsing into
  frozen, noisy, or runaway dynamics.
- Architectural changes improve tick latency, memory bounds, and failure
  containment under load.
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

## RPG Interpretation

As an RPG system, Topogenesis is best understood as cognition middleware for
living-world simulation, not as a full game engine. Its natural role is to model:

- NPC needs and pressure
- affective instability
- rumor and belief propagation
- social memory and reputation
- anticipation and hesitation
- cooperation, distrust, and deception
- faction and culture-like dynamics

The intended result is not prettier dialogue trees. It is NPC behavior shaped by
self-maintenance, uncertainty, memory, social pressure, and simulated futures.

## North Star

The near-term goal is a hardened cognition laboratory where agents maintain
viability, remember, communicate, model consequences, regulate internal
pressure, and scale under measurable computational pressure.

The open question is whether pressure-driven cognition can produce richer,
longer-lived, and more diverse agent behavior than scripts, reflexes, or pure
reward optimization alone.

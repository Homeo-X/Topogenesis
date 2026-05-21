# Functional Roles

Topogenesis should be evaluated by functional role, not by whether its internal
names sound cognitive or biological. A subsystem counts as part of the agent's
intelligence only when it has a causal role in behavior and a measurable effect
on outcomes.

The standard is:

```text
internal state -> functional role -> behavioral consequence -> measured outcome
```

If removing a subsystem does not change behavior or metrics, that subsystem is
ornamental until its role is made causal.

## Functionalist Standard

A Topogenesis agent becomes functionally intelligent when it:

- maintains viability under environmental pressure
- uses internal state to regulate action and cognition
- predicts consequences before acting
- learns from outcomes
- adapts differently across contexts
- communicates to alter social behavior
- degrades in specific ways when key functions are ablated

The goal is not to prove that the agent has human-like mental states. The goal
is to show that its internal states perform the same kinds of regulatory,
predictive, social, and adaptive roles that make those states useful in living
systems.

## Role Contracts

| Subsystem | Input | Functional Role | Output | Behavioral Effect | Metric | Ablation |
| --- | --- | --- | --- | --- | --- | --- |
| Viability state | Energy, membrane, health, structural integrity, hazard exposure | Track pressure on continuity | Viability deficits and collapse risk | Forces tradeoffs between survival, exploration, repair, and social action | survival steps, collapse rate, recovery rate | Fixed-viability agent |
| Need pressure | Viability deficits, resource access, social instability | Convert deficits into action urgency | Dominant need and need intensity | Biases agents toward food, safety, repair, social contact, or information | need satisfaction time, resource efficiency | No-needs agent |
| Affect field | Need pressure, prediction error, threat, social rupture | Dynamically reweight risk, attention, memory, planning, and motor urgency | Continuous affect dimensions | Produces hesitation, urgency, avoidance, recovery, and stress responses | hazard avoidance, overreaction rate, recovery time | No-affect agent |
| Reflex layer | Immediate threat and viability collapse risk | Protect agent under short-horizon danger | Fast action bias | Overrides slower cognition when delay is dangerous | near-hazard survival, reaction latency | No-reflex agent |
| Perception and interoception | World state, body state, field state | Build the current decision context | Observation vector and internal body signal | Makes action sensitive to location, hazards, resources, and bodily condition | observation error, state coverage | Blind/interoception-off agent |
| Memory | State, action, outcome, social events | Preserve useful traces from past experience | Retrieved priors and salient events | Avoids repeated harm, repeats useful strategies, changes relationships | memory usefulness, repeated-error rate | No-memory agent |
| World model | Observation history, action history, current state | Predict likely consequences | Expected next state and uncertainty | Lets agents prefer actions with better expected viability | prediction error, uncertainty calibration | No-world-model agent |
| Future simulation | Candidate actions, world model, affect, needs | Compare short possible futures before acting | Ranked action futures | Enables anticipation, caution, resource planning, and social hesitation | regret rate, risky-action rate, survival gain | No-imagination agent |
| Policy and action synthesis | Reflex, learned policy, memory prior, affect, field bias | Compose final action from competing pressures | Motor/action command | Balances survival, learning, exploration, and social action | action entropy, contribution logs, task success | Reflex-only or policy-only agent |
| Communication | Speaker state, listener model, social memory, intent | Attempt to alter another agent's predictive state | Communication intent and surface line | Produces persuasion, warning, deception, alliance, rumor, or silence | listener belief change, trust change, social stability | No-communication agent |
| Social model | Other-agent observations, reputation, interaction memory | Estimate other minds and relationships | Trust, threat, attachment, expected response | Makes agents behave differently toward different actors | relationship stability, betrayal response, cooperation rate | No-social-model agent |
| Field coupling | Local sigma field, lineage traces, agent pumping | Carry environmental memory and inherited context | Field bias and stability signal | Lets places become safer, unstable, inherited, or behaviorally meaningful | field persistence, niche reuse, lineage locality | No-field agent |
| Genome and development | Parent genome, mutation, age, regulatory state | Bias body, cognition, metabolism, and life-stage changes | Developmental parameters and inherited priors | Enables heritable pressure-management strategies | parent-child similarity, trait retention, generation survival | No-mutation or no-development agent |
| Experiment layer | Seeds, configs, logs, checkpoints | Make claims reproducible and falsifiable | Metrics, traces, summaries, checkpoints | Converts behavior into evidence instead of anecdote | deterministic replay, smoke pass rate, long-run stability | No-logging run |

## Required Evidence Pattern

Every major claim should eventually follow this pattern:

```text
Claim:
  Affect improves survival under hazard pressure.

Mechanism:
  Threat and instability reduce risk tolerance and increase avoidance urgency.

Experiment:
  Compare full agent against no-affect and reflex-only baselines across fixed
  seeds and hazard/resource layouts.

Expected evidence:
  Full agent survives longer, takes fewer fatal hazard crossings, and recovers
  faster after near-collapse without simply freezing in place.
```

## Ablation Ladder

Use ablations from simple to rich:

1. Random action baseline
2. Reflex-only baseline
3. Viability plus reflex
4. Viability plus needs
5. Needs plus affect
6. Affect plus memory
7. Memory plus world model
8. World model plus future simulation
9. Full agent without communication
10. Full agent with communication
11. Full agent with field coupling
12. Full agent with development and reproduction

This ladder prevents the project from claiming intelligence merely because many
modules are active at once. Each added layer must show what new functional work
it performs.

## Pass And Fail Criteria

A subsystem passes the functionalist standard when:

- its inputs are explicit
- its outputs are used by another subsystem
- removing it changes behavior in a predicted way
- its effect can be measured across seeds
- it does not only improve one hand-tuned scenario

A subsystem fails the standard when:

- it is computed but never changes action
- it only changes labels, UI, or narration
- its effect cannot be isolated by ablation
- it improves survival by freezing behavior or exploiting a bug
- it causes unstable runs, NaNs, runaway memory, or immediate extinction

## Engineering Implication

When adding or refactoring a subsystem, include:

- a role contract
- at least one metric
- at least one ablation switch or baseline comparison
- a smoke test proving the loop still runs
- a trace field showing how the subsystem affected action

The project should become less a collection of impressive mechanisms and more a
measurable ecology of causal functions.

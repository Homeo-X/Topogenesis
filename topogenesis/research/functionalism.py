from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FunctionalRoleContract:
    """A falsifiable role contract for a cognitive subsystem."""

    name: str
    inputs: tuple[str, ...]
    role: str
    outputs: tuple[str, ...]
    behavioral_effects: tuple[str, ...]
    metrics: tuple[str, ...]
    ablations: tuple[str, ...]

    def is_complete(self) -> bool:
        return all((
            bool(self.name.strip()),
            bool(self.inputs),
            bool(self.role.strip()),
            bool(self.outputs),
            bool(self.behavioral_effects),
            bool(self.metrics),
            bool(self.ablations),
        ))


@dataclass(frozen=True)
class EvidenceGate:
    """A minimum evidence gate before promoting an AGI claim."""

    claim: str
    required_baselines: tuple[str, ...]
    required_metrics: tuple[str, ...]
    failure_modes: tuple[str, ...]

    def is_testable(self) -> bool:
        return bool(self.claim and self.required_baselines and self.required_metrics)


def default_functionalist_ladder() -> tuple[FunctionalRoleContract, ...]:
    """Return the AGI branch's working ladder from viability to social cognition."""
    return (
        FunctionalRoleContract(
            name="viability",
            inputs=("energy", "integrity", "hazard_exposure"),
            role="track pressure on agent continuity",
            outputs=("collapse_risk", "viability_deficit"),
            behavioral_effects=("resource_seeking", "repair_priority"),
            metrics=("survival_steps", "recovery_rate", "collapse_rate"),
            ablations=("fixed_viability",),
        ),
        FunctionalRoleContract(
            name="affect",
            inputs=("need_pressure", "prediction_error", "threat"),
            role="reweight attention, memory, planning, and risk",
            outputs=("stability", "urgency", "threat_salience"),
            behavioral_effects=("avoidance", "hesitation", "recovery"),
            metrics=("hazard_avoidance", "overreaction_rate", "recovery_time"),
            ablations=("no_affect",),
        ),
        FunctionalRoleContract(
            name="world_model",
            inputs=("observation_history", "action_history", "current_state"),
            role="predict consequences before action",
            outputs=("expected_next_state", "uncertainty"),
            behavioral_effects=("anticipation", "risk_selection", "planning"),
            metrics=("prediction_error", "regret_rate", "uncertainty_calibration"),
            ablations=("no_world_model",),
        ),
        FunctionalRoleContract(
            name="memory",
            inputs=("events", "actions", "outcomes", "social_context"),
            role="preserve behaviorally useful traces",
            outputs=("episodic_priors", "semantic_priors"),
            behavioral_effects=("avoid_repeated_harm", "repeat_success"),
            metrics=("memory_usefulness", "repeated_error_rate"),
            ablations=("no_memory",),
        ),
        FunctionalRoleContract(
            name="social_model",
            inputs=("observed_others", "communication", "interaction_memory"),
            role="estimate other agents as pressure-bearing systems",
            outputs=("trust", "threat_estimate", "expected_response"),
            behavioral_effects=("cooperation", "avoidance", "warning"),
            metrics=("cooperation_rate", "betrayal_response", "belief_change"),
            ablations=("no_social_model", "no_communication"),
        ),
    )


def default_evidence_gates() -> tuple[EvidenceGate, ...]:
    """Minimum gates before calling a behavior functionally intelligent."""
    return (
        EvidenceGate(
            claim="affect improves survival under unstable hazard pressure",
            required_baselines=("no_affect", "reflex_only", "random_action"),
            required_metrics=("survival_steps", "hazard_avoidance", "recovery_time"),
            failure_modes=("freezing", "resource_avoidance", "single_seed_effect"),
        ),
        EvidenceGate(
            claim="memory changes future behavior through useful traces",
            required_baselines=("no_memory", "reflex_only"),
            required_metrics=("repeated_error_rate", "memory_usefulness"),
            failure_modes=("stored_but_unused", "unbounded_growth", "label_only_memory"),
        ),
        EvidenceGate(
            claim="social inference supports adaptive coordination",
            required_baselines=("no_social_model", "no_communication"),
            required_metrics=("cooperation_rate", "trust_change", "social_stability"),
            failure_modes=("scripted_dialogue_only", "perfect_information_transfer"),
        ),
    )


def incomplete_contracts(
    contracts: tuple[FunctionalRoleContract, ...] | list[FunctionalRoleContract],
) -> tuple[str, ...]:
    """Return contract names that are not yet falsifiable enough to test."""
    return tuple(contract.name for contract in contracts if not contract.is_complete())

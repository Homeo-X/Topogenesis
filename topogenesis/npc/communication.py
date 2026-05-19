from __future__ import annotations

from dataclasses import dataclass

from ._math import clamp01, non_empty_text
from .affect import AffectField
from .social import OtherMindModel


@dataclass(frozen=True)
class CommunicationIntent:
    """A social action intended to modify another agent's world model."""

    target_agent: str
    intended_effect: str
    belief_to_modify: str
    confidence: float
    emotional_weight: float = 0.0
    social_risk: float = 0.0
    urgency: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "target_agent",
            non_empty_text(self.target_agent, field_name="target_agent"))
        object.__setattr__(
            self, "intended_effect",
            non_empty_text(self.intended_effect, field_name="intended_effect"))
        object.__setattr__(
            self, "belief_to_modify",
            non_empty_text(self.belief_to_modify, field_name="belief_to_modify"))
        object.__setattr__(self, "confidence", clamp01(self.confidence))
        object.__setattr__(self, "emotional_weight", clamp01(self.emotional_weight))
        object.__setattr__(self, "social_risk", clamp01(self.social_risk))
        object.__setattr__(self, "urgency", clamp01(self.urgency))

    def expected_utility(self, listener: OtherMindModel) -> float:
        receptivity = listener.expected_receptivity(
            self.emotional_weight, self.social_risk)
        return (
            receptivity * (0.55 * self.confidence + 0.45 * self.urgency)
            - 0.35 * self.social_risk
        )

    def should_speak(self, listener: OtherMindModel, threshold: float = 0.12) -> bool:
        return self.expected_utility(listener) > threshold


@dataclass(frozen=True)
class MessageInterpretation:
    accepted_confidence: float
    suspicion_delta: float
    affect_delta: AffectField


def interpret_intent(
    intent: CommunicationIntent,
    listener: OtherMindModel,
    affect: AffectField,
) -> MessageInterpretation:
    """Interpret speech through trust, uncertainty, and affective distortion."""
    trust_gate = clamp01(listener.trust * 0.65 + listener.respect * 0.25
                         + (1.0 - listener.fear) * 0.10)
    distortion = clamp01(listener.uncertainty * 0.35
                         + affect.threat_salience * 0.30
                         + intent.social_risk * 0.35)
    accepted = clamp01(intent.confidence * trust_gate * (1.0 - 0.5 * distortion))
    suspicion = clamp01(distortion + (1.0 - trust_gate) * 0.35)

    updated_affect = affect.copy()
    updated_affect.update(
        prediction_error=distortion,
        uncertainty=listener.uncertainty,
        threat=max(intent.social_risk, affect.threat_salience),
        social_support=listener.trust,
        control_feedback=accepted - suspicion,
    )
    return MessageInterpretation(
        accepted_confidence=accepted,
        suspicion_delta=suspicion,
        affect_delta=updated_affect,
    )

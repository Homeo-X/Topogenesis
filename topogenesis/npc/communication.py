from __future__ import annotations

from dataclasses import dataclass

from .affect import AffectField
from .social import OtherMindModel


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


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

    def expected_utility(self, listener: OtherMindModel) -> float:
        receptivity = listener.expected_receptivity(
            self.emotional_weight, self.social_risk)
        return (
            receptivity * (0.55 * _clamp01(self.confidence)
                           + 0.45 * _clamp01(self.urgency))
            - 0.35 * _clamp01(self.social_risk)
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
    trust_gate = _clamp01(listener.trust * 0.65 + listener.respect * 0.25
                          + (1.0 - listener.fear) * 0.10)
    distortion = _clamp01(listener.uncertainty * 0.35
                          + affect.threat_salience * 0.30
                          + intent.social_risk * 0.35)
    accepted = _clamp01(intent.confidence * trust_gate * (1.0 - 0.5 * distortion))
    suspicion = _clamp01(distortion + (1.0 - trust_gate) * 0.35)

    updated_affect = AffectField(*affect.as_vector())
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

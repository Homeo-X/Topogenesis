from __future__ import annotations

from dataclasses import dataclass

from ._math import clamp, clamp01, non_empty_text
from .affect import AffectField
from .communication import CommunicationIntent
from .needs import NeedPressure
from .social import OtherMindModel


@dataclass(frozen=True)
class FutureOutcome:
    action: str
    stability_delta: float
    social_delta: float
    expected_risk: float
    confidence: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", non_empty_text(self.action, field_name="action"))
        object.__setattr__(
            self, "stability_delta", clamp(self.stability_delta, -1.0, 1.0))
        object.__setattr__(
            self, "social_delta", clamp(self.social_delta, -1.0, 1.0))
        object.__setattr__(self, "expected_risk", clamp01(self.expected_risk))
        object.__setattr__(self, "confidence", clamp01(self.confidence))

    @property
    def value(self) -> float:
        return (
            0.45 * self.stability_delta
            + 0.25 * self.social_delta
            + 0.20 * self.confidence
            - 0.35 * self.expected_risk
        )


def simulate_future(
    *,
    action: str,
    needs: NeedPressure,
    affect: AffectField,
    listener: OtherMindModel | None = None,
    intent: CommunicationIntent | None = None,
) -> FutureOutcome:
    """Small controlled-imagination primitive for action comparison."""
    base_risk = max(needs.safety, needs.repair, affect.threat_salience)
    stability_gain = 0.25 * (1.0 - needs.total) - 0.20 * affect.cognitive_load
    social_gain = 0.0
    action = non_empty_text(action, field_name="action")
    confidence = clamp01(affect.prediction_coherence * (1.0 - affect.cognitive_load))

    if intent is not None and listener is not None:
        utility = intent.expected_utility(listener)
        social_gain = max(-0.3, min(0.5, utility))
        base_risk = clamp01(base_risk + intent.social_risk * 0.35
                            - listener.trust * 0.15)
        confidence = clamp01(0.5 * confidence + 0.5 * listener.expected_receptivity(
            intent.emotional_weight, intent.social_risk))

    if action in {"seek_food", "rest", "repair"}:
        stability_gain += 0.35 * max(needs.metabolic, needs.repair)
    elif action in {"investigate", "verify"}:
        stability_gain += 0.30 * needs.epistemic
        base_risk = clamp01(base_risk + 0.10)
    elif action in {"withdraw", "hide"}:
        stability_gain += 0.20 * needs.safety
        social_gain -= 0.10

    return FutureOutcome(
        action=action,
        stability_delta=clamp(stability_gain, -1.0, 1.0),
        social_delta=clamp(social_gain, -1.0, 1.0),
        expected_risk=clamp01(base_risk),
        confidence=confidence,
    )

from __future__ import annotations

from dataclasses import dataclass

from .affect import AffectField
from .communication import CommunicationIntent
from .needs import NeedPressure
from .social import OtherMindModel


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class FutureOutcome:
    action: str
    stability_delta: float
    social_delta: float
    expected_risk: float
    confidence: float

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
    confidence = _clamp01(affect.prediction_coherence * (1.0 - affect.cognitive_load))

    if intent is not None and listener is not None:
        utility = intent.expected_utility(listener)
        social_gain = max(-0.3, min(0.5, utility))
        base_risk = _clamp01(base_risk + intent.social_risk * 0.35
                             - listener.trust * 0.15)
        confidence = _clamp01(0.5 * confidence + 0.5 * listener.expected_receptivity(
            intent.emotional_weight, intent.social_risk))

    if action in {"seek_food", "rest", "repair"}:
        stability_gain += 0.35 * max(needs.metabolic, needs.repair)
    elif action in {"investigate", "verify"}:
        stability_gain += 0.30 * needs.epistemic
        base_risk = _clamp01(base_risk + 0.10)
    elif action in {"withdraw", "hide"}:
        stability_gain += 0.20 * needs.safety
        social_gain -= 0.10

    return FutureOutcome(
        action=action,
        stability_delta=max(-1.0, min(1.0, stability_gain)),
        social_delta=max(-1.0, min(1.0, social_gain)),
        expected_risk=_clamp01(base_risk),
        confidence=confidence,
    )

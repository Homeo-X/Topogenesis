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
    depth: int = 1
    viability_delta: float = 0.0
    cognitive_cost: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", non_empty_text(self.action, field_name="action"))
        object.__setattr__(
            self, "stability_delta", clamp(self.stability_delta, -1.0, 1.0))
        object.__setattr__(
            self, "social_delta", clamp(self.social_delta, -1.0, 1.0))
        object.__setattr__(self, "expected_risk", clamp01(self.expected_risk))
        object.__setattr__(self, "confidence", clamp01(self.confidence))
        object.__setattr__(self, "depth", int(clamp(self.depth, 1, 3, default=1)))
        object.__setattr__(
            self, "viability_delta", clamp(self.viability_delta, -1.0, 1.0))
        object.__setattr__(self, "cognitive_cost", clamp01(self.cognitive_cost))

    @property
    def value(self) -> float:
        return (
            0.45 * self.stability_delta
            + 0.25 * self.social_delta
            + 0.25 * self.viability_delta
            + 0.20 * self.confidence
            - 0.35 * self.expected_risk
            - 0.30 * self.cognitive_cost
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
        depth=1,
        viability_delta=clamp(stability_gain - base_risk * 0.25, -1.0, 1.0),
        cognitive_cost=0.04,
    )


def simulate_hierarchical_future(
    *,
    action: str,
    needs: NeedPressure,
    affect: AffectField,
    listener: OtherMindModel | None = None,
    intent: CommunicationIntent | None = None,
    depth: int = 3,
) -> FutureOutcome:
    """Simulate immediate, social, and longer viability consequences.

    Depth 1 estimates direct pressure change. Depth 2 adds other-agent reaction.
    Depth 3 adds projected viability cost/benefit over the next short horizon.
    Each deeper level adds cognitive cost so planning is useful but not free.
    """
    depth = int(clamp(depth, 1, 3, default=1))
    immediate = simulate_future(
        action=action,
        needs=needs,
        affect=affect,
        listener=listener,
        intent=intent,
    )
    stability = immediate.stability_delta
    social = immediate.social_delta
    risk = immediate.expected_risk
    confidence = immediate.confidence
    viability = immediate.viability_delta
    cognitive_cost = 0.035 * depth + 0.055 * max(0, depth - 1) * (1.0 + affect.cognitive_load)

    if depth >= 2:
        receptivity = 0.45
        if listener is not None:
            receptivity = listener.expected_receptivity(
                intent.emotional_weight if intent is not None else affect.threat_salience,
                intent.social_risk if intent is not None else needs.safety,
            )
        if action in {"communicate", "share_information", "coordinate", "ask_help"}:
            social += 0.30 * receptivity - 0.10 * needs.safety
            confidence = 0.5 * confidence + 0.5 * receptivity
        elif action in {"withdraw", "hide"}:
            social -= 0.10 + 0.15 * (1.0 - receptivity)
        else:
            social += 0.08 * receptivity - 0.06 * affect.threat_salience
        risk = clamp01(risk + 0.10 * (1.0 - receptivity) - 0.05 * social)

    if depth >= 3:
        if action in {"seek_food", "rest", "repair"}:
            viability += 0.22 * max(needs.metabolic, needs.repair)
            risk = clamp01(risk - 0.08 * affect.safety)
        elif action in {"seek_shelter", "withdraw", "hide"}:
            viability += 0.18 * needs.safety
            social -= 0.04
        elif action in {"investigate", "verify"}:
            viability += 0.16 * needs.epistemic
            risk = clamp01(risk + 0.08 * (1.0 - affect.control))
        elif action in {"communicate", "share_information", "coordinate", "ask_help"}:
            viability += 0.12 * max(needs.social, needs.attachment)
            social += 0.10
        viability -= 0.18 * cognitive_cost
        confidence = clamp01(confidence - 0.08 * affect.cognitive_load + 0.04 * depth)

    return FutureOutcome(
        action=action,
        stability_delta=clamp(stability, -1.0, 1.0),
        social_delta=clamp(social, -1.0, 1.0),
        expected_risk=clamp01(risk),
        confidence=clamp01(confidence),
        depth=depth,
        viability_delta=clamp(viability, -1.0, 1.0),
        cognitive_cost=cognitive_cost,
    )

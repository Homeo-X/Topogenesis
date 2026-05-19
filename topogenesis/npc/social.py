from __future__ import annotations

from dataclasses import dataclass, field


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass
class Attachment:
    entity_id: str
    importance: float
    dependency: float = 0.0
    trust: float = 0.5

    def rupture_strength(self, observed_absence: float) -> float:
        return _clamp01(observed_absence * self.importance
                        * (0.5 + 0.5 * self.dependency))


@dataclass
class OtherMindModel:
    """Compact model of another agent used for social prediction."""

    agent_id: str
    trust: float = 0.5
    fear: float = 0.0
    respect: float = 0.5
    uncertainty: float = 0.5
    beliefs: dict[str, float] = field(default_factory=dict)
    goals: dict[str, float] = field(default_factory=dict)

    def update_belief(self, claim: str, confidence: float, source_trust: float) -> None:
        confidence = _clamp01(confidence)
        source_trust = _clamp01(source_trust)
        prior = self.beliefs.get(claim, 0.5)
        weight = 0.15 + 0.70 * source_trust
        self.beliefs[claim] = _clamp01(prior * (1.0 - weight) + confidence * weight)
        self.uncertainty = _clamp01(self.uncertainty * 0.9 + (1.0 - source_trust) * 0.1)

    def expected_receptivity(self, emotional_weight: float, social_risk: float) -> float:
        return _clamp01(
            0.35 * self.trust
            + 0.25 * self.respect
            + 0.20 * (1.0 - self.uncertainty)
            + 0.10 * emotional_weight
            - 0.25 * self.fear
            - 0.20 * social_risk
        )


@dataclass
class SocialMemory:
    """Persistent social traces: promises, insults, aid, betrayal, rumors."""

    events: list[dict[str, object]] = field(default_factory=list)

    def remember(self, *, agent_id: str, kind: str, salience: float,
                 valence: float, claim: str | None = None) -> None:
        self.events.append({
            "agent_id": agent_id,
            "kind": kind,
            "salience": _clamp01(salience),
            "valence": max(-1.0, min(1.0, float(valence))),
            "claim": claim,
        })

    def reputation(self, agent_id: str) -> float:
        matching = [event for event in self.events if event["agent_id"] == agent_id]
        if not matching:
            return 0.0
        weighted = sum(float(event["valence"]) * float(event["salience"])
                       for event in matching)
        denom = sum(float(event["salience"]) for event in matching) + 1e-8
        return max(-1.0, min(1.0, weighted / denom))

from __future__ import annotations

from dataclasses import dataclass, field

from ._math import clamp, clamp01, non_empty_text, normalized_confidence_map


@dataclass
class Attachment:
    entity_id: str
    importance: float
    dependency: float = 0.0
    trust: float = 0.5

    def __post_init__(self) -> None:
        self.entity_id = non_empty_text(self.entity_id, field_name="entity_id")
        self.importance = clamp01(self.importance)
        self.dependency = clamp01(self.dependency)
        self.trust = clamp01(self.trust, default=0.5)

    def rupture_strength(self, observed_absence: float) -> float:
        return clamp01(observed_absence * self.importance
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

    def __post_init__(self) -> None:
        self.agent_id = non_empty_text(self.agent_id, field_name="agent_id")
        self.trust = clamp01(self.trust, default=0.5)
        self.fear = clamp01(self.fear)
        self.respect = clamp01(self.respect, default=0.5)
        self.uncertainty = clamp01(self.uncertainty, default=0.5)
        self.beliefs = normalized_confidence_map(self.beliefs)
        self.goals = normalized_confidence_map(self.goals)

    def update_belief(self, claim: str, confidence: float, source_trust: float) -> None:
        claim = non_empty_text(claim, field_name="claim")
        confidence = clamp01(confidence)
        source_trust = clamp01(source_trust)
        prior = self.beliefs.get(claim, 0.5)
        weight = 0.15 + 0.70 * source_trust
        self.beliefs[claim] = clamp01(prior * (1.0 - weight) + confidence * weight)
        self.uncertainty = clamp01(
            self.uncertainty * 0.9 + (1.0 - source_trust) * 0.1)

    def expected_receptivity(self, emotional_weight: float, social_risk: float) -> float:
        return clamp01(
            0.35 * self.trust
            + 0.25 * self.respect
            + 0.20 * (1.0 - self.uncertainty)
            + 0.10 * clamp01(emotional_weight)
            - 0.25 * self.fear
            - 0.20 * clamp01(social_risk)
        )


@dataclass
class SocialMemory:
    """Persistent social traces: promises, insults, aid, betrayal, rumors."""

    events: list[dict[str, object]] = field(default_factory=list)
    max_events: int = 1024

    def __post_init__(self) -> None:
        self.max_events = max(1, int(clamp(self.max_events, 1, 1_000_000, default=1024)))
        clean_events = []
        for event in self.events:
            try:
                clean_events.append(self._normalize_event(event))
            except (KeyError, TypeError, ValueError):
                continue
        self.events = clean_events[-self.max_events:]

    def _normalize_event(self, event: dict[str, object]) -> dict[str, object]:
        claim = event.get("claim")
        return {
            "agent_id": non_empty_text(str(event["agent_id"]), field_name="agent_id"),
            "kind": non_empty_text(str(event["kind"]), field_name="kind"),
            "salience": clamp01(float(event.get("salience", 0.0))),
            "valence": clamp(float(event.get("valence", 0.0)), -1.0, 1.0),
            "claim": None if claim is None else non_empty_text(str(claim), field_name="claim"),
        }

    def remember(self, *, agent_id: str, kind: str, salience: float,
                 valence: float, claim: str | None = None) -> None:
        agent_id = non_empty_text(agent_id, field_name="agent_id")
        kind = non_empty_text(kind, field_name="kind")
        self.events.append(self._normalize_event({
            "agent_id": agent_id,
            "kind": kind,
            "salience": clamp01(salience),
            "valence": clamp(valence, -1.0, 1.0),
            "claim": None if claim is None else non_empty_text(claim, field_name="claim"),
        }))
        if len(self.events) > self.max_events:
            del self.events[:len(self.events) - self.max_events]

    def reputation(self, agent_id: str) -> float:
        matching = [event for event in self.events if event["agent_id"] == agent_id]
        if not matching:
            return 0.0
        weighted = sum(float(event["valence"]) * float(event["salience"])
                       for event in matching)
        denom = sum(float(event["salience"]) for event in matching) + 1e-8
        return max(-1.0, min(1.0, weighted / denom))

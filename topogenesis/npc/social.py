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
    pressure_estimates: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.agent_id = non_empty_text(self.agent_id, field_name="agent_id")
        self.trust = clamp01(self.trust, default=0.5)
        self.fear = clamp01(self.fear)
        self.respect = clamp01(self.respect, default=0.5)
        self.uncertainty = clamp01(self.uncertainty, default=0.5)
        self.beliefs = normalized_confidence_map(self.beliefs)
        self.goals = normalized_confidence_map(self.goals)
        self.pressure_estimates = normalized_confidence_map(self.pressure_estimates)

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

    def observe_pressure(self, *, need: str, intensity: float, affect: float) -> None:
        need = non_empty_text(need, field_name="need")
        intensity = clamp01(intensity)
        affect = clamp01(affect, default=0.5)
        prior = self.pressure_estimates.get(need, 0.5)
        self.pressure_estimates[need] = clamp01(0.72 * prior + 0.28 * intensity)
        self.uncertainty = clamp01(0.92 * self.uncertainty + 0.08 * (1.0 - affect))


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


@dataclass
class EpisodicSemanticMemory:
    """Event memory plus consolidated general patterns."""

    episodes: list[dict[str, object]] = field(default_factory=list)
    semantics: dict[str, dict[str, float]] = field(default_factory=dict)
    max_episodes: int = 256
    decay_rate: float = 0.015
    consolidation_threshold: float = 0.58

    def __post_init__(self) -> None:
        self.max_episodes = max(1, int(clamp(self.max_episodes, 1, 1_000_000, default=256)))
        self.decay_rate = clamp01(self.decay_rate, default=0.015)
        self.consolidation_threshold = clamp01(self.consolidation_threshold, default=0.58)
        clean = []
        for episode in self.episodes:
            try:
                clean.append(self._normalize_episode(episode))
            except (KeyError, TypeError, ValueError):
                continue
        self.episodes = clean[-self.max_episodes:]
        self.semantics = {
            non_empty_text(str(key), field_name="semantic_key"): {
                "confidence": clamp01(float(value.get("confidence", 0.0))),
                "valence": clamp(float(value.get("valence", 0.0)), -1.0, 1.0),
            }
            for key, value in self.semantics.items()
            if isinstance(value, dict)
        }

    def _normalize_episode(self, episode: dict[str, object]) -> dict[str, object]:
        agents = episode.get("agents", [])
        if not isinstance(agents, list):
            agents = [str(agents)]
        clean_agents = [
            non_empty_text(str(agent), field_name="agent")
            for agent in agents
            if str(agent).strip()
        ]
        return {
            "tick": int(clamp(float(episode.get("tick", 0.0)), 0.0, 1_000_000_000.0)),
            "kind": non_empty_text(str(episode["kind"]), field_name="kind"),
            "agents": clean_agents,
            "place": non_empty_text(str(episode.get("place", "local")), field_name="place"),
            "salience": clamp01(float(episode.get("salience", 0.0))),
            "valence": clamp(float(episode.get("valence", 0.0)), -1.0, 1.0),
            "affect_intensity": clamp01(float(episode.get("affect_intensity", 0.0))),
            "claim": None if episode.get("claim") is None else non_empty_text(str(episode.get("claim")), field_name="claim"),
        }

    def remember_episode(
        self,
        *,
        tick: int,
        kind: str,
        agents: list[str],
        place: str,
        salience: float,
        valence: float,
        affect_intensity: float,
        claim: str | None = None,
    ) -> None:
        self.episodes.append(self._normalize_episode({
            "tick": tick,
            "kind": kind,
            "agents": agents,
            "place": place,
            "salience": salience,
            "valence": valence,
            "affect_intensity": affect_intensity,
            "claim": claim,
        }))
        if len(self.episodes) > self.max_episodes:
            del self.episodes[:len(self.episodes) - self.max_episodes]
        self.consolidate()

    def decay(self) -> None:
        kept = []
        for episode in self.episodes:
            affect = float(episode["affect_intensity"])
            episode["salience"] = clamp01(float(episode["salience"]) * (1.0 - self.decay_rate * (1.0 - affect)))
            if float(episode["salience"]) > 0.03:
                kept.append(episode)
        self.episodes = kept[-self.max_episodes:]

    def consolidate(self) -> None:
        for episode in self.episodes:
            strength = float(episode["salience"]) * (0.55 + 0.45 * float(episode["affect_intensity"]))
            if strength < self.consolidation_threshold:
                continue
            keys = [f"place:{episode['place']}:{episode['kind']}"]
            keys.extend(f"agent:{agent}:{episode['kind']}" for agent in episode["agents"])
            for key in keys:
                prior = self.semantics.get(key, {"confidence": 0.0, "valence": 0.0})
                confidence = clamp01(0.80 * prior["confidence"] + 0.20 * strength)
                valence = clamp(0.80 * prior["valence"] + 0.20 * float(episode["valence"]), -1.0, 1.0)
                self.semantics[key] = {"confidence": confidence, "valence": valence}

    def semantic_confidence(self, key: str) -> float:
        return clamp01(self.semantics.get(key, {}).get("confidence", 0.0))

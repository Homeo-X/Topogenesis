from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any

from topogenesis.npc._math import clamp, clamp01, non_empty_text


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out or out in (float("inf"), float("-inf")):
        return default
    return out


def _clean_features(features: dict[str, Any]) -> dict[str, float]:
    return {
        non_empty_text(str(key), field_name="feature"): _finite_float(value)
        for key, value in features.items()
        if str(key).strip()
    }


@dataclass(frozen=True)
class SensoryObservation:
    """A bounded sensory update from perception into the cognitive loop."""

    entity_id: str
    features: dict[str, float]
    tick: int
    modality: str = "generic"
    confidence: float = 1.0
    salience: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_id", non_empty_text(self.entity_id, field_name="entity_id"))
        object.__setattr__(self, "features", _clean_features(self.features))
        object.__setattr__(self, "tick", int(max(0, _finite_float(self.tick))))
        object.__setattr__(self, "modality", non_empty_text(self.modality, field_name="modality"))
        object.__setattr__(self, "confidence", clamp01(self.confidence, default=1.0))
        object.__setattr__(self, "salience", clamp01(self.salience, default=0.5))


@dataclass(frozen=True)
class Prediction:
    entity_id: str
    expected_features: dict[str, float]
    confidence: float
    source_tick: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_id", non_empty_text(self.entity_id, field_name="entity_id"))
        object.__setattr__(self, "expected_features", _clean_features(self.expected_features))
        object.__setattr__(self, "confidence", clamp01(self.confidence, default=0.5))
        object.__setattr__(self, "source_tick", int(max(0, _finite_float(self.source_tick))))


@dataclass(frozen=True)
class ConfirmationSignal:
    entity_id: str
    prediction_error: float
    confirmation: float
    contradiction: float
    novelty: float
    update_weight: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_id", non_empty_text(self.entity_id, field_name="entity_id"))
        object.__setattr__(self, "prediction_error", clamp01(self.prediction_error))
        object.__setattr__(self, "confirmation", clamp01(self.confirmation))
        object.__setattr__(self, "contradiction", clamp01(self.contradiction))
        object.__setattr__(self, "novelty", clamp01(self.novelty))
        object.__setattr__(self, "update_weight", clamp01(self.update_weight))


@dataclass
class WorldEntity:
    """Persistent object in the agent's world model."""

    entity_id: str
    beliefs: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.25
    uncertainty: float = 0.75
    last_confirmed_tick: int = 0
    confirmation_count: int = 0
    contradiction_count: int = 0

    def __post_init__(self) -> None:
        self.entity_id = non_empty_text(self.entity_id, field_name="entity_id")
        self.beliefs = _clean_features(self.beliefs)
        self.confidence = clamp01(self.confidence, default=0.25)
        self.uncertainty = clamp01(self.uncertainty, default=1.0 - self.confidence)
        self.last_confirmed_tick = int(max(0, _finite_float(self.last_confirmed_tick)))
        self.confirmation_count = max(0, int(self.confirmation_count))
        self.contradiction_count = max(0, int(self.contradiction_count))

    def predict(self, tick: int) -> Prediction:
        age = max(0, int(tick) - self.last_confirmed_tick)
        age_decay = clamp01(1.0 / (1.0 + 0.03 * age))
        return Prediction(
            entity_id=self.entity_id,
            expected_features=dict(self.beliefs),
            confidence=self.confidence * age_decay,
            source_tick=self.last_confirmed_tick,
        )

    def apply_observation(
        self,
        observation: SensoryObservation,
        signal: ConfirmationSignal,
    ) -> None:
        weight = signal.update_weight
        for key, observed in observation.features.items():
            prior = self.beliefs.get(key, observed)
            self.beliefs[key] = prior * (1.0 - weight) + observed * weight
        if signal.confirmation >= signal.contradiction:
            self.confirmation_count += 1
        else:
            self.contradiction_count += 1
        confirmation_gain = observation.confidence * signal.confirmation
        contradiction_penalty = signal.contradiction * 0.12
        self.confidence = clamp01(
            self.confidence
            + 0.28 * confirmation_gain
            + 0.04 * observation.confidence * (1.0 - signal.prediction_error)
            - contradiction_penalty
        )
        self.uncertainty = clamp01(
            self.uncertainty * 0.75
            + signal.prediction_error * 0.20
            + signal.novelty * 0.10
            - signal.confirmation * 0.15
        )
        self.last_confirmed_tick = max(self.last_confirmed_tick, observation.tick)


@dataclass
class PersistentMemory:
    """Bounded episodic and semantic memory updated by sensory evidence."""

    episodes: list[dict[str, Any]] = field(default_factory=list)
    semantics: dict[str, dict[str, float]] = field(default_factory=dict)
    max_episodes: int = 512
    consolidation_threshold: float = 0.62

    def __post_init__(self) -> None:
        self.max_episodes = max(1, int(clamp(self.max_episodes, 1, 1_000_000, default=512)))
        self.consolidation_threshold = clamp01(self.consolidation_threshold, default=0.62)
        self.episodes = self.episodes[-self.max_episodes:]

    def record(
        self,
        observation: SensoryObservation,
        signal: ConfirmationSignal,
    ) -> None:
        episode = {
            "tick": observation.tick,
            "entity_id": observation.entity_id,
            "features": dict(observation.features),
            "modality": observation.modality,
            "salience": observation.salience,
            "confirmation": signal.confirmation,
            "contradiction": signal.contradiction,
            "prediction_error": signal.prediction_error,
        }
        self.episodes.append(episode)
        if len(self.episodes) > self.max_episodes:
            del self.episodes[:len(self.episodes) - self.max_episodes]
        self._consolidate(observation, signal)

    def _consolidate(
        self,
        observation: SensoryObservation,
        signal: ConfirmationSignal,
    ) -> None:
        strength = observation.salience * observation.confidence * max(
            signal.confirmation,
            signal.contradiction,
            signal.novelty,
        )
        if strength < self.consolidation_threshold:
            return
        for key, value in observation.features.items():
            semantic_key = f"{observation.entity_id}:{key}"
            prior = self.semantics.get(semantic_key, {"value": value, "confidence": 0.0})
            confidence = clamp01(prior["confidence"] * 0.82 + strength * 0.18)
            merged_value = prior["value"] * (1.0 - strength) + value * strength
            self.semantics[semantic_key] = {
                "value": merged_value,
                "confidence": confidence,
            }

    def semantic_confidence(self, semantic_key: str) -> float:
        return clamp01(self.semantics.get(semantic_key, {}).get("confidence", 0.0))


@dataclass
class PersistentWorldModel:
    """A persistent predictive model updated only through evidence signals."""

    entities: dict[str, WorldEntity] = field(default_factory=dict)

    def predict(self, entity_id: str, tick: int) -> Prediction:
        entity_id = non_empty_text(entity_id, field_name="entity_id")
        entity = self.entities.get(entity_id)
        if entity is None:
            return Prediction(entity_id=entity_id, expected_features={}, confidence=0.0, source_tick=0)
        return entity.predict(tick)

    def update_from_observation(
        self,
        observation: SensoryObservation,
        signal: ConfirmationSignal,
    ) -> WorldEntity:
        entity = self.entities.get(observation.entity_id)
        if entity is None:
            entity = WorldEntity(
                entity_id=observation.entity_id,
                beliefs=dict(observation.features),
                confidence=observation.confidence * max(0.20, signal.confirmation),
                uncertainty=1.0 - observation.confidence,
                last_confirmed_tick=observation.tick,
            )
            self.entities[observation.entity_id] = entity
            return entity
        entity.apply_observation(observation, signal)
        return entity


def compare_prediction(
    prediction: Prediction,
    observation: SensoryObservation,
) -> ConfirmationSignal:
    """Compare predicted features with sensory input and compute update gates."""
    predicted_keys = set(prediction.expected_features)
    observed_keys = set(observation.features)
    shared = predicted_keys & observed_keys
    novel = observed_keys - predicted_keys
    if not prediction.expected_features:
        error = 1.0 if observation.features else 0.0
        novelty = 1.0 if observation.features else 0.0
    elif shared:
        squared = [
            (prediction.expected_features[key] - observation.features[key]) ** 2
            for key in shared
        ]
        error = clamp01(sqrt(sum(squared) / max(1, len(squared))))
        novelty = clamp01(len(novel) / max(1, len(observed_keys)))
    else:
        error = 1.0
        novelty = 1.0
    confirmation = clamp01((1.0 - error) * prediction.confidence * observation.confidence)
    contradiction = clamp01(error * prediction.confidence * observation.confidence)
    update_weight = clamp01(
        0.10
        + 0.55 * observation.confidence
        + 0.20 * novelty
        + 0.15 * contradiction
    )
    return ConfirmationSignal(
        entity_id=observation.entity_id,
        prediction_error=error,
        confirmation=confirmation,
        contradiction=contradiction,
        novelty=novelty,
        update_weight=update_weight,
    )


@dataclass
class SensoryConfirmationLoop:
    """Predictive cognition loop: predict, sense, compare, update, remember."""

    world_model: PersistentWorldModel = field(default_factory=PersistentWorldModel)
    memory: PersistentMemory = field(default_factory=PersistentMemory)
    tick: int = 0

    def observe(self, observation: SensoryObservation) -> ConfirmationSignal:
        self.tick = max(self.tick, observation.tick)
        prediction = self.world_model.predict(observation.entity_id, observation.tick)
        signal = compare_prediction(prediction, observation)
        self.world_model.update_from_observation(observation, signal)
        self.memory.record(observation, signal)
        return signal

    def predict(self, entity_id: str, tick: int | None = None) -> Prediction:
        return self.world_model.predict(entity_id, self.tick if tick is None else tick)

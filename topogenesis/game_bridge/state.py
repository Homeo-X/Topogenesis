from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from topogenesis.npc import (
    AffectField,
    CommunicationIntent,
    NeedPressure,
    OtherMindModel,
    SocialMemory,
    ViabilityState,
    interpret_intent,
    simulate_future,
)


def _clamp(value: Any, low: float = 0.0, high: float = 1.0, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = default
    if out != out or out in (float("inf"), float("-inf")):
        out = default
    return max(low, min(high, out))


@dataclass
class BridgeNpc:
    npc_id: str
    display_name: str
    affect: AffectField = field(default_factory=AffectField)
    mind: OtherMindModel = field(init=False)
    memory: SocialMemory = field(default_factory=lambda: SocialMemory(max_events=128))
    energy: float = 1.0
    bodily_integrity: float = 1.0
    prediction_coherence: float = 0.78
    social_stability: float = 0.58
    environmental_safety: float = 0.76
    trust_player: float = 0.5
    future_action: str = "observe"
    need_total: float = 0.12
    dominant_need: str = "epistemic"

    def __post_init__(self) -> None:
        self.mind = OtherMindModel(agent_id=self.npc_id, trust=self.trust_player)

    def step(self, delta: float, pressure: dict[str, Any]) -> dict[str, Any]:
        hazard = _clamp(pressure.get("hazard", 0.0))
        resource = _clamp(pressure.get("resource", 0.0))
        player_help = _clamp(pressure.get("player_help", 0.0))
        player_threat = _clamp(pressure.get("player_threat", 0.0))

        self.energy = _clamp(self.energy - 0.006 * delta + 0.018 * resource * delta)
        self.environmental_safety = _clamp(1.0 - hazard)
        self.prediction_coherence = _clamp(
            0.95 * self.prediction_coherence
            + 0.05 * (1.0 - 0.55 * hazard + 0.15 * resource)
        )
        self.trust_player = _clamp(
            self.trust_player + 0.09 * player_help * delta - 0.12 * player_threat * delta
        )
        self.mind.trust = self.trust_player

        viability = ViabilityState(
            energy=self.energy,
            bodily_integrity=self.bodily_integrity,
            memory_integrity=1.0,
            prediction_coherence=self.prediction_coherence,
            social_stability=self.social_stability,
            attachment_integrity=self.trust_player,
            environmental_safety=self.environmental_safety,
        )
        needs = NeedPressure.from_viability(viability)
        self.need_total = needs.total
        self.dominant_need = needs.dominant
        self.affect.update(
            prediction_error=needs.epistemic,
            uncertainty=max(needs.epistemic, needs.safety),
            threat=max(needs.safety, player_threat),
            social_support=self.social_stability,
            control_feedback=1.0 - needs.total,
        )

        intent = CommunicationIntent(
            target_agent="player",
            intended_effect="coordinate",
            belief_to_modify="local_pressure",
            confidence=self.prediction_coherence,
            emotional_weight=self.affect.threat_salience,
            social_risk=needs.safety,
            urgency=max(needs.metabolic, needs.repair),
        )
        interpretation = interpret_intent(intent, self.mind, self.affect)
        if player_help > 0.0:
            self.memory.remember(
                agent_id="player",
                kind="interaction",
                salience=interpretation.accepted_confidence,
                valence=self.trust_player - needs.total,
                claim=needs.dominant,
            )

        action = "seek_food" if needs.metabolic >= needs.epistemic else "verify"
        future = simulate_future(
            action=action,
            needs=needs,
            affect=self.affect,
            listener=self.mind,
            intent=intent,
        )
        self.future_action = future.action
        return self.to_snapshot()

    def to_snapshot(self) -> dict[str, Any]:
        return {
            "display_name": self.display_name,
            "energy": self.energy,
            "bodily_integrity": self.bodily_integrity,
            "prediction_coherence": self.prediction_coherence,
            "social_stability": self.social_stability,
            "environmental_safety": self.environmental_safety,
            "need_total": self.need_total,
            "dominant_need": self.dominant_need,
            "affect_stability": self.affect.stability,
            "threat_salience": self.affect.threat_salience,
            "trust_player": self.trust_player,
            "future_action": self.future_action,
            "memory_events": list(self.memory.events),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        self.display_name = str(snapshot.get("display_name", self.display_name))
        self.energy = _clamp(snapshot.get("energy", self.energy))
        self.bodily_integrity = _clamp(snapshot.get("bodily_integrity", self.bodily_integrity))
        self.prediction_coherence = _clamp(
            snapshot.get("prediction_coherence", self.prediction_coherence))
        self.social_stability = _clamp(snapshot.get("social_stability", self.social_stability))
        self.environmental_safety = _clamp(
            snapshot.get("environmental_safety", self.environmental_safety))
        self.trust_player = _clamp(snapshot.get("trust_player", self.trust_player))
        self.memory = SocialMemory(
            events=list(snapshot.get("memory_events", [])),
            max_events=128,
        )
        self.mind.trust = self.trust_player


@dataclass
class GameBridgeState:
    npcs: dict[str, BridgeNpc] = field(default_factory=dict)
    tick: int = 0

    def register_npc(self, npc_id: str, display_name: str | None = None) -> None:
        npc_id = str(npc_id).strip()
        if not npc_id:
            raise ValueError("npc_id must be non-empty")
        if npc_id not in self.npcs:
            self.npcs[npc_id] = BridgeNpc(
                npc_id=npc_id,
                display_name=display_name or npc_id,
            )

    def step(self, payload: dict[str, Any]) -> dict[str, Any]:
        delta = _clamp(payload.get("delta", 0.016), 0.0, 1.0, default=0.016)
        snapshot = payload.get("snapshot", {})
        if isinstance(snapshot, dict):
            self.restore(snapshot)
        pressures = payload.get("pressures", {})
        if not isinstance(pressures, dict):
            pressures = {}
        for npc_id, pressure in pressures.items():
            self.register_npc(str(npc_id))
            self.npcs[str(npc_id)].step(
                delta,
                pressure if isinstance(pressure, dict) else {},
            )
        self.tick += 1
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        return {
            "version": 1,
            "tick": self.tick,
            "npcs": {
                npc_id: npc.to_snapshot()
                for npc_id, npc in self.npcs.items()
            },
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        npcs = snapshot.get("npcs", snapshot)
        if not isinstance(npcs, dict):
            return
        for npc_id, npc_snapshot in npcs.items():
            if not isinstance(npc_snapshot, dict):
                continue
            self.register_npc(str(npc_id), str(npc_snapshot.get("display_name", npc_id)))
            self.npcs[str(npc_id)].restore(npc_snapshot)

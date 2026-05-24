from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Any

from topogenesis.npc import (
    AffectField,
    CommunicationIntent,
    EpisodicSemanticMemory,
    NeedPressure,
    OtherMindModel,
    SocialMemory,
    ViabilityState,
    interpret_intent,
    simulate_future,
    simulate_hierarchical_future,
)
from topogenesis.world import OfflineConfig, OfflineSimulator, PopulationConfig, PopulationManager, WorldState
from topogenesis.world.core_engine_runtime import CoreEngineRuntime


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
    world_memory: EpisodicSemanticMemory = field(default_factory=lambda: EpisodicSemanticMemory(max_episodes=128))
    energy: float = 1.0
    bodily_integrity: float = 1.0
    prediction_coherence: float = 0.78
    social_stability: float = 0.58
    environmental_safety: float = 0.76
    trust_player: float = 0.5
    future_action: str = "observe"
    future_depth: int = 1
    imagination_fatigue: float = 0.0
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
        self.imagination_fatigue = _clamp(self.imagination_fatigue * 0.985)
        self.affect.update(
            prediction_error=needs.epistemic,
            uncertainty=max(needs.epistemic, needs.safety),
            threat=max(needs.safety, player_threat),
            social_support=self.social_stability,
            control_feedback=1.0 - needs.total,
        )
        self.mind.observe_pressure(
            need=needs.dominant,
            intensity=needs.total,
            affect=self.affect.stability,
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
            self.world_memory.remember_episode(
                tick=0,
                kind="player_help",
                agents=["player"],
                place="local",
                salience=interpretation.accepted_confidence,
                valence=self.trust_player - needs.total,
                affect_intensity=1.0 - self.affect.stability,
                claim=needs.dominant,
            )
        if player_threat > 0.0:
            self.world_memory.remember_episode(
                tick=0,
                kind="threat",
                agents=["player"],
                place="local",
                salience=max(player_threat, needs.safety),
                valence=-max(player_threat, needs.safety),
                affect_intensity=max(self.affect.threat_salience, player_threat),
                claim="player_threat",
            )
        self.world_memory.decay()

        options = self._generate_action_options(needs, intent)
        depth = 3 if needs.total > 0.34 and self.imagination_fatigue < 0.72 else 2
        futures = [
            simulate_hierarchical_future(
                action=action,
                needs=needs,
                affect=self.affect,
                listener=self.mind,
                intent=intent if action in {"communicate", "share_information", "coordinate", "ask_help"} else None,
                depth=depth,
            )
            for action in options
        ]
        future = max(futures, key=lambda item: item.value)
        self.imagination_fatigue = _clamp(self.imagination_fatigue + future.cognitive_cost * delta)
        self.energy = _clamp(self.energy - 0.0015 * future.cognitive_cost * delta)
        self.future_action = future.action
        self.future_depth = future.depth
        return self.to_snapshot()

    def _generate_action_options(self, needs: NeedPressure, intent: CommunicationIntent) -> list[str]:
        options = ["verify", "rest"]
        if needs.metabolic > 0.12:
            options.append("seek_food")
        if needs.repair > 0.14:
            options.append("repair")
        if needs.safety > 0.14:
            options.extend(["seek_shelter", "withdraw"])
        if max(needs.social, needs.attachment) > 0.12 or intent.should_speak(self.mind):
            options.extend(["communicate", "coordinate"])
        return list(dict.fromkeys(options))

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
            "future_depth": self.future_depth,
            "imagination_fatigue": self.imagination_fatigue,
            "memory_events": list(self.memory.events),
            "semantic_memory": dict(self.world_memory.semantics),
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
        self.world_memory = EpisodicSemanticMemory(
            semantics=dict(snapshot.get("semantic_memory", {})),
            max_episodes=128,
        )
        self.mind.trust = self.trust_player


@dataclass
class GameBridgeState:
    npcs: dict[str, BridgeNpc] = field(default_factory=dict)
    tick: int = 0
    full_engine_enabled: bool = False
    full_engine_interval: int = 8
    core_runtime: CoreEngineRuntime | None = None
    _core_thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _core_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _core_last_started_tick: int = field(default=-1, init=False, repr=False)
    offline: OfflineSimulator = field(default_factory=lambda: OfflineSimulator(
        world=WorldState.default(ticks_per_day=60, seed=11),
        population_manager=PopulationManager(
            PopulationConfig(initial_population=300, max_population=800),
            seed=11,
        ),
        config=OfflineConfig(ticks_per_day=60, metrics_interval=60, seed=11),
    ))

    def __post_init__(self) -> None:
        if self.full_engine_enabled and self.core_runtime is None:
            self.core_runtime = CoreEngineRuntime(agent_count=1, world_size=16, seed=11)

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
        self.offline.run_ticks(1)
        self._maybe_start_core_step()
        self.tick += 1
        return self.snapshot()

    def _maybe_start_core_step(self) -> None:
        if not self.full_engine_enabled or self.core_runtime is None:
            return
        interval = max(1, int(self.full_engine_interval))
        if self.tick % interval != 0 or self._core_last_started_tick == self.tick:
            return
        if self._core_thread is not None and self._core_thread.is_alive():
            return
        self._core_last_started_tick = self.tick
        self._core_thread = threading.Thread(
            target=self._run_core_step_background,
            name="topogenesis-core-engine",
            daemon=True,
        )
        self._core_thread.start()

    def _run_core_step_background(self) -> None:
        if self.core_runtime is None:
            return
        with self._core_lock:
            self.core_runtime.step()

    def world_snapshot(self, visible_limit: int = 48) -> dict[str, Any]:
        snapshot = self.offline.snapshot(visible_limit=visible_limit)
        if self.full_engine_enabled and self.core_runtime is not None:
            core_snapshot = self.core_runtime.snapshot()
            snapshot["core_engine"] = core_snapshot
            if core_snapshot.get("agents") and snapshot.get("npcs"):
                core_agent = core_snapshot["agents"][0]
                metrics = core_agent.get("metrics", {})
                snapshot["npcs"][0].update({
                    "id": "core_0",
                    "display_name": core_agent.get("display_name", "Core Organism"),
                    "calling": "full core organism",
                    "position": core_agent.get("position", snapshot["npcs"][0].get("position")),
                    "energy": core_agent.get("energy", snapshot["npcs"][0].get("energy")),
                    "bodily_integrity": core_agent.get("health", snapshot["npcs"][0].get("bodily_integrity")),
                    "affect_stability": max(0.0, min(1.0, float(core_agent.get("membrane", 1.0)))),
                    "need_total": metrics.get("need_total", snapshot["npcs"][0].get("need_total")),
                    "dominant_need": metrics.get("dominant_need", snapshot["npcs"][0].get("dominant_need")),
                    "future_action": metrics.get("future_action", snapshot["npcs"][0].get("future_action")),
                    "engine_mode": "full_core",
                })
        else:
            snapshot["core_engine"] = {
                "enabled": False,
                "mode": "offline_population_only",
                "initialized": False,
                "tick": 0,
                "agent_count": 0,
                "last_error": "",
                "agents": [],
            }
        return snapshot

    def director_snapshot(self) -> dict[str, Any]:
        """Return game-facing signals for objectives, UI, and encounter pacing."""
        world = self.world_snapshot(visible_limit=24)
        npcs = world.get("npcs", [])
        world_meta = world.get("world", {})
        living_count = int(world_meta.get("population", len(npcs)) or 0)
        average_affect = _clamp(world_meta.get("average_affect", 0.5), default=0.5)
        food_ratio = _clamp(world_meta.get("food_ratio", 0.5), default=0.5)
        water_ratio = _clamp(world_meta.get("water_ratio", 0.5), default=0.5)

        need_counts: dict[str, int] = {}
        for npc in npcs:
            need = str(npc.get("dominant_need", "unknown"))
            need_counts[need] = need_counts.get(need, 0) + 1
        dominant_need = max(need_counts, key=need_counts.get) if need_counts else "unknown"

        pressure_score = _clamp(
            (1.0 - average_affect) * 0.5
            + (1.0 - food_ratio) * 0.3
            + (1.0 - water_ratio) * 0.2,
            default=0.5,
        )
        if living_count <= 0:
            objective = "restore population continuity"
            tone = "collapse"
        elif pressure_score > 0.62:
            objective = f"stabilize {dominant_need} pressure"
            tone = "crisis"
        elif pressure_score > 0.38:
            objective = f"investigate {dominant_need} pressure"
            tone = "uneasy"
        else:
            objective = "strengthen village bonds"
            tone = "stable"

        return {
            "version": 1,
            "tick": self.tick,
            "tone": tone,
            "objective": objective,
            "pressure_score": pressure_score,
            "dominant_need": dominant_need,
            "need_counts": need_counts,
            "population": living_count,
            "food_ratio": food_ratio,
            "water_ratio": water_ratio,
            "average_affect": average_affect,
            "suggested_ui": {
                "headline": objective,
                "subhead": f"Population {living_count} | pressure {pressure_score:.2f}",
                "focus_need": dominant_need,
            },
        }

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

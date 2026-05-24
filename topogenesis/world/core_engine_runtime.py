from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CoreAgentSnapshot:
    id: str
    display_name: str
    position: list[float]
    energy: float
    health: float
    membrane: float
    alive: bool
    metrics: dict[str, Any]


@dataclass
class CoreEngineRuntime:
    """Small full-engine organism layer used by the game bridge.

    The batched offline ecology remains the scalable 100-300 NPC substrate.
    This runtime keeps one or more genuine TopogenesisAgent.self_maintain loops
    alive so the online/offline bridge is anchored to the full reference engine
    instead of only to the lightweight population approximation.
    """

    agent_count: int = 1
    world_size: int = 16
    seed: int = 0
    initialized: bool = False
    tick: int = 0
    last_error: str = ""
    _engine: Any = field(default=None, init=False, repr=False)
    _config: Any = field(default=None, init=False, repr=False)
    _world: Any = field(default=None, init=False, repr=False)
    _agents: list[Any] = field(default_factory=list, init=False, repr=False)
    _bodies: list[Any] = field(default_factory=list, init=False, repr=False)
    _actions: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _last_metrics: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _alive: list[bool] = field(default_factory=list, init=False, repr=False)

    def initialize(self) -> None:
        if self.initialized:
            return
        import jax
        from topogenesis import engine

        self._engine = engine
        self._config = engine.TopogenesisConfig()
        self._world = engine.World3D(
            size=(self.world_size, self.world_size, self.world_size),
            n_resources=min(8, self._config.cognition.n_resources),
            n_hazards=min(4, self._config.cognition.n_hazards),
            seed=self.seed,
            membrane_repair_rate=self._config.cognition.membrane_repair_rate,
            membrane_decay_rate=self._config.cognition.membrane_decay_rate,
            ground_resource_frac=self._config.cognition.ground_resource_frac,
            max_spawn_height=self._config.cognition.max_spawn_height,
            ground_locomotion_gain=self._config.cognition.ground_locomotion_gain,
            interaction_radius=self._config.cognition.interaction_radius,
            safe_spawn_radius=self._config.cognition.safe_spawn_radius,
            starter_resource_patch=self._config.cognition.starter_resource_patch,
            energy_decay=self._config.cognition.energy_decay,
            resource_energy_gain=self._config.cognition.resource_energy_gain,
            resource_repair_gain=self._config.cognition.resource_repair_gain,
            force_metabolic_cost=self._config.cognition.force_metabolic_cost,
            resource_regen_interval=self._config.cognition.resource_regen_interval,
            resource_regen_count=self._config.cognition.resource_regen_count,
            starter_regen_count=self._config.cognition.starter_regen_count,
        )
        self._agents = []
        self._bodies = []
        self._actions = []
        self._last_metrics = []
        self._alive = []
        center = self.world_size // 2
        for idx in range(max(1, int(self.agent_count))):
            body = engine.AgentBodyPhys(start_pos=(center + idx, center, 1))
            agent = engine.TopogenesisAgent(
                self._config,
                jax.random.PRNGKey(self.seed + idx),
                num_agents=max(1, int(self.agent_count)),
                self_idx=idx,
            )
            self._agents.append(agent)
            self._bodies.append(body)
            self._actions.append(np.zeros(engine.MAX_MOTORS, dtype=np.float32))
            self._last_metrics.append({})
            self._alive.append(True)
        self.initialized = True

    def step(self) -> dict[str, Any]:
        try:
            self.initialize()
            self._world.advance_field(self._bodies)
            for idx, agent in enumerate(self._agents):
                if not self._alive[idx]:
                    continue
                alive, action, metrics = agent.self_maintain(
                    self._world,
                    self._bodies[idx],
                    self._bodies,
                    self._actions[idx],
                )
                self._actions[idx] = action
                self._last_metrics[idx] = dict(metrics)
                self._alive[idx] = bool(alive)
            self.tick += 1
            self.last_error = ""
        except Exception as exc:  # pragma: no cover - diagnostic fallback
            self.last_error = f"{type(exc).__name__}: {exc}"
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        agents = []
        if self.initialized:
            for idx, body in enumerate(self._bodies):
                metrics = self._last_metrics[idx] if idx < len(self._last_metrics) else {}
                agents.append(CoreAgentSnapshot(
                    id=f"core_{idx}",
                    display_name="Core Organism" if idx == 0 else f"Core Organism {idx + 1}",
                    position=[
                        round(float((body.pos[0] - self.world_size * 0.5) * 6.0), 4),
                        round(float((body.pos[1] - self.world_size * 0.5) * 6.0), 4),
                    ],
                    energy=round(float(body.energy), 4),
                    health=round(float(body.health), 4),
                    membrane=round(float(body.membrane_integrity), 4),
                    alive=bool(self._alive[idx]),
                    metrics={
                        "need_total": metrics.get("npc_need_total"),
                        "dominant_need": metrics.get("npc_need_dominant"),
                        "future_action": metrics.get("npc_future_action"),
                        "free_energy": metrics.get("free_energy"),
                        "phase": metrics.get("phase"),
                        "step": metrics.get("step"),
                    },
                ).__dict__)
        return {
            "enabled": True,
            "mode": "full_topogenesis_agent_self_maintain",
            "initialized": self.initialized,
            "tick": self.tick,
            "agent_count": max(1, int(self.agent_count)),
            "last_error": self.last_error,
            "agents": agents,
        }

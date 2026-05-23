"""Offline and scalable world simulation primitives."""

from .metrics import OfflineMetrics, OfflineSummary
from .population import NPCBatch, PopulationConfig, PopulationManager
from .world_state import Location, WorldClock, WorldState

__all__ = [
    "Location",
    "NPCBatch",
    "OfflineConfig",
    "OfflineMetrics",
    "OfflineSimulator",
    "OfflineSummary",
    "PopulationConfig",
    "PopulationManager",
    "WorldClock",
    "WorldState",
]


def __getattr__(name: str):
    if name in {"OfflineConfig", "OfflineSimulator"}:
        from .offline_sim import OfflineConfig, OfflineSimulator
        return {"OfflineConfig": OfflineConfig, "OfflineSimulator": OfflineSimulator}[name]
    raise AttributeError(name)

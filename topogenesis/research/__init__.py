"""Research-only scaffolding for AGI-oriented Topogenesis experiments."""

from .functionalism import (
    EvidenceGate,
    FunctionalRoleContract,
    ScalingGate,
    default_evidence_gates,
    default_functionalist_ladder,
    default_scaling_gates,
    incomplete_contracts,
)
from .predictive_cognition import (
    ConfirmationSignal,
    PersistentMemory,
    PersistentWorldModel,
    Prediction,
    SensoryConfirmationLoop,
    SensoryObservation,
    WorldEntity,
    compare_prediction,
)

__all__ = [
    "EvidenceGate",
    "FunctionalRoleContract",
    "ScalingGate",
    "default_evidence_gates",
    "default_functionalist_ladder",
    "default_scaling_gates",
    "incomplete_contracts",
    "ConfirmationSignal",
    "PersistentMemory",
    "PersistentWorldModel",
    "Prediction",
    "SensoryConfirmationLoop",
    "SensoryObservation",
    "WorldEntity",
    "compare_prediction",
]

from __future__ import annotations

from dataclasses import dataclass

from ._math import clamp01


@dataclass(frozen=True)
class ViabilityState:
    """Substrate variables whose degradation creates intrinsic pressure."""

    energy: float = 1.0
    bodily_integrity: float = 1.0
    memory_integrity: float = 1.0
    prediction_coherence: float = 1.0
    social_stability: float = 0.6
    attachment_integrity: float = 0.6
    environmental_safety: float = 0.7

    def normalized(self) -> "ViabilityState":
        return ViabilityState(**{
            key: clamp01(value)
            for key, value in self.__dict__.items()
        })


@dataclass(frozen=True)
class NeedPressure:
    """Emergent pressure profile derived from viability deficits."""

    metabolic: float
    repair: float
    mnemonic: float
    epistemic: float
    social: float
    attachment: float
    safety: float

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            object.__setattr__(
                self, field_name, clamp01(getattr(self, field_name)))

    @classmethod
    def from_viability(cls, viability: ViabilityState) -> "NeedPressure":
        v = viability.normalized()
        return cls(
            metabolic=1.0 - v.energy,
            repair=1.0 - v.bodily_integrity,
            mnemonic=1.0 - v.memory_integrity,
            epistemic=1.0 - v.prediction_coherence,
            social=1.0 - v.social_stability,
            attachment=1.0 - v.attachment_integrity,
            safety=1.0 - v.environmental_safety,
        )

    @property
    def total(self) -> float:
        weights = {
            "metabolic": 0.18,
            "repair": 0.18,
            "mnemonic": 0.12,
            "epistemic": 0.16,
            "social": 0.11,
            "attachment": 0.10,
            "safety": 0.15,
        }
        return sum(getattr(self, key) * weight for key, weight in weights.items())

    @property
    def dominant(self) -> str:
        return max(self.__dict__, key=lambda key: getattr(self, key))

    def cognitive_modulators(self) -> dict[str, float]:
        pressure = clamp01(self.total)
        return {
            "planning_depth": max(1.0, 6.0 - 4.0 * pressure),
            "memory_noise": clamp01(0.05 + 0.55 * self.mnemonic + 0.25 * pressure),
            "risk_tolerance": clamp01(0.75 - 0.55 * max(self.safety, self.repair)),
            "verification_bias": clamp01(0.25 + 0.55 * self.epistemic),
            "social_seeking": clamp01(0.25 + 0.45 * self.social + 0.30 * self.attachment),
        }

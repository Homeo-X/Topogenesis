from __future__ import annotations

import math
from collections.abc import Mapping


def finite_float(value: float, *, default: float = 0.0) -> float:
    """Convert user/sim input to a finite float with a deterministic fallback."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def clamp(value: float, low: float, high: float, *, default: float = 0.0) -> float:
    out = finite_float(value, default=default)
    return max(low, min(high, out))


def clamp01(value: float, *, default: float = 0.0) -> float:
    return clamp(value, 0.0, 1.0, default=default)


def non_empty_text(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def normalized_confidence_map(values: Mapping[str, float]) -> dict[str, float]:
    return {
        non_empty_text(key, field_name="map key"): clamp01(value, default=0.5)
        for key, value in values.items()
    }

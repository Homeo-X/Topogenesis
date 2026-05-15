from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class ExperimentPreset:
    name: str
    description: str
    args: Tuple[str, ...]
    expected: str


PRESETS: Dict[str, ExperimentPreset] = {
    "smoke": ExperimentPreset(
        name="smoke",
        description="Tiny run that verifies the engine starts and exits.",
        args=("--steps", "2", "--agents", "1", "--world_size", "16", "--log_every", "1"),
        expected="Completes without exception and prints a summary.",
    ),
    "physics_only": ExperimentPreset(
        name="physics_only",
        description="Minimal organism pressure check with one agent.",
        args=("--steps", "100", "--agents", "1", "--world_size", "16", "--log_every", "25"),
        expected="No NaNs; body state remains bounded.",
    ),
    "single_agent_survival": ExperimentPreset(
        name="single_agent_survival",
        description="One agent survival, metabolism, and resource loop.",
        args=("--steps", "1000", "--agents", "1", "--world_size", "32", "--log_every", "100"),
        expected="Agent survives longer than baseline random motion.",
    ),
    "lifetime_learning": ExperimentPreset(
        name="lifetime_learning",
        description="Single-agent learning without population pressure.",
        args=("--steps", "3000", "--agents", "1", "--world_size", "32", "--log_every", "250"),
        expected="Prediction error and survival behavior improve over life.",
    ),
    "reproduction_basic": ExperimentPreset(
        name="reproduction_basic",
        description="Small population run focused on birth/death mechanics.",
        args=("--steps", "2000", "--agents", "3", "--world_size", "32", "--log_every", "100"),
        expected="Births occur only under viable body and field conditions.",
    ),
    "evolutionary_run": ExperimentPreset(
        name="evolutionary_run",
        description="Population run for heredity, mutation, and lineage metrics.",
        args=("--steps", "10000", "--agents", "6", "--world_size", "48", "--log_every", "500"),
        expected="Lineages diverge without immediate collapse.",
    ),
    "open_ended_ecology": ExperimentPreset(
        name="open_ended_ecology",
        description="Long ecological run for novelty and diversity tracking.",
        args=("--steps", "50000", "--agents", "8", "--world_size", "64", "--log_every", "1000"),
        expected="Novelty, diversity, and lineage persistence remain measurable.",
    ),
}


def preset_args(name: str) -> Tuple[str, ...]:
    try:
        return PRESETS[name].args
    except KeyError as exc:
        names = ", ".join(sorted(PRESETS))
        raise KeyError(f"Unknown experiment '{name}'. Available: {names}") from exc


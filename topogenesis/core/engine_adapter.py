from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable, Optional


DEFAULT_ENGINE_PATH = Path(__file__).resolve().parents[1] / "engine.py"


def load_engine(path: Optional[str | Path] = None) -> ModuleType:
    """Load the Topogenesis engine module from this project or an explicit path."""
    engine_path = Path(path) if path is not None else DEFAULT_ENGINE_PATH
    if not engine_path.exists():
        raise FileNotFoundError(f"Topogenesis engine file not found: {engine_path}")

    spec = importlib.util.spec_from_file_location("topogenesis_engine_runtime", engine_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {engine_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_engine(argv: Iterable[str], path: Optional[str | Path] = None):
    """Run the engine main function with CLI-style arguments."""
    module = load_engine(path)
    if not hasattr(module, "main"):
        raise AttributeError("Loaded engine does not expose main(argv=None)")
    return module.main(list(argv))

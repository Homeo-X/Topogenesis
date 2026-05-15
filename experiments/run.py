from __future__ import annotations

import argparse

from topogenesis.core.engine_adapter import DEFAULT_ENGINE_PATH, run_engine
from topogenesis.core.experiments import PRESETS, preset_args


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run Topogenesis experiment presets.")
    parser.add_argument("--experiment", choices=sorted(PRESETS), default="smoke")
    parser.add_argument("--engine-path", default=str(DEFAULT_ENGINE_PATH))
    parser.add_argument(
        "--override",
        nargs=argparse.REMAINDER,
        help="Optional raw engine args after --override.",
    )
    args = parser.parse_args(argv)

    preset = PRESETS[args.experiment]
    engine_args = tuple(args.override) if args.override else preset_args(args.experiment)

    print(f"[topogenesis-run] experiment={preset.name}")
    print(f"[topogenesis-run] description={preset.description}")
    print(f"[topogenesis-run] expected={preset.expected}")
    print(f"[topogenesis-run] args={' '.join(engine_args)}")
    return run_engine(engine_args, args.engine_path)


if __name__ == "__main__":
    main()

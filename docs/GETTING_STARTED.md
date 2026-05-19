# Getting Started

This guide is for someone arriving at Topogenesis today who wants the fastest
path from clone to meaningful feedback.

## 1. Verify The Project

```bash
git clone https://github.com/Homeo-X/Topogenesis.git
cd Topogenesis
python -m pip install -e .

python -m unittest discover -s tests -p "test_*.py"
python -m experiments.run --experiment smoke
```

The `smoke` preset is the right first run. It verifies that the reference engine
starts, steps, emits summaries, and exits without exceptions.

For the Godot RPG vertical slice, you can also run the optional cognition bridge:

```bash
python -m topogenesis.game_bridge.server
```

Then open `godot/` in Godot 4.6.2. If the server is not running, the Godot slice
falls back to its local cognition model.

## 2. Try More Interesting Presets

After `smoke` passes, try:

```bash
python -m experiments.run --experiment single_agent_survival
python -m experiments.run --experiment lifetime_learning
python -m experiments.run --experiment reproduction_basic
```

Longer presets may be slow on CPU because the integrated reference engine is
still heavy. Start small before attempting `evolutionary_run` or
`open_ended_ecology`.

## 3. Read The Core Docs

Recommended order:

1. `docs/PROJECT_THESIS.md`
2. `docs/KNOWN_LIMITS.md`
3. `docs/STRESS_VALIDATION.md`
4. `docs/PRODUCTION_READINESS.md`

The thesis explains what the project is trying to prove. The limits and
validation docs explain what is currently hardened and what remains
research-grade.

## 4. Highest-Leverage Contributions

- Add ablation baselines: no-affect, no-memory, reflex-only, no-field.
- Add dashboards or simple plots for energy, viability, affect pressure, field
  state, and population.
- Improve long-run stability and performance.
- Add social communication stress experiments.
- Compare pressure-driven cognition against simpler baselines.
- Continue extracting `topogenesis/engine.py` into smaller testable modules.

## 5. Research Question

The central question is:

> Does coupling affect, needs, communication, and future simulation to viability
> pressure produce more robust or interesting behavior than scripted,
> reflex-only, or pure reward-driven baselines?

Good contributions should make that question easier to test, not merely add more
mechanisms.

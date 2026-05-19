# Godot Production Notes

This folder is a production-shaped vertical slice, not a finished AAA RPG.

## Upgraded Foundation

- Godot 4.6 project with a real main scene.
- Runtime input action setup instead of raw key polling.
- Third-person player controller with sprint.
- Pause, debug toggle, interaction prompt, and dialogue HUD.
- Small readable 3D village layout with huts, path, trees, and pressure zones.
- NPC overhead state markers for dominant need and pressure.
- NPC movement and dialogue are driven by the `TopogenesisBridge` pressure model.
- Headless Godot validation is part of the local check workflow.

## Not Yet AAA

- No authored character models, animation sets, combat, inventory, quest system,
  save/load, audio mix, localization, accessibility pass, asset streaming,
  performance budgets, art direction bible, or console certification work.
- The Godot-side bridge currently mirrors the Topogenesis cognition language
  locally; it is not yet a live Python backend connection.

## Next Production Milestones

1. Replace capsule placeholders with authored character rigs and animation trees.
2. Add a Python snapshot bridge for live Topogenesis NPC state.
3. Add navigation meshes and proper NPC pathfinding.
4. Add interaction state machines, UI focus management, and save/load.
5. Add performance telemetry and frame-budget checks.
6. Add art/audio direction and content pipelines.

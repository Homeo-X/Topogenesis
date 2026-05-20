# Godot Production Notes

This folder is a production-shaped vertical slice, not a finished AAA RPG.

## Upgraded Foundation

- Godot 4.6 project with a real main scene.
- Runtime input action setup instead of raw key polling.
- Third-person player controller with sprint.
- Pause, debug toggle, interaction prompt, and dialogue HUD.
- Objective state, JSON save/load, and bridge snapshots.
- Small readable 3D village layout using included Quaternius CC0 modular
  village, prop, and nature assets with generated primitive fallback paths.
- Runtime character loading for Quaternius CC0 rigged FBX villagers/player,
  with generated humanoid fallback bodies if import metadata is unavailable.
- NPC overhead state markers for dominant need and pressure.
- NPC movement and dialogue are driven by the `TopogenesisBridge` pressure model.
- Headless Godot validation is part of the local check workflow.

## Not Yet AAA

- The included assets are CC0/standard-pack production placeholders, not a final
  bespoke AAA art pass with custom facial blend shapes, scanned PBR humans, or
  authored cinematic animation.
- No combat, inventory, quest system, audio mix, localization, accessibility
  pass, asset streaming, performance budgets, art direction bible, or console
  certification work.
- The Python bridge can run via `run_with_bridge.ps1`; the in-Godot local
  fallback remains available when the backend is offline.

## Next Production Milestones

1. Promote imported FBX characters into authored Godot scenes with
   `AnimationTree`, retargeted locomotion, and need-state blend spaces.
2. Add navigation meshes and proper NPC pathfinding.
3. Add interaction state machines, inventory, and village resource actions.
4. Add performance telemetry, LOD import settings, and frame-budget checks.
5. Add bespoke art/audio direction and content pipelines.

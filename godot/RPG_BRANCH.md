# RPG Branch Charter

`game-rpg` is the playable branch for **Topogenesis: Echoes of the Veil**.

This branch is allowed to contain Godot assets, game launchers, player-facing
UX, bridge code, controller scripts, and visual experiments. Stable Python
engine improvements can be promoted back to `main` later, but game-specific
systems should remain here.

## Current Game-Facing Bridge Endpoints

- `GET /health`: bridge availability
- `GET /snapshot`: compact NPC cognition state
- `GET /world_snapshot`: offline ecology and visible population state
- `GET /director_snapshot`: RPG-facing objective, tone, pressure score, and UI
  hints
- `POST /step`: advance bridge cognition from Godot pressure payloads
- `POST /restore`: restore saved bridge state

## Vertical Slice Priorities

1. Keep click-to-move and WASD both playable.
2. Keep NPC pressure visible without turning the HUD into a debug wall.
3. Make the Python bridge provide game-readable signals, not just raw metrics.
4. Keep the local Godot fallback playable when the bridge is offline.
5. Treat `director_snapshot` as the main handshake for objectives, tone, and
   village pressure presentation.

## Near-Term Enrichment Targets

- Use `director_snapshot.objective` for the top-left quest line.
- Use `director_snapshot.tone` to drive lighting, fog, and ambient color.
- Use `pressure_score` to pace village events.
- Use `dominant_need` to pick NPC barks and interaction prompts.
- Promote imported character assets into authored scenes with `AnimationTree`.

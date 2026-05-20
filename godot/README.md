# Topogenesis RPG Vertical Slice

This is a Godot 4.6 prototype for turning Topogenesis into a 3D RPG-style
experience.

## Run

Open the `godot/` folder in Godot 4.6.2, or run:

```powershell
& 'C:\Users\rsijr\Downloads\Godot_v4.6.2-stable_win64.exe\Godot_v4.6.2-stable_win64_console.exe' --path godot
```

Optional Python cognition bridge:

```powershell
python -m topogenesis.game_bridge.server
```

Or launch Godot and the bridge together:

```powershell
powershell -ExecutionPolicy Bypass -File godot/run_with_bridge.ps1
```

When the bridge is running, Godot polls `http://127.0.0.1:8765/step` and imports
NPC cognition snapshots. If it is not running, the local Godot fallback remains
playable.

## Controls

- Left mouse click: move to ground position
- Mouse wheel: zoom camera in/out
- `Shift`: sprint; when held near an NPC it creates threat pressure
- `E`: interact with nearby NPC
- `F1`: toggle debug HUD
- `F5`: save vertical-slice state
- `F9`: load vertical-slice state
- `Esc`: pause

## Current Prototype

- third-person player controller
- small 3D village scene with Quaternius CC0 modular village pieces,
  props, trees, foliage, hazard/resource pressure zones
- warmer medieval-fantasy village dressing: torches, well, market stall,
  campfire, chapel marker, blacksmith point, herb garden, fences, fog, and
  dynamic day/night lighting
- three NPCs
- CC0 rigged animated character variants are used when Godot can import the
  FBX character pack; generated humanoid bodies remain as runtime fallback
- action-mapped input and pause/debug controls
- objective state and JSON save/load
- optional localhost Python cognition bridge
- NPC affect/need/future-action state
- overhead NPC state markers
- immersive focus panel, interaction prompt, dialogue line, and debug HUD
- `TopogenesisBridge` autoload as the future connection point to the Python
  Topogenesis backend

The current bridge is a lightweight in-Godot simulation of the same design
language as `topogenesis.npc`: viability pressure, affect stability, dominant
need, trust, memory events, and future action. The next milestone is replacing
or augmenting that local bridge with Python-backed Topogenesis state snapshots.

See `PRODUCTION_NOTES.md` for what is production-shaped now and what remains
before this could honestly be called AAA-quality.

See `ASSET_CREDITS.md` for the included CC0 asset packs and license notes.

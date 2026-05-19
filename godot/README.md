# Topogenesis RPG Prototype

This is a Godot 4.6 prototype for turning Topogenesis into a 3D RPG-style
experience.

## Run

Open the `godot/` folder in Godot 4.6.2, or run:

```powershell
& 'C:\Users\rsijr\Downloads\Godot_v4.6.2-stable_win64.exe\Godot_v4.6.2-stable_win64_console.exe' --path godot
```

## Controls

- `WASD`: move
- `E`: interact with nearby NPC
- hold `Shift` near an NPC: create threat pressure for that NPC

## Current Prototype

- third-person player controller
- small 3D village scene
- three NPCs
- resource and hazard pressure zones
- NPC affect/need/future-action state
- dialogue/debug HUD
- `TopogenesisBridge` autoload as the future connection point to the Python
  Topogenesis backend

The current bridge is a lightweight in-Godot simulation of the same design
language as `topogenesis.npc`: viability pressure, affect stability, dominant
need, trust, memory events, and future action. The next milestone is replacing
or augmenting that local bridge with Python-backed Topogenesis state snapshots.

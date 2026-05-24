# Branch Map

Topogenesis now has three intentionally different public branches.

## `main`

Stable neutral base.

- core Python engine
- experiment presets
- tests and validation docs
- branch-neutral architecture notes

`main` should stay runnable, compact, and conservative. It should not contain
Godot game assets, branch-specific launchers, the offline RPG population/world
simulator, or speculative research-only interfaces unless they have become
stable enough to serve the whole project.

## `game-rpg`

Playable RPG branch.

- Godot project
- player/NPC controllers
- visual assets
- game bridge and launch scripts
- offline population/world simulator
- RPG-facing UX, camera, movement, world, and villager presentation

Game features should land here first. Stable engine improvements can later be
merged back into `main` when they are useful outside the game.

## `agi-research`

Speculative cognition and AGI research branch.

- deeper functionalist cognition experiments
- richer world-model and memory hypotheses
- ablation protocols
- research-only scaffolding
- experimental agent interfaces

Research mechanisms should land here first. Only validated, stable, generally
useful pieces should graduate back to `main`.

## Promotion Rule

Move code toward `main` only when it is:

- covered by a smoke test or unit test
- documented in branch-neutral language
- useful outside one branch's specialty
- free of branch-specific dependencies

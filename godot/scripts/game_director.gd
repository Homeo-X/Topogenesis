extends Node

const SAVE_PATH := "user://topogenesis_save.json"

var interactions := 0
var village_stability := 0.5
var last_status := "Find villagers and stabilize local pressure."


func record_interaction(npc_id: String, npc_state: Dictionary) -> void:
	interactions += 1
	var affect := float(npc_state.get("affect_stability", 0.5))
	var need := float(npc_state.get("need_total", 0.5))
	village_stability = clampf(lerpf(village_stability, affect * (1.0 - 0.35 * need), 0.18), 0.0, 1.0)
	last_status = "Spoke with %s | village stability %.2f" % [npc_id, village_stability]


func objective_text() -> String:
	if interactions < 1:
		return "Objective: speak with a villager."
	if village_stability < 0.45:
		return "Objective: reduce hazard pressure and rebuild trust."
	if interactions < 3:
		return "Objective: compare each villager's pressure state."
	return "Objective: village pressure mapped. Prototype loop complete."


func status_text() -> String:
	return "%s | interactions:%d | stability:%.2f" % [last_status, interactions, village_stability]


func snapshot() -> Dictionary:
	return {
		"version": 1,
		"interactions": interactions,
		"village_stability": village_stability,
		"last_status": last_status,
		"bridge": TopogenesisBridge.export_snapshot(),
	}


func restore(snapshot_data: Dictionary) -> void:
	interactions = int(snapshot_data.get("interactions", 0))
	village_stability = clampf(float(snapshot_data.get("village_stability", 0.5)), 0.0, 1.0)
	last_status = str(snapshot_data.get("last_status", "Save restored."))
	TopogenesisBridge.import_snapshot(snapshot_data.get("bridge", {}))


func save_game() -> bool:
	var file := FileAccess.open(SAVE_PATH, FileAccess.WRITE)
	if file == null:
		last_status = "Save failed: %s" % FileAccess.get_open_error()
		return false
	file.store_string(JSON.stringify(snapshot(), "\t"))
	last_status = "Saved game state."
	return true


func load_game() -> bool:
	if not FileAccess.file_exists(SAVE_PATH):
		last_status = "No save file found."
		return false
	var file := FileAccess.open(SAVE_PATH, FileAccess.READ)
	if file == null:
		last_status = "Load failed: %s" % FileAccess.get_open_error()
		return false
	var parsed = JSON.parse_string(file.get_as_text())
	if typeof(parsed) != TYPE_DICTIONARY:
		last_status = "Load failed: invalid save data."
		return false
	restore(parsed)
	last_status = "Loaded game state."
	return true

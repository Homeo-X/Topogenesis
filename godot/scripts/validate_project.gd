extends SceneTree


func _init() -> void:
	var failures: Array[String] = []
	if not ProjectSettings.has_setting("application/run/main_scene"):
		failures.append("Missing main scene setting.")
	for autoload_name in ["TopogenesisBridge", "GameDirector"]:
		if not ProjectSettings.has_setting("autoload/%s" % autoload_name):
			failures.append("Missing autoload: %s" % autoload_name)
	for path in [
		"res://scenes/main.tscn",
		"res://scripts/main.gd",
		"res://scripts/player_controller.gd",
		"res://scripts/npc_controller.gd",
		"res://scripts/topogenesis_bridge.gd",
		"res://scripts/game_director.gd",
	]:
		if not ResourceLoader.exists(path):
			failures.append("Missing resource: %s" % path)
	if failures.is_empty():
		print("Godot project validation passed.")
		quit(0)
	else:
		for failure in failures:
			push_error(failure)
		quit(1)

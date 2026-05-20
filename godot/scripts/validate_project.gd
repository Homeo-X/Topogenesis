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
		"res://assets/quaternius/medieval_village/glTF/Wall_Plaster_Door_Round.gltf",
		"res://assets/quaternius/medieval_village/glTF/Roof_RoundTiles_4x4.gltf",
		"res://assets/quaternius/fantasy_props/Exports/glTF/Stall_Empty.gltf",
		"res://assets/quaternius/fantasy_props/Exports/glTF/Anvil_Log.gltf",
		"res://assets/quaternius/stylized_nature/glTF/CommonTree_1.gltf",
		"res://assets/quaternius/animated_characters/Ultimate Animated Character Pack - Nov 2019/FBX/Witch.fbx",
	]:
		if not ResourceLoader.exists(path) and not FileAccess.file_exists(path):
			failures.append("Missing resource: %s" % path)
	if failures.is_empty():
		print("Godot project validation passed.")
		quit(0)
	else:
		for failure in failures:
			push_error(failure)
		quit(1)

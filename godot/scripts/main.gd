extends Node3D

var hud: CanvasLayer
var nearest_npc: Node = null
var interact_was_down := false
var is_paused := false
var sun_light: DirectionalLight3D
var world_environment: WorldEnvironment
var torch_lights: Array[OmniLight3D] = []
var time_of_day := 0.68
var day_speed := 0.018


func _ready() -> void:
	_configure_input_actions()
	_build_lighting()
	_build_ground()
	_build_village()
	_spawn_player()
	_spawn_npcs()
	_spawn_pressure_markers()
	_spawn_hud()


func _process(delta: float) -> void:
	if Input.is_action_just_pressed("pause"):
		_toggle_pause()
	if Input.is_action_just_pressed("save_game"):
		_show_system_result(GameDirector.save_game(), "Saved.", "Save failed.")
	if Input.is_action_just_pressed("load_game"):
		_show_system_result(GameDirector.load_game(), "Loaded.", "Load failed.")
	if is_paused:
		return
	_update_lighting(delta)
	nearest_npc = _find_nearest_npc()
	if nearest_npc != null:
		hud.set_prompt("Press E to speak with %s" % nearest_npc.name)
		if nearest_npc.has_method("current_state"):
			hud.set_focus_state(str(nearest_npc.name), nearest_npc.current_state())
	else:
		hud.set_prompt("")
		hud.clear_focus_state()
	var interact_down := Input.is_action_pressed("interact")
	if nearest_npc != null and interact_down and not interact_was_down:
		if nearest_npc.has_method("interact"):
			var line: String = nearest_npc.interact()
			if nearest_npc.has_method("current_state"):
				GameDirector.record_interaction(nearest_npc.name, nearest_npc.current_state())
			hud.show_dialogue(line)
	interact_was_down = interact_down


func _show_system_result(ok: bool, success: String, failure: String) -> void:
	hud.show_dialogue(success if ok else "%s %s" % [failure, GameDirector.status_text()])


func _configure_input_actions() -> void:
	_ensure_key_action("move_forward", KEY_W)
	_ensure_key_action("move_back", KEY_S)
	_ensure_key_action("move_left", KEY_A)
	_ensure_key_action("move_right", KEY_D)
	_ensure_key_action("interact", KEY_E)
	_ensure_key_action("sprint", KEY_SHIFT)
	_ensure_key_action("pause", KEY_ESCAPE)
	_ensure_key_action("toggle_debug", KEY_F1)
	_ensure_key_action("save_game", KEY_F5)
	_ensure_key_action("load_game", KEY_F9)


func _ensure_key_action(action_name: StringName, keycode: Key) -> void:
	if not InputMap.has_action(action_name):
		InputMap.add_action(action_name)
	if InputMap.action_get_events(action_name).is_empty():
		var event := InputEventKey.new()
		event.keycode = keycode
		InputMap.action_add_event(action_name, event)


func _toggle_pause() -> void:
	is_paused = not is_paused
	get_tree().paused = is_paused
	hud.process_mode = Node.PROCESS_MODE_ALWAYS
	hud.show_pause(is_paused)


func _build_lighting() -> void:
	sun_light = DirectionalLight3D.new()
	sun_light.rotation_degrees = Vector3(-48.0, -32.0, 0.0)
	sun_light.light_energy = 2.4
	sun_light.light_color = Color(1.0, 0.82, 0.58)
	sun_light.shadow_enabled = true
	add_child(sun_light)

	world_environment = WorldEnvironment.new()
	var env := Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.background_color = Color(0.44, 0.55, 0.58)
	env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.ambient_light_color = Color(0.46, 0.49, 0.43)
	env.ambient_light_energy = 0.65
	env.fog_enabled = true
	env.fog_density = 0.016
	env.fog_light_color = Color(0.50, 0.56, 0.50)
	env.glow_enabled = true
	env.glow_intensity = 0.18
	world_environment.environment = env
	add_child(world_environment)
	_update_lighting(0.0)


func _build_ground() -> void:
	var ground := MeshInstance3D.new()
	var plane := PlaneMesh.new()
	plane.size = Vector2(44.0, 44.0)
	ground.mesh = plane
	var mat := StandardMaterial3D.new()
	mat.albedo_color = Color(0.30, 0.46, 0.31)
	mat.roughness = 1.0
	ground.material_override = mat
	add_child(ground)
	_add_path()

	var body := StaticBody3D.new()
	var collision := CollisionShape3D.new()
	var shape := BoxShape3D.new()
	shape.size = Vector3(44.0, 0.2, 44.0)
	collision.shape = shape
	collision.position.y = -0.1
	body.add_child(collision)
	add_child(body)


func _build_village() -> void:
	for item in [
		[Vector3(-5.0, 0.0, 4.0), Color(0.45, 0.33, 0.24)],
		[Vector3(3.0, 0.0, 5.5), Color(0.50, 0.38, 0.26)],
		[Vector3(6.5, 0.0, -1.5), Color(0.40, 0.30, 0.22)],
		[Vector3(-9.0, 0.0, -1.8), Color(0.43, 0.31, 0.23)],
		[Vector3(0.2, 0.0, -6.2), Color(0.36, 0.33, 0.28)],
	]:
		_add_hut(item[0], item[1])
	_add_well(Vector3(-1.0, 0.0, -0.8))
	_add_market_stall(Vector3(2.2, 0.0, -2.7))
	_add_blacksmith(Vector3(8.0, 0.0, 3.2))
	_add_chapel_marker(Vector3(-8.7, 0.0, 5.6))
	_add_campfire(Vector3(0.0, 0.0, 2.2))
	_add_herb_garden(Vector3(-7.8, 0.0, -5.7))
	for fence_x in [-4.0, -2.8, -1.6, 1.6, 2.8, 4.0]:
		_add_fence(Vector3(fence_x, 0.0, -4.7), 0.0)
	for i in range(26):
		var angle := TAU * float(i) / 26.0
		var radius_x := 15.0 + 2.5 * sin(float(i) * 1.7)
		var radius_z := 13.5 + 2.0 * cos(float(i) * 1.2)
		_add_tree(Vector3(cos(angle) * radius_x, 0.0, sin(angle) * radius_z))
	for i in range(36):
		var x := -17.0 + float((i * 7) % 34)
		var z := -17.0 + float((i * 11) % 34)
		if absf(x) > 3.0 or absf(z) > 5.0:
			_add_grass_patch(Vector3(x, 0.0, z))


func _add_path() -> void:
	var path := MeshInstance3D.new()
	var plane := PlaneMesh.new()
	plane.size = Vector2(5.0, 25.0)
	path.mesh = plane
	path.position = Vector3(0.0, 0.012, 0.0)
	path.rotation_degrees.y = 18.0
	var mat := StandardMaterial3D.new()
	mat.albedo_color = Color(0.48, 0.40, 0.28)
	mat.roughness = 1.0
	path.material_override = mat
	add_child(path)


func _add_tree(pos: Vector3) -> void:
	var trunk := MeshInstance3D.new()
	var trunk_mesh := CylinderMesh.new()
	trunk_mesh.top_radius = 0.18
	trunk_mesh.bottom_radius = 0.28
	trunk_mesh.height = 2.1
	trunk.mesh = trunk_mesh
	trunk.position = pos + Vector3(0.0, 1.05, 0.0)
	var trunk_mat := StandardMaterial3D.new()
	trunk_mat.albedo_color = Color(0.28, 0.17, 0.10)
	trunk.material_override = trunk_mat
	add_child(trunk)

	var crown := MeshInstance3D.new()
	var crown_mesh := SphereMesh.new()
	crown_mesh.radius = 1.15
	crown_mesh.height = 1.8
	crown.mesh = crown_mesh
	crown.position = pos + Vector3(0.0, 2.65, 0.0)
	var crown_mat := StandardMaterial3D.new()
	crown_mat.albedo_color = Color(0.14, 0.34, 0.18)
	crown_mat.roughness = 0.9
	crown.material_override = crown_mat
	add_child(crown)


func _add_hut(pos: Vector3, color: Color) -> void:
	var hut := MeshInstance3D.new()
	var box := BoxMesh.new()
	box.size = Vector3(2.6, 1.8, 2.6)
	hut.mesh = box
	hut.position = pos + Vector3(0.0, 0.9, 0.0)
	var mat := StandardMaterial3D.new()
	mat.albedo_color = color
	mat.roughness = 0.9
	hut.material_override = mat
	add_child(hut)

	var roof := MeshInstance3D.new()
	var roof_mesh := PrismMesh.new()
	roof_mesh.size = Vector3(3.0, 1.0, 3.0)
	roof.mesh = roof_mesh
	roof.position = pos + Vector3(0.0, 2.05, 0.0)
	roof.rotation_degrees.y = 90.0
	var roof_mat := StandardMaterial3D.new()
	roof_mat.albedo_color = Color(0.22, 0.16, 0.12)
	roof.material_override = roof_mat
	add_child(roof)

	var body := StaticBody3D.new()
	var collision := CollisionShape3D.new()
	var shape := BoxShape3D.new()
	shape.size = Vector3(2.8, 2.0, 2.8)
	collision.shape = shape
	collision.position = pos + Vector3(0.0, 1.0, 0.0)
	body.add_child(collision)
	add_child(body)


func _add_well(pos: Vector3) -> void:
	var base := MeshInstance3D.new()
	var base_mesh := CylinderMesh.new()
	base_mesh.top_radius = 0.78
	base_mesh.bottom_radius = 0.86
	base_mesh.height = 0.68
	base.mesh = base_mesh
	base.position = pos + Vector3(0.0, 0.34, 0.0)
	base.material_override = _material(Color(0.38, 0.37, 0.34), 0.95)
	add_child(base)

	var beam := MeshInstance3D.new()
	var beam_mesh := BoxMesh.new()
	beam_mesh.size = Vector3(1.55, 0.15, 0.15)
	beam.mesh = beam_mesh
	beam.position = pos + Vector3(0.0, 1.26, 0.0)
	beam.material_override = _material(Color(0.24, 0.14, 0.08), 0.85)
	add_child(beam)


func _add_market_stall(pos: Vector3) -> void:
	var table := MeshInstance3D.new()
	var table_mesh := BoxMesh.new()
	table_mesh.size = Vector3(2.2, 0.18, 1.0)
	table.mesh = table_mesh
	table.position = pos + Vector3(0.0, 0.72, 0.0)
	table.material_override = _material(Color(0.33, 0.20, 0.10), 0.82)
	add_child(table)

	var awning := MeshInstance3D.new()
	var awning_mesh := PrismMesh.new()
	awning_mesh.size = Vector3(2.6, 0.55, 1.25)
	awning.mesh = awning_mesh
	awning.position = pos + Vector3(0.0, 1.65, 0.0)
	awning.rotation_degrees.y = 90.0
	awning.material_override = _material(Color(0.45, 0.11, 0.10), 0.78)
	add_child(awning)
	_add_torch(pos + Vector3(-1.25, 0.0, -0.65))


func _add_blacksmith(pos: Vector3) -> void:
	var anvil := MeshInstance3D.new()
	var anvil_mesh := BoxMesh.new()
	anvil_mesh.size = Vector3(0.86, 0.42, 0.38)
	anvil.mesh = anvil_mesh
	anvil.position = pos + Vector3(0.0, 0.42, 0.0)
	anvil.material_override = _material(Color(0.16, 0.16, 0.15), 0.5)
	add_child(anvil)
	_add_torch(pos + Vector3(0.9, 0.0, -0.3))


func _add_chapel_marker(pos: Vector3) -> void:
	var stone := MeshInstance3D.new()
	var stone_mesh := BoxMesh.new()
	stone_mesh.size = Vector3(0.55, 1.8, 0.38)
	stone.mesh = stone_mesh
	stone.position = pos + Vector3(0.0, 0.9, 0.0)
	stone.material_override = _material(Color(0.42, 0.42, 0.39), 0.96)
	add_child(stone)
	var cross := MeshInstance3D.new()
	var cross_mesh := BoxMesh.new()
	cross_mesh.size = Vector3(1.0, 0.18, 0.20)
	cross.mesh = cross_mesh
	cross.position = pos + Vector3(0.0, 1.38, -0.02)
	cross.material_override = stone.material_override
	add_child(cross)


func _add_campfire(pos: Vector3) -> void:
	for i in range(6):
		var rock := MeshInstance3D.new()
		var rock_mesh := SphereMesh.new()
		rock_mesh.radius = 0.14
		rock_mesh.height = 0.18
		rock.mesh = rock_mesh
		var angle := TAU * float(i) / 6.0
		rock.position = pos + Vector3(cos(angle) * 0.55, 0.12, sin(angle) * 0.55)
		rock.material_override = _material(Color(0.28, 0.27, 0.24), 0.92)
		add_child(rock)
	_add_torch(pos)


func _add_herb_garden(pos: Vector3) -> void:
	var soil := MeshInstance3D.new()
	var soil_mesh := BoxMesh.new()
	soil_mesh.size = Vector3(2.4, 0.08, 1.5)
	soil.mesh = soil_mesh
	soil.position = pos + Vector3(0.0, 0.04, 0.0)
	soil.material_override = _material(Color(0.18, 0.11, 0.07), 1.0)
	add_child(soil)
	for i in range(8):
		_add_grass_patch(pos + Vector3(-0.9 + 0.25 * float(i), 0.05, -0.35 + 0.18 * float(i % 3)))


func _add_fence(pos: Vector3, yaw: float) -> void:
	var fence := MeshInstance3D.new()
	var mesh := BoxMesh.new()
	mesh.size = Vector3(1.0, 0.36, 0.12)
	fence.mesh = mesh
	fence.position = pos + Vector3(0.0, 0.42, 0.0)
	fence.rotation_degrees.y = yaw
	fence.material_override = _material(Color(0.30, 0.19, 0.10), 0.88)
	add_child(fence)


func _add_grass_patch(pos: Vector3) -> void:
	var grass := MeshInstance3D.new()
	var mesh := CylinderMesh.new()
	mesh.top_radius = 0.02
	mesh.bottom_radius = 0.09
	mesh.height = 0.42
	grass.mesh = mesh
	grass.position = pos + Vector3(0.0, 0.21, 0.0)
	grass.rotation_degrees = Vector3(0.0, float(int(pos.x * 17.0) % 180), 8.0)
	grass.material_override = _material(Color(0.18, 0.32, 0.16), 0.95)
	add_child(grass)


func _add_torch(pos: Vector3) -> void:
	var post := MeshInstance3D.new()
	var post_mesh := CylinderMesh.new()
	post_mesh.top_radius = 0.045
	post_mesh.bottom_radius = 0.06
	post_mesh.height = 1.25
	post.mesh = post_mesh
	post.position = pos + Vector3(0.0, 0.65, 0.0)
	post.material_override = _material(Color(0.20, 0.12, 0.06), 0.84)
	add_child(post)

	var light := OmniLight3D.new()
	light.position = pos + Vector3(0.0, 1.34, 0.0)
	light.light_color = Color(1.0, 0.55, 0.22)
	light.light_energy = 1.4
	light.omni_range = 5.0
	torch_lights.append(light)
	add_child(light)


func _material(color: Color, roughness: float) -> StandardMaterial3D:
	var mat := StandardMaterial3D.new()
	mat.albedo_color = color
	mat.roughness = roughness
	return mat


func _update_lighting(delta: float) -> void:
	time_of_day = fmod(time_of_day + day_speed * delta, 1.0)
	var sun_phase := sin(time_of_day * TAU)
	var daylight := clampf(sun_phase * 0.55 + 0.55, 0.08, 1.0)
	if sun_light != null:
		sun_light.rotation_degrees.x = lerpf(-16.0, -68.0, daylight)
		sun_light.rotation_degrees.y = -35.0 + 24.0 * cos(time_of_day * TAU)
		sun_light.light_energy = lerpf(0.08, 2.7, daylight)
		sun_light.light_color = Color(1.0, lerpf(0.52, 0.84, daylight), lerpf(0.36, 0.68, daylight))
	if world_environment != null and world_environment.environment != null:
		var env := world_environment.environment
		env.background_color = Color(
			lerpf(0.05, 0.44, daylight),
			lerpf(0.07, 0.55, daylight),
			lerpf(0.10, 0.58, daylight)
		)
		env.ambient_light_energy = lerpf(0.18, 0.72, daylight)
		env.fog_density = lerpf(0.030, 0.012, daylight)
	for light in torch_lights:
		if light != null:
			light.light_energy = lerpf(2.4, 0.55, daylight)


func _spawn_player() -> void:
	var player := CharacterBody3D.new()
	player.name = "Player"
	player.set_script(load("res://scripts/player_controller.gd"))
	player.position = Vector3(0.0, 0.2, 9.0)
	add_child(player)


func _spawn_npcs() -> void:
	var script := load("res://scripts/npc_controller.gd")
	var data := [
		["npc_mara", "Mara", Vector3(-2.0, 0.2, 1.0)],
		["npc_ovan", "Ovan", Vector3(4.0, 0.2, 2.0)],
		["npc_sera", "Sera", Vector3(-6.0, 0.2, -4.0)],
	]
	for row in data:
		var npc := CharacterBody3D.new()
		npc.name = row[1]
		npc.set_script(script)
		npc.npc_id = row[0]
		npc.display_name = row[1]
		npc.home_position = row[2]
		npc.position = row[2]
		add_child(npc)


func _spawn_pressure_markers() -> void:
	_add_marker(Vector3(-8.0, 0.1, -6.0), Color(0.20, 0.78, 0.36), "Resource")
	_add_marker(Vector3(7.0, 0.1, -7.0), Color(0.88, 0.18, 0.12), "Hazard")


func _add_marker(pos: Vector3, color: Color, label_text: String) -> void:
	var mesh := MeshInstance3D.new()
	var cylinder := CylinderMesh.new()
	cylinder.top_radius = 0.8
	cylinder.bottom_radius = 0.8
	cylinder.height = 0.12
	mesh.mesh = cylinder
	mesh.position = pos
	var mat := StandardMaterial3D.new()
	mat.albedo_color = color
	mat.emission_enabled = true
	mat.emission = color
	mat.emission_energy_multiplier = 0.35
	mesh.material_override = mat
	add_child(mesh)

	var label := Label3D.new()
	label.text = label_text
	label.position = pos + Vector3(0.0, 0.8, 0.0)
	label.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	add_child(label)


func _spawn_hud() -> void:
	hud = load("res://scripts/debug_hud.gd").new()
	add_child(hud)


func _find_nearest_npc() -> Node:
	var player := get_tree().get_first_node_in_group("player") as Node3D
	if player == null:
		return null
	var best: Node = null
	var best_dist := 2.6
	for npc in get_tree().get_nodes_in_group("npc"):
		var node := npc as Node3D
		var dist := player.global_position.distance_to(node.global_position)
		if dist < best_dist:
			best = npc
			best_dist = dist
	return best

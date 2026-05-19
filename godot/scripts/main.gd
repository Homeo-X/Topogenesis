extends Node3D

var hud: CanvasLayer
var nearest_npc: Node = null
var interact_was_down := false
var is_paused := false


func _ready() -> void:
	_configure_input_actions()
	_build_lighting()
	_build_ground()
	_build_village()
	_spawn_player()
	_spawn_npcs()
	_spawn_pressure_markers()
	_spawn_hud()


func _process(_delta: float) -> void:
	if Input.is_action_just_pressed("pause"):
		_toggle_pause()
	if is_paused:
		return
	nearest_npc = _find_nearest_npc()
	if nearest_npc != null:
		hud.set_prompt("Press E to speak with %s" % nearest_npc.name)
	else:
		hud.set_prompt("")
	var interact_down := Input.is_action_pressed("interact")
	if nearest_npc != null and interact_down and not interact_was_down:
		if nearest_npc.has_method("interact"):
			hud.show_dialogue(nearest_npc.interact())
	interact_was_down = interact_down


func _configure_input_actions() -> void:
	_ensure_key_action("move_forward", KEY_W)
	_ensure_key_action("move_back", KEY_S)
	_ensure_key_action("move_left", KEY_A)
	_ensure_key_action("move_right", KEY_D)
	_ensure_key_action("interact", KEY_E)
	_ensure_key_action("sprint", KEY_SHIFT)
	_ensure_key_action("pause", KEY_ESCAPE)
	_ensure_key_action("toggle_debug", KEY_F1)


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
	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-52.0, -28.0, 0.0)
	sun.light_energy = 2.5
	add_child(sun)

	var world := WorldEnvironment.new()
	var env := Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.background_color = Color(0.47, 0.62, 0.70)
	env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.ambient_light_color = Color(0.55, 0.60, 0.58)
	env.ambient_light_energy = 0.8
	env.fog_enabled = true
	env.fog_density = 0.018
	env.fog_light_color = Color(0.50, 0.62, 0.58)
	world.environment = env
	add_child(world)


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
	]:
		_add_hut(item[0], item[1])
	for i in range(14):
		var angle := TAU * float(i) / 14.0
		_add_tree(Vector3(cos(angle) * 16.0, 0.0, sin(angle) * 14.0))


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

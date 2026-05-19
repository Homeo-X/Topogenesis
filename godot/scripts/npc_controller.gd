extends CharacterBody3D

@export var npc_id := "npc"
@export var display_name := "Villager"
@export var home_position := Vector3.ZERO
@export var walk_speed := 1.65
@export var turn_speed := 8.0

var wander_angle := 0.0
var retarget_timer := 0.0
var current_target := Vector3.ZERO
var state: Dictionary = {}
var overhead_label: Label3D
var pressure_ring: MeshInstance3D
var body_material: StandardMaterial3D
var cloak_material: StandardMaterial3D


func _ready() -> void:
	add_to_group("npc")
	TopogenesisBridge.register_npc(npc_id, display_name)
	current_target = home_position
	_build_body()


func _physics_process(delta: float) -> void:
	var player := get_tree().get_first_node_in_group("player") as Node3D
	var distance_to_player := 999.0
	if player != null:
		distance_to_player = global_position.distance_to(player.global_position)

	var hazard := _hazard_pressure()
	var resource := _resource_pressure()
	state = TopogenesisBridge.step_npc(npc_id, delta, {
		"hazard": hazard,
		"resource": resource,
		"player_help": 0.0,
		"player_threat": 1.0 if distance_to_player < 1.8 and Input.is_action_pressed("sprint") else 0.0,
	})
	_update_visual_state()

	retarget_timer -= delta
	if retarget_timer <= 0.0 or global_position.distance_to(current_target) < 0.85:
		current_target = _desired_position(player, distance_to_player)
		retarget_timer = 0.75 + 0.15 * float(npc_id.length() % 4)

	var direction := global_position.direction_to(current_target)
	direction.y = 0.0
	if global_position.distance_to(current_target) > 0.65:
		var target_velocity := direction.normalized() * walk_speed
		velocity.x = move_toward(velocity.x, target_velocity.x, 5.5 * delta)
		velocity.z = move_toward(velocity.z, target_velocity.z, 5.5 * delta)
	else:
		velocity.x = move_toward(velocity.x, 0.0, 5.5 * delta)
		velocity.z = move_toward(velocity.z, 0.0, 5.5 * delta)
	velocity.y = -0.1
	move_and_slide()

	if Vector2(velocity.x, velocity.z).length() > 0.05:
		var target_yaw := atan2(-velocity.x, -velocity.z)
		rotation.y = lerp_angle(rotation.y, target_yaw, minf(1.0, turn_speed * delta))


func interact() -> String:
	TopogenesisBridge.remember(npc_id, "player_interaction", 0.25, "player_spoke")
	state = TopogenesisBridge.step_npc(npc_id, 0.1, {
		"hazard": _hazard_pressure(),
		"resource": _resource_pressure(),
		"player_help": 0.8,
		"player_threat": 0.0,
	})
	_update_visual_state()
	return TopogenesisBridge.interaction_line(npc_id)


func debug_summary() -> String:
	if state.is_empty():
		state = TopogenesisBridge.step_npc(npc_id, 0.0, {})
	return "%s | need:%s %.2f | affect:%.2f | future:%s | trust:%.2f" % [
		display_name,
		state.get("dominant_need", "unknown"),
		state.get("need_total", 0.0),
		state.get("affect_stability", 0.0),
		state.get("future_action", "observe"),
		state.get("trust_player", 0.5),
	]


func current_state() -> Dictionary:
	return state.duplicate(true)


func _desired_position(player: Node3D, distance_to_player: float) -> Vector3:
	var need := str(state.get("dominant_need", "epistemic"))
	if need == "safety" and player != null and distance_to_player < 5.0:
		var away := (global_position - player.global_position)
		away.y = 0.0
		if away.length() < 0.1:
			away = Vector3.RIGHT
		return _bounded_world(global_position + away.normalized() * 4.0)
	if need == "metabolic":
		return Vector3(-8.0, 0.0, -6.0)
	if need == "epistemic" and player != null and distance_to_player < 8.0:
		var offset_seed := float(absi(hash(npc_id)) % 628) / 100.0
		var offset := Vector3(cos(offset_seed), 0.0, sin(offset_seed)) * 2.2
		return _bounded_world(player.global_position + offset)
	wander_angle += 0.015
	var radius := 2.2 + 0.3 * float(npc_id.length() % 3)
	return _bounded_world(home_position + Vector3(
		cos(wander_angle + float(npc_id.length())),
		0.0,
		sin(wander_angle * 0.8 + float(npc_id.length()))
	) * radius)


func _bounded_world(pos: Vector3) -> Vector3:
	return Vector3(clampf(pos.x, -18.0, 18.0), 0.2, clampf(pos.z, -18.0, 18.0))


func _hazard_pressure() -> float:
	var hazard_center := Vector3(7.0, 0.0, -7.0)
	return clampf(1.0 - global_position.distance_to(hazard_center) / 14.0, 0.0, 1.0)


func _resource_pressure() -> float:
	var resource_center := Vector3(-8.0, 0.0, -6.0)
	return clampf(1.0 - global_position.distance_to(resource_center) / 10.0, 0.0, 1.0)


func _build_body() -> void:
	var palette := _palette()
	body_material = _make_material(palette["body"], 0.78)
	cloak_material = _make_material(palette["cloak"], 0.88)

	var torso := MeshInstance3D.new()
	torso.name = "Torso"
	var torso_mesh := CapsuleMesh.new()
	torso_mesh.radius = 0.34
	torso_mesh.height = 1.22
	torso.mesh = torso_mesh
	torso.position.y = 0.86
	torso.scale = Vector3(0.95, 1.0, 0.72)
	torso.material_override = body_material
	add_child(torso)

	var cloak := MeshInstance3D.new()
	cloak.name = "Cloak"
	var cloak_mesh := BoxMesh.new()
	cloak_mesh.size = Vector3(0.78, 1.05, 0.10)
	cloak.mesh = cloak_mesh
	cloak.position = Vector3(0.0, 0.92, 0.30)
	cloak.material_override = cloak_material
	add_child(cloak)

	var head := MeshInstance3D.new()
	head.name = "Head"
	var head_mesh := SphereMesh.new()
	head_mesh.radius = 0.27
	head_mesh.height = 0.42
	head.mesh = head_mesh
	head.position.y = 1.66
	head.material_override = _make_material(palette["skin"], 0.65)
	add_child(head)

	var hood := MeshInstance3D.new()
	hood.name = "Hood"
	var hood_mesh := TorusMesh.new()
	hood_mesh.inner_radius = 0.24
	hood_mesh.outer_radius = 0.31
	hood.mesh = hood_mesh
	hood.position = Vector3(0.0, 1.67, 0.02)
	hood.rotation_degrees.x = 90.0
	hood.material_override = cloak_material
	add_child(hood)

	_add_eye(Vector3(-0.09, 1.68, -0.245))
	_add_eye(Vector3(0.09, 1.68, -0.245))
	_add_side_satchel(palette["accent"])

	var shape := CollisionShape3D.new()
	var capsule_shape := CapsuleShape3D.new()
	capsule_shape.radius = 0.32
	capsule_shape.height = 1.45
	shape.shape = capsule_shape
	shape.position.y = 0.78
	add_child(shape)

	pressure_ring = MeshInstance3D.new()
	var ring := TorusMesh.new()
	ring.inner_radius = 0.42
	ring.outer_radius = 0.48
	pressure_ring.mesh = ring
	pressure_ring.position.y = 0.04
	add_child(pressure_ring)

	overhead_label = Label3D.new()
	overhead_label.text = display_name
	overhead_label.position.y = 2.05
	overhead_label.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	overhead_label.modulate = Color(0.95, 0.92, 0.78)
	add_child(overhead_label)


func _add_eye(pos: Vector3) -> void:
	var eye := MeshInstance3D.new()
	var mesh := SphereMesh.new()
	mesh.radius = 0.028
	mesh.height = 0.04
	eye.mesh = mesh
	eye.position = pos
	eye.material_override = _make_material(Color(0.08, 0.07, 0.05), 0.4)
	add_child(eye)


func _add_side_satchel(color: Color) -> void:
	var satchel := MeshInstance3D.new()
	var mesh := BoxMesh.new()
	mesh.size = Vector3(0.22, 0.28, 0.10)
	satchel.mesh = mesh
	satchel.position = Vector3(0.34, 0.82, -0.03)
	satchel.rotation_degrees.z = -8.0
	satchel.material_override = _make_material(color, 0.8)
	add_child(satchel)


func _make_material(color: Color, roughness: float) -> StandardMaterial3D:
	var mat := StandardMaterial3D.new()
	mat.albedo_color = color
	mat.roughness = roughness
	return mat


func _palette() -> Dictionary:
	var idx: int = absi(hash(npc_id)) % 3
	if idx == 0:
		return {
			"body": Color(0.64, 0.47, 0.28),
			"cloak": Color(0.24, 0.37, 0.32),
			"skin": Color(0.72, 0.55, 0.42),
			"accent": Color(0.84, 0.67, 0.24),
		}
	if idx == 1:
		return {
			"body": Color(0.44, 0.38, 0.62),
			"cloak": Color(0.26, 0.22, 0.35),
			"skin": Color(0.62, 0.46, 0.34),
			"accent": Color(0.35, 0.63, 0.82),
		}
	return {
		"body": Color(0.58, 0.35, 0.26),
		"cloak": Color(0.34, 0.18, 0.16),
		"skin": Color(0.78, 0.61, 0.45),
		"accent": Color(0.53, 0.76, 0.39),
	}


func _update_visual_state() -> void:
	if overhead_label == null or pressure_ring == null:
		return
	var need := str(state.get("dominant_need", "unknown"))
	var affect := float(state.get("affect_stability", 0.5))
	var threat := float(state.get("threat_salience", 0.0))
	overhead_label.text = "%s\n%s %.2f" % [display_name, need, state.get("need_total", 0.0)]
	var mat := StandardMaterial3D.new()
	mat.emission_enabled = true
	if need == "safety":
		mat.albedo_color = Color(0.90, 0.20, 0.16)
	elif need == "metabolic":
		mat.albedo_color = Color(0.20, 0.82, 0.36)
	elif need == "epistemic":
		mat.albedo_color = Color(0.35, 0.58, 1.00)
	else:
		mat.albedo_color = Color(0.86, 0.72, 0.25)
	mat.emission = mat.albedo_color
	mat.emission_energy_multiplier = 0.3 + 0.7 * threat
	mat.roughness = 0.7
	pressure_ring.material_override = mat
	pressure_ring.scale = Vector3.ONE * lerpf(0.85, 1.25, 1.0 - affect)

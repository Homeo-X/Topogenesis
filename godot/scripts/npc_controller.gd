extends CharacterBody3D

@export var npc_id := "npc"
@export var display_name := "Villager"
@export var home_position := Vector3.ZERO

var wander_angle := 0.0
var state: Dictionary = {}
var overhead_label: Label3D
var pressure_ring: MeshInstance3D


func _ready() -> void:
	add_to_group("npc")
	TopogenesisBridge.register_npc(npc_id, display_name)
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

	var target := _desired_position(player, distance_to_player)
	var direction := global_position.direction_to(target)
	direction.y = 0.0
	if global_position.distance_to(target) > 0.4:
		velocity.x = direction.x * 2.0
		velocity.z = direction.z * 2.0
	else:
		velocity.x = move_toward(velocity.x, 0.0, 8.0 * delta)
		velocity.z = move_toward(velocity.z, 0.0, 8.0 * delta)
	velocity.y = -0.1
	move_and_slide()

	if Vector2(velocity.x, velocity.z).length() > 0.05:
		rotation.y = atan2(-velocity.x, -velocity.z)


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


func _desired_position(player: Node3D, distance_to_player: float) -> Vector3:
	var need := str(state.get("dominant_need", "epistemic"))
	if need == "safety" and player != null and distance_to_player < 5.0:
		return global_position + (global_position - player.global_position).normalized() * 4.0
	if need == "metabolic":
		return Vector3(-8.0, 0.0, -6.0)
	if need == "epistemic" and player != null and distance_to_player < 8.0:
		return player.global_position + Vector3(1.8, 0.0, 1.8)
	wander_angle += 0.015
	return home_position + Vector3(cos(wander_angle + float(npc_id.length())), 0.0, sin(wander_angle)) * 2.0


func _hazard_pressure() -> float:
	var hazard_center := Vector3(7.0, 0.0, -7.0)
	return clampf(1.0 - global_position.distance_to(hazard_center) / 14.0, 0.0, 1.0)


func _resource_pressure() -> float:
	var resource_center := Vector3(-8.0, 0.0, -6.0)
	return clampf(1.0 - global_position.distance_to(resource_center) / 10.0, 0.0, 1.0)


func _build_body() -> void:
	var mesh := MeshInstance3D.new()
	var capsule := CapsuleMesh.new()
	capsule.radius = 0.32
	capsule.height = 1.45
	mesh.mesh = capsule
	mesh.position.y = 0.78
	var material := StandardMaterial3D.new()
	material.albedo_color = Color(0.72, 0.58, 0.34)
	material.roughness = 0.85
	mesh.material_override = material
	add_child(mesh)

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

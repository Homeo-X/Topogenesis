extends CharacterBody3D

@export var speed := 6.0
@export var sprint_speed := 8.5
@export var acceleration := 16.0
@export var braking := 18.0
@export var gravity := 18.0
@export var arrival_radius := 0.28
@export var turn_speed := 10.0
@export var zoom_step := 1.1
@export var min_zoom := 4.8
@export var max_zoom := 14.0

const CHARACTER_ASSET_ROOT := "res://assets/quaternius/animated_characters/Ultimate Animated Character Pack - Nov 2019/FBX/"

var look_target: Vector3 = Vector3.FORWARD
var camera: Camera3D
var move_target := Vector3.ZERO
var has_move_target := false
var camera_distance := 8.0
var target_camera_distance := 8.0


func _ready() -> void:
	add_to_group("player")
	move_target = global_position
	_build_body()


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseButton and event.pressed:
		var mouse_event := event as InputEventMouseButton
		if mouse_event.button_index == MOUSE_BUTTON_LEFT:
			_set_move_target(mouse_event.position)
		elif mouse_event.button_index == MOUSE_BUTTON_WHEEL_UP:
			target_camera_distance = clampf(target_camera_distance - zoom_step, min_zoom, max_zoom)
		elif mouse_event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			target_camera_distance = clampf(target_camera_distance + zoom_step, min_zoom, max_zoom)


func _physics_process(delta: float) -> void:
	var to_target := move_target - global_position
	to_target.y = 0.0
	var distance := to_target.length()
	var input_dir := Vector3.ZERO
	if has_move_target and distance > arrival_radius:
		input_dir = to_target.normalized()
	elif has_move_target:
		has_move_target = false

	var active_speed := sprint_speed if Input.is_action_pressed("sprint") else speed
	var slow_factor := clampf(distance / 1.7, 0.25, 1.0) if has_move_target else 0.0

	if has_move_target:
		var target_velocity := input_dir * active_speed * slow_factor
		velocity.x = move_toward(velocity.x, target_velocity.x, acceleration * delta)
		velocity.z = move_toward(velocity.z, target_velocity.z, acceleration * delta)
	else:
		velocity.x = move_toward(velocity.x, 0.0, braking * delta)
		velocity.z = move_toward(velocity.z, 0.0, braking * delta)
	if not is_on_floor():
		velocity.y -= gravity * delta
	else:
		velocity.y = -0.1

	move_and_slide()
	if Vector2(velocity.x, velocity.z).length() > 0.04:
		look_target = Vector3(velocity.x, 0.0, velocity.z).normalized()
		var target_yaw := atan2(-look_target.x, -look_target.z)
		rotation.y = lerp_angle(rotation.y, target_yaw, minf(1.0, turn_speed * delta))
	_update_camera(delta)


func _update_camera(delta: float) -> void:
	if camera == null:
		return
	camera_distance = lerpf(camera_distance, target_camera_distance, minf(1.0, 9.0 * delta))
	var height := lerpf(3.8, 7.8, inverse_lerp(min_zoom, max_zoom, camera_distance))
	var desired := Vector3(0.0, height, camera_distance)
	camera.position = camera.position.lerp(desired, minf(1.0, 10.0 * delta))


func _set_move_target(screen_position: Vector2) -> void:
	if camera == null:
		return
	var origin := camera.project_ray_origin(screen_position)
	var ray := camera.project_ray_normal(screen_position)
	if absf(ray.y) < 0.001:
		return
	var distance_to_ground := (0.2 - origin.y) / ray.y
	if distance_to_ground <= 0.0:
		return
	move_target = origin + ray * distance_to_ground
	move_target.x = clampf(move_target.x, -19.5, 19.5)
	move_target.y = 0.2
	move_target.z = clampf(move_target.z, -19.5, 19.5)
	has_move_target = true


func _build_body() -> void:
	if _add_character_model() == null:
		var coat := _make_material(Color(0.18, 0.36, 0.72), 0.72)
		var leather := _make_material(Color(0.30, 0.20, 0.13), 0.84)
		var brass := _make_material(Color(0.92, 0.68, 0.28), 0.56)
		var skin := _make_material(Color(0.70, 0.52, 0.39), 0.64)

		var torso := MeshInstance3D.new()
		torso.name = "ExplorerCoat"
		var capsule := CapsuleMesh.new()
		capsule.radius = 0.36
		capsule.height = 1.55
		torso.mesh = capsule
		torso.position.y = 0.86
		torso.scale = Vector3(0.95, 1.0, 0.74)
		torso.material_override = coat
		add_child(torso)

		var sash := MeshInstance3D.new()
		sash.name = "FieldSash"
		var sash_mesh := BoxMesh.new()
		sash_mesh.size = Vector3(0.88, 0.16, 0.08)
		sash.mesh = sash_mesh
		sash.position = Vector3(0.0, 1.08, -0.31)
		sash.rotation_degrees.z = -15.0
		sash.material_override = brass
		add_child(sash)

		var head := MeshInstance3D.new()
		head.name = "Head"
		var head_mesh := SphereMesh.new()
		head_mesh.radius = 0.28
		head_mesh.height = 0.42
		head.mesh = head_mesh
		head.position.y = 1.68
		head.material_override = skin
		add_child(head)

		var collar := MeshInstance3D.new()
		collar.name = "RaisedCollar"
		var collar_mesh := TorusMesh.new()
		collar_mesh.inner_radius = 0.25
		collar_mesh.outer_radius = 0.34
		collar.mesh = collar_mesh
		collar.position = Vector3(0.0, 1.43, 0.02)
		collar.rotation_degrees.x = 90.0
		collar.material_override = leather
		add_child(collar)

		var pack := MeshInstance3D.new()
		pack.name = "CognitionPack"
		var pack_mesh := BoxMesh.new()
		pack_mesh.size = Vector3(0.58, 0.70, 0.18)
		pack.mesh = pack_mesh
		pack.position = Vector3(0.0, 0.98, 0.42)
		pack.material_override = leather
		add_child(pack)

		var core := MeshInstance3D.new()
		core.name = "SigmaCore"
		var core_mesh := SphereMesh.new()
		core_mesh.radius = 0.075
		core_mesh.height = 0.11
		core.mesh = core_mesh
		core.position = Vector3(0.0, 1.13, -0.39)
		var core_mat := _make_material(Color(0.42, 0.92, 1.00), 0.35)
		core_mat.emission_enabled = true
		core_mat.emission = Color(0.28, 0.74, 1.00)
		core_mat.emission_energy_multiplier = 1.1
		core.material_override = core_mat
		add_child(core)

		_add_eye(Vector3(-0.09, 1.70, -0.25))
		_add_eye(Vector3(0.09, 1.70, -0.25))

	var shape := CollisionShape3D.new()
	var capsule_shape := CapsuleShape3D.new()
	capsule_shape.radius = 0.35
	capsule_shape.height = 1.65
	shape.shape = capsule_shape
	shape.position.y = 0.85
	add_child(shape)

	camera = Camera3D.new()
	camera.name = "Camera3D"
	camera.position = Vector3(0.0, 5.8, 8.0)
	camera.rotation_degrees = Vector3(-38.0, 0.0, 0.0)
	camera.current = true
	add_child(camera)


func _add_character_model() -> Node3D:
	var path: String = CHARACTER_ASSET_ROOT + "Witch.fbx"
	if not FileAccess.file_exists(path):
		return null
	var packed = load(path)
	if packed == null or not packed is PackedScene:
		return null
	var model := (packed as PackedScene).instantiate() as Node3D
	if model == null:
		return null
	model.name = "RiggedPlayer"
	model.scale = Vector3.ONE * 0.95
	model.rotation_degrees.y = 180.0
	add_child(model)
	return model


func _add_eye(pos: Vector3) -> void:
	var eye := MeshInstance3D.new()
	var eye_mesh := SphereMesh.new()
	eye_mesh.radius = 0.028
	eye_mesh.height = 0.04
	eye.mesh = eye_mesh
	eye.position = pos
	eye.material_override = _make_material(Color(0.06, 0.05, 0.04), 0.42)
	add_child(eye)


func _make_material(color: Color, roughness: float) -> StandardMaterial3D:
	var material := StandardMaterial3D.new()
	material.albedo_color = color
	material.roughness = roughness
	return material

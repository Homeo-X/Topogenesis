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
@export var mouse_sensitivity := 0.12
@export var min_camera_pitch := 24.0
@export var max_camera_pitch := 62.0

const CHARACTER_ASSET_ROOT := "res://assets/quaternius/animated_characters/Ultimate Animated Character Pack - Nov 2019/FBX/"
const WORLD_HALF_EXTENT := 50.0

var look_target: Vector3 = Vector3.FORWARD
var camera: Camera3D
var visual_root: Node3D
var imported_model: Node3D
var move_target := Vector3.ZERO
var has_move_target := false
var camera_distance := 8.0
var target_camera_distance := 8.0
var camera_yaw := 0.0
var camera_pitch := 38.0
var mouse_look_pressed := false
var walk_phase := 0.0
var fallback_left_arm: Node3D
var fallback_right_arm: Node3D
var fallback_left_leg: Node3D
var fallback_right_leg: Node3D


func _ready() -> void:
	add_to_group("player")
	move_target = global_position
	_build_body()


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseButton:
		var mouse_event := event as InputEventMouseButton
		if mouse_event.button_index == MOUSE_BUTTON_RIGHT:
			mouse_look_pressed = mouse_event.pressed
		elif mouse_event.pressed and mouse_event.button_index == MOUSE_BUTTON_LEFT:
			_set_move_target(mouse_event.position)
		elif mouse_event.pressed and mouse_event.button_index == MOUSE_BUTTON_WHEEL_UP:
			target_camera_distance = clampf(target_camera_distance - zoom_step, min_zoom, max_zoom)
		elif mouse_event.pressed and mouse_event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			target_camera_distance = clampf(target_camera_distance + zoom_step, min_zoom, max_zoom)
	elif event is InputEventMouseMotion and mouse_look_pressed:
		var motion := event as InputEventMouseMotion
		camera_yaw -= motion.relative.x * mouse_sensitivity * 0.01
		camera_pitch = clampf(
			camera_pitch - motion.relative.y * mouse_sensitivity,
			min_camera_pitch,
			max_camera_pitch
		)


func _physics_process(delta: float) -> void:
	var keyboard_dir := _keyboard_move_direction()
	var to_target := move_target - global_position
	to_target.y = 0.0
	var distance := to_target.length()
	var input_dir := Vector3.ZERO
	var using_keyboard := keyboard_dir.length() > 0.01
	if using_keyboard:
		input_dir = keyboard_dir
		has_move_target = false
	elif has_move_target and distance > arrival_radius:
		input_dir = to_target.normalized()
	elif has_move_target:
		has_move_target = false

	var active_speed := sprint_speed if Input.is_action_pressed("sprint") else speed
	var slow_factor := 1.0 if using_keyboard else (clampf(distance / 1.7, 0.25, 1.0) if has_move_target else 0.0)

	if has_move_target or using_keyboard:
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
	_update_body_motion(delta)
	_update_camera(delta)


func _update_camera(delta: float) -> void:
	if camera == null:
		return
	camera_distance = lerpf(camera_distance, target_camera_distance, minf(1.0, 9.0 * delta))
	var pitch_rad := deg_to_rad(camera_pitch)
	var desired := Vector3(
		sin(camera_yaw) * cos(pitch_rad) * camera_distance,
		sin(pitch_rad) * camera_distance,
		cos(camera_yaw) * cos(pitch_rad) * camera_distance
	)
	camera.position = camera.position.lerp(desired, minf(1.0, 10.0 * delta))
	camera.look_at(global_position + Vector3(0.0, 1.05, 0.0), Vector3.UP)


func _keyboard_move_direction() -> Vector3:
	var input_2d := Input.get_vector("move_left", "move_right", "move_forward", "move_back")
	if input_2d.length() <= 0.01 or camera == null:
		return Vector3.ZERO
	var forward := -camera.global_transform.basis.z
	forward.y = 0.0
	forward = forward.normalized()
	var right := camera.global_transform.basis.x
	right.y = 0.0
	right = right.normalized()
	return (right * input_2d.x + forward * -input_2d.y).normalized()


func _update_body_motion(delta: float) -> void:
	if visual_root == null:
		return
	var horizontal_speed := Vector2(velocity.x, velocity.z).length()
	var moving := horizontal_speed > 0.08
	if moving:
		walk_phase += delta * lerpf(5.5, 9.5, clampf(horizontal_speed / sprint_speed, 0.0, 1.0))
	var stride := sin(walk_phase)
	var counter_stride := sin(walk_phase + PI)
	var bob := absf(stride) * 0.045 if moving else sin(Time.get_ticks_msec() * 0.002) * 0.012
	var lean := clampf(horizontal_speed / sprint_speed, 0.0, 1.0) * -0.08
	visual_root.position = visual_root.position.lerp(Vector3(0.0, bob, 0.0), minf(1.0, 12.0 * delta))
	visual_root.rotation.x = lerp_angle(visual_root.rotation.x, lean, minf(1.0, 10.0 * delta))
	visual_root.rotation.z = lerp_angle(
		visual_root.rotation.z,
		stride * 0.035 if moving else 0.0,
		minf(1.0, 10.0 * delta)
	)
	_animate_fallback_limb(fallback_left_arm, counter_stride, moving, delta, 0.55)
	_animate_fallback_limb(fallback_right_arm, stride, moving, delta, 0.55)
	_animate_fallback_limb(fallback_left_leg, stride, moving, delta, 0.42)
	_animate_fallback_limb(fallback_right_leg, counter_stride, moving, delta, 0.42)


func _animate_fallback_limb(limb: Node3D, stride: float, moving: bool, delta: float, amount: float) -> void:
	if limb == null:
		return
	var target := stride * amount if moving else 0.0
	limb.rotation.x = lerp_angle(limb.rotation.x, target, minf(1.0, 12.0 * delta))


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
	move_target.x = clampf(move_target.x, -WORLD_HALF_EXTENT, WORLD_HALF_EXTENT)
	move_target.y = 0.2
	move_target.z = clampf(move_target.z, -WORLD_HALF_EXTENT, WORLD_HALF_EXTENT)
	has_move_target = true


func _build_body() -> void:
	visual_root = Node3D.new()
	visual_root.name = "VisualRoot"
	add_child(visual_root)
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
		visual_root.add_child(torso)

		var sash := MeshInstance3D.new()
		sash.name = "FieldSash"
		var sash_mesh := BoxMesh.new()
		sash_mesh.size = Vector3(0.88, 0.16, 0.08)
		sash.mesh = sash_mesh
		sash.position = Vector3(0.0, 1.08, -0.31)
		sash.rotation_degrees.z = -15.0
		sash.material_override = brass
		visual_root.add_child(sash)

		var head := MeshInstance3D.new()
		head.name = "Head"
		var head_mesh := SphereMesh.new()
		head_mesh.radius = 0.28
		head_mesh.height = 0.42
		head.mesh = head_mesh
		head.position.y = 1.68
		head.material_override = skin
		visual_root.add_child(head)

		var collar := MeshInstance3D.new()
		collar.name = "RaisedCollar"
		var collar_mesh := TorusMesh.new()
		collar_mesh.inner_radius = 0.25
		collar_mesh.outer_radius = 0.34
		collar.mesh = collar_mesh
		collar.position = Vector3(0.0, 1.43, 0.02)
		collar.rotation_degrees.x = 90.0
		collar.material_override = leather
		visual_root.add_child(collar)

		var pack := MeshInstance3D.new()
		pack.name = "CognitionPack"
		var pack_mesh := BoxMesh.new()
		pack_mesh.size = Vector3(0.58, 0.70, 0.18)
		pack.mesh = pack_mesh
		pack.position = Vector3(0.0, 0.98, 0.42)
		pack.material_override = leather
		visual_root.add_child(pack)

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
		visual_root.add_child(core)

		_add_eye(Vector3(-0.09, 1.70, -0.25))
		_add_eye(Vector3(0.09, 1.70, -0.25))
		_add_fallback_limbs(coat, leather)

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
	camera.current = true
	add_child(camera)
	_update_camera(1.0)


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
	visual_root.add_child(model)
	imported_model = model
	_play_first_imported_animation(imported_model)
	return model


func _play_first_imported_animation(root: Node) -> void:
	if root == null:
		return
	if root is AnimationPlayer:
		var player := root as AnimationPlayer
		var animations := player.get_animation_list()
		for animation_name in animations:
			if animation_name != "RESET":
				player.play(animation_name)
				return
	for child in root.get_children():
		_play_first_imported_animation(child)


func _add_fallback_limbs(coat: StandardMaterial3D, leather: StandardMaterial3D) -> void:
	fallback_left_arm = _add_limb("LeftArm", Vector3(-0.42, 1.03, -0.02), coat, 0.12, 0.66)
	fallback_right_arm = _add_limb("RightArm", Vector3(0.42, 1.03, -0.02), coat, 0.12, 0.66)
	fallback_left_leg = _add_limb("LeftLeg", Vector3(-0.16, 0.30, 0.0), leather, 0.13, 0.64)
	fallback_right_leg = _add_limb("RightLeg", Vector3(0.16, 0.30, 0.0), leather, 0.13, 0.64)


func _add_limb(limb_name: String, pos: Vector3, material: StandardMaterial3D, radius: float, height: float) -> Node3D:
	var pivot := Node3D.new()
	pivot.name = limb_name
	pivot.position = pos
	var mesh_instance := MeshInstance3D.new()
	var mesh := CapsuleMesh.new()
	mesh.radius = radius
	mesh.height = height
	mesh_instance.mesh = mesh
	mesh_instance.position.y = -height * 0.35
	mesh_instance.material_override = material
	pivot.add_child(mesh_instance)
	visual_root.add_child(pivot)
	return pivot


func _add_eye(pos: Vector3) -> void:
	var eye := MeshInstance3D.new()
	var eye_mesh := SphereMesh.new()
	eye_mesh.radius = 0.028
	eye_mesh.height = 0.04
	eye.mesh = eye_mesh
	eye.position = pos
	eye.material_override = _make_material(Color(0.06, 0.05, 0.04), 0.42)
	visual_root.add_child(eye)


func _make_material(color: Color, roughness: float) -> StandardMaterial3D:
	var material := StandardMaterial3D.new()
	material.albedo_color = color
	material.roughness = roughness
	return material

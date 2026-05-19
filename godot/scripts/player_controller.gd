extends CharacterBody3D

@export var speed := 6.0
@export var sprint_speed := 8.5
@export var acceleration := 16.0
@export var gravity := 18.0

var look_target: Vector3 = Vector3.FORWARD
var camera: Camera3D


func _ready() -> void:
	add_to_group("player")
	_build_body()


func _physics_process(delta: float) -> void:
	var input_2d := Input.get_vector("move_left", "move_right", "move_forward", "move_back")
	var input_dir := Vector3(input_2d.x, 0.0, input_2d.y).normalized()
	var active_speed := sprint_speed if Input.is_action_pressed("sprint") else speed

	var target_velocity := input_dir * active_speed
	velocity.x = move_toward(velocity.x, target_velocity.x, acceleration * delta)
	velocity.z = move_toward(velocity.z, target_velocity.z, acceleration * delta)
	if not is_on_floor():
		velocity.y -= gravity * delta
	else:
		velocity.y = -0.1

	move_and_slide()
	if input_dir.length() > 0.01:
		look_target = input_dir
		rotation.y = atan2(-look_target.x, -look_target.z)
	_update_camera(delta)


func _update_camera(delta: float) -> void:
	if camera == null:
		return
	var desired := Vector3(0.0, 5.8, 8.0)
	camera.position = camera.position.lerp(desired, minf(1.0, 10.0 * delta))


func _build_body() -> void:
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

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
	var mesh := MeshInstance3D.new()
	mesh.name = "Body"
	var capsule := CapsuleMesh.new()
	capsule.radius = 0.35
	capsule.height = 1.65
	mesh.mesh = capsule
	mesh.position.y = 0.85
	var material := StandardMaterial3D.new()
	material.albedo_color = Color(0.28, 0.54, 0.92)
	material.roughness = 0.72
	mesh.material_override = material
	add_child(mesh)

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

extends CanvasLayer

var info_label: Label
var dialogue_label: Label
var prompt_label: Label
var debug_visible := true
var dialogue_timer := 0.0


func _ready() -> void:
	info_label = Label.new()
	info_label.position = Vector2(16, 16)
	info_label.size = Vector2(760, 260)
	info_label.add_theme_font_size_override("font_size", 16)
	add_child(info_label)

	dialogue_label = Label.new()
	dialogue_label.position = Vector2(16, 604)
	dialogue_label.size = Vector2(1100, 80)
	dialogue_label.add_theme_font_size_override("font_size", 20)
	dialogue_label.text = "WASD move | Shift sprint/threat | E interact | F1 debug | Esc pause"
	add_child(dialogue_label)

	prompt_label = Label.new()
	prompt_label.position = Vector2(16, 560)
	prompt_label.size = Vector2(900, 40)
	prompt_label.add_theme_font_size_override("font_size", 18)
	add_child(prompt_label)


func _process(delta: float) -> void:
	dialogue_timer = maxf(0.0, dialogue_timer - delta)
	if Input.is_action_just_pressed("toggle_debug"):
		debug_visible = not debug_visible
		info_label.visible = debug_visible
	if dialogue_timer <= 0.0 and not dialogue_label.text.begins_with("WASD"):
		dialogue_label.text = "WASD move | Shift sprint/threat | E interact | F1 debug | Esc pause"

	var lines: Array[String] = ["Topogenesis RPG Vertical Slice"]
	for npc in get_tree().get_nodes_in_group("npc"):
		if npc.has_method("debug_summary"):
			lines.append(npc.debug_summary())
	info_label.text = "\n".join(lines)


func show_dialogue(text: String) -> void:
	dialogue_label.text = text
	dialogue_timer = 5.0


func set_prompt(text: String) -> void:
	prompt_label.text = text


func show_pause(paused: bool) -> void:
	if paused:
		dialogue_label.text = "Paused | Esc resume"
		dialogue_timer = 999.0
	else:
		dialogue_timer = 0.0

extends CanvasLayer

var info_label: Label
var objective_label: Label
var dialogue_label: Label
var prompt_label: Label
var focus_panel: ColorRect
var focus_label: Label
var debug_visible := true
var dialogue_timer := 0.0


func _ready() -> void:
	var top_shadow := ColorRect.new()
	top_shadow.color = Color(0.03, 0.025, 0.02, 0.42)
	top_shadow.position = Vector2(0, 0)
	top_shadow.size = Vector2(1280, 108)
	add_child(top_shadow)

	objective_label = Label.new()
	objective_label.position = Vector2(16, 12)
	objective_label.size = Vector2(1000, 64)
	objective_label.add_theme_font_size_override("font_size", 18)
	add_child(objective_label)

	info_label = Label.new()
	info_label.position = Vector2(16, 78)
	info_label.size = Vector2(760, 260)
	info_label.add_theme_font_size_override("font_size", 16)
	add_child(info_label)

	dialogue_label = Label.new()
	dialogue_label.position = Vector2(16, 604)
	dialogue_label.size = Vector2(1100, 80)
	dialogue_label.add_theme_font_size_override("font_size", 20)
	dialogue_label.text = "Left click move | WASD move | Right-drag rotate | Wheel zoom | E interact | F1 debug"
	add_child(dialogue_label)

	prompt_label = Label.new()
	prompt_label.position = Vector2(16, 560)
	prompt_label.size = Vector2(900, 40)
	prompt_label.add_theme_font_size_override("font_size", 18)
	add_child(prompt_label)

	focus_panel = ColorRect.new()
	focus_panel.color = Color(0.03, 0.025, 0.02, 0.68)
	focus_panel.position = Vector2(912, 112)
	focus_panel.size = Vector2(340, 188)
	focus_panel.visible = false
	add_child(focus_panel)

	focus_label = Label.new()
	focus_label.position = Vector2(930, 126)
	focus_label.size = Vector2(304, 160)
	focus_label.add_theme_font_size_override("font_size", 17)
	focus_label.visible = false
	add_child(focus_label)


func _process(delta: float) -> void:
	dialogue_timer = maxf(0.0, dialogue_timer - delta)
	if Input.is_action_just_pressed("toggle_debug"):
		debug_visible = not debug_visible
		info_label.visible = debug_visible
	if dialogue_timer <= 0.0 and not dialogue_label.text.begins_with("Left click"):
		dialogue_label.text = "Left click move | WASD move | Right-drag rotate | Wheel zoom | E interact | F1 debug"

	objective_label.text = "%s\n%s\n%s" % [
		GameDirector.objective_text(),
		GameDirector.status_text(),
		TopogenesisBridge.backend_status(),
	]
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


func set_focus_state(npc_name: String, state: Dictionary) -> void:
	var need := str(state.get("dominant_need", "unknown"))
	var need_total := float(state.get("need_total", 0.0))
	var affect := float(state.get("affect_stability", 0.0))
	var future := str(state.get("future_action", "observe"))
	var trust := float(state.get("trust_player", 0.5))
	var memory_events: Array = state.get("memory_events", [])
	var memory_text := "none"
	if not memory_events.is_empty() and typeof(memory_events[-1]) == TYPE_DICTIONARY:
		memory_text = str(memory_events[-1].get("claim", "recent pressure"))
	focus_label.text = "%s\nNeed: %s %.2f\nAffect stability: %.2f\nIntention: %s\nTrust: %.2f\nRecent memory: %s" % [
		npc_name,
		need,
		need_total,
		affect,
		future.replace("_", " "),
		trust,
		memory_text,
	]
	focus_panel.visible = true
	focus_label.visible = true


func clear_focus_state() -> void:
	focus_panel.visible = false
	focus_label.visible = false


func show_pause(paused: bool) -> void:
	if paused:
		dialogue_label.text = "Paused | Esc resume"
		dialogue_timer = 999.0
	else:
		dialogue_timer = 0.0

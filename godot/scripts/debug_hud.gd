extends CanvasLayer

var info_label: Label
var objective_label: Label
var world_label: Label
var dialogue_label: Label
var prompt_label: Label
var controls_label: Label
var focus_panel: ColorRect
var focus_label: Label
var debug_visible := false
var dialogue_timer := 0.0

const MAX_DEBUG_NPCS := 6


func _ready() -> void:
	var top_shadow := ColorRect.new()
	top_shadow.color = Color(0.03, 0.025, 0.02, 0.42)
	top_shadow.position = Vector2(0, 0)
	top_shadow.size = Vector2(1280, 84)
	add_child(top_shadow)

	objective_label = Label.new()
	objective_label.position = Vector2(16, 12)
	objective_label.size = Vector2(760, 58)
	objective_label.add_theme_font_size_override("font_size", 18)
	add_child(objective_label)

	world_label = Label.new()
	world_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	world_label.position = Vector2(782, 14)
	world_label.size = Vector2(470, 52)
	world_label.add_theme_font_size_override("font_size", 14)
	world_label.modulate = Color(0.88, 0.86, 0.76)
	add_child(world_label)

	info_label = Label.new()
	info_label.position = Vector2(16, 94)
	info_label.size = Vector2(560, 132)
	info_label.add_theme_font_size_override("font_size", 12)
	info_label.visible = false
	add_child(info_label)

	dialogue_label = Label.new()
	dialogue_label.position = Vector2(16, 548)
	dialogue_label.size = Vector2(850, 48)
	dialogue_label.add_theme_font_size_override("font_size", 19)
	dialogue_label.text = ""
	add_child(dialogue_label)

	prompt_label = Label.new()
	prompt_label.position = Vector2(16, 500)
	prompt_label.size = Vector2(760, 36)
	prompt_label.add_theme_font_size_override("font_size", 18)
	add_child(prompt_label)

	controls_label = Label.new()
	controls_label.position = Vector2(16, 622)
	controls_label.size = Vector2(1040, 32)
	controls_label.add_theme_font_size_override("font_size", 15)
	controls_label.modulate = Color(0.90, 0.88, 0.78)
	controls_label.text = "LMB move    WASD move    RMB rotate    Wheel zoom    E interact    F1 diagnostics"
	add_child(controls_label)

	focus_panel = ColorRect.new()
	focus_panel.color = Color(0.025, 0.022, 0.018, 0.78)
	focus_panel.position = Vector2(890, 100)
	focus_panel.size = Vector2(362, 214)
	focus_panel.visible = false
	add_child(focus_panel)

	focus_label = Label.new()
	focus_label.position = Vector2(910, 116)
	focus_label.size = Vector2(322, 182)
	focus_label.add_theme_font_size_override("font_size", 16)
	focus_label.visible = false
	add_child(focus_label)


func _process(delta: float) -> void:
	dialogue_timer = maxf(0.0, dialogue_timer - delta)
	if Input.is_action_just_pressed("toggle_debug"):
		debug_visible = not debug_visible
		info_label.visible = debug_visible
	if dialogue_timer <= 0.0:
		dialogue_label.text = ""

	objective_label.text = "%s\n%s" % [
		GameDirector.objective_text(),
		GameDirector.status_text(),
	]
	world_label.text = _world_status_text()
	var lines: Array[String] = ["Topogenesis RPG Vertical Slice"]
	var snapshot := TopogenesisBridge.current_world_snapshot()
	if not snapshot.is_empty():
		var clock: Dictionary = snapshot.get("clock", {})
		var world: Dictionary = snapshot.get("world", {})
		var summary: Dictionary = snapshot.get("summary", {})
		lines.append("Offline world day:%s pop:%s food:%.0f%% water:%.0f%%" % [
			str(clock.get("day", 0)),
			str(summary.get("population_final", "?")),
			100.0 * float(world.get("food_stock", 1.0)),
			100.0 * float(world.get("water_stock", 1.0)),
		])
	var npc_count := 0
	for npc in get_tree().get_nodes_in_group("npc"):
		npc_count += 1
		if lines.size() >= MAX_DEBUG_NPCS + 2:
			continue
		if npc.has_method("debug_summary"):
			lines.append(_compact_debug_summary(npc.debug_summary()))
	if npc_count > MAX_DEBUG_NPCS:
		lines.append("... %d more villagers | focus one for details" % [npc_count - MAX_DEBUG_NPCS])
	info_label.text = "\n".join(lines)


func show_dialogue(text: String) -> void:
	dialogue_label.text = text
	dialogue_timer = 5.0


func set_prompt(text: String) -> void:
	prompt_label.text = text


func _compact_debug_summary(summary: String) -> String:
	var parts := summary.split(" | ")
	if parts.size() < 4:
		return summary
	var name := parts[0]
	var need := parts[1].replace("need:", "")
	var affect := parts[2].replace("affect:", "aff:")
	return "%s | %s | %s" % [name, need, affect]


func _world_status_text() -> String:
	var bridge := "Live cognition" if TopogenesisBridge.backend_status().find("online") >= 0 else "Local fallback"
	var snapshot := TopogenesisBridge.current_world_snapshot()
	if snapshot.is_empty():
		return "%s\nF1 diagnostics" % bridge
	var clock: Dictionary = snapshot.get("clock", {})
	var world: Dictionary = snapshot.get("world", {})
	var summary: Dictionary = snapshot.get("summary", {})
	return "%s\nDay %s   Pop %s   Food %.0f%%   Water %.0f%%" % [
		bridge,
		str(clock.get("day", 0)),
		str(summary.get("population_final", "?")),
		100.0 * float(world.get("food_stock", 1.0)),
		100.0 * float(world.get("water_stock", 1.0)),
	]


func set_focus_state(npc_name: String, state: Dictionary) -> void:
	var need := str(state.get("dominant_need", "unknown"))
	var need_total := float(state.get("need_total", 0.0))
	var calling := str(state.get("calling", "villager"))
	var affect := float(state.get("affect_stability", 0.0))
	var future := str(state.get("future_action", "observe"))
	var trust := float(state.get("trust_player", 0.5))
	var memory_events: Array = state.get("memory_events", [])
	var memory_text := "none"
	if not memory_events.is_empty() and typeof(memory_events[-1]) == TYPE_DICTIONARY:
		memory_text = str(memory_events[-1].get("claim", "recent pressure"))
	focus_label.text = "%s\n%s\n\nNeed        %s %.2f\nStability   %.2f\nIntention   %s\nTrust       %.2f\nMemory      %s" % [
		npc_name,
		calling,
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

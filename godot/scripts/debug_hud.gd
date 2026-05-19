extends CanvasLayer

var info_label: Label
var dialogue_label: Label
var dialogue_timer := 0.0


func _ready() -> void:
	info_label = Label.new()
	info_label.position = Vector2(16, 16)
	info_label.size = Vector2(760, 260)
	info_label.add_theme_font_size_override("font_size", 16)
	add_child(info_label)

	dialogue_label = Label.new()
	dialogue_label.position = Vector2(16, 620)
	dialogue_label.size = Vector2(1100, 80)
	dialogue_label.add_theme_font_size_override("font_size", 20)
	dialogue_label.text = "WASD move | E interact | Hold Shift near NPC to create threat pressure"
	add_child(dialogue_label)


func _process(delta: float) -> void:
	dialogue_timer = maxf(0.0, dialogue_timer - delta)
	if dialogue_timer <= 0.0 and not dialogue_label.text.begins_with("WASD"):
		dialogue_label.text = "WASD move | E interact | Hold Shift near NPC to create threat pressure"

	var lines: Array[String] = ["Topogenesis RPG Prototype"]
	for npc in get_tree().get_nodes_in_group("npc"):
		if npc.has_method("debug_summary"):
			lines.append(npc.debug_summary())
	info_label.text = "\n".join(lines)


func show_dialogue(text: String) -> void:
	dialogue_label.text = text
	dialogue_timer = 5.0

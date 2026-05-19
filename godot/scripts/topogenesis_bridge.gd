extends Node

const MAX_MEMORY_EVENTS := 128

var npc_states: Dictionary = {}


func register_npc(npc_id: String, display_name: String) -> void:
	if npc_states.has(npc_id):
		return
	npc_states[npc_id] = {
		"display_name": display_name,
		"energy": 1.0,
		"bodily_integrity": 1.0,
		"prediction_coherence": 0.78,
		"social_stability": 0.58,
		"environmental_safety": 0.76,
		"need_total": 0.12,
		"dominant_need": "epistemic",
		"affect_stability": 0.72,
		"threat_salience": 0.08,
		"trust_player": 0.5,
		"future_action": "observe",
		"memory_events": [],
	}


func step_npc(npc_id: String, delta: float, world_pressure: Dictionary) -> Dictionary:
	if not npc_states.has(npc_id):
		register_npc(npc_id, npc_id)

	var state: Dictionary = npc_states[npc_id]
	var hazard: float = clampf(float(world_pressure.get("hazard", 0.0)), 0.0, 1.0)
	var resource: float = clampf(float(world_pressure.get("resource", 0.0)), 0.0, 1.0)
	var player_help: float = clampf(float(world_pressure.get("player_help", 0.0)), 0.0, 1.0)
	var player_threat: float = clampf(float(world_pressure.get("player_threat", 0.0)), 0.0, 1.0)

	state.energy = clampf(state.energy - 0.006 * delta + 0.018 * resource * delta, 0.0, 1.0)
	state.environmental_safety = clampf(1.0 - hazard, 0.0, 1.0)
	state.prediction_coherence = clampf(
		lerpf(state.prediction_coherence, 1.0 - 0.55 * hazard + 0.15 * resource, 0.05),
		0.0,
		1.0
	)
	state.trust_player = clampf(
		state.trust_player + 0.09 * player_help * delta - 0.12 * player_threat * delta,
		0.0,
		1.0
	)

	var metabolic: float = 1.0 - state.energy
	var epistemic: float = 1.0 - state.prediction_coherence
	var safety: float = 1.0 - state.environmental_safety
	var social: float = 1.0 - state.social_stability
	state.need_total = clampf(
		0.28 * metabolic + 0.22 * epistemic + 0.30 * safety + 0.20 * social,
		0.0,
		1.0
	)
	state.dominant_need = _dominant_need({
		"metabolic": metabolic,
		"epistemic": epistemic,
		"safety": safety,
		"social": social,
	})
	state.threat_salience = clampf(lerpf(state.threat_salience, maxf(safety, player_threat), 0.08), 0.0, 1.0)
	state.affect_stability = clampf(
		0.35 * state.environmental_safety
		+ 0.25 * state.prediction_coherence
		+ 0.20 * state.energy
		+ 0.20 * state.social_stability
		- 0.20 * state.threat_salience,
		0.0,
		1.0
	)
	state.future_action = _choose_future_action(metabolic, epistemic, safety, state.trust_player)
	return state


func remember(npc_id: String, kind: String, valence: float, claim: String) -> void:
	if not npc_states.has(npc_id):
		register_npc(npc_id, npc_id)
	var memory: Array = npc_states[npc_id].memory_events
	memory.append({
		"kind": kind,
		"valence": clampf(valence, -1.0, 1.0),
		"claim": claim,
	})
	if memory.size() > MAX_MEMORY_EVENTS:
		memory.pop_front()


func interaction_line(npc_id: String) -> String:
	var state: Dictionary = npc_states.get(npc_id, {})
	var name: String = str(state.get("display_name", npc_id))
	var need: String = str(state.get("dominant_need", "unknown"))
	var trust: float = float(state.get("trust_player", 0.5))
	var future: String = str(state.get("future_action", "observe"))
	if trust > 0.68:
		return "%s: I remember your help. I should %s next." % [name, future]
	if trust < 0.32:
		return "%s: Keep your distance. My pressure is %s." % [name, need]
	return "%s: Something feels unstable. I need to handle %s." % [name, need]


func export_snapshot() -> Dictionary:
	var snapshot := {}
	for npc_id in npc_states:
		var source: Dictionary = npc_states[npc_id]
		snapshot[npc_id] = source.duplicate(true)
	return snapshot


func import_snapshot(snapshot: Dictionary) -> void:
	npc_states.clear()
	for npc_id in snapshot:
		var source: Dictionary = snapshot[npc_id]
		register_npc(str(npc_id), str(source.get("display_name", npc_id)))
		for key in source:
			npc_states[str(npc_id)][key] = source[key]
		var memory: Array = npc_states[str(npc_id)].get("memory_events", [])
		while memory.size() > MAX_MEMORY_EVENTS:
			memory.pop_front()


func _dominant_need(values: Dictionary) -> String:
	var best_key := "unknown"
	var best_value := -INF
	for key in values:
		var value := float(values[key])
		if value > best_value:
			best_value = value
			best_key = str(key)
	return best_key


func _choose_future_action(metabolic: float, epistemic: float, safety: float, trust_player: float) -> String:
	if safety > 0.48:
		return "seek_shelter"
	if metabolic > 0.42:
		return "seek_food"
	if epistemic > 0.36:
		return "verify_rumor"
	if trust_player > 0.65:
		return "share_information"
	return "patrol"

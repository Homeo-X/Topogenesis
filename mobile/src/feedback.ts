import { AppState } from "./types";

/**
 * Tactile + audio feedback scaffold.
 *
 * HAPTICS: opt-in (default off — some people find vibration aversive, a real
 * wellbeing consideration). Gated by preferences.hapticsEnabled. Uses
 * expo-haptics, imported lazily so the app runs even if the module is absent.
 *
 * AUDIO: scaffold only. There are NO bundled sound files (none can be shipped
 * without licensing). playCue() is a graceful no-op until the developer drops
 * licensed files into assets/audio/ and wires them in loadCues(). This keeps
 * the call sites stable without claiming sound that doesn't exist.
 */

type HapticKind = "light" | "success" | "warning";

let haptics: typeof import("expo-haptics") | null | undefined;

async function getHaptics() {
  if (haptics !== undefined) return haptics;
  try {
    haptics = await import("expo-haptics");
  } catch {
    haptics = null; // module not installed / unavailable
  }
  return haptics;
}

/** Fire a haptic if the user has enabled it. Always safe to call. */
export async function tap(state: AppState, kind: HapticKind = "light"): Promise<void> {
  if (!state.profile.preferences.hapticsEnabled) return;
  try {
    const h = await getHaptics();
    if (!h) return;
    if (kind === "success") await h.notificationAsync(h.NotificationFeedbackType.Success);
    else if (kind === "warning") await h.notificationAsync(h.NotificationFeedbackType.Warning);
    else await h.impactAsync(h.ImpactFeedbackStyle.Light);
  } catch {
    // Haptics unavailable on this device/build: silently skip.
  }
}

/**
 * Audio cues: original synthesized chiptune WAVs bundled in assets/audio/
 * (generated for this project — no licensing concerns). Played via expo-audio,
 * imported lazily so the app still runs if the module is absent. Gated by
 * preferences.soundEnabled. Always safe to call.
 */
export type AudioCue = "accept" | "complete" | "levelup";

const CUE_SOURCES: Record<AudioCue, number> = {
  accept: require("../assets/audio/accept.wav"),
  complete: require("../assets/audio/complete.wav"),
  levelup: require("../assets/audio/levelup.wav")
};

let audioModule: typeof import("expo-audio") | null | undefined;

async function getAudio() {
  if (audioModule !== undefined) return audioModule;
  try {
    audioModule = await import("expo-audio");
  } catch {
    audioModule = null; // module not installed / unavailable
  }
  return audioModule;
}

export async function playCue(state: AppState, cue: AudioCue): Promise<void> {
  if (!state.profile.preferences.soundEnabled) return;
  try {
    const audio = await getAudio();
    if (!audio) return;
    const player = audio.createAudioPlayer(CUE_SOURCES[cue]);
    player.play();
    // Release after the longest cue could have finished.
    setTimeout(() => { try { player.release(); } catch { /* already gone */ } }, 2000);
  } catch {
    // Audio unavailable on this device/build: silently skip.
  }
}

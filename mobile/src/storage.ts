import AsyncStorage from "@react-native-async-storage/async-storage";
import { Platform } from "react-native";
import { AppState } from "./types";

const storageKey = "forge_system_os_v1";

/**
 * Minimal structural check: enough to reject corrupt or foreign JSON before it
 * reaches ensureStateShape. We do not validate deeply here — ensureStateShape
 * back-fills missing fields — we only confirm this looks like an AppState object.
 */
function looksLikeAppState(value: unknown): value is AppState {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Record<string, unknown>;
  return (
    typeof candidate.profile === "object" &&
    typeof candidate.progression === "object" &&
    typeof candidate.host === "object"
  );
}

function safeParse(raw: string | null | undefined): AppState | null {
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as unknown;
    return looksLikeAppState(parsed) ? parsed : null;
  } catch {
    // Corrupt payload: treat as no saved state rather than crashing on launch.
    return null;
  }
}

export async function loadAppState(): Promise<AppState | null> {
  try {
    if (Platform.OS === "web") {
      return safeParse(globalThis.localStorage?.getItem(storageKey));
    }
    return safeParse(await AsyncStorage.getItem(storageKey));
  } catch {
    // Storage backend failure (quota, unavailable, permissions): start fresh.
    return null;
  }
}

export async function saveAppState(state: AppState): Promise<void> {
  try {
    const serialized = JSON.stringify(state);
    if (Platform.OS === "web") {
      globalThis.localStorage?.setItem(storageKey, serialized);
      return;
    }
    await AsyncStorage.setItem(storageKey, serialized);
  } catch {
    // Best-effort persistence: a failed save must never crash the app.
  }
}

export async function clearAppState(): Promise<void> {
  try {
    if (Platform.OS === "web") {
      globalThis.localStorage?.removeItem(storageKey);
      return;
    }
    await AsyncStorage.removeItem(storageKey);
  } catch {
    // Ignore: clearing is best-effort.
  }
}

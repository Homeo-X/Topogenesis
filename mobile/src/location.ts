import { AppState } from "./types";

/**
 * Location layer for territory discovery and location-backed evidence.
 *
 * Access model: on-demand only. A single position read happens when the
 * player opens the World map or completes a quest with location on — never
 * continuously, never in the background.
 *
 * Privacy architecture (the load-bearing decision): raw coordinates are
 * quantized to ~400m grid cells IMMEDIATELY on read and then discarded. Only
 * cell ids are stored, synced, or attached to evidence. A cell id says "was
 * somewhere in this 400m square at some point" — it cannot reconstruct a
 * route, a home address, or a timeline.
 *
 * Native access goes through lazy imports with full fallbacks (the same
 * pattern as haptics/audio), so the app runs unchanged when expo-location is
 * absent, permission is denied, or the platform is web.
 */

/** Cell edge length in meters. ~400m ≈ a small neighborhood block cluster. */
export const CELL_SIZE_M = 400;

const METERS_PER_DEG_LAT = 111_320;

/** Pure: quantize a coordinate to its grid cell id. Deterministic; rows are
 *  fixed latitude bands and column width adapts to the band's latitude so
 *  cells stay roughly square anywhere on Earth. */
export function cellOf(latitude: number, longitude: number): string {
  const latStep = CELL_SIZE_M / METERS_PER_DEG_LAT;
  const row = Math.floor(latitude / latStep);
  const rowCenterLat = (row + 0.5) * latStep;
  const cosLat = Math.max(0.01, Math.cos((rowCenterLat * Math.PI) / 180));
  const lonStep = CELL_SIZE_M / (METERS_PER_DEG_LAT * cosLat);
  const col = Math.floor(longitude / lonStep);
  return `c${row}_${col}`;
}

/** Pure: parse a cell id back to grid indices (for fog-of-war rendering). */
export function parseCell(id: string): { row: number; col: number } | null {
  const match = /^c(-?\d+)_(-?\d+)$/.exec(id);
  if (!match) return null;
  return { row: Number(match[1]), col: Number(match[2]) };
}

type LocationModule = typeof import("expo-location");

let locationModule: LocationModule | null | undefined;

async function getLocation(): Promise<LocationModule | null> {
  if (locationModule !== undefined) return locationModule;
  try {
    locationModule = await import("expo-location");
  } catch {
    locationModule = null;
  }
  return locationModule;
}

/** Request foreground permission. Safe everywhere; false on any failure. */
export async function ensureForegroundPermission(): Promise<boolean> {
  try {
    const Location = await getLocation();
    if (!Location) return false;
    const { status } = await Location.requestForegroundPermissionsAsync();
    return status === "granted";
  } catch {
    return false;
  }
}

/**
 * One foreground sample → cell id, or null on any failure. Bounded by a
 * timeout so quest submission can await this without ever feeling stuck.
 */
export async function sampleCellOnce(timeoutMs = 4000): Promise<string | null> {
  try {
    const Location = await getLocation();
    if (!Location) return null;
    const granted = await ensureForegroundPermission();
    if (!granted) return null;
    const fix = await Promise.race([
      Location.getCurrentPositionAsync({ accuracy: Location.Accuracy.Balanced }),
      new Promise<null>((resolve) => setTimeout(() => resolve(null), timeoutMs))
    ]);
    if (!fix) return null;
    return cellOf(fix.coords.latitude, fix.coords.longitude);
  } catch {
    return null;
  }
}

/** Whether location features are consented to in preferences. */
export function locationEnabled(state: AppState): boolean {
  return Boolean(state.profile.preferences.locationQuests);
}

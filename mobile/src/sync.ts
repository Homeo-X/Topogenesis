import { createClient, SupabaseClient, User } from "@supabase/supabase-js";
import { AppState } from "./types";
import { ensureStateShape } from "./engine";
import { SUPABASE_ANON_KEY, SUPABASE_URL, isSyncConfigured } from "./config";

/**
 * Sync layer (Supabase). Entirely additive and failure-tolerant: the app is
 * fully usable signed-out and local-only. Every remote call degrades to a safe
 * no-op / null on misconfiguration, network failure, or auth absence, so sync
 * can never break the app — same principle as the local storage hardening.
 *
 * Storage model: one row per user in `forge_state` (user_id, state jsonb,
 * updated_at), last-write-wins. RLS guarantees a user only ever touches their
 * own row.
 */

const TABLE = "forge_state";

let client: SupabaseClient | null = null;
function getClient(): SupabaseClient | null {
  if (!isSyncConfigured()) return null;
  if (!client) {
    // AsyncStorage is required lazily: it touches React Native native modules at
    // import time, so loading it here (only on a configured device, never in the
    // node test environment where isSyncConfigured() is false) keeps Jest green.
    // It is what makes the auth session survive an app restart on the phone.
    const AsyncStorage = require("@react-native-async-storage/async-storage").default;
    client = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
      auth: {
        storage: AsyncStorage,
        persistSession: true,
        autoRefreshToken: true,
        // No deep-link/URL session to parse in a native app.
        detectSessionInUrl: false
      }
    });
  }
  return client;
}

export function syncAvailable(): boolean {
  return getClient() !== null;
}

export async function currentUser(): Promise<User | null> {
  const c = getClient();
  if (!c) return null;
  try {
    const { data } = await c.auth.getUser();
    return data.user ?? null;
  } catch {
    return null;
  }
}

export async function currentAccessToken(): Promise<string | null> {
  const c = getClient();
  if (!c) return null;
  try {
    const { data } = await c.auth.getSession();
    return data.session?.access_token ?? null;
  } catch {
    return null;
  }
}

export interface AuthResult {
  ok: boolean;
  message: string;
}

export async function signUp(email: string, password: string): Promise<AuthResult> {
  const c = getClient();
  if (!c) return { ok: false, message: "Sync is not configured." };
  try {
    const { error } = await c.auth.signUp({ email, password });
    if (error) return { ok: false, message: error.message };
    return { ok: true, message: "Account created. Check your email if confirmation is required." };
  } catch (e) {
    return { ok: false, message: e instanceof Error ? e.message : "Sign-up failed." };
  }
}

export async function signIn(email: string, password: string): Promise<AuthResult> {
  const c = getClient();
  if (!c) return { ok: false, message: "Sync is not configured." };
  try {
    const { error } = await c.auth.signInWithPassword({ email, password });
    if (error) return { ok: false, message: error.message };
    return { ok: true, message: "Signed in. Your progress will sync." };
  } catch (e) {
    return { ok: false, message: e instanceof Error ? e.message : "Sign-in failed." };
  }
}

export async function signOut(): Promise<void> {
  const c = getClient();
  if (!c) return;
  try {
    await c.auth.signOut();
  } catch {
    // best-effort
  }
}

/** Load this user's remote state, normalized. Null if absent/unconfigured/failed. */
export async function loadRemoteState(): Promise<AppState | null> {
  const c = getClient();
  if (!c) return null;
  try {
    const user = await currentUser();
    if (!user) return null;
    const { data, error } = await c.from(TABLE).select("state").eq("user_id", user.id).maybeSingle();
    if (error || !data?.state) return null;
    return ensureStateShape(data.state as AppState);
  } catch {
    return null;
  }
}

/** Upsert this user's state. Fire-and-forget; never throws to the caller. */
export async function saveRemoteState(state: AppState): Promise<void> {
  const c = getClient();
  if (!c) return;
  try {
    const user = await currentUser();
    if (!user) return;
    await c.from(TABLE).upsert(
      { user_id: user.id, state, updated_at: new Date().toISOString() },
      { onConflict: "user_id" }
    );
  } catch {
    // Offline / transient failure: local cache remains source of truth.
  }
}

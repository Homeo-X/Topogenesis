/**
 * Supabase configuration for the backend (email/password auth + cloud sync of
 * progress + automatic LLM refinement).
 *
 * Provide credentials in EITHER of two ways — whichever is easier for your build:
 *
 *   1. Environment (recommended for `eas build`): set
 *        EXPO_PUBLIC_SUPABASE_URL       = https://YOUR_REF.supabase.co
 *        EXPO_PUBLIC_SUPABASE_ANON_KEY  = your-anon-key
 *      Expo inlines any EXPO_PUBLIC_* variable into the bundle at build time, so
 *      `expo start` (via a local .env file) and `eas build` (via eas.json `env`
 *      or EAS environment variables) both pick these up automatically.
 *
 *   2. Inline: paste the values into the two *_FALLBACK constants below. Safe to
 *      commit — the anon key is public by design.
 *
 * The anon key is public-safe: Row-Level Security on the forge_state table is
 * what protects user data, not key secrecy. NEVER put the service-role key here.
 * Leave everything blank to run fully local-only (all remote calls no-op).
 */

// Option 2: inline fallback. Used only when the matching env var is unset/empty.
const SUPABASE_URL_FALLBACK = "";
const SUPABASE_ANON_KEY_FALLBACK = "";

const pick = (fromEnv: string | undefined, fallback: string): string =>
  (fromEnv ?? "").trim() || fallback.trim();

export const SUPABASE_URL = pick(process.env.EXPO_PUBLIC_SUPABASE_URL, SUPABASE_URL_FALLBACK);
export const SUPABASE_ANON_KEY = pick(process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY, SUPABASE_ANON_KEY_FALLBACK);

/** True only when both values are present; gates all remote behavior (auth, sync, LLM). */
export const isSyncConfigured = (): boolean => SUPABASE_URL.length > 0 && SUPABASE_ANON_KEY.length > 0;

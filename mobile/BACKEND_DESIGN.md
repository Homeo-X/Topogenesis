# Forge System OS — Backend Design (v1, for approval)

**Status:** DESIGN ONLY. No implementation until approved. Personal-use scope.
**Decisions locked:** (1b) sync + LLM proxy · (2a) real auth · (3a) single JSONB row · (4a) last-write-wins.

This document is the shape I'll implement once you approve it. I cannot run Supabase
in my environment, so you will run and verify every piece. Where something is a risk
or a guess, it's flagged.

---

## 1. Stack

- **Supabase** (managed Postgres + Auth + Edge Functions + secrets).
- **Auth:** Supabase email/password (real accounts; supports multi-device).
- **Storage:** one `forge_state` row per user, `state JSONB` = the entire `AppState`.
- **Sync:** last-write-wins (newest client save overwrites the row).
- **LLM proxy:** one Edge Function holding the model API key in Supabase secrets,
  serving the three endpoints the client already calls.

---

## 2. Database schema

```sql
-- One row per user holding the whole AppState blob.
create table public.forge_state (
  user_id     uuid primary key references auth.users(id) on delete cascade,
  state       jsonb not null,
  updated_at  timestamptz not null default now()
);

-- Row-Level Security: a user can only ever touch their own row.
alter table public.forge_state enable row level security;

create policy "own row select" on public.forge_state
  for select using (auth.uid() = user_id);
create policy "own row insert" on public.forge_state
  for insert with check (auth.uid() = user_id);
create policy "own row update" on public.forge_state
  for update using (auth.uid() = user_id) with check (auth.uid() = user_id);
```

That's the entire schema. The blob maps 1:1 to the current `AppState`; `ensureStateShape`
already handles migration on load, so schema changes in the app don't require DB migrations.

**Risk flagged:** last-write-wins means if you save on device A (offline), then save on
device B, then device A reconnects and saves, A's older state overwrites B. For a single
active device this never bites. For true multi-device you'd want the timestamp guard
(decision 4b) — you chose 4a, which I'll honor, but I'm noting the trade so it's not a
surprise later. Adding the guard later is a ~5-line change.

---

## 3. Sync client (`src/sync.ts`, new)

A thin module the store calls. Interface:

```ts
loadRemoteState(): Promise<AppState | null>   // SELECT own row, ensureStateShape it
saveRemoteState(state: AppState): Promise<void> // UPSERT own row, set updated_at
signIn(email, password) / signUp / signOut / currentUser()
```

**Integration with the store (the one real client change to existing code):**
- On launch, if signed in: `loadRemoteState()` → if present, hydrate; else fall back to
  local. Local AsyncStorage stays as the offline cache and first-write source.
- On every committed state change (the existing save effect): also `saveRemoteState()`,
  fire-and-forget, failure-tolerant (offline must never break the app — same principle
  as the current storage hardening).
- A small auth screen (sign in / sign up) gates sync; **the app remains fully usable
  signed-out**, purely local — sync is additive, not required. This preserves the
  current offline-first behavior.

**Honest note:** this adds `@supabase/supabase-js` to the app and a SUPABASE_URL +
anon key (the anon key is public-safe by design; RLS is what protects data).

---

## 4. LLM proxy Edge Function (`supabase/functions/forge-llm/index.ts`)

One function, routed internally by a `task` field, holding the model key server-side.

**Contract mismatch to resolve (flagged):** the app today calls bare paths
`/quest/refine`, `/goal/arc`, `/outcome/interpret` with **no auth header**. Supabase
Edge Functions live at `https://{ref}.functions.supabase.co/forge-llm` and require an
`Authorization: Bearer {access_token}` header. So the client needs a small change:
- point the `backend` provider endpoint at the function URL,
- send the Supabase access token,
- send a `task: "quest" | "arc" | "outcome"` discriminator instead of three paths.

I'll make this change behind the existing provider abstraction so the `ollama` route and
all the validation/fallback logic are untouched. The function:

1. Verifies the JWT (Supabase does this automatically when `verify_jwt` is on).
2. Reads `task` + payload.
3. Calls the model API (key from `Deno.env.get("LLM_API_KEY")`).
4. Returns JSON in the **exact shape the client validators already expect** — so
   `validatePatch`, `mergeArc`, `validateOutcomeSuggestion` need no changes.
5. On any error, returns a 4xx/5xx → client's existing silent fallback to the
   deterministic engine kicks in. The safety/guardrail layer is unchanged and still
   client-side, which is correct: the model never decides risk/reward/outcome.

**Key safety:** `LLM_API_KEY` lives in `supabase secrets set` — never in the app binary,
never in the repo. This is the whole reason the LLM features move server-side.

---

## 5. What YOU run to stand it up (I can't)

1. Create a Supabase project; note the project ref, URL, anon key.
2. Run the schema SQL (section 2) in the SQL editor.
3. Enable Email auth in Auth settings.
4. `supabase functions deploy forge-llm` and `supabase secrets set LLM_API_KEY=...`.
5. Put SUPABASE_URL + anon key into the app config I'll provide.
6. Verify: sign up, confirm a `forge_state` row appears, kill/reinstall the app, sign
   back in, confirm state restored. Toggle LLM provider to `backend`, generate a quest,
   confirm refinement (or graceful fallback).

I'll provide exact commands and the config file; the verification is yours because only
a real device + real project exercises it.

---

## 6. What I can vs. cannot verify

- **Can (in my env):** the client-side TypeScript compiles; the sync module's shape and
  fallback logic; that existing tests still pass; that the Edge Function is valid
  TypeScript/Deno syntactically.
- **Cannot (yours):** that auth works, that RLS actually isolates, that the function
  reaches the model, that sync round-trips, that deploy succeeds. All of section 5.

---

## 7. Build order (once approved)

1. `src/sync.ts` + auth screen + store integration (client; I can type-check).
2. Client `backend`-provider change (auth header + task discriminator; type-checked,
   existing LLM tests must still pass).
3. `supabase/functions/forge-llm/index.ts` (Deno; syntax-only verification from me).
4. Schema SQL file + a SETUP.md with your exact commands.
5. Delivery with a precise verified/unverified split.

---

## Open questions before I build

- **A.** Model provider for the proxy — OpenAI-compatible? Anthropic? Something else?
  (Determines the fetch call inside the function. The client shape is provider-agnostic.)
- **B.** OK that the app stays **fully usable signed-out** (local-only), with sign-in
  purely enabling sync? (Recommended; preserves offline-first.)
- **C.** OK with the small client change in section 4 (auth header + `task` field)? It's
  required — Supabase functions can't match the current bare-path, no-auth contract.

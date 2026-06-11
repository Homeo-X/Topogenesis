# Forge Backend — Setup (what YOU run)

I cannot run Supabase, so this is the part you execute and verify. ~15 minutes.

## 1. Create the project
1. Create a Supabase project at supabase.com. Note **Project URL** and **anon key**
   (Settings → API).

## 2. Schema + auth
2. SQL editor → paste and run `supabase/schema.sql`.
3. Auth → Providers → enable **Email**. (For personal use you may also turn off
   email confirmation in Auth settings so sign-up is instant.)

## 3. Deploy the LLM function

### 3a. Install the Supabase CLI (Windows)
`npm install -g supabase` is **NOT supported**. Install it as a project
dev-dependency and call it through `npx` (examples below use `npx.cmd` for
PowerShell):
```
npm.cmd install supabase --save-dev
```
(Scoop alternative: `scoop bucket add supabase https://github.com/supabase/scoop-bucket.git`
then `scoop install supabase`, after which you can drop the `npx` prefix.)

### 3b. Log in and deploy — RUN FROM THE PROJECT ROOT
The deploy reads `supabase/functions/forge-llm/index.ts` **relative to your current
directory**. If you run it from your home folder you get
`unexpected deploy status 400: Entrypoint path does not exist`. `cd` into the
folder that contains `supabase/` first:
```
cd path\to\forge-system-mobile-authoritative
npx.cmd supabase login
npx.cmd supabase functions deploy forge-llm --project-ref cswwxwckwaygvfkshupu
```
Real-run notes:
- **Pass `--project-ref cswwxwckwaygvfkshupu` explicitly.** This package has no
  `supabase/config.toml` (it was never `supabase init`-ed), so the CLI can't infer
  the project from a link — and a `link` you ran from another folder won't apply
  here.
- `WARNING: Docker is not running` is **harmless** — the deploy bundles
  server-side. Only if a real Docker/bundling error appears, add `--use-api`.
- A trailing `posthog ... context deadline exceeded` line is just the CLI's own
  telemetry timing out. **Ignore it.** Success is the line
  `Deployed Functions on project cswwxwckwaygvfkshupu: forge-llm`.

### 3c. Set provider keys (server-side secrets)
Get free keys from https://console.groq.com/keys and
https://aistudio.google.com/app/apikey. These are **secret** — they live ONLY in
Supabase secrets, never in the app, the APK, `eas.json`, or any committed file.
Unlike the publishable anon key, these must not be shared or checked in.

`secrets set` works from any directory (it only needs the project ref, not local
files — which is why it succeeds even when a deploy from the wrong folder fails).
Set at least one; both can go in a single command:
```
npx.cmd supabase secrets set GROQ_API_KEY=gsk_xxx GEMINI_API_KEY=AIxxx
```
Success prints `Finished supabase secrets set.` The chain tries Groq first, then
Gemini; a provider with no key is skipped. With no keys set, refinement 502s and
the app falls back to its deterministic engine. Secrets are read at runtime — no
redeploy needed after changing them.

### 3d. Confirm it actually fires
There is no in-app "LLM on" indicator (refinement is auto/hidden), so the **only**
place to confirm it works is the dashboard: **Edge Functions → forge-llm → Logs**.
Sign in, generate a quest, and look for a `200`. `401` = you weren't signed in;
`502` = both providers failed (the app silently used the deterministic engine —
nothing breaks, but re-check a key is valid).

## 4. Point the app at your project
Give the app your Project URL + anon key. Pick **one** method:

**A. Environment (recommended for builds).** Copy `.env.example` → `.env` and fill in:
```
EXPO_PUBLIC_SUPABASE_URL=https://YOUR_REF.supabase.co
EXPO_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
```
Expo inlines `EXPO_PUBLIC_*` at build time. `expo start` reads `.env` locally, so
this is enough for `npm run start` / Expo Go.

**Gotcha for `eas build` (cloud):** `.env` is gitignored, and EAS cloud builds
**skip gitignored files** — so a cloud APK built from `.env` alone ships with the
backend silently OFF. The creds must live somewhere that gets uploaded. This repo
already handles it by baking the two public-safe values into **`eas.json`** under
`build.preview.env` and `build.production.env`:
```
"env": {
  "EXPO_PUBLIC_SUPABASE_URL": "https://YOUR_REF.supabase.co",
  "EXPO_PUBLIC_SUPABASE_ANON_KEY": "your-anon-key"
}
```
`eas.json` is committed (not gitignored), so the build sees these. The anon /
publishable key is public by design, so committing it is safe. If you'd rather not
commit it, remove those `env` blocks and register the vars on EAS instead:
```
eas env:create --environment production --name EXPO_PUBLIC_SUPABASE_URL --value https://YOUR_REF.supabase.co
eas env:create --environment production --name EXPO_PUBLIC_SUPABASE_ANON_KEY --value your-anon-key
```

**B. Inline.** Paste the same two values into the `*_FALLBACK` constants in
`src/config.ts`. Safe to commit (anon key is public).

That's all the app needs. **There is no LLM settings screen** — once the URL/key
are present and you sign in, cloud sync and backend LLM refinement turn on
automatically. The LLM function URL is derived for you
(`https://YOUR_REF.supabase.co/functions/v1/forge-llm`); refinement is silent and
falls back to the deterministic engine if a provider is down. Sign in/out lives in
**System tab → Account & Cloud Sync**.

> Build the APK with: `eas build -p android --profile preview` (produces an
> installable `.apk`). Credentials are baked in at build time, so rebuild after
> changing them.

## 5. Verify (the real test — only you can do this)
- [ ] In **System → Account & Cloud Sync**, create an account; confirm a row
      appears in `forge_state` (Table editor).
- [ ] Add a goal / complete a quest; confirm the row's `state` updates.
- [ ] Reinstall the app (or use a second device); sign in; confirm progress restored.
- [ ] In Table editor, confirm you can only see your own row (RLS).
- [ ] While signed in, generate a quest → confirm refinement (if a provider key is
      set), OR confirm graceful, invisible fallback if a provider is down.
- [ ] Sign out → app still works fully local-only.

## Notes / honest caveats
- **Last-write-wins:** a stale device that saves later overwrites newer data. Fine
  for one active device; for true multi-device, ask me to add the timestamp guard
  (~5 lines).
- **Free tiers rotate model names and rate-limit.** If Groq deprecates the model,
  update `GROQ_MODEL` in `supabase/functions/forge-llm/index.ts` and redeploy.
- **The anon key is meant to be public**; RLS is the protection. Do not put the
  *service-role* key in the app — it bypasses RLS.
- I verified the client TypeScript compiles and tests pass. I could NOT verify
  auth, RLS isolation, deploy, or that the providers respond — those are steps 1–5.

# Forge System OS Mobile - Authoritative Build

This is the clean source package for the Forge System OS mobile app.

It is production-ready as a local-first Expo/React Native Android app: the app runs without a backend, persists locally, generates ranked quests, supports notifications, and passes the included TypeScript and Jest verification suite.

Backend sync and hosted LLM refinement are included as optional Supabase scaffolding. They are disabled until `src/config.ts` is filled with your Supabase project URL and anon key.

## Included

- Expo SDK 54 / React Native 0.81 / React 19.
- Modular app shell with screens, components, reducer/store, and error boundary.
- Ranked quest generation with top offers, risk gating, difficulty bands, and explicit tier-3 confirmation.
- Event director for emergency, hidden, and boss quest surfacing.
- Nuanced 11-outcome evidence flow with LLM suggest-with-confirmation.
- Adaptive calibration, assumptions, streak shields, skill tree, world progression, and bounded System voice.
- Body-composition input that only derives capability/behavior signals, never aesthetic or weight-target pressure.
- State-aware local notifications with Android `POST_NOTIFICATIONS` permission and `expo-notifications` plugin.
- Local AsyncStorage persistence.
- Optional Supabase sync and Supabase Edge Function LLM proxy.
- Jest coverage for engine, director, body signals, skills, reducer, framing, voice, LLM validation, and composition.

## Verify

Install dependencies:

```bash
npm install
```

Run the full local verification suite:

```bash
npm run verify
```

Equivalent commands:

```bash
npm run typecheck
npm test -- --runInBand
```

Expected result:

- TypeScript: 0 diagnostics.
- Jest: 9 suites passing, 73 tests passing.

## Run

```bash
npm run start
```

Then scan with Expo Go or press `a` in the Expo terminal for Android.

On Windows PowerShell, use the `.cmd` executable if script execution blocks npm:

```powershell
npm.cmd run start
```

## Build APK

For a downloadable Android APK:

```bash
eas build -p android --profile preview
```

On Windows PowerShell:

```powershell
eas.cmd build -p android --profile preview
```

For Play Store AAB:

```bash
eas build -p android --profile production
```

## Optional Supabase Backend

The app is fully usable without Supabase. To enable email/password account sync and hosted LLM refinement:

1. Follow `supabase/SETUP.md`.
2. Run `supabase/schema.sql` in your Supabase SQL editor.
3. Deploy `supabase/functions/forge-llm`.
4. Provide your Project URL + anon key, either via env (`.env` / EAS env vars,
   names in `.env.example`) or by pasting into the `*_FALLBACK` constants in `src/config.ts`.
5. Build the APK (`eas build -p android --profile preview`), then in the app open
   **System → Account & Cloud Sync** to create an account / sign in.

There is no LLM settings screen: once credentials are present and you sign in,
cloud sync and backend LLM refinement activate automatically. The Edge Function
URL is derived from your Project URL, and refinement silently falls back to the
deterministic engine if the backend is unreachable.

The anon key is public by design. Row-Level Security protects user data. Never put a Supabase service-role key in the app.

## Production Caveats

- Full Python TSCS and ForgeGoal multi-agent orchestration are deliberately not bundled into this mobile build.
- Supabase auth, RLS isolation, and Edge Function provider keys must be verified in your Supabase project.
- EAS build requires an Expo account and remote build credentials.
- The included notification behavior is local scheduled notification behavior, not alarm/full-screen intent behavior.

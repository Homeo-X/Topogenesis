-- Forge System OS — backend schema (run once in Supabase SQL editor).
-- One row per user holding the entire AppState blob. Last-write-wins.

create table if not exists public.forge_state (
  user_id     uuid primary key references auth.users(id) on delete cascade,
  state       jsonb not null,
  updated_at  timestamptz not null default now()
);

alter table public.forge_state enable row level security;

-- A user may only read/write their own row. RLS — not key secrecy — is what
-- protects the data, which is why the anon key is safe to ship in the app.
drop policy if exists "own row select" on public.forge_state;
drop policy if exists "own row insert" on public.forge_state;
drop policy if exists "own row update" on public.forge_state;

create policy "own row select" on public.forge_state
  for select using (auth.uid() = user_id);
create policy "own row insert" on public.forge_state
  for insert with check (auth.uid() = user_id);
create policy "own row update" on public.forge_state
  for update using (auth.uid() = user_id) with check (auth.uid() = user_id);

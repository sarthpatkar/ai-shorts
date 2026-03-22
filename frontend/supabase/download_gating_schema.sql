-- AI Shorts: download gating, entitlement, and payment schema for Supabase
-- Run this entire script in Supabase SQL Editor.

create extension if not exists pgcrypto;

create table if not exists public.user_profiles (
  user_id uuid primary key references auth.users(id) on delete cascade,
  plan_tier text not null default 'free' check (plan_tier in ('free', 'premium')),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.job_unlocks (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  job_id text not null,
  unlock_type text not null check (unlock_type in ('free_480_first_job', 'pay_per_job')),
  unlocked_qualities text[] not null default '{}',
  source text not null default 'system',
  stripe_checkout_session_id text,
  stripe_payment_intent_id text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint job_unlocks_user_job_unique unique (user_id, job_id)
);

create unique index if not exists job_unlocks_one_free_480_per_user_idx
  on public.job_unlocks (user_id)
  where unlock_type = 'free_480_first_job';

create index if not exists job_unlocks_user_idx on public.job_unlocks (user_id);
create index if not exists job_unlocks_user_job_idx on public.job_unlocks (user_id, job_id);

create table if not exists public.stripe_customers (
  user_id uuid primary key references auth.users(id) on delete cascade,
  stripe_customer_id text not null unique,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.subscriptions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  stripe_customer_id text not null,
  stripe_subscription_id text not null unique,
  stripe_price_id text not null,
  status text not null,
  current_period_start timestamptz,
  current_period_end timestamptz,
  cancel_at_period_end boolean not null default false,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists subscriptions_user_idx on public.subscriptions (user_id);
create index if not exists subscriptions_user_status_idx on public.subscriptions (user_id, status);
create index if not exists subscriptions_period_end_idx on public.subscriptions (current_period_end);

create table if not exists public.stripe_webhook_events (
  event_id text primary key,
  event_type text not null,
  payload jsonb not null default '{}'::jsonb,
  received_at timestamptz not null default now()
);

create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists user_profiles_set_updated_at on public.user_profiles;
create trigger user_profiles_set_updated_at
before update on public.user_profiles
for each row
execute function public.set_updated_at();

drop trigger if exists job_unlocks_set_updated_at on public.job_unlocks;
create trigger job_unlocks_set_updated_at
before update on public.job_unlocks
for each row
execute function public.set_updated_at();

drop trigger if exists stripe_customers_set_updated_at on public.stripe_customers;
create trigger stripe_customers_set_updated_at
before update on public.stripe_customers
for each row
execute function public.set_updated_at();

drop trigger if exists subscriptions_set_updated_at on public.subscriptions;
create trigger subscriptions_set_updated_at
before update on public.subscriptions
for each row
execute function public.set_updated_at();

create or replace function public.handle_new_user_profile()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  insert into public.user_profiles (user_id, plan_tier)
  values (new.id, 'free')
  on conflict (user_id) do nothing;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created_profile on auth.users;
create trigger on_auth_user_created_profile
after insert on auth.users
for each row
execute function public.handle_new_user_profile();

insert into public.user_profiles (user_id, plan_tier)
select u.id, 'free'
from auth.users u
left join public.user_profiles p on p.user_id = u.id
where p.user_id is null;

alter table public.user_profiles enable row level security;
alter table public.job_unlocks enable row level security;
alter table public.stripe_customers enable row level security;
alter table public.subscriptions enable row level security;
alter table public.stripe_webhook_events enable row level security;

drop policy if exists user_profiles_select_own on public.user_profiles;
create policy user_profiles_select_own
on public.user_profiles
for select
to authenticated
using (auth.uid() = user_id);

drop policy if exists user_profiles_update_own on public.user_profiles;
create policy user_profiles_update_own
on public.user_profiles
for update
to authenticated
using (auth.uid() = user_id)
with check (
  auth.uid() = user_id
  and plan_tier = (
    select up.plan_tier
    from public.user_profiles up
    where up.user_id = auth.uid()
    limit 1
  )
);

drop policy if exists user_profiles_insert_own on public.user_profiles;
create policy user_profiles_insert_own
on public.user_profiles
for insert
to authenticated
with check (auth.uid() = user_id);

drop policy if exists job_unlocks_select_own on public.job_unlocks;
create policy job_unlocks_select_own
on public.job_unlocks
for select
to authenticated
using (auth.uid() = user_id);

drop policy if exists stripe_customers_select_own on public.stripe_customers;
create policy stripe_customers_select_own
on public.stripe_customers
for select
to authenticated
using (auth.uid() = user_id);

drop policy if exists subscriptions_select_own on public.subscriptions;
create policy subscriptions_select_own
on public.subscriptions
for select
to authenticated
using (auth.uid() = user_id);

create or replace function public.get_download_entitlement(
  p_user_id uuid,
  p_job_id text,
  p_quality text
)
returns table (
  allowed boolean,
  reason text,
  should_grant_first_free_480 boolean
)
language plpgsql
security definer
set search_path = public
as $$
declare
  v_quality text := lower(regexp_replace(coalesce(p_quality, ''), '\\s+', '', 'g'));
  v_has_active_subscription boolean := false;
  v_has_job_unlock boolean := false;
  v_has_any_first_free_480 boolean := false;
  v_has_this_job_first_free_480 boolean := false;
begin
  if p_user_id is null then
    return query select false, 'needs_auth', false;
    return;
  end if;

  if v_quality in ('240', '240p') then
    v_quality := '240p';
  elsif v_quality in ('360', '360p') then
    v_quality := '360p';
  elsif v_quality in ('480', '480p') then
    v_quality := '480p';
  elsif v_quality in ('720', '720p') then
    v_quality := '720p';
  elsif v_quality in ('1080', '1080p') then
    v_quality := '1080p';
  else
    return query select false, 'unsupported_quality', false;
    return;
  end if;

  if v_quality in ('240p', '360p') then
    return query select true, 'free_quality', false;
    return;
  end if;

  select exists (
    select 1
    from public.subscriptions s
    where s.user_id = p_user_id
      and s.status in ('active', 'trialing')
      and (s.current_period_end is null or s.current_period_end > now())
  )
  into v_has_active_subscription;

  if v_has_active_subscription then
    return query select true, 'subscription_active', false;
    return;
  end if;

  select exists (
    select 1
    from public.job_unlocks ju
    where ju.user_id = p_user_id
      and ju.job_id = p_job_id
      and v_quality = any(ju.unlocked_qualities)
  )
  into v_has_job_unlock;

  if v_has_job_unlock then
    return query select true, 'job_unlocked', false;
    return;
  end if;

  if v_quality = '480p' then
    select exists (
      select 1
      from public.job_unlocks ju
      where ju.user_id = p_user_id
        and ju.unlock_type = 'free_480_first_job'
    )
    into v_has_any_first_free_480;

    select exists (
      select 1
      from public.job_unlocks ju
      where ju.user_id = p_user_id
        and ju.job_id = p_job_id
        and ju.unlock_type = 'free_480_first_job'
    )
    into v_has_this_job_first_free_480;

    if v_has_this_job_first_free_480 then
      return query select true, 'first_free_480_granted', false;
      return;
    end if;

    if not v_has_any_first_free_480 then
      return query select true, 'first_free_480_eligible', true;
      return;
    end if;

    return query select false, 'free_480_already_used', false;
    return;
  end if;

  return query select false, 'premium_unlock_required', false;
end;
$$;

create or replace function public.grant_first_free_480_unlock(
  p_user_id uuid,
  p_job_id text
)
returns boolean
language plpgsql
security definer
set search_path = public
as $$
declare
  v_job_id text := trim(coalesce(p_job_id, ''));
  v_inserted boolean := false;
begin
  if p_user_id is null or v_job_id = '' then
    return false;
  end if;

  insert into public.job_unlocks (
    user_id,
    job_id,
    unlock_type,
    unlocked_qualities,
    source
  )
  values (
    p_user_id,
    v_job_id,
    'free_480_first_job',
    array['480p'],
    'first_free_480'
  )
  on conflict do nothing;

  select exists (
    select 1
    from public.job_unlocks ju
    where ju.user_id = p_user_id
      and ju.job_id = v_job_id
      and ju.unlock_type = 'free_480_first_job'
  )
  into v_inserted;

  return v_inserted;
end;
$$;

revoke all on function public.get_download_entitlement(uuid, text, text) from public;
revoke all on function public.grant_first_free_480_unlock(uuid, text) from public;

grant execute on function public.get_download_entitlement(uuid, text, text) to service_role;
grant execute on function public.grant_first_free_480_unlock(uuid, text) to service_role;

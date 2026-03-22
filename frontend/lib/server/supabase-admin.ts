import { serverEnv } from "@/lib/server/env";

export type AuthUser = {
  id: string;
  email: string | null;
};

export type DownloadEntitlement = {
  allowed: boolean;
  reason: string;
  should_grant_first_free_480: boolean;
};

type RestMethod = "GET" | "POST" | "PATCH";

function buildRestUrl(path: string): string {
  if (!serverEnv.supabaseUrl) {
    throw new Error("Missing NEXT_PUBLIC_SUPABASE_URL");
  }
  const base = serverEnv.supabaseUrl.endsWith("/")
    ? serverEnv.supabaseUrl
    : `${serverEnv.supabaseUrl}/`;
  const normalized = path.replace(/^\//, "");
  return new URL(normalized, base).toString();
}

async function parseJsonSafe(response: Response): Promise<unknown> {
  const text = await response.text();
  if (!text) {
    return null;
  }
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return null;
  }
}

function errorMessage(payload: unknown, fallback: string): string {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    return fallback;
  }
  const record = payload as Record<string, unknown>;
  const message = typeof record.message === "string" ? record.message : null;
  const error = typeof record.error === "string" ? record.error : null;
  return (message || error || fallback).trim();
}

async function supabaseRest(
  method: RestMethod,
  path: string,
  options?: {
    body?: unknown;
    headers?: Record<string, string>;
    useServiceRole?: boolean;
  }
): Promise<{ response: Response; payload: unknown }> {
  const useServiceRole = options?.useServiceRole ?? true;
  const apikey = useServiceRole
    ? serverEnv.supabaseServiceRoleKey
    : serverEnv.supabaseAnonKey;
  if (!apikey) {
    throw new Error(
      useServiceRole
        ? "Missing SUPABASE_SERVICE_ROLE_KEY"
        : "Missing NEXT_PUBLIC_SUPABASE_ANON_KEY"
    );
  }
  const headers: Record<string, string> = {
    apikey,
    Authorization: `Bearer ${apikey}`,
    ...options?.headers,
  };
  if (options?.body !== undefined) {
    headers["Content-Type"] = "application/json";
  }

  const response = await fetch(buildRestUrl(path), {
    method,
    headers,
    cache: "no-store",
    body: options?.body === undefined ? undefined : JSON.stringify(options.body),
  });

  const payload = await parseJsonSafe(response);
  return { response, payload };
}

export async function verifyAccessToken(
  accessToken: string
): Promise<AuthUser | null> {
  const token = accessToken.trim();
  if (!token) {
    return null;
  }

  if (!serverEnv.supabaseAnonKey) {
    throw new Error("Missing NEXT_PUBLIC_SUPABASE_ANON_KEY");
  }
  const response = await fetch(buildRestUrl("/auth/v1/user"), {
    method: "GET",
    cache: "no-store",
    headers: {
      apikey: serverEnv.supabaseAnonKey,
      Authorization: `Bearer ${token}`,
    },
  });
  if (!response.ok) {
    return null;
  }

  const payload = await parseJsonSafe(response);
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    return null;
  }
  const record = payload as Record<string, unknown>;
  const id = typeof record.id === "string" ? record.id.trim() : "";
  if (!id) {
    return null;
  }
  const email = typeof record.email === "string" ? record.email : null;
  return { id, email };
}

export async function getDownloadEntitlement(
  userId: string,
  jobId: string,
  quality: string
): Promise<DownloadEntitlement> {
  const { response, payload } = await supabaseRest(
    "POST",
    "/rest/v1/rpc/get_download_entitlement",
    {
      body: {
        p_user_id: userId,
        p_job_id: jobId,
        p_quality: quality,
      },
    }
  );

  if (!response.ok) {
    throw new Error(
      errorMessage(payload, "Failed to evaluate download entitlement.")
    );
  }

  if (!Array.isArray(payload) || payload.length === 0) {
    return {
      allowed: false,
      reason: "entitlement_unavailable",
      should_grant_first_free_480: false,
    };
  }

  const row = payload[0] as Record<string, unknown>;
  return {
    allowed: Boolean(row.allowed),
    reason:
      typeof row.reason === "string" && row.reason.trim()
        ? row.reason.trim()
        : "unknown",
    should_grant_first_free_480: Boolean(row.should_grant_first_free_480),
  };
}

export async function grantFirstFree480Unlock(
  userId: string,
  jobId: string
): Promise<boolean> {
  const { response, payload } = await supabaseRest(
    "POST",
    "/rest/v1/rpc/grant_first_free_480_unlock",
    {
      body: {
        p_user_id: userId,
        p_job_id: jobId,
      },
    }
  );
  if (!response.ok) {
    throw new Error(
      errorMessage(payload, "Failed to grant one-time free 480 unlock.")
    );
  }
  if (typeof payload === "boolean") {
    return payload;
  }
  if (Array.isArray(payload) && typeof payload[0] === "boolean") {
    return payload[0];
  }
  return false;
}

export async function insertJobUnlock(params: {
  userId: string;
  jobId: string;
  unlockType: "pay_per_job" | "free_480_first_job";
  unlockedQualities: string[];
  source: string;
  stripeCheckoutSessionId?: string;
  stripePaymentIntentId?: string;
}): Promise<void> {
  const { response, payload } = await supabaseRest("POST", "/rest/v1/job_unlocks", {
    headers: {
      Prefer: "resolution=merge-duplicates,return=minimal",
    },
    body: {
      user_id: params.userId,
      job_id: params.jobId,
      unlock_type: params.unlockType,
      unlocked_qualities: params.unlockedQualities,
      source: params.source,
      stripe_checkout_session_id: params.stripeCheckoutSessionId ?? null,
      stripe_payment_intent_id: params.stripePaymentIntentId ?? null,
    },
  });

  if (!response.ok) {
    throw new Error(errorMessage(payload, "Failed to persist job unlock."));
  }
}

export async function upsertStripeCustomer(
  userId: string,
  stripeCustomerId: string
): Promise<void> {
  const { response, payload } = await supabaseRest(
    "POST",
    "/rest/v1/stripe_customers",
    {
      headers: {
        Prefer: "resolution=merge-duplicates,return=minimal",
      },
      body: {
        user_id: userId,
        stripe_customer_id: stripeCustomerId,
      },
    }
  );
  if (!response.ok) {
    throw new Error(errorMessage(payload, "Failed to upsert stripe customer."));
  }
}

export async function getStripeCustomerIdForUser(
  userId: string
): Promise<string | null> {
  const { response, payload } = await supabaseRest(
    "GET",
    `/rest/v1/stripe_customers?select=stripe_customer_id&user_id=eq.${encodeURIComponent(
      userId
    )}&limit=1`
  );
  if (!response.ok) {
    throw new Error(errorMessage(payload, "Failed to load stripe customer."));
  }
  if (!Array.isArray(payload) || payload.length === 0) {
    return null;
  }
  const row = payload[0] as Record<string, unknown>;
  return typeof row.stripe_customer_id === "string"
    ? row.stripe_customer_id
    : null;
}

export async function getUserIdByStripeCustomerId(
  stripeCustomerId: string
): Promise<string | null> {
  const { response, payload } = await supabaseRest(
    "GET",
    `/rest/v1/stripe_customers?select=user_id&stripe_customer_id=eq.${encodeURIComponent(
      stripeCustomerId
    )}&limit=1`
  );
  if (!response.ok) {
    throw new Error(errorMessage(payload, "Failed to map stripe customer."));
  }
  if (!Array.isArray(payload) || payload.length === 0) {
    return null;
  }
  const row = payload[0] as Record<string, unknown>;
  return typeof row.user_id === "string" ? row.user_id : null;
}

export async function upsertSubscriptionFromStripe(params: {
  userId: string;
  stripeCustomerId: string;
  stripeSubscriptionId: string;
  stripePriceId: string;
  status: string;
  currentPeriodStart?: number | null;
  currentPeriodEnd?: number | null;
  cancelAtPeriodEnd?: boolean;
  metadata?: Record<string, string>;
}): Promise<void> {
  const toIso = (value?: number | null): string | null => {
    if (!value || Number.isNaN(value) || value <= 0) {
      return null;
    }
    return new Date(value * 1000).toISOString();
  };

  const { response, payload } = await supabaseRest(
    "POST",
    "/rest/v1/subscriptions",
    {
      headers: {
        Prefer: "resolution=merge-duplicates,return=minimal",
      },
      body: {
        user_id: params.userId,
        stripe_customer_id: params.stripeCustomerId,
        stripe_subscription_id: params.stripeSubscriptionId,
        stripe_price_id: params.stripePriceId,
        status: params.status,
        current_period_start: toIso(params.currentPeriodStart),
        current_period_end: toIso(params.currentPeriodEnd),
        cancel_at_period_end: Boolean(params.cancelAtPeriodEnd),
        metadata: params.metadata ?? {},
      },
    }
  );

  if (!response.ok) {
    throw new Error(errorMessage(payload, "Failed to upsert subscription."));
  }
}

export async function setUserPlanTier(
  userId: string,
  tier: "free" | "premium"
): Promise<void> {
  const { response, payload } = await supabaseRest(
    "POST",
    "/rest/v1/user_profiles",
    {
      headers: {
        Prefer: "resolution=merge-duplicates,return=minimal",
      },
      body: {
        user_id: userId,
        plan_tier: tier,
      },
    }
  );

  if (!response.ok) {
    throw new Error(errorMessage(payload, "Failed to update user plan tier."));
  }
}

export async function isNewWebhookEvent(
  eventId: string,
  eventType: string,
  payload: unknown
): Promise<boolean> {
  if (!serverEnv.supabaseServiceRoleKey) {
    throw new Error("Missing SUPABASE_SERVICE_ROLE_KEY");
  }
  const response = await fetch(buildRestUrl("/rest/v1/stripe_webhook_events"), {
    method: "POST",
    cache: "no-store",
    headers: {
      apikey: serverEnv.supabaseServiceRoleKey,
      Authorization: `Bearer ${serverEnv.supabaseServiceRoleKey}`,
      "Content-Type": "application/json",
      Prefer: "resolution=ignore-duplicates,return=representation",
    },
    body: JSON.stringify({
      event_id: eventId,
      event_type: eventType,
      payload,
    }),
  });

  if (response.status === 201) {
    return true;
  }
  if (response.status === 200) {
    const body = await parseJsonSafe(response);
    return Array.isArray(body) && body.length > 0;
  }
  if (response.status === 409) {
    return false;
  }
  if (response.ok) {
    return true;
  }

  const data = await parseJsonSafe(response);
  throw new Error(errorMessage(data, "Failed to persist webhook event."));
}

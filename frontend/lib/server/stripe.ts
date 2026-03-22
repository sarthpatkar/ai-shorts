import crypto from "node:crypto";
import { serverEnv } from "@/lib/server/env";

export type StripeCheckoutMode = "subscription" | "payment";

export type StripeCheckoutParams = {
  mode: StripeCheckoutMode;
  customerId: string;
  userId: string;
  successUrl: string;
  cancelUrl: string;
  monthlyPriceId?: string;
  jobUnlockPriceId?: string;
  jobId?: string;
};

export type StripeEvent = {
  id: string;
  type: string;
  data: {
    object: Record<string, unknown>;
  };
};

function requireStripeKey(): string {
  const key = serverEnv.stripeSecretKey;
  if (!key) {
    throw new Error("Missing STRIPE_SECRET_KEY");
  }
  return key;
}

async function stripeFormRequest(
  path: string,
  form: URLSearchParams
): Promise<Record<string, unknown>> {
  const key = requireStripeKey();
  const response = await fetch(`https://api.stripe.com/v1/${path.replace(/^\//, "")}`, {
    method: "POST",
    cache: "no-store",
    headers: {
      Authorization: `Bearer ${key}`,
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: form.toString(),
  });

  const payload = (await response.json()) as Record<string, unknown>;
  if (!response.ok) {
    const message =
      (payload?.error as Record<string, unknown> | undefined)?.message ??
      payload?.message ??
      "Stripe request failed.";
    throw new Error(String(message));
  }
  return payload;
}

export async function createStripeCustomer(params: {
  userId: string;
  email?: string | null;
}): Promise<string> {
  const form = new URLSearchParams();
  if (params.email) {
    form.set("email", params.email);
  }
  form.set("metadata[user_id]", params.userId);
  const payload = await stripeFormRequest("customers", form);
  const id = typeof payload.id === "string" ? payload.id : "";
  if (!id) {
    throw new Error("Stripe customer id missing in response.");
  }
  return id;
}

export async function createCheckoutSession(
  params: StripeCheckoutParams
): Promise<{ id: string; url: string }> {
  const form = new URLSearchParams();
  form.set("mode", params.mode);
  form.set("customer", params.customerId);
  form.set("client_reference_id", params.userId);
  form.set("success_url", params.successUrl);
  form.set("cancel_url", params.cancelUrl);
  form.set("allow_promotion_codes", "true");

  if (params.mode === "subscription") {
    const priceId = params.monthlyPriceId || serverEnv.stripeMonthlyPriceId;
    if (!priceId) {
      throw new Error("Missing STRIPE_MONTHLY_PRICE_ID");
    }
    form.set("line_items[0][price]", priceId);
    form.set("line_items[0][quantity]", "1");
    form.set("metadata[purchase_type]", "monthly_subscription");
    form.set("metadata[user_id]", params.userId);
    form.set("subscription_data[metadata][purchase_type]", "monthly_subscription");
    form.set("subscription_data[metadata][user_id]", params.userId);
  } else {
    const priceId = params.jobUnlockPriceId || serverEnv.stripeJobUnlockPriceId;
    if (!priceId) {
      throw new Error("Missing STRIPE_JOB_UNLOCK_PRICE_ID");
    }
    if (!params.jobId) {
      throw new Error("jobId is required for pay-per-video checkout.");
    }
    form.set("line_items[0][price]", priceId);
    form.set("line_items[0][quantity]", "1");
    form.set("metadata[purchase_type]", "job_unlock");
    form.set("metadata[user_id]", params.userId);
    form.set("metadata[job_id]", params.jobId);
    form.set("payment_intent_data[metadata][purchase_type]", "job_unlock");
    form.set("payment_intent_data[metadata][user_id]", params.userId);
    form.set("payment_intent_data[metadata][job_id]", params.jobId);
  }

  const payload = await stripeFormRequest("checkout/sessions", form);
  const id = typeof payload.id === "string" ? payload.id : "";
  const url = typeof payload.url === "string" ? payload.url : "";
  if (!id || !url) {
    throw new Error("Stripe checkout session response missing id/url.");
  }
  return { id, url };
}

function secureEqualHex(a: string, b: string): boolean {
  const left = Buffer.from(a, "utf8");
  const right = Buffer.from(b, "utf8");
  if (left.length !== right.length) {
    return false;
  }
  return crypto.timingSafeEqual(left, right);
}

export function verifyWebhookSignature(rawBody: string, signatureHeader: string): void {
  const secret = serverEnv.stripeWebhookSecret;
  if (!secret) {
    throw new Error("Missing STRIPE_WEBHOOK_SECRET");
  }

  const parts = signatureHeader
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean);
  const timestampPart = parts.find((part) => part.startsWith("t="));
  const signatureParts = parts.filter((part) => part.startsWith("v1="));
  if (!timestampPart || signatureParts.length === 0) {
    throw new Error("Invalid Stripe signature header.");
  }

  const timestampRaw = timestampPart.replace(/^t=/, "");
  const timestamp = Number(timestampRaw);
  if (!Number.isFinite(timestamp) || timestamp <= 0) {
    throw new Error("Invalid Stripe signature timestamp.");
  }
  const toleranceInput = Number(process.env.STRIPE_WEBHOOK_TOLERANCE_SECONDS ?? "300");
  const toleranceSeconds =
    Number.isFinite(toleranceInput) && toleranceInput > 0
      ? Math.floor(toleranceInput)
      : 300;
  const nowEpochSeconds = Math.floor(Date.now() / 1000);
  if (Math.abs(nowEpochSeconds - Math.floor(timestamp)) > toleranceSeconds) {
    throw new Error("Stale Stripe webhook timestamp.");
  }

  const signedPayload = `${timestampRaw}.${rawBody}`;
  const expected = crypto
    .createHmac("sha256", secret)
    .update(signedPayload, "utf8")
    .digest("hex");

  const matched = signatureParts.some((part) => {
    const signature = part.replace(/^v1=/, "");
    return secureEqualHex(expected, signature);
  });
  if (!matched) {
    throw new Error("Stripe signature verification failed.");
  }
}

export function parseStripeEvent(rawBody: string): StripeEvent {
  let parsed: unknown;
  try {
    parsed = JSON.parse(rawBody);
  } catch {
    throw new Error("Invalid Stripe webhook payload.");
  }

  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("Invalid Stripe webhook payload.");
  }

  const record = parsed as Record<string, unknown>;
  const id = typeof record.id === "string" ? record.id : "";
  const type = typeof record.type === "string" ? record.type : "";
  const data =
    record.data && typeof record.data === "object" && !Array.isArray(record.data)
      ? (record.data as Record<string, unknown>)
      : null;
  const object =
    data && data.object && typeof data.object === "object" && !Array.isArray(data.object)
      ? (data.object as Record<string, unknown>)
      : null;

  if (!id || !type || !object) {
    throw new Error("Invalid Stripe event shape.");
  }

  return {
    id,
    type,
    data: {
      object,
    },
  };
}

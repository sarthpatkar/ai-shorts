import { NextResponse } from "next/server";
import {
  getUserIdByStripeCustomerId,
  insertJobUnlock,
  isNewWebhookEvent,
  setUserPlanTier,
  upsertSubscriptionFromStripe,
} from "@/lib/server/supabase-admin";
import {
  parseStripeEvent,
  verifyWebhookSignature,
  type StripeEvent,
} from "@/lib/server/stripe";
import { assertServerEnv } from "@/lib/server/env";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";
assertServerEnv("payments_webhook");

function isActiveSubscriptionStatus(status: string): boolean {
  return status === "active" || status === "trialing";
}

function asRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}

function asString(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

async function handleCheckoutCompleted(event: StripeEvent): Promise<void> {
  const object = event.data.object;
  const metadata = asRecord(object.metadata);
  const purchaseType = asString(metadata?.purchase_type);

  if (purchaseType !== "job_unlock") {
    return;
  }

  const userId = asString(metadata?.user_id);
  const jobId = asString(metadata?.job_id);
  const sessionId = asString(object.id);
  const paymentIntentId = asString(object.payment_intent);

  if (!userId || !jobId) {
    return;
  }

  await insertJobUnlock({
    userId,
    jobId,
    unlockType: "pay_per_job",
    unlockedQualities: ["480p", "720p", "1080p"],
    source: "stripe_checkout",
    stripeCheckoutSessionId: sessionId || undefined,
    stripePaymentIntentId: paymentIntentId || undefined,
  });
}

async function handleSubscriptionEvent(event: StripeEvent): Promise<void> {
  const object = event.data.object;
  const metadata = asRecord(object.metadata);
  const stripeCustomerId = asString(object.customer);
  const stripeSubscriptionId = asString(object.id);
  const status = asString(object.status);
  if (!stripeCustomerId || !stripeSubscriptionId || !status) {
    return;
  }

  let userId = asString(metadata?.user_id);
  if (!userId) {
    userId = (await getUserIdByStripeCustomerId(stripeCustomerId)) ?? "";
  }
  if (!userId) {
    return;
  }

  const items = asRecord(object.items);
  const data = Array.isArray(items?.data) ? items?.data : [];
  const firstItem = asRecord(data[0]);
  const price = asRecord(firstItem?.price);
  const stripePriceId = asString(price?.id);

  await upsertSubscriptionFromStripe({
    userId,
    stripeCustomerId,
    stripeSubscriptionId,
    stripePriceId: stripePriceId || "unknown",
    status,
    currentPeriodStart:
      typeof object.current_period_start === "number"
        ? object.current_period_start
        : null,
    currentPeriodEnd:
      typeof object.current_period_end === "number"
        ? object.current_period_end
        : null,
    cancelAtPeriodEnd: Boolean(object.cancel_at_period_end),
    metadata: metadata
      ? Object.fromEntries(
          Object.entries(metadata).map(([key, value]) => [key, String(value ?? "")])
        )
      : {},
  });

  await setUserPlanTier(userId, isActiveSubscriptionStatus(status) ? "premium" : "free");
}

export async function POST(request: Request) {
  const signature = request.headers.get("stripe-signature") ?? "";
  if (!signature.trim()) {
    return NextResponse.json({ error: "Missing Stripe signature." }, { status: 400 });
  }

  const rawBody = await request.text();

  let event: StripeEvent;
  try {
    verifyWebhookSignature(rawBody, signature);
    event = parseStripeEvent(rawBody);
  } catch (error) {
    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : "Invalid Stripe webhook signature.",
      },
      { status: 400 }
    );
  }

  try {
    const isNew = await isNewWebhookEvent(event.id, event.type, event);
    if (!isNew) {
      return NextResponse.json({ ok: true, duplicate: true }, { status: 200 });
    }

    if (event.type === "checkout.session.completed") {
      await handleCheckoutCompleted(event);
    }

    if (
      event.type === "customer.subscription.created" ||
      event.type === "customer.subscription.updated" ||
      event.type === "customer.subscription.deleted"
    ) {
      await handleSubscriptionEvent(event);
    }

    return NextResponse.json({ ok: true }, { status: 200 });
  } catch (error) {
    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : "Webhook processing failed.",
      },
      { status: 500 }
    );
  }
}

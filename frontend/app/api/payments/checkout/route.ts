import { NextResponse } from "next/server";
import { getRequestUser } from "@/lib/server/auth";
import {
  getStripeCustomerIdForUser,
  upsertStripeCustomer,
} from "@/lib/server/supabase-admin";
import {
  createCheckoutSession,
  createStripeCustomer,
} from "@/lib/server/stripe";
import { assertServerEnv, getAppBaseUrlFromRequest } from "@/lib/server/env";

export const dynamic = "force-dynamic";
assertServerEnv("payments_checkout");

type CheckoutBody = {
  purchaseType?: "monthly_subscription" | "job_unlock";
  jobId?: string;
};

export async function POST(request: Request) {
  const { user } = await getRequestUser(request);
  if (!user) {
    return NextResponse.json(
      { code: "needs_auth", error: "Login required." },
      { status: 401 }
    );
  }

  let body: CheckoutBody;
  try {
    body = (await request.json()) as CheckoutBody;
  } catch {
    return NextResponse.json({ error: "Invalid request payload." }, { status: 400 });
  }

  const purchaseType = body.purchaseType;
  const jobId = (body.jobId ?? "").trim();

  if (purchaseType !== "monthly_subscription" && purchaseType !== "job_unlock") {
    return NextResponse.json(
      { error: "purchaseType must be monthly_subscription or job_unlock." },
      { status: 400 }
    );
  }

  if (purchaseType === "job_unlock" && !jobId) {
    return NextResponse.json(
      { error: "jobId is required for job unlock checkout." },
      { status: 400 }
    );
  }

  try {
    let customerId = await getStripeCustomerIdForUser(user.id);
    if (!customerId) {
      customerId = await createStripeCustomer({
        userId: user.id,
        email: user.email,
      });
      await upsertStripeCustomer(user.id, customerId);
    }

    const appBase = getAppBaseUrlFromRequest(request);
    const successUrl = `${appBase}/?checkout=success&type=${encodeURIComponent(
      purchaseType
    )}${jobId ? `&jobId=${encodeURIComponent(jobId)}` : ""}`;
    const cancelUrl = `${appBase}/?checkout=cancel`;

    const session = await createCheckoutSession({
      mode: purchaseType === "monthly_subscription" ? "subscription" : "payment",
      customerId,
      userId: user.id,
      successUrl,
      cancelUrl,
      jobId: purchaseType === "job_unlock" ? jobId : undefined,
    });

    return NextResponse.json(
      {
        checkoutUrl: session.url,
        checkoutSessionId: session.id,
      },
      { status: 200 }
    );
  } catch (error) {
    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : "Unable to create checkout session.",
      },
      { status: 502 }
    );
  }
}

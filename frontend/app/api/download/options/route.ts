import { NextResponse } from "next/server";
import { getRequestUser } from "@/lib/server/auth";
import {
  getDownloadEntitlement,
  type DownloadEntitlement,
} from "@/lib/server/supabase-admin";
import {
  assertServerEnv,
  buildBackendUrl,
  normalizeBackendPath,
  serverEnv,
} from "@/lib/server/env";

export const dynamic = "force-dynamic";
assertServerEnv("download_proxy");

type BackendOption = {
  quality: string;
  available: boolean;
  height?: number;
};

type BackendOptionsResponse = {
  options?: BackendOption[];
  source_height?: number;
};

function asNonEmpty(value: string | null): string {
  return (value ?? "").trim();
}

function mapDeniedReason(reason: string): string {
  const lowered = reason.toLowerCase();
  if (lowered.includes("needs_auth")) {
    return "needs_auth";
  }
  if (lowered.includes("free_480_already_used")) {
    return "needs_purchase";
  }
  if (lowered.includes("premium_unlock_required")) {
    return "needs_purchase";
  }
  if (lowered.includes("unsupported_quality")) {
    return "unavailable";
  }
  if (lowered.includes("entitlement_unavailable")) {
    return "entitlement_unavailable";
  }
  return "needs_purchase";
}

export async function GET(request: Request) {
  const url = new URL(request.url);
  const jobId = asNonEmpty(url.searchParams.get("jobId"));
  const clipId = asNonEmpty(url.searchParams.get("clipId"));

  if (!jobId || !clipId) {
    return NextResponse.json(
      { error: "jobId and clipId are required." },
      { status: 400 }
    );
  }

  const backendPath = normalizeBackendPath(
    process.env.BACKEND_DOWNLOAD_OPTIONS_PATH ?? "",
    "/download/options"
  );

  let backendPayload: BackendOptionsResponse;
  try {
    const response = await fetch(buildBackendUrl(backendPath), {
      method: "POST",
      cache: "no-store",
      headers: {
        "Content-Type": "application/json",
        ...(serverEnv.backendInternalToken
          ? { "x-internal-token": serverEnv.backendInternalToken }
          : {}),
      },
      body: JSON.stringify({
        job_id: jobId,
        clip_id: clipId,
      }),
    });
    const payload = (await response.json()) as BackendOptionsResponse;
    if (!response.ok) {
      const message =
        payload && typeof payload === "object" && "detail" in payload
          ? String((payload as Record<string, unknown>).detail || "")
          : "Failed to load download options.";
      return NextResponse.json(
        { error: message || "Failed to load download options." },
        { status: response.status }
      );
    }
    backendPayload = payload;
  } catch {
    return NextResponse.json(
      { error: "Backend unavailable for download options." },
      { status: 503 }
    );
  }

  const backendOptions = Array.isArray(backendPayload.options)
    ? backendPayload.options
    : [];

  const { user } = await getRequestUser(request);

  if (!user) {
    return NextResponse.json(
      {
        userAuthenticated: false,
        jobId,
        clipId,
        sourceHeight: backendPayload.source_height ?? null,
        options: backendOptions.map((option) => ({
          quality: option.quality,
          available: Boolean(option.available),
          locked: true,
          reason: option.available ? "needs_auth" : "unavailable",
          height: option.height ?? null,
        })),
      },
      { status: 200 }
    );
  }

  const entitlements = new Map<string, DownloadEntitlement>();
  await Promise.all(
    backendOptions
      .filter((option) => Boolean(option.available) && typeof option.quality === "string")
      .map(async (option) => {
        try {
          const entitlement = await getDownloadEntitlement(
            user.id,
            jobId,
            option.quality
          );
          entitlements.set(option.quality, entitlement);
        } catch {
          entitlements.set(option.quality, {
            allowed: false,
            reason: "entitlement_unavailable",
            should_grant_first_free_480: false,
          });
        }
      })
  );

  return NextResponse.json(
    {
      userAuthenticated: true,
      jobId,
      clipId,
      sourceHeight: backendPayload.source_height ?? null,
      options: backendOptions.map((option) => {
        if (!option.available) {
          return {
            quality: option.quality,
            available: false,
            locked: true,
            reason: "unavailable",
            height: option.height ?? null,
          };
        }
        const entitlement = entitlements.get(option.quality);
        if (!entitlement) {
          return {
            quality: option.quality,
            available: true,
            locked: true,
            reason: "entitlement_unavailable",
            height: option.height ?? null,
          };
        }
        if (entitlement.allowed) {
          return {
            quality: option.quality,
            available: true,
            locked: false,
            reason: entitlement.reason,
            height: option.height ?? null,
          };
        }
        return {
          quality: option.quality,
          available: true,
          locked: true,
          reason: mapDeniedReason(entitlement.reason),
          height: option.height ?? null,
        };
      }),
    },
    { status: 200 }
  );
}

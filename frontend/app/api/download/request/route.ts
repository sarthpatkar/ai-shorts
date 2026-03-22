import { NextResponse } from "next/server";
import { getRequestUser } from "@/lib/server/auth";
import {
  getDownloadEntitlement,
  grantFirstFree480Unlock,
} from "@/lib/server/supabase-admin";
import {
  assertServerEnv,
  buildBackendUrl,
  normalizeBackendPath,
  serverEnv,
} from "@/lib/server/env";

export const dynamic = "force-dynamic";
assertServerEnv("download_proxy");

type DownloadRequestBody = {
  jobId?: string;
  clipId?: string;
  quality?: string;
};

type BackendDownloadResponse = {
  download_url?: string;
  quality?: string;
};

function normalizeQuality(raw: string): string {
  const cleaned = raw.trim().toLowerCase().replace(/\s+/g, "");
  if (cleaned === "240" || cleaned === "240p") {
    return "240p";
  }
  if (cleaned === "360" || cleaned === "360p") {
    return "360p";
  }
  if (cleaned === "480" || cleaned === "480p") {
    return "480p";
  }
  if (cleaned === "720" || cleaned === "720p") {
    return "720p";
  }
  if (cleaned === "1080" || cleaned === "1080p") {
    return "1080p";
  }
  throw new Error("unsupported_quality");
}

function purchaseReason(reason: string): string {
  const lowered = reason.toLowerCase();
  if (lowered.includes("unsupported_quality")) {
    return "unavailable";
  }
  if (lowered.includes("free_480_already_used")) {
    return "free_480_already_used";
  }
  if (lowered.includes("premium_unlock_required")) {
    return "premium_unlock_required";
  }
  if (lowered.includes("entitlement_unavailable")) {
    return "request_failed";
  }
  return "needs_purchase";
}

export async function POST(request: Request) {
  const { user } = await getRequestUser(request);
  if (!user) {
    return NextResponse.json(
      { code: "needs_auth", error: "Login is required for download." },
      { status: 401 }
    );
  }

  let body: DownloadRequestBody;
  try {
    body = (await request.json()) as DownloadRequestBody;
  } catch {
    return NextResponse.json({ error: "Invalid request payload." }, { status: 400 });
  }

  const jobId = (body.jobId ?? "").trim();
  const clipId = (body.clipId ?? "").trim();
  if (!jobId || !clipId) {
    return NextResponse.json(
      { error: "jobId and clipId are required." },
      { status: 400 }
    );
  }

  let quality: string;
  try {
    quality = normalizeQuality(body.quality ?? "");
  } catch {
    return NextResponse.json({ error: "Unsupported quality." }, { status: 400 });
  }

  try {
    const entitlement = await getDownloadEntitlement(user.id, jobId, quality);
    if (!entitlement.allowed) {
      const denialReason = purchaseReason(entitlement.reason);
      const denialCode =
        denialReason === "unavailable"
          ? "unavailable"
          : denialReason === "request_failed"
          ? "request_failed"
          : "needs_purchase";
      return NextResponse.json(
        {
          code: denialCode,
          reason: denialReason,
          error:
            denialReason === "unavailable"
              ? "This quality is not available for the selected clip."
              : "This quality requires premium unlock.",
        },
        { status: 403 }
      );
    }

    if (entitlement.should_grant_first_free_480 && quality === "480p") {
      await grantFirstFree480Unlock(user.id, jobId);
    }
  } catch (error) {
    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : "Failed to evaluate download entitlement.",
      },
      { status: 502 }
    );
  }

  const backendPath = normalizeBackendPath(
    process.env.BACKEND_DOWNLOAD_REQUEST_PATH ?? "",
    "/download/request"
  );

  let backendPayload: BackendDownloadResponse;
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
        quality,
      }),
    });

    const payload = (await response.json()) as BackendDownloadResponse;
    if (!response.ok) {
      const message =
        payload && typeof payload === "object" && "detail" in payload
          ? String((payload as Record<string, unknown>).detail || "")
          : "Download generation failed.";
      return NextResponse.json(
        {
          code: response.status === 400 ? "unavailable" : "request_failed",
          error: message || "Download generation failed.",
        },
        { status: response.status }
      );
    }
    backendPayload = payload;
  } catch {
    return NextResponse.json(
      { error: "Backend unavailable for download request." },
      { status: 503 }
    );
  }

  const downloadUrl =
    typeof backendPayload.download_url === "string"
      ? backendPayload.download_url.trim()
      : "";

  if (!downloadUrl) {
    return NextResponse.json(
      { error: "Backend did not return download URL." },
      { status: 502 }
    );
  }

  return NextResponse.json(
    {
      jobId,
      clipId,
      quality,
      downloadUrl,
    },
    { status: 200 }
  );
}

import { NextResponse } from "next/server";
import { normalizeGenerateResponse } from "@/lib/clip-normalizer";
import { assertServerEnv } from "@/lib/server/env";

export const dynamic = "force-dynamic";
assertServerEnv("core");

const DEFAULT_BACKEND_BASE_URL = "http://localhost:8000";
const DEFAULT_RESULT_PATH = "/result";
const REQUEST_TIMEOUT_MS = 1000 * 60;

function asNonEmptyString(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function normalizePath(path: string): string {
  return path.startsWith("/") ? path : `/${path}`;
}

function buildBackendUrl(baseUrl: string, path: string): string {
  const normalizedBase = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
  const normalizedPath = path.replace(/^\//, "");
  return new URL(normalizedPath, normalizedBase).toString();
}

function getBackendConfig(): { baseUrl: string; resultPath: string } {
  const baseUrl =
    asNonEmptyString(process.env.BACKEND_API_BASE_URL) ??
    asNonEmptyString(process.env.NEXT_PUBLIC_BACKEND_API_BASE_URL) ??
    DEFAULT_BACKEND_BASE_URL;

  const resultPath =
    asNonEmptyString(process.env.BACKEND_RESULT_PATH) ??
    asNonEmptyString(process.env.NEXT_PUBLIC_BACKEND_RESULT_PATH) ??
    DEFAULT_RESULT_PATH;

  return {
    baseUrl,
    resultPath: normalizePath(resultPath),
  };
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

function mapFailureMessage(error: string | null): string {
  const normalized = (error ?? "").toLowerCase();
  if (
    normalized.includes("youtube_bot_check_required") ||
    (normalized.includes("sign in to confirm") && normalized.includes("not a bot"))
  ) {
    return "This video appears restricted. Try another source.";
  }
  if (normalized.includes("youtube_private_video")) {
    return "This video is private or restricted.";
  }
  if (!normalized || normalized.includes("no_clips_uploaded")) {
    return "We couldn't extract clips from this source video.";
  }
  return "Clip generation is still in progress.";
}

export async function GET(request: Request) {
  const requestUrl = new URL(request.url);
  const jobId = asNonEmptyString(requestUrl.searchParams.get("jobId"));
  if (!jobId) {
    return NextResponse.json({ error: "jobId is required." }, { status: 400 });
  }

  const config = getBackendConfig();
  const endpoint = buildBackendUrl(
    config.baseUrl,
    `${config.resultPath.replace(/\/$/, "")}/${encodeURIComponent(jobId)}`
  );

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const response = await fetch(endpoint, {
      method: "GET",
      cache: "no-store",
      signal: controller.signal,
    });
    const payload = await parseJsonSafe(response);
    if (!response.ok) {
      return NextResponse.json(
        { error: "Could not refresh clip URLs right now." },
        { status: response.status }
      );
    }

    const normalized = normalizeGenerateResponse(payload, config.baseUrl);
    if (normalized.clips.length > 0) {
      return NextResponse.json(
        {
          clips: normalized.clips,
          runId: normalized.runId,
          jobId,
        },
        { status: 200 }
      );
    }

    let status = "processing";
    let errorMessage: string | null = null;
    if (payload && typeof payload === "object" && !Array.isArray(payload)) {
      const record = payload as Record<string, unknown>;
      status = asNonEmptyString(record.status) ?? "processing";
      errorMessage = asNonEmptyString(record.error);
    }

    if (status === "failed") {
      return NextResponse.json(
        { error: mapFailureMessage(errorMessage) },
        { status: 502 }
      );
    }

    return NextResponse.json({ clips: [], jobId }, { status: 200 });
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      return NextResponse.json(
        { error: "Refresh timed out. Please try again." },
        { status: 504 }
      );
    }
    return NextResponse.json(
      { error: "Backend is currently unavailable. Please try again shortly." },
      { status: 503 }
    );
  } finally {
    clearTimeout(timeoutId);
  }
}

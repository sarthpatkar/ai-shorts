import { NextResponse } from "next/server";
import { assertServerEnv } from "@/lib/server/env";

export const dynamic = "force-dynamic";
assertServerEnv("core");

const DEFAULT_FEEDBACK_PATH = "/feedback";
const REQUEST_TIMEOUT_MS = 1000 * 30;

function asNonEmptyString(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }

  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function normalizePath(path: string): string {
  if (!path.startsWith("/")) {
    return `/${path}`;
  }
  return path;
}

function buildBackendUrl(baseUrl: string, path: string): string {
  const normalizedBase = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
  const normalizedPath = path.replace(/^\//, "");
  return new URL(normalizedPath, normalizedBase).toString();
}

function getBackendConfig(): { baseUrl: string; feedbackPath: string } {
  const baseUrl =
    asNonEmptyString(process.env.NEXT_PUBLIC_API_URL) ??
    asNonEmptyString(process.env.BACKEND_API_BASE_URL) ??
    asNonEmptyString(process.env.NEXT_PUBLIC_BACKEND_API_BASE_URL);

  if (!baseUrl) {
    throw new Error("Missing backend API URL. Set NEXT_PUBLIC_API_URL.");
  }

  const feedbackPath =
    asNonEmptyString(process.env.BACKEND_FEEDBACK_PATH) ??
    asNonEmptyString(process.env.NEXT_PUBLIC_BACKEND_FEEDBACK_PATH) ??
    DEFAULT_FEEDBACK_PATH;

  return {
    baseUrl,
    feedbackPath: normalizePath(feedbackPath),
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

async function sendFeedback(
  endpoint: string,
  payload: Record<string, unknown>
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    return await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      cache: "no-store",
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
  } finally {
    clearTimeout(timeoutId);
  }
}

function feedbackError(status: number): string {
  if (status === 400 || status === 422) {
    return "Feedback payload is invalid.";
  }
  if (status === 404) {
    return "Feedback endpoint was not found.";
  }
  if (status >= 500) {
    return "Could not submit feedback right now. Please try again.";
  }
  return "Could not submit feedback.";
}

export async function POST(request: Request) {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json(
      { error: "Invalid request payload." },
      { status: 400 }
    );
  }

  const payload =
    body && typeof body === "object" && !Array.isArray(body)
      ? (body as Record<string, unknown>)
      : null;

  if (!payload) {
    return NextResponse.json(
      { error: "Feedback payload is invalid." },
      { status: 400 }
    );
  }

  const clipId =
    asNonEmptyString(payload.clip_id) ?? asNonEmptyString(payload.clipId);

  if (!clipId) {
    return NextResponse.json(
      { error: "clip_id is required." },
      { status: 400 }
    );
  }

  let config: ReturnType<typeof getBackendConfig>;
  try {
    config = getBackendConfig();
  } catch (error) {
    const message =
      error instanceof Error && error.message
        ? error.message
        : "Backend API URL is not configured.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
  const endpoint = buildBackendUrl(config.baseUrl, config.feedbackPath);

  try {
    let response = await sendFeedback(endpoint, payload);

    if (!response.ok && response.status === 422) {
      const fallbackPayload = {
        feedback: payload,
      };
      const fallbackResponse = await sendFeedback(endpoint, fallbackPayload);
      if (fallbackResponse.ok) {
        response = fallbackResponse;
      }
    }

    if (!response.ok) {
      const parsed = await parseJsonSafe(response);
      const backendMessage =
        parsed && typeof parsed === "object"
          ? asNonEmptyString((parsed as Record<string, unknown>).error)
          : null;

      return NextResponse.json(
        { error: backendMessage ?? feedbackError(response.status) },
        { status: response.status }
      );
    }

    return NextResponse.json({ ok: true }, { status: 200 });
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      return NextResponse.json(
        { error: "Feedback request timed out. Please try again." },
        { status: 504 }
      );
    }

    return NextResponse.json(
      { error: "Backend is currently unavailable. Please try again shortly." },
      { status: 503 }
    );
  }
}

import { NextResponse } from "next/server";
import { normalizeGenerateResponse } from "@/lib/clip-normalizer";
import { assertServerEnv } from "@/lib/server/env";

export const dynamic = "force-dynamic";
assertServerEnv("core");

const DEFAULT_BACKEND_BASE_URL = "http://localhost:8000";
const DEFAULT_GENERATE_PATH = "/generate";
const DEFAULT_GENERATE_UPLOAD_PATH = "/generate/upload";
const DEFAULT_RESULT_PATH = "/result";
const REQUEST_TIMEOUT_MS = 1000 * 60 * 8;
const JOB_POLL_INTERVAL_MS = 2500;

type BackendResult = {
  response: Response;
  payload: unknown;
};

type SourcePayload = {
  youtubeUrl: string;
  videoFile: File | null;
  userConfirmedRights: boolean;
};

class ClipGenerationFailedError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ClipGenerationFailedError";
  }
}

function asNonEmptyString(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function parseBooleanLike(value: unknown): boolean {
  if (typeof value === "boolean") {
    return value;
  }
  if (typeof value !== "string") {
    return false;
  }
  const normalized = value.trim().toLowerCase();
  return normalized === "true" || normalized === "1" || normalized === "yes";
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

function getBackendConfig(): {
  baseUrl: string;
  generatePath: string;
  generateUploadPath: string;
  resultPath: string;
} {
  const baseUrl =
    asNonEmptyString(process.env.BACKEND_API_BASE_URL) ??
    asNonEmptyString(process.env.NEXT_PUBLIC_BACKEND_API_BASE_URL) ??
    DEFAULT_BACKEND_BASE_URL;

  const generatePath =
    asNonEmptyString(process.env.BACKEND_GENERATE_PATH) ??
    asNonEmptyString(process.env.NEXT_PUBLIC_BACKEND_GENERATE_PATH) ??
    DEFAULT_GENERATE_PATH;

  const generateUploadPath =
    asNonEmptyString(process.env.BACKEND_GENERATE_UPLOAD_PATH) ??
    asNonEmptyString(process.env.NEXT_PUBLIC_BACKEND_GENERATE_UPLOAD_PATH) ??
    DEFAULT_GENERATE_UPLOAD_PATH;

  const resultPath =
    asNonEmptyString(process.env.BACKEND_RESULT_PATH) ??
    asNonEmptyString(process.env.NEXT_PUBLIC_BACKEND_RESULT_PATH) ??
    DEFAULT_RESULT_PATH;

  return {
    baseUrl,
    generatePath: normalizePath(generatePath),
    generateUploadPath: normalizePath(generateUploadPath),
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

async function requestBackendJson(
  endpoint: string,
  body: Record<string, unknown>
): Promise<BackendResult> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      cache: "no-store",
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    const payload = await parseJsonSafe(response);
    return { response, payload };
  } finally {
    clearTimeout(timeoutId);
  }
}

async function requestBackendForm(
  endpoint: string,
  formData: FormData
): Promise<BackendResult> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      cache: "no-store",
      body: formData,
      signal: controller.signal,
    });

    const payload = await parseJsonSafe(response);
    return { response, payload };
  } finally {
    clearTimeout(timeoutId);
  }
}

async function requestBackendGet(endpoint: string): Promise<BackendResult> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const response = await fetch(endpoint, {
      method: "GET",
      cache: "no-store",
      signal: controller.signal,
    });

    const payload = await parseJsonSafe(response);
    return { response, payload };
  } finally {
    clearTimeout(timeoutId);
  }
}

function toUserError(status: number): string {
  if (status === 400 || status === 422) {
    return "Please provide a valid video source.";
  }
  if (status === 404) {
    return "Clip generation endpoint was not found.";
  }
  if (status >= 500) {
    return "Failed to generate clips right now. Please try again.";
  }
  return "Failed to generate clips.";
}

function readJobId(payload: unknown): string | null {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    return null;
  }

  const record = payload as Record<string, unknown>;
  const value = asNonEmptyString(record.job_id) ?? asNonEmptyString(record.jobId);
  return value ?? null;
}

function readJobStatus(payload: unknown): string | null {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    return null;
  }

  const record = payload as Record<string, unknown>;
  return asNonEmptyString(record.status);
}

function readJobError(payload: unknown): string | null {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    return null;
  }

  const record = payload as Record<string, unknown>;
  return asNonEmptyString(record.error);
}

function mapJobFailureMessage(error: string | null): string {
  const normalized = (error ?? "").toLowerCase().trim();

  if (!normalized || normalized === "no_clips_uploaded") {
    return "No clips were generated for this source. Please try a different video.";
  }
  if (
    normalized.includes("youtube_bot_check_required") ||
    (normalized.includes("sign in to confirm") && normalized.includes("not a bot"))
  ) {
    return "YouTube blocked automated access for this video. Try another URL or configure backend cookies (YTDLP_COOKIES_FROM_BROWSER / YTDLP_COOKIES_FILE).";
  }
  if (normalized.includes("youtube_private_video")) {
    return "This video requires authentication. Configure backend cookies and retry.";
  }
  if (normalized.includes("youtube_video_unavailable")) {
    return "This YouTube video is unavailable. Please try a different URL.";
  }
  if (normalized.includes("pipeline_timeout")) {
    return "Clip generation is taking too long. Please try again.";
  }

  return "Clip generation failed. Please try a different source video.";
}

function hasExplicitClipArray(payload: unknown): boolean {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    return false;
  }
  const record = payload as Record<string, unknown>;
  return Array.isArray(record.clips);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function buildJobEndpoint(basePath: string, jobId: string): string {
  const normalizedBase = normalizePath(basePath).replace(/\/$/, "");
  return `${normalizedBase}/${encodeURIComponent(jobId)}`;
}

async function waitForResult(
  baseUrl: string,
  resultPath: string,
  jobId: string
): Promise<unknown> {
  const deadline = Date.now() + REQUEST_TIMEOUT_MS;

  while (Date.now() < deadline) {
    const endpoint = buildBackendUrl(baseUrl, buildJobEndpoint(resultPath, jobId));
    const result = await requestBackendGet(endpoint);

    if (!result.response.ok) {
      throw new Error(toUserError(result.response.status));
    }

    const status = readJobStatus(result.payload);
    const normalized = normalizeGenerateResponse(result.payload, baseUrl);

    if (normalized.clips.length > 0 || hasExplicitClipArray(result.payload)) {
      return result.payload;
    }

    if (status === "failed") {
      throw new ClipGenerationFailedError(
        mapJobFailureMessage(readJobError(result.payload))
      );
    }

    if (status === "completed") {
      return result.payload;
    }

    await sleep(JOB_POLL_INTERVAL_MS);
  }

  throw new Error("Clip generation is taking too long. Please try again.");
}

async function parseSourcePayload(request: Request): Promise<SourcePayload> {
  const contentType = request.headers.get("content-type") ?? "";

  if (contentType.includes("multipart/form-data")) {
    const form = await request.formData();
    const maybeFile =
      form.get("videoFile") ?? form.get("video_file") ?? form.get("file");
    const videoFile =
      maybeFile instanceof File && maybeFile.size > 0 ? maybeFile : null;
    const youtubeUrl = asNonEmptyString(form.get("youtubeUrl")) ?? "";
    const rightsValue =
      form.get("user_confirmed_rights") ?? form.get("userConfirmedRights");
    return {
      youtubeUrl,
      videoFile,
      userConfirmedRights: parseBooleanLike(rightsValue),
    };
  }

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    throw new Error("Invalid request payload.");
  }

  const record = (body as Record<string, unknown>) ?? {};
  return {
    youtubeUrl: asNonEmptyString(record.youtubeUrl) ?? "",
    videoFile: null,
    userConfirmedRights: parseBooleanLike(
      record.userConfirmedRights ?? record.user_confirmed_rights
    ),
  };
}

export async function POST(request: Request) {
  let source: SourcePayload;
  try {
    source = await parseSourcePayload(request);
  } catch (error) {
    const message =
      error instanceof Error && error.message
        ? error.message
        : "Invalid request payload.";
    return NextResponse.json({ error: message }, { status: 400 });
  }

  if (!source.userConfirmedRights) {
    return NextResponse.json(
      {
        error:
          "Please confirm rights before processing. Use only videos you own or have permission to process.",
      },
      { status: 400 }
    );
  }

  if (!source.youtubeUrl && !source.videoFile) {
    return NextResponse.json(
      { error: "Provide a YouTube URL or upload a video file." },
      { status: 400 }
    );
  }

  const config = getBackendConfig();
  const backendGenerateEndpoint = buildBackendUrl(
    config.baseUrl,
    config.generatePath
  );
  const backendUploadEndpoint = buildBackendUrl(
    config.baseUrl,
    config.generateUploadPath
  );

  try {
    let result: BackendResult;

    if (source.videoFile) {
      const uploadForm = new FormData();
      uploadForm.append("videoFile", source.videoFile);
      uploadForm.append("user_confirmed_rights", "true");
      result = await requestBackendForm(backendUploadEndpoint, uploadForm);
    } else {
      result = await requestBackendJson(backendGenerateEndpoint, {
        url: source.youtubeUrl,
        user_confirmed_rights: true,
      });

      if (!result.response.ok && result.response.status === 422) {
        const fallback = await requestBackendJson(backendGenerateEndpoint, {
          youtube_url: source.youtubeUrl,
          user_confirmed_rights: true,
        });
        if (fallback.response.ok) {
          result = fallback;
        }
      }
    }

    if (!result.response.ok) {
      return NextResponse.json(
        { error: toUserError(result.response.status) },
        { status: result.response.status }
      );
    }

    const immediate = normalizeGenerateResponse(result.payload, config.baseUrl);
    if (immediate.clips.length > 0 || hasExplicitClipArray(result.payload)) {
      return NextResponse.json(
        {
          clips: immediate.clips,
          runId: immediate.runId,
          jobId: readJobId(result.payload),
        },
        { status: 200 }
      );
    }

    const jobId = readJobId(result.payload);
    if (!jobId) {
      return NextResponse.json(
        { error: "Backend response did not include clips or a job id." },
        { status: 502 }
      );
    }

    const finalPayload = await waitForResult(config.baseUrl, config.resultPath, jobId);
    const normalized = normalizeGenerateResponse(finalPayload, config.baseUrl);
    return NextResponse.json(
      {
        clips: normalized.clips,
        runId: normalized.runId,
        jobId,
      },
      { status: 200 }
    );
  } catch (error) {
    if (error instanceof ClipGenerationFailedError) {
      return NextResponse.json({ error: error.message }, { status: 502 });
    }

    if (error instanceof Error && error.message) {
      if (error.message.includes("taking too long")) {
        return NextResponse.json({ error: error.message }, { status: 504 });
      }
      if (error.message.includes("Please provide a valid video source")) {
        return NextResponse.json({ error: error.message }, { status: 400 });
      }
      if (error.message.includes("endpoint was not found")) {
        return NextResponse.json({ error: error.message }, { status: 404 });
      }
      if (error.message.includes("Failed to generate clips")) {
        return NextResponse.json({ error: error.message }, { status: 502 });
      }
    }

    if (error instanceof Error && error.name === "AbortError") {
      return NextResponse.json(
        { error: "Clip generation is taking too long. Please try again." },
        { status: 504 }
      );
    }

    return NextResponse.json(
      { error: "Backend is currently unavailable. Please try again shortly." },
      { status: 503 }
    );
  }
}

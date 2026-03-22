import type {
  Clip,
  DownloadOption,
  DownloadQuality,
  FeedbackSubmissionInput,
  GenerateClipsResult,
} from "@/lib/types";

type GenerateResponse = {
  clips?: Clip[];
  runId?: string | null;
  jobId?: string | null;
};

type ApiErrorCode =
  | "validation"
  | "unavailable"
  | "timeout"
  | "request_failed"
  | "empty_response";

type GenerateClipsInput = {
  youtubeUrl?: string;
  videoFile?: File | null;
  userConfirmedRights: boolean;
};

const REQUEST_TIMEOUT_MS = 1000 * 60 * 8;

export class ApiError extends Error {
  code: ApiErrorCode;
  status?: number;

  constructor(message: string, code: ApiErrorCode, status?: number) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
  }
}

function withTimeout(signal?: AbortSignal): {
  signal: AbortSignal;
  cleanup: () => void;
} {
  if (signal) {
    return { signal, cleanup: () => {} };
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => {
    controller.abort();
  }, REQUEST_TIMEOUT_MS);

  return {
    signal: controller.signal,
    cleanup: () => clearTimeout(timeoutId),
  };
}

async function readJsonSafe(response: Response): Promise<unknown> {
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

function extractErrorMessage(payload: unknown): string | null {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    return null;
  }

  const error = (payload as Record<string, unknown>).error;
  if (typeof error !== "string") {
    return null;
  }

  const trimmed = error.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function mapErrorCode(status: number): ApiErrorCode {
  if (status === 400 || status === 422) {
    return "validation";
  }
  if (status === 503 || status === 502 || status === 504) {
    return "unavailable";
  }
  return "request_failed";
}

async function postJson<TResponse>(
  path: string,
  body: Record<string, unknown>,
  signal?: AbortSignal,
  headers?: Record<string, string>
): Promise<TResponse> {
  const timeout = withTimeout(signal);

  try {
    const response = await fetch(path, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(headers ?? {}),
      },
      cache: "no-store",
      body: JSON.stringify(body),
      signal: timeout.signal,
    });

    const payload = await readJsonSafe(response);

    if (!response.ok) {
      const message =
        extractErrorMessage(payload) ?? "Request failed. Please try again.";
      throw new ApiError(message, mapErrorCode(response.status), response.status);
    }

    return (payload as TResponse) ?? ({} as TResponse);
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }

    if (error instanceof Error && error.name === "AbortError") {
      throw new ApiError(
        "The request is taking too long. Please try again in a moment.",
        "timeout"
      );
    }

    throw new ApiError(
      "Backend is currently unavailable. Please try again shortly.",
      "unavailable"
    );
  } finally {
    timeout.cleanup();
  }
}

async function postForm<TResponse>(
  path: string,
  body: FormData,
  signal?: AbortSignal
): Promise<TResponse> {
  const timeout = withTimeout(signal);

  try {
    const response = await fetch(path, {
      method: "POST",
      cache: "no-store",
      body,
      signal: timeout.signal,
    });

    const payload = await readJsonSafe(response);

    if (!response.ok) {
      const message =
        extractErrorMessage(payload) ?? "Request failed. Please try again.";
      throw new ApiError(message, mapErrorCode(response.status), response.status);
    }

    return (payload as TResponse) ?? ({} as TResponse);
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }

    if (error instanceof Error && error.name === "AbortError") {
      throw new ApiError(
        "The request is taking too long. Please try again in a moment.",
        "timeout"
      );
    }

    throw new ApiError(
      "Backend is currently unavailable. Please try again shortly.",
      "unavailable"
    );
  } finally {
    timeout.cleanup();
  }
}

async function getJson<TResponse>(
  path: string,
  signal?: AbortSignal,
  headers?: Record<string, string>
): Promise<TResponse> {
  const timeout = withTimeout(signal);

  try {
    const response = await fetch(path, {
      method: "GET",
      cache: "no-store",
      headers: headers ?? undefined,
      signal: timeout.signal,
    });

    const payload = await readJsonSafe(response);

    if (!response.ok) {
      const message =
        extractErrorMessage(payload) ?? "Request failed. Please try again.";
      throw new ApiError(message, mapErrorCode(response.status), response.status);
    }

    return (payload as TResponse) ?? ({} as TResponse);
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }

    if (error instanceof Error && error.name === "AbortError") {
      throw new ApiError(
        "The request is taking too long. Please try again in a moment.",
        "timeout"
      );
    }

    throw new ApiError(
      "Backend is currently unavailable. Please try again shortly.",
      "unavailable"
    );
  } finally {
    timeout.cleanup();
  }
}

export async function generateClips(
  input: GenerateClipsInput
): Promise<GenerateClipsResult> {
  if (!input.userConfirmedRights) {
    throw new ApiError(
      "Please confirm rights before processing.",
      "validation",
      400
    );
  }

  const youtubeUrl = (input.youtubeUrl ?? "").trim();
  const videoFile = input.videoFile ?? null;
  if (!youtubeUrl && !videoFile) {
    throw new ApiError(
      "Provide a YouTube URL or upload a video file.",
      "validation",
      400
    );
  }

  let payload: GenerateResponse;
  if (videoFile) {
    const formData = new FormData();
    formData.append("videoFile", videoFile);
    formData.append("user_confirmed_rights", "true");
    if (youtubeUrl) {
      formData.append("youtubeUrl", youtubeUrl);
    }
    payload = await postForm<GenerateResponse>("/api/clips", formData);
  } else {
    payload = await postJson<GenerateResponse>("/api/clips", {
      youtubeUrl,
      userConfirmedRights: true,
    });
  }

  const clips = Array.isArray(payload.clips) ? payload.clips : [];
  if (clips.length === 0) {
    throw new ApiError(
      "No clips were found for this source. Try a different video.",
      "empty_response"
    );
  }

  return {
    clips,
    runId: typeof payload.runId === "string" ? payload.runId : null,
    jobId: typeof payload.jobId === "string" ? payload.jobId : null,
  };
}

export async function refreshClipUrls(jobId: string): Promise<Clip[]> {
  const cleaned = jobId.trim();
  if (!cleaned) {
    throw new ApiError("Missing job id.", "validation", 400);
  }

  const payload = await getJson<GenerateResponse>(
    `/api/clips/result?jobId=${encodeURIComponent(cleaned)}`
  );
  return Array.isArray(payload.clips) ? payload.clips : [];
}

export async function submitClipFeedback(
  input: FeedbackSubmissionInput
): Promise<void> {
  await postJson<{ ok: boolean }>("/api/feedback", {
    clip_id: input.clipId,
    action: input.action,
    liked: input.liked,
    reasons: input.reasons,
    note: input.note ?? "",
    trigger_type: input.triggerType,
    implicit: {
      watch_time_seconds: Number(input.implicit.watchTimeSeconds.toFixed(2)),
      play_count: input.implicit.playCount,
      download_clicks: input.implicit.downloadClicks,
      rate_clicks: input.implicit.rateClicks,
      interaction_events: input.implicit.interactionEvents,
    },
  });
}

function authHeader(accessToken?: string): Record<string, string> {
  const token = String(accessToken ?? "").trim();
  if (!token) {
    return {};
  }
  return { Authorization: `Bearer ${token}` };
}

type DownloadOptionsResponse = {
  options?: DownloadOption[];
  userAuthenticated?: boolean;
  sourceHeight?: number | null;
};

export async function fetchDownloadOptions(input: {
  jobId: string;
  clipId: string;
  accessToken?: string;
}): Promise<DownloadOptionsResponse> {
  const query = `jobId=${encodeURIComponent(input.jobId)}&clipId=${encodeURIComponent(
    input.clipId
  )}`;
  return getJson<DownloadOptionsResponse>(
    `/api/download/options?${query}`,
    undefined,
    authHeader(input.accessToken)
  );
}

type DownloadRequestResponse = {
  jobId: string;
  clipId: string;
  quality: DownloadQuality;
  downloadUrl: string;
  code?: string;
  reason?: string;
  error?: string;
};

export async function requestDownloadUrl(input: {
  jobId: string;
  clipId: string;
  quality: DownloadQuality;
  accessToken?: string;
}): Promise<DownloadRequestResponse> {
  return postJson<DownloadRequestResponse>(
    "/api/download/request",
    {
      jobId: input.jobId,
      clipId: input.clipId,
      quality: input.quality,
    },
    undefined,
    authHeader(input.accessToken)
  );
}

type CheckoutResponse = {
  checkoutUrl: string;
  checkoutSessionId: string;
};

export async function createCheckout(input: {
  purchaseType: "monthly_subscription" | "job_unlock";
  jobId?: string;
  accessToken?: string;
}): Promise<CheckoutResponse> {
  return postJson<CheckoutResponse>(
    "/api/payments/checkout",
    {
      purchaseType: input.purchaseType,
      jobId: input.jobId ?? "",
    },
    undefined,
    authHeader(input.accessToken)
  );
}

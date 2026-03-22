import type { Clip } from "@/lib/types";

type JsonRecord = Record<string, unknown>;

const CLIP_ARRAY_KEYS = ["clips", "outputs", "videos", "results", "items"];
const CLIP_NESTED_KEYS = ["data", "result", "payload"];

function asRecord(value: unknown): JsonRecord | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  return value as JsonRecord;
}

function asNonEmptyString(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function getBackendOrigin(baseUrl: string): string {
  try {
    return new URL(baseUrl).origin;
  } catch {
    return "";
  }
}

function resolveVideoUrl(source: string, backendBaseUrl: string): string {
  const normalized = source.replace(/\\/g, "/").trim();
  if (!normalized) {
    return "";
  }
  if (/^https?:\/\//i.test(normalized) || normalized.startsWith("blob:") || normalized.startsWith("data:")) {
    return normalized;
  }

  const origin = getBackendOrigin(backendBaseUrl);
  if (!origin) {
    return normalized;
  }

  const path = normalized.startsWith("/") ? normalized : `/${normalized}`;
  return new URL(path, origin).toString();
}

function extractClipArray(payload: unknown): unknown[] {
  const queue: unknown[] = [payload];
  const seen = new Set<unknown>();

  while (queue.length > 0) {
    const current = queue.shift();
    if (!current || seen.has(current)) {
      continue;
    }
    seen.add(current);

    if (Array.isArray(current)) {
      return current;
    }

    const record = asRecord(current);
    if (!record) {
      continue;
    }

    for (const key of CLIP_ARRAY_KEYS) {
      const candidate = record[key];
      if (Array.isArray(candidate)) {
        return candidate;
      }
    }

    for (const key of CLIP_NESTED_KEYS) {
      const nested = record[key];
      if (nested !== undefined) {
        queue.push(nested);
      }
    }
  }

  return [];
}

function extractRunId(payload: unknown): string | null {
  const queue: unknown[] = [payload];
  const seen = new Set<unknown>();

  while (queue.length > 0) {
    const current = queue.shift();
    if (!current || seen.has(current)) {
      continue;
    }
    seen.add(current);

    const record = asRecord(current);
    if (!record) {
      continue;
    }

    const runId = asNonEmptyString(record.run_id) ?? asNonEmptyString(record.runId);
    if (runId) {
      return runId;
    }

    for (const key of CLIP_NESTED_KEYS) {
      const nested = record[key];
      if (nested !== undefined) {
        queue.push(nested);
      }
    }
  }

  return null;
}

function normalizeClipItem(item: unknown, index: number, backendBaseUrl: string): Clip | null {
  const fallbackId = `clip-${index + 1}`;

  if (typeof item === "string") {
    const videoUrl = resolveVideoUrl(item, backendBaseUrl);
    if (!videoUrl) {
      return null;
    }
    return {
      id: fallbackId,
      videoUrl,
      caption: `Generated clip ${index + 1}`,
    };
  }

  const record = asRecord(item);
  if (!record) {
    return null;
  }

  const id =
    asNonEmptyString(record.id) ??
    asNonEmptyString(record.clip_id) ??
    asNonEmptyString(record.clipId) ??
    fallbackId;

  const videoSource =
    asNonEmptyString(record.video_url) ??
    asNonEmptyString(record.videoUrl) ??
    asNonEmptyString(record.output_path) ??
    asNonEmptyString(record.outputPath) ??
    asNonEmptyString(record.clip_url) ??
    asNonEmptyString(record.clipUrl) ??
    asNonEmptyString(record.url) ??
    asNonEmptyString(record.path) ??
    asNonEmptyString(record.file);

  if (!videoSource) {
    return null;
  }

  const videoUrl = resolveVideoUrl(videoSource, backendBaseUrl);
  if (!videoUrl) {
    return null;
  }

  const caption =
    asNonEmptyString(record.caption) ??
    asNonEmptyString(record.hook) ??
    asNonEmptyString(record.title) ??
    asNonEmptyString(record.text) ??
    `Generated clip ${index + 1}`;

  return {
    id,
    videoUrl,
    caption,
  };
}

export function normalizeGenerateResponse(
  payload: unknown,
  backendBaseUrl: string
): { clips: Clip[]; runId: string | null } {
  const clipArray = extractClipArray(payload);
  const clips = clipArray
    .map((item, index) => normalizeClipItem(item, index, backendBaseUrl))
    .filter((clip): clip is Clip => clip !== null);

  return {
    clips,
    runId: extractRunId(payload),
  };
}

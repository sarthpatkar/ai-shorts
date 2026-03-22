export type AuthSession = {
  accessToken: string;
  refreshToken: string;
  user: {
    id: string;
    email: string | null;
  };
};

const STORAGE_KEY = "ai_shorts_auth_session_v1";

function getSupabaseConfig(): { url: string; anonKey: string } {
  const url = (process.env.NEXT_PUBLIC_SUPABASE_URL ?? "").trim();
  const anonKey = (process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ?? "").trim();
  if (!url || !anonKey) {
    throw new Error("Supabase client env is missing (NEXT_PUBLIC_SUPABASE_URL / NEXT_PUBLIC_SUPABASE_ANON_KEY).");
  }
  return { url, anonKey };
}

function authUrl(path: string): string {
  const { url } = getSupabaseConfig();
  const base = url.endsWith("/") ? url : `${url}/`;
  return new URL(path.replace(/^\//, ""), base).toString();
}

function persistSession(session: AuthSession | null): void {
  if (typeof window === "undefined") {
    return;
  }
  if (!session) {
    window.sessionStorage.removeItem(STORAGE_KEY);
    return;
  }
  // TODO: migrate to httpOnly cookies for production security.
  window.sessionStorage.setItem(
    STORAGE_KEY,
    JSON.stringify({
      // Keep only short-lived access context; do not persist refresh token in browser storage.
      accessToken: session.accessToken,
      user: session.user,
    })
  );
}

function toSession(payload: unknown): AuthSession {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new Error("Invalid auth response.");
  }
  const record = payload as Record<string, unknown>;
  const accessToken = (
    typeof record.access_token === "string"
      ? record.access_token
      : typeof record.accessToken === "string"
      ? record.accessToken
      : ""
  ).trim();
  const refreshToken = (
    typeof record.refresh_token === "string"
      ? record.refresh_token
      : typeof record.refreshToken === "string"
      ? record.refreshToken
      : ""
  ).trim();
  const userRecord =
    record.user && typeof record.user === "object" && !Array.isArray(record.user)
      ? (record.user as Record<string, unknown>)
      : null;
  const userId = userRecord && typeof userRecord.id === "string" ? userRecord.id : "";
  const email = userRecord && typeof userRecord.email === "string" ? userRecord.email : null;

  if (!accessToken || !userId) {
    throw new Error("Auth response is missing access token/session data.");
  }

  return {
    accessToken,
    refreshToken,
    user: {
      id: userId,
      email,
    },
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

function authError(payload: unknown, fallback: string): string {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    return fallback;
  }
  const record = payload as Record<string, unknown>;
  if (typeof record.msg === "string" && record.msg.trim()) {
    return record.msg.trim();
  }
  if (typeof record.error_description === "string" && record.error_description.trim()) {
    return record.error_description.trim();
  }
  if (typeof record.error === "string" && record.error.trim()) {
    return record.error.trim();
  }
  if (typeof record.message === "string" && record.message.trim()) {
    return record.message.trim();
  }
  return fallback;
}

async function authPost(path: string, body: Record<string, unknown>): Promise<unknown> {
  const { anonKey } = getSupabaseConfig();
  const response = await fetch(authUrl(path), {
    method: "POST",
    cache: "no-store",
    headers: {
      "Content-Type": "application/json",
      apikey: anonKey,
    },
    body: JSON.stringify(body),
  });

  const payload = await parseJsonSafe(response);
  if (!response.ok) {
    throw new Error(authError(payload, "Authentication failed."));
  }
  return payload;
}

export function loadSession(): AuthSession | null {
  if (typeof window === "undefined") {
    return null;
  }
  const raw = window.sessionStorage.getItem(STORAGE_KEY);
  if (!raw) {
    return null;
  }
  try {
    return toSession(JSON.parse(raw));
  } catch {
    window.sessionStorage.removeItem(STORAGE_KEY);
    return null;
  }
}

export function clearSession(): void {
  persistSession(null);
}

export async function signInWithPassword(params: {
  email: string;
  password: string;
}): Promise<AuthSession> {
  const payload = await authPost("/auth/v1/token?grant_type=password", {
    email: params.email,
    password: params.password,
  });
  const session = toSession(payload);
  persistSession(session);
  return session;
}

export async function signUpWithPassword(params: {
  email: string;
  password: string;
}): Promise<AuthSession> {
  const payload = await authPost("/auth/v1/signup", {
    email: params.email,
    password: params.password,
  });
  try {
    const session = toSession(payload);
    persistSession(session);
    return session;
  } catch {
    if (
      payload &&
      typeof payload === "object" &&
      !Array.isArray(payload) &&
      (payload as Record<string, unknown>).user
    ) {
      throw new Error(
        "Signup created. Please verify your email, then login to continue."
      );
    }
    throw new Error("Signup failed. Please try again.");
  }
}

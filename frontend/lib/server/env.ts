function optional(name: string, fallback = ""): string {
  const value = process.env[name];
  if (!value || !value.trim()) {
    return fallback;
  }
  return value.trim();
}

const isProduction = process.env.NODE_ENV === "production";

export const serverEnv = {
  supabaseUrl: optional("NEXT_PUBLIC_SUPABASE_URL"),
  supabaseAnonKey: optional("NEXT_PUBLIC_SUPABASE_ANON_KEY"),
  supabaseServiceRoleKey: optional("SUPABASE_SERVICE_ROLE_KEY"),
  backendBaseUrl:
    optional("NEXT_PUBLIC_API_URL") ||
    optional("BACKEND_API_BASE_URL") ||
    optional("NEXT_PUBLIC_BACKEND_API_BASE_URL"),
  backendInternalToken: optional("BACKEND_INTERNAL_API_TOKEN"),
  stripeSecretKey: optional("STRIPE_SECRET_KEY"),
  stripeWebhookSecret: optional("STRIPE_WEBHOOK_SECRET"),
  stripeMonthlyPriceId: optional("STRIPE_MONTHLY_PRICE_ID"),
  stripeJobUnlockPriceId: optional("STRIPE_JOB_UNLOCK_PRICE_ID"),
  appBaseUrl:
    optional("NEXT_PUBLIC_APP_BASE_URL") || optional("NEXT_PUBLIC_SITE_URL"),
};

type EnvValidationScope =
  | "core"
  | "download_proxy"
  | "payments_checkout"
  | "payments_webhook";

function requireEnv(name: string, value: string, minLength = 1): void {
  if (value.trim().length < Math.max(1, minLength)) {
    throw new Error(`Missing or invalid ${name}.`);
  }
}

function requireValidUrl(name: string, value: string): URL {
  requireEnv(name, value);
  try {
    return new URL(value);
  } catch {
    throw new Error(`Missing or invalid ${name}.`);
  }
}

function isLocalHostname(hostname: string): boolean {
  const host = String(hostname || "").trim().toLowerCase();
  return (
    host === "localhost" ||
    host === "0.0.0.0" ||
    host === "::1" ||
    host === "[::1]" ||
    host.startsWith("127.")
  );
}

function requireProductionBackendUrl(): void {
  const parsed = requireValidUrl(
    "NEXT_PUBLIC_API_URL/BACKEND_API_BASE_URL/NEXT_PUBLIC_BACKEND_API_BASE_URL",
    serverEnv.backendBaseUrl
  );
  if (isLocalHostname(parsed.hostname)) {
    throw new Error(
      "Invalid backend base URL for production; localhost values are not allowed."
    );
  }
}

export function assertServerEnv(scope: EnvValidationScope): void {
  if (!isProduction) {
    return;
  }

  if (scope !== "payments_webhook") {
    requireProductionBackendUrl();
  }

  if (scope === "core") {
    return;
  }

  if (scope === "download_proxy" || scope === "payments_checkout") {
    requireValidUrl("NEXT_PUBLIC_SUPABASE_URL", serverEnv.supabaseUrl);
    requireEnv("NEXT_PUBLIC_SUPABASE_ANON_KEY", serverEnv.supabaseAnonKey, 20);
    requireEnv("SUPABASE_SERVICE_ROLE_KEY", serverEnv.supabaseServiceRoleKey, 20);
    requireEnv("BACKEND_INTERNAL_API_TOKEN", serverEnv.backendInternalToken, 32);
  }

  if (scope === "payments_checkout" || scope === "payments_webhook") {
    requireEnv("STRIPE_SECRET_KEY", serverEnv.stripeSecretKey, 20);
  }

  if (scope === "payments_checkout") {
    requireEnv("STRIPE_MONTHLY_PRICE_ID", serverEnv.stripeMonthlyPriceId, 4);
    requireEnv("STRIPE_JOB_UNLOCK_PRICE_ID", serverEnv.stripeJobUnlockPriceId, 4);
  }

  if (scope === "payments_webhook") {
    requireValidUrl("NEXT_PUBLIC_SUPABASE_URL", serverEnv.supabaseUrl);
    requireEnv("SUPABASE_SERVICE_ROLE_KEY", serverEnv.supabaseServiceRoleKey, 20);
    requireEnv("STRIPE_WEBHOOK_SECRET", serverEnv.stripeWebhookSecret, 20);
  }
}

export function normalizeBackendPath(path: string, fallback: string): string {
  const selected = (path || fallback).trim() || fallback;
  return selected.startsWith("/") ? selected : `/${selected}`;
}

export function buildBackendUrl(path: string): string {
  if (!serverEnv.backendBaseUrl) {
    throw new Error(
      "Missing backend API URL. Set NEXT_PUBLIC_API_URL."
    );
  }
  const base = serverEnv.backendBaseUrl.endsWith("/")
    ? serverEnv.backendBaseUrl
    : `${serverEnv.backendBaseUrl}/`;
  const normalized = path.replace(/^\//, "");
  return new URL(normalized, base).toString();
}

export function getAppBaseUrlFromRequest(request: Request): string {
  if (serverEnv.appBaseUrl) {
    return serverEnv.appBaseUrl;
  }
  const requestUrl = new URL(request.url);
  return requestUrl.origin;
}

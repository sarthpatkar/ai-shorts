import { verifyAccessToken, type AuthUser } from "@/lib/server/supabase-admin";

function extractBearerToken(request: Request): string {
  const authHeader = request.headers.get("authorization") ?? "";
  const [scheme, token] = authHeader.split(" ");
  if (!scheme || !token) {
    return "";
  }
  if (scheme.toLowerCase() !== "bearer") {
    return "";
  }
  return token.trim();
}

export async function getRequestUser(
  request: Request
): Promise<{ user: AuthUser | null; accessToken: string }> {
  const accessToken = extractBearerToken(request);
  if (!accessToken) {
    return { user: null, accessToken: "" };
  }
  const user = await verifyAccessToken(accessToken);
  return { user, accessToken };
}

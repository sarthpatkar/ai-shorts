from __future__ import annotations

import json
import hmac
import ipaddress
import logging
import os
import shutil
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel
from supabase import create_client

from feedback_store import save_clip_feedback
from main import run_pipeline

logger = logging.getLogger(__name__)
app = FastAPI(title="AI Shorts Backend", version="1.0.0")

try:
    import multipart  # type: ignore  # noqa: F401

    HAS_MULTIPART = True
except Exception:
    HAS_MULTIPART = False

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_BUCKET = "clips"
INTERNAL_API_TOKEN = str(os.getenv("INTERNAL_API_TOKEN", "")).strip()
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
MAX_CLIP_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_SOURCE_UPLOAD_BYTES = 250 * 1024 * 1024
SOURCE_UPLOAD_CHUNK_BYTES = 1024 * 1024
UPLOADS_ROOT = (BACKEND_DIR / "runs" / "_uploads").resolve()
ALLOWED_SOURCE_UPLOAD_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".m4v",
    ".webm",
    ".mkv",
    ".avi",
}
supabase = None

TTL_SECONDS = 600
CLEANUP_INTERVAL_SECONDS = 60
TERMINAL_STATUSES = {"completed", "failed"}

jobs: Dict[str, dict] = {}
_jobs_lock = threading.Lock()
_jobs_schema_lock = threading.Lock()
_jobs_optional_columns: Dict[str, bool] = {
    "updated_at": True,
    "error": True,
    "reason": True,
}
QUALITY_HEIGHTS: Dict[str, int] = {
    "240p": 240,
    "360p": 360,
    "480p": 480,
    "720p": 720,
    "1080p": 1080,
}
QUALITY_ORDER = ["240p", "360p", "480p", "720p", "1080p"]
ALLOWED_PUBLIC_SOURCE_HOSTS = {
    "youtube.com",
    "youtu.be",
    "youtube-nocookie.com",
}


class GenerateRequest(BaseModel):
    url: str
    user_confirmed_rights: bool


class DownloadOptionsRequest(BaseModel):
    job_id: str
    clip_id: str


class DownloadRequest(BaseModel):
    job_id: str
    clip_id: str
    quality: str


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; using default=%s", name, raw, default)
        return default


SIGNED_URL_EXPIRY_SECONDS = max(60, _env_int("SIGNED_URL_EXPIRY_SECONDS", 600))
SIGNED_URL_CACHE_SECONDS = max(
    60,
    min(_env_int("SIGNED_URL_CACHE_SECONDS", 300), SIGNED_URL_EXPIRY_SECONDS),
)
MAX_SOURCE_UPLOAD_BYTES = max(
    10 * 1024 * 1024,
    _env_int("MAX_SOURCE_UPLOAD_BYTES", MAX_SOURCE_UPLOAD_BYTES),
)
UPLOAD_RETRIES = max(0, _env_int("UPLOAD_RETRIES", 2))
UPLOAD_RETRY_DELAY_SECONDS = max(0.2, float(os.getenv("UPLOAD_RETRY_DELAY_SECONDS", "1.0")))
PIPELINE_RETRIES = max(0, _env_int("PIPELINE_RETRIES", 0))
PIPELINE_RETRY_DELAY_SECONDS = max(0.2, float(os.getenv("PIPELINE_RETRY_DELAY_SECONDS", "1.2")))
PIPELINE_TIMEOUT_SECONDS = max(60, _env_int("PIPELINE_TIMEOUT_SECONDS", 1800))
MAX_INFLIGHT_JOBS = max(1, _env_int("MAX_INFLIGHT_JOBS", 100))
JOB_WORKERS = max(1, min(_env_int("JOB_WORKERS", 8), MAX_INFLIGHT_JOBS))
_job_executor = ThreadPoolExecutor(max_workers=JOB_WORKERS, thread_name_prefix="ai-shorts-job")
_inflight_jobs_lock = threading.Lock()
_inflight_jobs: set[str] = set()
# Lightweight in-memory anti-abuse limiter for generation endpoints only.
RATE_LIMIT_MAX_REQUESTS = 10
RATE_LIMIT_WINDOW_SECONDS = 60.0
_rate_limit_lock = threading.Lock()
_rate_limit_hits: Dict[str, List[float]] = {}
FEEDBACK_RATE_LIMIT_MAX_REQUESTS = max(1, _env_int("FEEDBACK_RATE_LIMIT_MAX_REQUESTS", 30))
FEEDBACK_RATE_LIMIT_WINDOW_SECONDS = max(
    1.0,
    float(os.getenv("FEEDBACK_RATE_LIMIT_WINDOW_SECONDS", "60.0")),
)
MAX_FEEDBACK_PAYLOAD_BYTES = max(256, _env_int("MAX_FEEDBACK_PAYLOAD_BYTES", 16 * 1024))
MAX_FEEDBACK_NOTE_CHARS = max(64, _env_int("MAX_FEEDBACK_NOTE_CHARS", 1000))
_feedback_rate_limit_lock = threading.Lock()
_feedback_rate_limit_hits: Dict[str, List[float]] = {}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _extract_db_error(exc: Exception) -> str:
    return str(exc).strip() or exc.__class__.__name__


def _normalize_job_error(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    trimmed = value.strip()
    return trimmed if trimmed else None


def _normalize_job_reason(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    trimmed = value.strip()
    return trimmed if trimmed else None


def _is_youtube_blocked_error(error_message: Optional[str]) -> bool:
    lowered = str(error_message or "").lower()
    if not lowered:
        return False
    return any(
        token in lowered
        for token in (
            "youtube_blocked",
            "youtube_bot_check_required",
            "sign in to confirm",
            "not a bot",
            "http error 403",
            "http error 429",
            "status code: 403",
            "status code: 429",
        )
    )


def _derive_failure_reason(error_message: Optional[str]) -> Optional[str]:
    if _is_youtube_blocked_error(error_message):
        return "youtube_blocked"
    return None


def _jobs_column_enabled(column: str) -> bool:
    with _jobs_schema_lock:
        return bool(_jobs_optional_columns.get(column, False))


def _disable_jobs_column(column: str, reason: str) -> None:
    with _jobs_schema_lock:
        if not _jobs_optional_columns.get(column, False):
            return
        _jobs_optional_columns[column] = False
    logger.warning("jobs schema: disabling optional column '%s' (%s)", column, reason)


def _has_missing_jobs_column_error(error_message: str, column: str) -> bool:
    lowered = str(error_message or "").lower()
    column_lower = column.lower()
    if column_lower not in lowered:
        return False
    return (
        "schema cache" in lowered
        or "does not exist" in lowered
        or "could not find" in lowered
    )


def _apply_jobs_schema_hints(error_message: str) -> bool:
    changed = False
    for column in ("updated_at", "error", "reason"):
        if _jobs_column_enabled(column) and _has_missing_jobs_column_error(
            error_message, column
        ):
            _disable_jobs_column(column, reason=error_message)
            changed = True
    return changed


def _jobs_select_fields() -> str:
    fields = ["job_id", "status", "result", "created_at"]
    if _jobs_column_enabled("updated_at"):
        fields.append("updated_at")
    if _jobs_column_enabled("error"):
        fields.append("error")
    if _jobs_column_enabled("reason"):
        fields.append("reason")
    return ",".join(fields)


def _build_job_insert_payload(job_id: str, status: str, now_iso: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "job_id": job_id,
        "status": status,
        "created_at": now_iso,
        "result": None,
    }
    if _jobs_column_enabled("updated_at"):
        payload["updated_at"] = now_iso
    if _jobs_column_enabled("error"):
        payload["error"] = None
    if _jobs_column_enabled("reason"):
        payload["reason"] = None
    return payload


def _build_job_update_payload(
    status: str,
    result: Optional[List[str]] = None,
    error: Optional[str] = None,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "status": status,
    }
    if _jobs_column_enabled("updated_at"):
        payload["updated_at"] = _utc_now_iso()
    if result is not None:
        payload["result"] = result
    if error is not None and _jobs_column_enabled("error"):
        payload["error"] = error
    if reason is not None and _jobs_column_enabled("reason"):
        payload["reason"] = reason
    return payload


def _log_job_event(job_id: str, event: str, **details: Any) -> None:
    logger.info(json.dumps({"event": event, "job_id": job_id, **details}, ensure_ascii=True))


def _init_supabase_client() -> None:
    global supabase
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        logger.error(
            "Supabase disabled: missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY"
        )
        supabase = None
        return
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    except Exception as exc:
        logger.error("Supabase client init failed: %s", _extract_db_error(exc))
        supabase = None


def _is_production_environment() -> bool:
    candidates = (
        os.getenv("APP_ENV", ""),
        os.getenv("ENVIRONMENT", ""),
        os.getenv("PYTHON_ENV", ""),
        os.getenv("FASTAPI_ENV", ""),
        os.getenv("NODE_ENV", ""),
    )
    for raw in candidates:
        normalized = str(raw or "").strip().lower()
        if normalized in {"prod", "production"}:
            return True
    return False


def _require_nonempty_env(name: str, min_length: int = 1) -> str:
    value = str(os.getenv(name, "")).strip()
    if len(value) < max(1, int(min_length)):
        raise RuntimeError(f"missing_or_invalid_{name}")
    return value


def _require_valid_url_env(name: str) -> str:
    value = _require_nonempty_env(name)
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError(f"missing_or_invalid_{name}")
    return value


def _validate_backend_env_or_fail() -> None:
    if not _is_production_environment():
        return
    _require_valid_url_env("SUPABASE_URL")
    _require_nonempty_env("SUPABASE_SERVICE_ROLE_KEY", min_length=32)
    _require_nonempty_env("INTERNAL_API_TOKEN", min_length=32)


def _insert_job_in_db(job_id: str, status: str) -> None:
    if not supabase:
        return
    now_iso = _utc_now_iso()
    for attempt in range(1, 3):
        payload = _build_job_insert_payload(job_id, status, now_iso)
        try:
            supabase.table("jobs").insert(payload).execute()
            return
        except Exception as exc:
            error_message = _extract_db_error(exc)
            logger.warning(
                "jobs.insert failed for %s attempt=%s: %s",
                job_id,
                attempt,
                error_message,
            )
            if attempt == 1 and _apply_jobs_schema_hints(error_message):
                continue
            break

    try:
        supabase.table("jobs").insert(
            {
                "job_id": job_id,
                "status": status,
                "created_at": now_iso,
                "result": None,
            }
        ).execute()
    except Exception as fallback_exc:
        logger.warning(
            "jobs.insert fallback failed for %s: %s",
            job_id,
            _extract_db_error(fallback_exc),
        )


def _update_job_in_db(
    job_id: str,
    status: str,
    result: Optional[List[str]] = None,
    error: Optional[str] = None,
    reason: Optional[str] = None,
) -> None:
    if not supabase:
        return
    for attempt in range(1, 3):
        payload = _build_job_update_payload(
            status,
            result=result,
            error=error,
            reason=reason,
        )
        try:
            supabase.table("jobs").update(payload).eq("job_id", job_id).execute()
            return
        except Exception as exc:
            error_message = _extract_db_error(exc)
            logger.warning(
                "jobs.update failed for %s attempt=%s: %s",
                job_id,
                attempt,
                error_message,
            )
            if attempt == 1 and _apply_jobs_schema_hints(error_message):
                continue
            break

    fallback_payload: Dict[str, Any] = {"status": status}
    if result is not None:
        fallback_payload["result"] = result
    try:
        supabase.table("jobs").update(fallback_payload).eq("job_id", job_id).execute()
    except Exception as exc:
        logger.warning(
            "jobs.update fallback failed for %s: %s", job_id, _extract_db_error(exc)
        )


def _fetch_job_from_db(job_id: str) -> Optional[Dict[str, Any]]:
    if not supabase:
        return None
    for attempt in range(1, 3):
        select_fields = _jobs_select_fields()
        try:
            resp = (
                supabase.table("jobs")
                .select(select_fields)
                .eq("job_id", job_id)
                .limit(1)
                .execute()
            )
            rows = getattr(resp, "data", None) or []
            if rows:
                row = rows[0]
                return {
                    "job_id": str(row.get("job_id", job_id)),
                    "status": str(row.get("status", "unknown")),
                    "result": row.get("result"),
                    "created_at": row.get("created_at"),
                    "updated_at": row.get("updated_at"),
                    "error": row.get("error"),
                    "reason": row.get("reason"),
                }
            return None
        except Exception as exc:
            error_message = _extract_db_error(exc)
            logger.warning(
                "jobs.select failed for %s attempt=%s: %s",
                job_id,
                attempt,
                error_message,
            )
            if attempt == 1 and _apply_jobs_schema_hints(error_message):
                continue
            break

    try:
        resp = (
            supabase.table("jobs")
            .select("job_id,status,result,created_at")
            .eq("job_id", job_id)
            .limit(1)
            .execute()
        )
        rows = getattr(resp, "data", None) or []
        if rows:
            row = rows[0]
            return {
                "job_id": str(row.get("job_id", job_id)),
                "status": str(row.get("status", "unknown")),
                "result": row.get("result"),
                "created_at": row.get("created_at"),
                "updated_at": None,
                "error": None,
                "reason": None,
            }
    except Exception as fallback_exc:
        logger.warning(
            "jobs.select fallback failed for %s: %s",
            job_id,
            _extract_db_error(fallback_exc),
        )
    return None


def _set_job_state(
    job_id: str,
    *,
    status: Optional[str] = None,
    result: Optional[List[str]] = None,
    storage_paths: Optional[List[str]] = None,
    run_dir: Optional[str] = None,
    error: Optional[str] = None,
    reason: Optional[str] = None,
    persist_db: bool = False,
) -> None:
    now_ts = time.time()
    db_status = "processing"
    db_result: Optional[List[str]] = None
    db_error: Optional[str] = None
    db_reason: Optional[str] = None
    with _jobs_lock:
        job = jobs.get(job_id)
        if not job:
            job = {
                "job_id": job_id,
                "status": "processing",
                "result": None,
                "created_ts": now_ts,
                "updated_ts": now_ts,
                "created_at": _utc_now_iso(),
                "run_dir": None,
                "storage_paths": [],
                "error": None,
                "reason": None,
            }
            jobs[job_id] = job

        if status is not None:
            job["status"] = status
        if result is not None:
            job["result"] = list(result)
        if storage_paths is not None:
            job["storage_paths"] = list(storage_paths)
        if run_dir is not None:
            job["run_dir"] = run_dir
        if error is not None:
            job["error"] = error
        if reason is not None:
            job["reason"] = reason
        job["updated_ts"] = now_ts

        db_status = str(job.get("status", "processing"))
        db_result = _to_unique_storage_paths(job.get("storage_paths") or job.get("result") or [])
        raw_error = job.get("error")
        db_error = str(raw_error).strip() if isinstance(raw_error, str) and raw_error.strip() else None
        db_reason = _normalize_job_reason(job.get("reason"))

    if persist_db:
        _update_job_in_db(
            job_id,
            status=db_status,
            result=db_result,
            error=db_error,
            reason=db_reason,
        )


def _rehydrate_job_from_db(job_id: str) -> Optional[Dict[str, Any]]:
    db_job = _fetch_job_from_db(job_id)
    if not db_job:
        return None

    status = str(db_job.get("status", "unknown"))
    storage_paths = _to_unique_storage_paths(db_job.get("result") or [])
    db_error = _normalize_job_error(db_job.get("error"))
    db_reason = _normalize_job_reason(db_job.get("reason"))
    now_ts = time.time()

    with _jobs_lock:
        existing = jobs.get(job_id)
        if existing:
            existing_status = str(existing.get("status", "unknown"))
            if (
                existing_status in TERMINAL_STATUSES
                and status not in TERMINAL_STATUSES
            ):
                status = existing_status
            if not storage_paths:
                storage_paths = _to_unique_storage_paths(
                    existing.get("storage_paths") or existing.get("result") or []
                )
            if db_error is None:
                db_error = _normalize_job_error(existing.get("error"))
            if db_reason is None:
                db_reason = _normalize_job_reason(existing.get("reason"))
            run_dir = existing.get("run_dir")
            created_at = db_job.get("created_at") or existing.get("created_at")
        else:
            run_dir = None
            created_at = db_job.get("created_at")

        rebuilt = {
            "job_id": job_id,
            "status": status,
            "result": storage_paths,
            "storage_paths": storage_paths,
            "run_dir": run_dir,
            "error": db_error,
            "reason": db_reason,
            "created_at": created_at,
            "created_ts": now_ts,
            "updated_ts": now_ts,
        }
        jobs[job_id] = rebuilt
    return rebuilt


def _call_with_timeout(fn: Callable[[], Any], timeout_seconds: float) -> Any:
    if timeout_seconds <= 0:
        return fn()
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn)
    try:
        return future.result(timeout=timeout_seconds)
    except FutureTimeout:
        future.cancel()
        raise
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _run_pipeline_with_retries(job_id: str, url: str) -> Dict[str, Any]:
    attempts = PIPELINE_RETRIES + 1
    last_error: Optional[str] = None
    for attempt in range(1, attempts + 1):
        _log_job_event(job_id, "stage_start", stage="pipeline", attempt=attempt)
        try:
            result = _call_with_timeout(
                lambda: run_pipeline(url),
                timeout_seconds=float(PIPELINE_TIMEOUT_SECONDS),
            )
            if not isinstance(result, dict):
                raise RuntimeError("pipeline returned invalid response")
            _log_job_event(job_id, "stage_complete", stage="pipeline", attempt=attempt)
            return result
        except FutureTimeout:
            last_error = "pipeline_timeout"
        except Exception as exc:
            last_error = _extract_db_error(exc)

        _log_job_event(
            job_id,
            "stage_failed",
            stage="pipeline",
            attempt=attempt,
            error=last_error,
        )
        if _is_youtube_blocked_error(last_error):
            _log_job_event(
                job_id,
                "stage_abort",
                stage="pipeline",
                attempt=attempt,
                reason="non_retryable",
                error=last_error,
            )
            break
        if attempt < attempts:
            _log_job_event(
                job_id,
                "stage_retry",
                stage="pipeline",
                next_attempt=attempt + 1,
                delay_seconds=PIPELINE_RETRY_DELAY_SECONDS,
            )
            time.sleep(PIPELINE_RETRY_DELAY_SECONDS)

    raise RuntimeError(last_error or "pipeline_failed")


def _ensure_rights_confirmed(confirmed: bool) -> None:
    if bool(confirmed):
        return
    raise HTTPException(
        status_code=400,
        detail=(
            "user_confirmed_rights must be true. "
            "Use only videos you own or have permission to process."
        ),
    )


def _ensure_public_source_url_allowed(raw_url: str) -> None:
    value = str(raw_url or "").strip()
    parsed = urlparse(value)
    scheme = str(parsed.scheme or "").strip().lower()
    host = str(parsed.hostname or "").strip().lower()

    if scheme not in {"http", "https"} or not host:
        raise HTTPException(status_code=400, detail="url must be a valid public YouTube URL")

    # Reject local/internal hosts and private IPs for public generate requests.
    if host in {"localhost"} or host.startswith("127.") or host == "::1":
        raise HTTPException(status_code=400, detail="url must be a valid public YouTube URL")
    try:
        ip = ipaddress.ip_address(host)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
        ):
            raise HTTPException(
                status_code=400, detail="url must be a valid public YouTube URL"
            )
    except ValueError:
        pass

    allowed = host in ALLOWED_PUBLIC_SOURCE_HOSTS or any(
        host.endswith(f".{root}") for root in ALLOWED_PUBLIC_SOURCE_HOSTS
    )
    if not allowed:
        raise HTTPException(status_code=400, detail="only YouTube URLs are supported")


def _require_internal_token(token: str) -> None:
    configured = INTERNAL_API_TOKEN
    # Fail closed: internal routes must not be accessible without a configured secret.
    if not configured:
        raise HTTPException(status_code=503, detail="internal auth not configured")
    if hmac.compare_digest(str(token or "").strip(), configured):
        return
    raise HTTPException(status_code=403, detail="forbidden")


def _request_client_ip(request: Request) -> str:
    forwarded_for = str(request.headers.get("x-forwarded-for", "")).strip()
    if forwarded_for:
        first = forwarded_for.split(",")[0].strip()
        if first:
            return first
    if request.client and request.client.host:
        return str(request.client.host)
    return "unknown"


def _enforce_generate_rate_limit(request: Request) -> None:
    # Per-IP 10 requests/minute (POST /generate and POST /generate/upload only).
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS
    client_ip = _request_client_ip(request)
    with _rate_limit_lock:
        recent_hits = [
            ts for ts in _rate_limit_hits.get(client_ip, []) if ts >= cutoff
        ]
        if len(recent_hits) >= RATE_LIMIT_MAX_REQUESTS:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"rate limit exceeded: max {RATE_LIMIT_MAX_REQUESTS} "
                    "requests per minute per IP"
                ),
            )
        recent_hits.append(now)
        _rate_limit_hits[client_ip] = recent_hits


def _enforce_feedback_rate_limit(request: Request) -> None:
    now = time.time()
    cutoff = now - FEEDBACK_RATE_LIMIT_WINDOW_SECONDS
    client_ip = _request_client_ip(request)
    with _feedback_rate_limit_lock:
        recent_hits = [
            ts for ts in _feedback_rate_limit_hits.get(client_ip, []) if ts >= cutoff
        ]
        if len(recent_hits) >= FEEDBACK_RATE_LIMIT_MAX_REQUESTS:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"feedback rate limit exceeded: max {FEEDBACK_RATE_LIMIT_MAX_REQUESTS} "
                    "requests per minute per IP"
                ),
            )
        recent_hits.append(now)
        _feedback_rate_limit_hits[client_ip] = recent_hits


def _validate_feedback_payload_limits(payload: Dict[str, Any]) -> None:
    try:
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        serialized = str(payload)
    if len(serialized.encode("utf-8")) > MAX_FEEDBACK_PAYLOAD_BYTES:
        raise HTTPException(status_code=413, detail="feedback payload is too large")

    note = payload.get("note")
    if isinstance(note, str) and len(note) > MAX_FEEDBACK_NOTE_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"feedback note is too long (max {MAX_FEEDBACK_NOTE_CHARS} characters)",
        )


def _normalize_quality(raw_quality: str) -> str:
    value = str(raw_quality or "").strip().lower().replace(" ", "")
    if value in {"240", "240p"}:
        return "240p"
    if value in {"360", "360p"}:
        return "360p"
    if value in {"480", "480p"}:
        return "480p"
    if value in {"720", "720p"}:
        return "720p"
    if value in {"1080", "1080p"}:
        return "1080p"
    raise HTTPException(status_code=400, detail="unsupported quality")


def _ffprobe_video_height(path: Path) -> int:
    ffprobe_bin = str(os.getenv("FFPROBE_BIN", "ffprobe")).strip() or "ffprobe"
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=height",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=20,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe_not_found") from exc
    except Exception as exc:
        raise RuntimeError(f"ffprobe_failed: {_extract_db_error(exc)}") from exc

    if proc.returncode != 0:
        raise RuntimeError(
            f"ffprobe_failed: {(proc.stderr or proc.stdout or '').strip() or 'unknown_error'}"
        )
    try:
        height = int(str(proc.stdout or "").strip().splitlines()[0])
    except Exception as exc:
        raise RuntimeError("ffprobe_invalid_height") from exc
    if height <= 0:
        raise RuntimeError("ffprobe_invalid_height")
    return height


def _available_qualities_for_height(height: int) -> Dict[str, bool]:
    return {
        quality: bool(height >= required)
        for quality, required in QUALITY_HEIGHTS.items()
    }


def _ffmpeg_transcode_quality(source: Path, target: Path, quality: str) -> None:
    ffmpeg_bin = str(os.getenv("FFMPEG_BIN", "ffmpeg")).strip() or "ffmpeg"
    height = QUALITY_HEIGHTS[quality]
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(source),
        "-vf",
        f"scale=-2:{height}:flags=lanczos",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(target),
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=max(60, PIPELINE_TIMEOUT_SECONDS),
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg_not_found") from exc
    except Exception as exc:
        raise RuntimeError(f"ffmpeg_transcode_failed: {_extract_db_error(exc)}") from exc

    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg_transcode_failed: {(proc.stderr or proc.stdout or '').strip() or 'unknown_error'}"
        )
    if not target.exists() or target.stat().st_size <= 0:
        raise RuntimeError("ffmpeg_transcode_empty_output")


def _storage_path_to_clip_id(storage_path: str) -> str:
    normalized = str(storage_path or "").strip()
    if not normalized:
        return ""
    return Path(normalized).stem


def _sign_storage_path(storage_path: str, job_id: str) -> str:
    signed = _sign_storage_paths([storage_path], job_id)
    if not signed:
        return ""
    return str(signed[0]).strip()


def _resolve_job_source_clip_path(job_id: str, clip_id: str) -> Path:
    resolved_clip_id = str(clip_id or "").strip()
    if not resolved_clip_id:
        raise HTTPException(status_code=400, detail="clip_id is required")

    with _jobs_lock:
        job = jobs.get(job_id)
        run_dir_raw = str((job or {}).get("run_dir", "")).strip()
    if not run_dir_raw:
        raise HTTPException(status_code=404, detail="job assets not available")

    run_dir = _resolve_local_path(run_dir_raw)
    source_path = (run_dir / f"{resolved_clip_id}.mp4").resolve()
    if not source_path.exists() or not source_path.is_file():
        raise HTTPException(status_code=404, detail="source clip not available")
    return source_path


def _build_result_clip_entries(job_id: str, storage_paths: List[str]) -> List[Dict[str, str]]:
    signed_urls = _sign_storage_paths(storage_paths, job_id)
    entries: List[Dict[str, str]] = []
    for storage_path, signed_url in zip(storage_paths, signed_urls):
        clip_id = _storage_path_to_clip_id(storage_path)
        if not clip_id or not signed_url:
            continue
        entries.append(
            {
                "id": clip_id,
                "clip_id": clip_id,
                "video_url": signed_url,
            }
        )
    return entries


def _cleanup_uploaded_source(path_like: Optional[str]) -> None:
    raw_path = str(path_like or "").strip()
    if not raw_path:
        return

    try:
        source_path = Path(raw_path).expanduser().resolve()
    except Exception:
        return

    try:
        uploads_root = UPLOADS_ROOT
        if source_path.exists() and source_path.is_file():
            source_path.unlink()
        parent = source_path.parent
        while (
            parent != uploads_root
            and str(parent).startswith(str(uploads_root))
            and parent.exists()
        ):
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
        if uploads_root.exists():
            try:
                uploads_root.rmdir()
            except OSError:
                pass
    except Exception as exc:
        logger.warning(
            "uploaded source cleanup failed for %s: %s",
            raw_path,
            _extract_db_error(exc),
        )


def _save_uploaded_source(job_id: str, video_file: Any) -> str:
    file_name = str(video_file.filename or "").strip()
    suffix = Path(file_name).suffix.lower()
    content_type = str(video_file.content_type or "").lower().strip()
    if (
        suffix
        and suffix not in ALLOWED_SOURCE_UPLOAD_EXTENSIONS
        and not content_type.startswith("video/")
    ):
        raise HTTPException(status_code=400, detail="uploaded file must be a video")

    upload_dir = (UPLOADS_ROOT / job_id).resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / f"source{suffix or '.mp4'}"

    total_bytes = 0
    try:
        with open(target, "wb") as output:
            while True:
                chunk = video_file.file.read(SOURCE_UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_SOURCE_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "uploaded file is too large "
                            f"(max {MAX_SOURCE_UPLOAD_BYTES // (1024 * 1024)} MB)"
                        ),
                    )
                output.write(chunk)
    except HTTPException:
        _cleanup_uploaded_source(str(target))
        raise
    except Exception as exc:
        _cleanup_uploaded_source(str(target))
        raise HTTPException(
            status_code=400,
            detail=f"failed to read uploaded file: {_extract_db_error(exc)}",
        ) from exc
    finally:
        try:
            video_file.file.close()
        except Exception:
            pass

    if total_bytes <= 0:
        _cleanup_uploaded_source(str(target))
        raise HTTPException(status_code=400, detail="uploaded file is empty")

    return str(target)


def _resolve_local_path(path_like: str) -> Path:
    path = Path(str(path_like or "")).expanduser()
    if path.is_absolute():
        return path

    project_candidate = (PROJECT_ROOT / path).resolve()
    if project_candidate.exists():
        return project_candidate

    backend_candidate = (BACKEND_DIR / path).resolve()
    if backend_candidate.exists():
        return backend_candidate

    return project_candidate


def _is_valid_clip_file(local_file: Path) -> bool:
    if not local_file.exists():
        logger.warning("clip upload skipped: file missing (%s)", local_file)
        return False
    if not local_file.is_file():
        logger.warning("clip upload skipped: not a file (%s)", local_file)
        return False
    if local_file.suffix.lower() != ".mp4":
        logger.warning("clip upload skipped: invalid extension (%s)", local_file)
        return False
    try:
        file_size = local_file.stat().st_size
    except Exception as exc:
        logger.warning(
            "clip upload skipped: cannot stat file (%s): %s",
            local_file,
            _extract_db_error(exc),
        )
        return False
    if file_size > MAX_CLIP_UPLOAD_BYTES:
        logger.warning(
            "clip upload skipped: file too large (%s bytes) for %s",
            file_size,
            local_file,
        )
        return False
    return True


def _upload_clip_to_supabase(job_id: str, local_file: Path) -> Optional[str]:
    if not supabase:
        return None
    if not _is_valid_clip_file(local_file):
        return None

    storage_file_name = f"{local_file.stem}.mp4"
    storage_path = f"clips/{job_id}/{storage_file_name}"
    attempts = UPLOAD_RETRIES + 1
    for attempt in range(1, attempts + 1):
        try:
            with open(local_file, "rb") as file_obj:
                supabase.storage.from_(SUPABASE_BUCKET).upload(
                    storage_path,
                    file_obj,
                    {
                        "content-type": "video/mp4",
                        "upsert": "true",
                        "cache-control": f"public, max-age={SIGNED_URL_CACHE_SECONDS}, s-maxage={SIGNED_URL_CACHE_SECONDS}",
                    },
                )
            return storage_path
        except Exception as exc:
            logger.warning(
                "storage.upload failed for %s (%s) attempt=%s/%s: %s",
                storage_path,
                local_file,
                attempt,
                attempts,
                _extract_db_error(exc),
            )
            if attempt < attempts:
                time.sleep(UPLOAD_RETRY_DELAY_SECONDS)
    return None


def _upload_variant_to_supabase(storage_path: str, local_file: Path) -> Optional[str]:
    if not supabase:
        return None
    if not local_file.exists() or not local_file.is_file():
        return None
    attempts = UPLOAD_RETRIES + 1
    for attempt in range(1, attempts + 1):
        try:
            with open(local_file, "rb") as file_obj:
                supabase.storage.from_(SUPABASE_BUCKET).upload(
                    storage_path,
                    file_obj,
                    {
                        "content-type": "video/mp4",
                        "upsert": "true",
                        "cache-control": f"public, max-age={SIGNED_URL_CACHE_SECONDS}, s-maxage={SIGNED_URL_CACHE_SECONDS}",
                    },
                )
            return storage_path
        except Exception as exc:
            logger.warning(
                "storage.upload variant failed for %s attempt=%s/%s: %s",
                storage_path,
                attempt,
                attempts,
                _extract_db_error(exc),
            )
            if attempt < attempts:
                time.sleep(UPLOAD_RETRY_DELAY_SECONDS)
    return None


def _extract_signed_url(payload: Any) -> str:
    def _normalize_signed_url(value: str) -> str:
        signed_url = str(value or "").strip()
        if not signed_url:
            return ""
        if signed_url.startswith("http://") or signed_url.startswith("https://"):
            return signed_url
        if not SUPABASE_URL:
            return ""
        base = SUPABASE_URL.rstrip("/")
        if signed_url.startswith("/"):
            return f"{base}{signed_url}"
        return f"{base}/{signed_url}"

    if isinstance(payload, str):
        return _normalize_signed_url(payload)
    if not isinstance(payload, dict):
        return ""

    for key in ("signedURL", "signedUrl"):
        value = payload.get(key)
        if isinstance(value, str):
            return _normalize_signed_url(value)

    nested = payload.get("data")
    if isinstance(nested, dict):
        for key in ("signedURL", "signedUrl"):
            value = nested.get(key)
            if isinstance(value, str):
                return _normalize_signed_url(value)
    return ""


def _to_unique_storage_paths(items: Any) -> List[str]:
    if not isinstance(items, list):
        return []
    seen = set()
    unique: List[str] = []
    for item in items:
        value = str(item).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _sign_storage_paths(storage_paths: List[str], job_id: str) -> List[str]:
    if not supabase:
        return []

    signed_urls: List[str] = []
    fail_count = 0
    for storage_path in storage_paths:
        try:
            payload = supabase.storage.from_(SUPABASE_BUCKET).create_signed_url(
                storage_path,
                SIGNED_URL_EXPIRY_SECONDS,
            )
            signed_url = _extract_signed_url(payload)
            if not signed_url:
                fail_count += 1
                logger.warning("signed URL missing for %s", storage_path)
                continue
            signed_urls.append(signed_url)
        except Exception as exc:
            fail_count += 1
            logger.warning(
                "signed URL generation failed for job_id=%s path=%s: %s",
                job_id,
                storage_path,
                _extract_db_error(exc),
            )

    logger.info(
        "signed URL generation: job_id=%s total=%s success=%s failed=%s",
        job_id,
        len(storage_paths),
        len(signed_urls),
        fail_count,
    )
    return signed_urls


def _ensure_quality_variant_signed_url(job_id: str, clip_id: str, quality: str) -> str:
    normalized_quality = _normalize_quality(quality)
    source_path = _resolve_job_source_clip_path(job_id, clip_id)
    source_height = _ffprobe_video_height(source_path)
    available = _available_qualities_for_height(source_height)
    if not available.get(normalized_quality, False):
        raise HTTPException(status_code=400, detail="quality unavailable for this clip")

    if normalized_quality == "1080p" and source_height <= QUALITY_HEIGHTS["1080p"]:
        # For source clips already at or below 1080p, the source acts as 1080 output.
        normalized_quality = "1080p"

    variants_dir = source_path.parent / "_variants"
    variant_local = (variants_dir / f"{clip_id}_{normalized_quality}.mp4").resolve()
    variant_storage_path = f"clips/{job_id}/variants/{clip_id}_{normalized_quality}.mp4"

    existing_signed = _sign_storage_path(variant_storage_path, job_id)
    if existing_signed:
        return existing_signed

    if normalized_quality == "1080p" and source_height <= QUALITY_HEIGHTS["1080p"]:
        try:
            variants_dir.mkdir(parents=True, exist_ok=True)
            if not variant_local.exists():
                shutil.copy2(source_path, variant_local)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"failed to prepare quality variant: {_extract_db_error(exc)}",
            ) from exc
    else:
        _ffmpeg_transcode_quality(source_path, variant_local, normalized_quality)

    uploaded = _upload_variant_to_supabase(variant_storage_path, variant_local)
    if not uploaded:
        raise HTTPException(status_code=500, detail="failed to upload quality variant")
    signed_url = _sign_storage_path(uploaded, job_id)
    if not signed_url:
        raise HTTPException(status_code=500, detail="failed to sign quality variant")
    return signed_url


def run_job(job_id: str, url: str, uploaded_source_path: Optional[str] = None) -> None:
    _log_job_event(job_id, "job_started")
    _set_job_state(job_id, status="processing", persist_db=True)

    try:
        pipeline_result = _run_pipeline_with_retries(job_id, url)
        local_clips_raw = pipeline_result.get("clips", [])
        run_dir_raw = str(pipeline_result.get("run_dir", "") or "")

        local_clips = local_clips_raw if isinstance(local_clips_raw, list) else []
        _set_job_state(job_id, run_dir=run_dir_raw, persist_db=True)

        storage_paths: List[str] = []
        upload_failures = 0
        _log_job_event(job_id, "stage_start", stage="upload", clip_count=len(local_clips))
        for clip_path in local_clips:
            local_file = _resolve_local_path(str(clip_path))
            uploaded = _upload_clip_to_supabase(job_id, local_file)
            if uploaded:
                storage_paths.append(str(uploaded))
            else:
                upload_failures += 1

        if storage_paths:
            error_message = None
            if upload_failures > 0:
                error_message = f"partial_upload_failure:{upload_failures}"
            _set_job_state(
                job_id,
                status="completed",
                result=storage_paths,
                storage_paths=storage_paths,
                error=error_message,
                reason=None,
                persist_db=True,
            )
            _log_job_event(
                job_id,
                "job_completed",
                uploaded_clips=len(storage_paths),
                upload_failures=upload_failures,
            )
            return

        _set_job_state(
            job_id,
            status="failed",
            result=[],
            storage_paths=[],
            error="no_clips_uploaded",
            reason="no_clips_uploaded",
            persist_db=True,
        )
        _log_job_event(
            job_id,
            "job_failed",
            reason="no_clips_uploaded",
            pipeline_clips=len(local_clips),
            upload_failures=upload_failures,
        )
    except Exception as exc:
        error_message = _extract_db_error(exc)
        failure_reason = _derive_failure_reason(error_message)
        _set_job_state(
            job_id,
            status="failed",
            result=[],
            storage_paths=[],
            error=error_message,
            reason=failure_reason,
            persist_db=True,
        )
        _log_job_event(
            job_id,
            "job_failed",
            reason=failure_reason or error_message,
            error=error_message,
        )
    finally:
        if uploaded_source_path:
            _cleanup_uploaded_source(uploaded_source_path)


def _try_reserve_inflight_slot(job_id: str) -> bool:
    with _inflight_jobs_lock:
        if len(_inflight_jobs) >= MAX_INFLIGHT_JOBS:
            return False
        _inflight_jobs.add(job_id)
        return True


def _release_inflight_slot(job_id: str) -> None:
    with _inflight_jobs_lock:
        _inflight_jobs.discard(job_id)


def _run_job_with_slot(
    job_id: str, url: str, uploaded_source_path: Optional[str] = None
) -> None:
    try:
        run_job(job_id, url, uploaded_source_path=uploaded_source_path)
    finally:
        _release_inflight_slot(job_id)


def cleanup_worker() -> None:
    while True:
        time.sleep(CLEANUP_INTERVAL_SECONDS)
        try:
            now_ts = time.time()
            expired: List[Dict[str, Any]] = []
            with _jobs_lock:
                for job_id, job in jobs.items():
                    status = str(job.get("status", ""))
                    if status not in TERMINAL_STATUSES:
                        continue
                    updated_ts = float(job.get("updated_ts", job.get("created_ts", now_ts)))
                    if (now_ts - updated_ts) >= TTL_SECONDS:
                        expired.append(
                            {
                                "job_id": job_id,
                                "run_dir": job.get("run_dir"),
                                "storage_paths": list(job.get("storage_paths") or []),
                            }
                        )

            for item in expired:
                storage_paths = [
                    str(p).strip()
                    for p in (item.get("storage_paths") or [])
                    if str(p).strip()
                ]
                if storage_paths and supabase:
                    try:
                        supabase.storage.from_(SUPABASE_BUCKET).remove(storage_paths)
                    except Exception as exc:
                        logger.warning(
                            "supabase cleanup failed for job_id=%s: %s",
                            item.get("job_id"),
                            _extract_db_error(exc),
                        )

                run_dir_raw = str(item.get("run_dir") or "").strip()
                if run_dir_raw:
                    try:
                        run_dir = _resolve_local_path(run_dir_raw)
                        if run_dir.exists() and run_dir.is_dir():
                            shutil.rmtree(run_dir)
                    except Exception as exc:
                        logger.warning(
                            "cleanup failed for run_dir=%s: %s",
                            run_dir_raw,
                            _extract_db_error(exc),
                        )
                with _jobs_lock:
                    jobs.pop(str(item["job_id"]), None)
        except Exception as exc:
            logger.warning("cleanup_worker iteration failed: %s", _extract_db_error(exc))


@app.post("/generate")
def generate_clips(
    req: GenerateRequest,
    background_tasks: BackgroundTasks,
    request: Request,
) -> Dict[str, Any]:
    _enforce_generate_rate_limit(request)
    _ensure_rights_confirmed(req.user_confirmed_rights)
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="url is required")
    _ensure_public_source_url_allowed(url)

    job_id = uuid.uuid4().hex
    if not _try_reserve_inflight_slot(job_id):
        raise HTTPException(
            status_code=429,
            detail=f"too many in-flight jobs ({MAX_INFLIGHT_JOBS})",
        )

    _insert_job_in_db(job_id, status="processing")
    _set_job_state(job_id, status="processing", result=[], persist_db=True)
    try:
        _job_executor.submit(_run_job_with_slot, job_id, url)
    except Exception as exc:
        _release_inflight_slot(job_id)
        error_message = _extract_db_error(exc)
        failure_reason = _derive_failure_reason(error_message)
        _set_job_state(
            job_id,
            status="failed",
            result=[],
            storage_paths=[],
            error=error_message,
            reason=failure_reason,
            persist_db=True,
        )
        raise HTTPException(status_code=503, detail="job scheduler unavailable")

    _log_job_event(job_id, "job_queued")
    return {"job_id": job_id}


if HAS_MULTIPART:
    from fastapi import File, Form, UploadFile

    @app.post("/generate/upload")
    def generate_clips_from_upload(
        background_tasks: BackgroundTasks,
        request: Request,
        user_confirmed_rights: bool = Form(...),
        video_file: UploadFile = File(..., alias="videoFile"),
    ) -> Dict[str, Any]:
        _enforce_generate_rate_limit(request)
        _ensure_rights_confirmed(user_confirmed_rights)

        job_id = uuid.uuid4().hex
        uploaded_source_path = _save_uploaded_source(job_id, video_file)

        if not _try_reserve_inflight_slot(job_id):
            _cleanup_uploaded_source(uploaded_source_path)
            raise HTTPException(
                status_code=429,
                detail=f"too many in-flight jobs ({MAX_INFLIGHT_JOBS})",
            )

        _insert_job_in_db(job_id, status="processing")
        _set_job_state(job_id, status="processing", result=[], persist_db=True)
        try:
            _job_executor.submit(
                _run_job_with_slot,
                job_id,
                uploaded_source_path,
                uploaded_source_path,
            )
        except Exception as exc:
            _release_inflight_slot(job_id)
            _cleanup_uploaded_source(uploaded_source_path)
            error_message = _extract_db_error(exc)
            failure_reason = _derive_failure_reason(error_message)
            _set_job_state(
                job_id,
                status="failed",
                result=[],
                storage_paths=[],
                error=error_message,
                reason=failure_reason,
                persist_db=True,
            )
            raise HTTPException(status_code=503, detail="job scheduler unavailable")

        _log_job_event(job_id, "job_queued", source="upload")
        return {"job_id": job_id}
else:

    @app.post("/generate/upload")
    def generate_clips_from_upload_unavailable(request: Request) -> Dict[str, Any]:
        _enforce_generate_rate_limit(request)
        raise HTTPException(
            status_code=503,
            detail=(
                "File upload is unavailable because python-multipart is not installed. "
                "Install it with: pip install python-multipart"
            ),
        )


@app.get("/status/{job_id}")
def get_status(job_id: str) -> Dict[str, Any]:
    with _jobs_lock:
        job = jobs.get(job_id)
        if job:
            return {
                "job_id": job_id,
                "status": str(job.get("status", "unknown")),
                "created_at": job.get("created_at"),
            }

    rebuilt = _rehydrate_job_from_db(job_id)
    if rebuilt:
        return {
            "job_id": job_id,
            "status": str(rebuilt.get("status", "unknown")),
            "created_at": rebuilt.get("created_at"),
        }
    raise HTTPException(status_code=404, detail="job not found")


@app.get("/result/{job_id}")
def get_result(job_id: str) -> Dict[str, Any]:
    with _jobs_lock:
        job = jobs.get(job_id)
        if job:
            status = str(job.get("status", "unknown"))
            if status != "completed":
                response: Dict[str, Any] = {
                    "job_id": job_id,
                    "status": status,
                }
                reason = _normalize_job_reason(job.get("reason"))
                if status == "failed":
                    response["reason"] = reason or _derive_failure_reason(
                        _normalize_job_error(job.get("error"))
                    )
                return response
            raw_paths = job.get("storage_paths") or job.get("result") or []
            storage_paths = _to_unique_storage_paths(raw_paths)
            clips = _build_result_clip_entries(job_id, storage_paths)
            return {"job_id": job_id, "status": "completed", "clips": clips}

    rebuilt = _rehydrate_job_from_db(job_id)
    if rebuilt:
        status = str(rebuilt.get("status", "unknown"))
        if status != "completed":
            response = {
                "job_id": job_id,
                "status": status,
            }
            reason = _normalize_job_reason(rebuilt.get("reason"))
            if status == "failed":
                response["reason"] = reason or _derive_failure_reason(
                    _normalize_job_error(rebuilt.get("error"))
                )
            return response
        raw_paths = rebuilt.get("storage_paths") or rebuilt.get("result") or []
        storage_paths = _to_unique_storage_paths(raw_paths)
        clips = _build_result_clip_entries(job_id, storage_paths)
        return {"job_id": job_id, "status": "completed", "clips": clips}
    raise HTTPException(status_code=404, detail="job not found")


@app.post("/download/options")
def get_download_options(
    payload: DownloadOptionsRequest,
    x_internal_token: str = Header(default="", alias="x-internal-token"),
) -> Dict[str, Any]:
    _require_internal_token(x_internal_token)
    job_id = str(payload.job_id or "").strip()
    clip_id = str(payload.clip_id or "").strip()
    if not job_id or not clip_id:
        raise HTTPException(status_code=400, detail="job_id and clip_id are required")

    with _jobs_lock:
        job = jobs.get(job_id)
    if not job:
        job = _rehydrate_job_from_db(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if str(job.get("status", "unknown")) != "completed":
        raise HTTPException(status_code=409, detail="job not completed")

    source_path = _resolve_job_source_clip_path(job_id, clip_id)
    source_height = _ffprobe_video_height(source_path)
    available_map = _available_qualities_for_height(source_height)
    options = [
        {
            "quality": quality,
            "available": bool(available_map.get(quality, False)),
            "height": QUALITY_HEIGHTS[quality],
        }
        for quality in QUALITY_ORDER
    ]
    return {
        "job_id": job_id,
        "clip_id": clip_id,
        "source_height": source_height,
        "options": options,
    }


@app.post("/download/request")
def request_download_quality(
    payload: DownloadRequest,
    x_internal_token: str = Header(default="", alias="x-internal-token"),
) -> Dict[str, Any]:
    _require_internal_token(x_internal_token)
    job_id = str(payload.job_id or "").strip()
    clip_id = str(payload.clip_id or "").strip()
    quality = _normalize_quality(payload.quality)
    if not job_id or not clip_id:
        raise HTTPException(status_code=400, detail="job_id and clip_id are required")

    with _jobs_lock:
        job = jobs.get(job_id)
    if not job:
        job = _rehydrate_job_from_db(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if str(job.get("status", "unknown")) != "completed":
        raise HTTPException(status_code=409, detail="job not completed")

    signed_url = _ensure_quality_variant_signed_url(job_id, clip_id, quality)
    return {
        "job_id": job_id,
        "clip_id": clip_id,
        "quality": quality,
        "download_url": signed_url,
    }


@app.post("/feedback")
def feedback(request: Request, payload: Dict[str, Any]) -> Dict[str, Any]:
    _enforce_feedback_rate_limit(request)
    _validate_feedback_payload_limits(payload)
    clip_id = str(payload.get("clip_id", "")).strip()
    if not clip_id:
        raise HTTPException(status_code=400, detail="clip_id is required")

    record = dict(payload)
    record.setdefault("received_at", _utc_now_iso())
    save_clip_feedback(record)
    return {"success": True}


def _mount_static_dirs() -> None:
    # Intentionally disabled: local run files should not be publicly exposed.
    return


_validate_backend_env_or_fail()
_init_supabase_client()
_mount_static_dirs()
threading.Thread(target=cleanup_worker, daemon=True).start()

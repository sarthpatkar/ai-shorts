import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse

import yt_dlp


_BROWSER_COOKIE_SPEC_RE = re.compile(
    r"""(?x)
    (?P<name>[^+:]+)
    (?:\s*\+\s*(?P<keyring>[^:]+))?
    (?:\s*:\s*(?!:)(?P<profile>.+?))?
    (?:\s*::\s*(?P<container>.+))?
    """
)


def _parse_js_runtimes(raw_value: str) -> Dict[str, Dict[str, str]]:
    value = str(raw_value or "").strip()
    runtimes: Dict[str, Dict[str, str]] = {}

    if value:
        for token in value.split(","):
            item = token.strip()
            if not item:
                continue
            name, path = (item.split(":", 1) + [""])[:2]
            runtime_name = name.strip().lower()
            runtime_path = path.strip()
            if not runtime_name:
                continue
            runtimes[runtime_name] = {"path": runtime_path} if runtime_path else {}

    if not runtimes:
        runtimes["deno"] = {}
        node_path = shutil.which("node")
        if node_path:
            runtimes["node"] = {"path": node_path}
    return runtimes


def _parse_cookies_from_browser(
    raw_value: str,
) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    value = str(raw_value or "").strip()
    match = _BROWSER_COOKIE_SPEC_RE.fullmatch(value)
    if not match:
        raise ValueError("invalid_YTDLP_COOKIES_FROM_BROWSER")

    browser_name = str(match.group("name") or "").strip().lower()
    profile = match.group("profile")
    keyring = match.group("keyring")
    container = match.group("container")

    normalized_profile = str(profile).strip() if profile else None
    normalized_keyring = str(keyring).strip().upper() if keyring else None
    normalized_container = str(container).strip() if container else None
    return (
        browser_name,
        normalized_profile,
        normalized_keyring,
        normalized_container,
    )


def _normalize_download_error(exc: Exception) -> str:
    message = str(exc).strip()
    lowered = message.lower()

    if "http error 429" in lowered or "status code: 429" in lowered:
        return "http_429_too_many_requests"
    if "http error 403" in lowered or "status code: 403" in lowered:
        return "http_403_forbidden"
    if "sign in to confirm" in lowered and "not a bot" in lowered:
        return (
            "youtube_bot_check_required: configure YTDLP_COOKIES_FROM_BROWSER "
            "(for example, chrome) or YTDLP_COOKIES_FILE, then retry"
        )
    if "private video" in lowered:
        return "youtube_private_video: this video requires authentication"
    if "video unavailable" in lowered:
        return "youtube_video_unavailable: this video is unavailable"
    if "unsupported url" in lowered:
        return "unsupported_url"
    if any(
        token in lowered
        for token in (
            "timed out",
            "temporary failure in name resolution",
            "name or service not known",
            "network is unreachable",
            "connection reset",
            "remote end closed connection",
            "connection refused",
            "tlsv1 alert",
            "ssl:",
            "eof occurred in violation of protocol",
        )
    ):
        return f"network_error: {message or 'network_failure'}"
    if any(
        token in lowered
        for token in (
            "unable to download webpage",
            "service unavailable",
            "internal server error",
            "bad gateway",
            "gateway timeout",
            "extractor error",
        )
    ):
        return f"transient_error: {message or 'transient_download_failure'}"
    return message or exc.__class__.__name__


def _is_nonempty_file(path: Path) -> bool:
    try:
        return path.exists() and path.is_file() and path.stat().st_size > 0
    except Exception:
        return False


def _resolve_local_source_path(raw_source: str) -> Optional[Path]:
    source = str(raw_source or "").strip()
    if not source:
        return None

    if source.lower().startswith("file://"):
        parsed = urlparse(source)
        candidate = Path(unquote(parsed.path)).expanduser().resolve()
        return candidate if candidate.exists() and candidate.is_file() else None

    candidate = Path(source).expanduser()
    if candidate.exists() and candidate.is_file():
        return candidate.resolve()
    return None


def _copy_local_source(source: Path, target: Path) -> str:
    if not source.exists() or not source.is_file():
        raise RuntimeError("uploaded_source_missing_or_invalid")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(source, target)
    except Exception as exc:
        raise RuntimeError(f"uploaded_source_copy_failed: {exc}") from exc
    if not _is_nonempty_file(target):
        raise RuntimeError("uploaded_source_empty_or_unreadable")
    return str(target)


def _build_ydl_opts(
    target: Path,
    *,
    quality_format: str,
    include_cookies: bool,
    cookies_file: str,
    cookies_from_browser: str,
) -> Dict[str, Any]:
    opts: Dict[str, Any] = {
        "format": quality_format,
        "format_sort": [
            "res:1080",
            "fps",
            "vcodec:av01",
            "vcodec:vp9.2",
            "vcodec:vp9",
            "vcodec:h264",
            "acodec:opus",
            "acodec:aac",
            "br",
        ],
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "outtmpl": str(target),
        "js_runtimes": _parse_js_runtimes(os.environ.get("YTDLP_JS_RUNTIMES", "")),
    }

    if include_cookies and cookies_file:
        opts["cookiefile"] = str(Path(cookies_file).expanduser())
    if include_cookies and cookies_from_browser:
        opts["cookiesfrombrowser"] = _parse_cookies_from_browser(
            cookies_from_browser
        )
    return opts


def _download_once(url: str, ydl_opts: Dict[str, Any]) -> None:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def download_video(url: str, output_path: str) -> str:
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    local_source = _resolve_local_source_path(url)
    if local_source is not None:
        return _copy_local_source(local_source, target)

    normal_quality_format = os.environ.get(
        "YTDLP_FORMAT",
        "bv*[height<=1080][vcodec!=none]+ba[acodec!=none]/b[height<=1080]/best",
    )
    fallback_quality_format = os.environ.get(
        "YTDLP_FALLBACK_FORMAT",
        "bv*[height<=720][vcodec!=none]+ba[acodec!=none]/b[height<=720]/best[height<=720]/best",
    )
    cookies_file = str(os.environ.get("YTDLP_COOKIES_FILE", "")).strip()
    cookies_from_browser = str(
        os.environ.get("YTDLP_COOKIES_FROM_BROWSER", "")
    ).strip()
    cookies_available = bool(cookies_file or cookies_from_browser)

    strategy_steps: List[Tuple[str, bool, str]] = [
        ("normal", False, normal_quality_format),
    ]
    if cookies_available:
        strategy_steps.append(("cookies", True, normal_quality_format))
    strategy_steps.append(
        ("lower_quality", cookies_available, fallback_quality_format)
    )

    seen = set()
    deduped_steps: List[Tuple[str, bool, str]] = []
    for label, with_cookies, quality in strategy_steps:
        key = (with_cookies, quality)
        if key in seen:
            continue
        seen.add(key)
        deduped_steps.append((label, with_cookies, quality))

    last_error: Optional[str] = None
    for _, with_cookies, quality in deduped_steps:
        if target.exists():
            try:
                target.unlink()
            except Exception:
                pass
        ydl_opts = _build_ydl_opts(
            target=target,
            quality_format=quality,
            include_cookies=with_cookies,
            cookies_file=cookies_file,
            cookies_from_browser=cookies_from_browser,
        )
        try:
            _download_once(url, ydl_opts)
            if _is_nonempty_file(target):
                return str(target)
            last_error = "downloaded_file_missing_or_empty"
        except Exception as exc:
            last_error = _normalize_download_error(exc)

    if target.exists() and not _is_nonempty_file(target):
        try:
            target.unlink()
        except Exception:
            pass
    raise RuntimeError(last_error or "download_failed")

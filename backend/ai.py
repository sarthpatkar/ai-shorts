import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from threading import Lock, Semaphore
from typing import Any, Dict, List, Sequence

from dotenv import load_dotenv

from cache import get_cached_ai_response, set_cached_ai_response

try:
    from groq import Groq
except ImportError:
    Groq = None

load_dotenv()
logger = logging.getLogger(__name__)

_client = None
if Groq is not None and os.environ.get("GROQ_API_KEY"):
    _client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

_EXECUTOR = ThreadPoolExecutor(max_workers=1)
_INFLIGHT = Semaphore(max(1, int(os.environ.get("AI_MAX_INFLIGHT", "1"))))
_RATE_LOCK = Lock()
_LAST_REQUEST_TS = 0.0
_TOKEN_LOCK = Lock()
_TOKEN_WINDOW_START = 0.0
_TOKEN_WINDOW_USED = 0

_METRICS_LOCK = Lock()
_METRICS = {
    "ai_calls_count": 0,
    "ai_judge_calls_count": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "fallback_usage": 0,
    "rate_limit_waits": 0,
}


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _track_metric(name: str, delta: int = 1) -> None:
    with _METRICS_LOCK:
        _METRICS[name] = int(_METRICS.get(name, 0)) + delta


def get_ai_metrics_snapshot() -> Dict[str, int]:
    with _METRICS_LOCK:
        return {k: int(v) for k, v in _METRICS.items()}


def _normalize_text(text: str, max_chars: int = 340) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())[:max_chars]


def _safe_caption_from_text(text: str) -> str:
    words = _normalize_text(text, max_chars=200).split()[:12]
    if len(words) < 3:
        words = ["WATCH", "THIS", "NOW"]
    return " ".join(words).upper()


def fallback_enrichment(text: str, error: str = "") -> Dict[str, Any]:
    clean = _normalize_text(text)
    first_words = clean.split()[:10]
    improved_hook = " ".join(first_words).strip()
    if not improved_hook:
        improved_hook = "WATCH THIS MOMENT"

    output = {
        "hook": improved_hook,
        "caption": _safe_caption_from_text(clean),
        "pacing": {
            "start_trim": 0.0,
            "end_trim": 0.0,
            "cut_style": "normal",
        },
        "title": "VIRAL MOMENT",
        "thumbnail_text": "WATCH THIS",
        "source": "fallback",
        "error": error,
        "raw_response": "",
    }
    _track_metric("fallback_usage", 1)
    return output


def _clamp_trim(value: Any) -> float:
    try:
        num = float(value)
    except Exception:
        num = 0.0
    return round(max(0.0, min(2.0, num)), 3)


def _clamp_score(value: Any, default: float = 5.0) -> float:
    try:
        numeric = float(value)
    except Exception:
        numeric = default
    return round(max(0.0, min(10.0, numeric)), 3)


def _validate_enrichment_item(item: Dict[str, Any], fallback_text: str) -> Dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError("enrichment_item_not_object")

    hook = str(item.get("hook", "")).strip()
    caption = str(item.get("caption", "")).strip()
    pacing = item.get("pacing")
    if not hook:
        raise ValueError("missing_hook")
    if not caption:
        raise ValueError("missing_caption")
    if not isinstance(pacing, dict):
        raise ValueError("missing_pacing")

    caption_words = caption.split()
    if len(caption_words) < 3 or len(caption_words) > 12 or re.search(r"[!?]{3,}", caption):
        caption = _safe_caption_from_text(fallback_text)
    else:
        caption = " ".join(caption_words).upper()

    cut_style = str(pacing.get("cut_style", "normal")).strip().lower()
    if cut_style not in {"fast", "normal"}:
        cut_style = "normal"

    title = str(item.get("title", "")).strip()
    thumbnail_text = str(item.get("thumbnail_text", "")).strip()

    return {
        "hook": hook,
        "caption": caption,
        "pacing": {
            "start_trim": _clamp_trim(pacing.get("start_trim", 0.0)),
            "end_trim": _clamp_trim(pacing.get("end_trim", 0.0)),
            "cut_style": cut_style,
        },
        "title": " ".join(title.split()[:8]) if title else "",
        "thumbnail_text": " ".join(thumbnail_text.split()[:6]).upper() if thumbnail_text else "",
        "source": "groq",
    }


def _build_judge_prompt(texts: Sequence[str]) -> str:
    body_lines = []
    for i, txt in enumerate(texts, start=1):
        body_lines.append(f"Clip {i}: {txt}")
    clips_block = "\n".join(body_lines)

    return f"""
You are a pragmatic short-form video editor optimizing for quality and throughput.

For each input clip, output ONLY a JSON array in exactly the same order.

Input clips:
{clips_block}

Return strictly:
[
  {{
    "hook": "improved opening hook",
    "caption": "UP TO 12 WORDS, ALL CAPS",
    "pacing": {{
      "start_trim": 0.0,
      "end_trim": 0.0,
      "cut_style": "fast" or "normal"
    }},
    "scores": {{
      "hook_strength": 0-10,
      "emotional_impact": 0-10,
      "standalone_clarity": 0-10,
      "curiosity_retention": 0-10,
      "retention_potential": 0-10,
      "narrative_completeness": 0-10,
      "payoff_satisfaction": 0-10
    }},
    "overall_score": 0-10,
    "pass": true or false,
    "reason": "short reason"
  }}
]

Rules:
- Prioritize retention_potential first, then hook_strength, then overall engagement.
- A clip can pass even if narrative_completeness or payoff_satisfaction is weaker.
- Prefer "good enough and engaging" over "perfect but rare".
- Return STRICT JSON ARRAY ONLY.
- NO markdown, NO code fences, NO extra text.
""".strip()


def _build_duration_prompt(clips: Sequence[Dict[str, Any]]) -> str:
    rows: List[str] = []
    for idx, clip in enumerate(clips, start=1):
        rows.append(
            "Clip {idx}: start={start:.3f}, end={end:.3f}, duration={dur:.3f}, "
            "retention={ret:.2f}, hook={hook:.2f}, confidence={conf:.2f}, text=\"{text}\"".format(
                idx=idx,
                start=float(clip.get("start", 0.0)),
                end=float(clip.get("end", 0.0)),
                dur=float(clip.get("duration", 0.0)),
                ret=float(clip.get("retention_score", 0.0)),
                hook=float(clip.get("hook_score", 0.0)),
                conf=float(clip.get("confidence_score", 0.0)),
                text=_normalize_text(str(clip.get("text", "")), max_chars=300),
            )
        )
    clips_block = "\n".join(rows)
    return f"""
You are optimizing short-form clips for retention and completion.
Return STRICT JSON ARRAY in the same order as input clips.

Input clips:
{clips_block}

Return:
[
  {{
    "quality_score": 0-10,
    "suggested_duration_seconds": 15-59,
    "start_trim_seconds": 0-6,
    "end_trim_seconds": 0-6,
    "reason": "short reason"
  }}
]

Rules:
- Prioritize retention potential and hook over narrative perfection.
- Avoid mid-sentence cuts; favor complete thoughts.
- Keep outputs compact and deterministic.
- JSON array only. No markdown.
""".strip()


def _validate_judge_item(item: Dict[str, Any], fallback_text: str) -> Dict[str, Any]:
    base = _validate_enrichment_item(item, fallback_text=fallback_text)
    raw_scores = item.get("scores", {}) if isinstance(item.get("scores"), dict) else {}
    hook_strength = _clamp_score(raw_scores.get("hook_strength", item.get("hook_strength", 5.0)))
    emotional_impact = _clamp_score(raw_scores.get("emotional_impact", item.get("emotional_impact", 5.0)))
    standalone_clarity = _clamp_score(raw_scores.get("standalone_clarity", item.get("standalone_clarity", 5.0)))
    curiosity_retention = _clamp_score(raw_scores.get("curiosity_retention", item.get("curiosity_retention", 5.0)))
    retention_potential = _clamp_score(raw_scores.get("retention_potential", item.get("retention_potential", curiosity_retention)))
    narrative_completeness = _clamp_score(raw_scores.get("narrative_completeness", item.get("narrative_completeness", standalone_clarity)))
    payoff_satisfaction = _clamp_score(raw_scores.get("payoff_satisfaction", item.get("payoff_satisfaction", standalone_clarity)))
    inferred_overall = (
        (hook_strength * 0.24)
        + (emotional_impact * 0.12)
        + (standalone_clarity * 0.12)
        + (curiosity_retention * 0.17)
        + (retention_potential * 0.23)
        + (narrative_completeness * 0.06)
        + (payoff_satisfaction * 0.06)
    )
    overall = _clamp_score(item.get("overall_score", inferred_overall), default=inferred_overall)
    strong_threshold = float(os.environ.get("AI_JUDGE_MIN_SCORE", "4.8"))
    soft_threshold = float(os.environ.get("AI_JUDGE_SOFT_MIN_SCORE", "4.8"))
    hook_floor = float(os.environ.get("AI_JUDGE_MIN_COMPONENT", "4.7"))
    raw_pass = item.get("pass", overall >= strong_threshold)
    pass_flag = raw_pass if isinstance(raw_pass, bool) else _as_bool(raw_pass, default=overall >= strong_threshold)
    retention_floor = float(os.environ.get("AI_JUDGE_RETENTION_MIN_COMPONENT", "4.6"))
    soft_blend = _clamp_score(
        (retention_potential * 0.50)
        + (hook_strength * 0.30)
        + (curiosity_retention * 0.10)
        + (overall * 0.10)
    )
    if not pass_flag:
        pass_flag = bool(
            overall >= strong_threshold
            or soft_blend >= soft_threshold
            or (retention_potential >= retention_floor and hook_strength >= hook_floor)
        )
    if retention_potential < max(3.2, retention_floor - 1.4):
        pass_flag = False
    reason = _normalize_text(str(item.get("reason", "")), max_chars=160)

    return {
        **base,
        "scores": {
            "hook_strength": hook_strength,
            "emotional_impact": emotional_impact,
            "standalone_clarity": standalone_clarity,
            "curiosity_retention": curiosity_retention,
            "retention_potential": retention_potential,
            "narrative_completeness": narrative_completeness,
            "payoff_satisfaction": payoff_satisfaction,
            "retention_hook_blend": soft_blend,
        },
        "overall_score": overall,
        "pass": pass_flag,
        "reason": reason,
        "source": "groq_judge",
    }


def _validate_duration_item(item: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError("duration_item_not_object")
    quality_score = _clamp_score(item.get("quality_score", 5.0))
    suggested_duration_seconds = max(
        15.0,
        min(59.0, round(float(item.get("suggested_duration_seconds", 30.0)), 3)),
    )
    start_trim_seconds = max(0.0, min(6.0, round(float(item.get("start_trim_seconds", 0.0)), 3)))
    end_trim_seconds = max(0.0, min(6.0, round(float(item.get("end_trim_seconds", 0.0)), 3)))
    reason = _normalize_text(str(item.get("reason", "")), max_chars=120)
    return {
        "quality_score": quality_score,
        "suggested_duration_seconds": suggested_duration_seconds,
        "start_trim_seconds": start_trim_seconds,
        "end_trim_seconds": end_trim_seconds,
        "reason": reason,
        "source": "groq_duration",
    }


def fallback_judge(text: str, error: str = "") -> Dict[str, Any]:
    try:
        from scoring import score_breakdown

        breakdown = score_breakdown(text)
        hook_strength = _clamp_score(breakdown.get("hook_score", 4.5))
        emotional_impact = _clamp_score(breakdown.get("emotional_score", 4.5))
        standalone_clarity = _clamp_score(breakdown.get("clarity_score", 4.5))
        curiosity_retention = _clamp_score(
            (float(breakdown.get("curiosity_score", 4.5)) + float(breakdown.get("retention_score", 4.5))) / 2.0
        )
        retention_potential = _clamp_score((float(breakdown.get("retention_score", 4.5)) * 0.75) + (curiosity_retention * 0.25))
        narrative_completeness = _clamp_score((standalone_clarity * 0.65) + (retention_potential * 0.35))
        payoff_satisfaction = _clamp_score((retention_potential * 0.55) + (float(breakdown.get("hook_score", 4.5)) * 0.25) + (float(breakdown.get("curiosity_score", 4.5)) * 0.20))
        inferred = (
            (hook_strength * 0.24)
            + (emotional_impact * 0.12)
            + (standalone_clarity * 0.12)
            + (curiosity_retention * 0.17)
            + (retention_potential * 0.23)
            + (narrative_completeness * 0.06)
            + (payoff_satisfaction * 0.06)
        )
        overall = _clamp_score(inferred)
    except Exception:
        hook_strength, emotional_impact, standalone_clarity, curiosity_retention, retention_potential, narrative_completeness, payoff_satisfaction, overall = (
            4.8,
            4.8,
            5.5,
            4.8,
            4.8,
            5.0,
            4.8,
            5.0,
        )

    strong_threshold = float(os.environ.get("AI_JUDGE_MIN_SCORE", "4.8"))
    soft_threshold = float(os.environ.get("AI_JUDGE_SOFT_MIN_SCORE", "4.8"))
    hook_floor = float(os.environ.get("AI_JUDGE_MIN_COMPONENT", "4.7"))
    retention_floor = float(os.environ.get("AI_JUDGE_RETENTION_MIN_COMPONENT", "4.6"))
    soft_blend = _clamp_score(
        (retention_potential * 0.50)
        + (hook_strength * 0.30)
        + (curiosity_retention * 0.10)
        + (overall * 0.10)
    )
    pass_flag = bool(
        overall >= strong_threshold
        or soft_blend >= soft_threshold
        or (retention_potential >= retention_floor and hook_strength >= hook_floor)
    )
    if retention_potential < max(3.2, retention_floor - 1.4):
        pass_flag = False
    fallback = fallback_enrichment(text, error=error or "judge_fallback")
    return {
        **fallback,
        "scores": {
            "hook_strength": hook_strength,
            "emotional_impact": emotional_impact,
            "standalone_clarity": standalone_clarity,
            "curiosity_retention": curiosity_retention,
            "retention_potential": retention_potential,
            "narrative_completeness": narrative_completeness,
            "payoff_satisfaction": payoff_satisfaction,
            "retention_hook_blend": soft_blend,
        },
        "overall_score": overall,
        "pass": pass_flag,
        "reason": error[:160] if error else "heuristic_fallback",
        "source": "fallback_judge",
    }


def _rate_limit_wait() -> None:
    global _LAST_REQUEST_TS
    min_interval = float(os.environ.get("AI_MIN_INTERVAL_SECONDS", "0.6"))
    with _RATE_LOCK:
        now = time.monotonic()
        wait = max(0.0, min_interval - (now - _LAST_REQUEST_TS))
        if wait > 0:
            _track_metric("rate_limit_waits", 1)
            logger.info(json.dumps({"event": "ai_rate_limit_wait", "wait_seconds": round(wait, 3)}, ensure_ascii=True))
            time.sleep(wait)
        _LAST_REQUEST_TS = time.monotonic()


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 4))


def _token_budget_wait(prompt: str, max_output_tokens: int) -> None:
    global _TOKEN_WINDOW_START, _TOKEN_WINDOW_USED

    tpm_limit = int(os.environ.get("AI_TPM_LIMIT", "5500"))
    need = _estimate_tokens(prompt) + int(max_output_tokens)
    if tpm_limit <= 0:
        return

    with _TOKEN_LOCK:
        now = time.monotonic()
        if _TOKEN_WINDOW_START <= 0 or now - _TOKEN_WINDOW_START >= 60.0:
            _TOKEN_WINDOW_START = now
            _TOKEN_WINDOW_USED = 0

        projected = _TOKEN_WINDOW_USED + need
        if projected > tpm_limit:
            sleep_for = max(0.0, 60.0 - (now - _TOKEN_WINDOW_START))
            if sleep_for > 0:
                _track_metric("rate_limit_waits", 1)
                logger.info(
                    json.dumps(
                        {"event": "ai_token_budget_wait", "sleep_seconds": round(sleep_for, 3), "need_tokens": need},
                        ensure_ascii=True,
                    )
                )
                time.sleep(sleep_for)
            _TOKEN_WINDOW_START = time.monotonic()
            _TOKEN_WINDOW_USED = 0

        _TOKEN_WINDOW_USED += need


def _build_batch_prompt(texts: Sequence[str]) -> str:
    body_lines = []
    for i, txt in enumerate(texts, start=1):
        body_lines.append(f"Clip {i}: {txt}")
    clips_block = "\n".join(body_lines)

    return f"""
You are an expert in viral short-form video optimization.

For each clip, return ONLY a JSON array in the same order as input.

Input clips:
{clips_block}

Return strictly:
[
  {{
    "hook": "improved hook",
    "caption": "UP TO 12 WORDS, ALL CAPS",
    "pacing": {{
      "start_trim": 0.0,
      "end_trim": 0.0,
      "cut_style": "fast" or "normal"
    }},
    "title": "optional <=8 words",
    "thumbnail_text": "optional <=6 words"
  }}
]

Rules:
- Return STRICT JSON ARRAY ONLY.
- NO markdown.
- NO code fences.
- NO explanations.
- Keep deterministic.
""".strip()


def _strip_markdown_fences(raw: str) -> str:
    content = (raw or "").strip()
    if not content.startswith("```"):
        return content

    content = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", content)
    content = re.sub(r"\s*```$", "", content.strip())
    return content.strip()


def _extract_partial_json_array(content: str) -> List[Dict[str, Any]]:
    start = content.find("[")
    if start < 0:
        return []

    body = content[start + 1 :]
    decoder = json.JSONDecoder()
    idx = 0
    items: List[Dict[str, Any]] = []

    while idx < len(body):
        while idx < len(body) and body[idx] in " \n\r\t,":
            idx += 1
        if idx >= len(body) or body[idx] == "]":
            break
        try:
            obj, new_idx = decoder.raw_decode(body, idx)
        except json.JSONDecodeError:
            break
        if isinstance(obj, dict):
            items.append(obj)
        idx = new_idx

    return items


def safe_extract_json(raw: str) -> List[Dict[str, Any]]:
    content = _strip_markdown_fences(raw)
    if not content:
        return []

    try:
        parsed = json.loads(content)
    except Exception:
        start = content.find("[")
        end = content.rfind("]")
        if start < 0 or end <= start:
            partial = _extract_partial_json_array(content)
            if partial:
                logger.warning(
                    json.dumps(
                        {"event": "ai_partial_json_recovered", "items": len(partial)},
                        ensure_ascii=True,
                    )
                )
                return partial
            return []
        try:
            parsed = json.loads(content[start : end + 1])
        except Exception:
            partial = _extract_partial_json_array(content[start : end + 1])
            if partial:
                return partial
            return []

    if not isinstance(parsed, list):
        return []
    return [x for x in parsed if isinstance(x, dict)]


def _request_batch_with_retry(texts: Sequence[str]) -> List[Dict[str, Any]]:
    if _client is None:
        raise RuntimeError("groq_not_configured")

    model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
    timeout_seconds = float(os.environ.get("AI_TIMEOUT_SECONDS", "18"))
    retries = max(1, int(os.environ.get("AI_RETRIES", "2")))
    consistent_mode = _as_bool(os.environ.get("CONSISTENT_MODE", "1"), True)

    prompt = _build_batch_prompt(texts)
    per_clip_tokens = int(os.environ.get("AI_OUTPUT_TOKENS_PER_CLIP", "95"))
    max_output_cap = int(os.environ.get("AI_MAX_OUTPUT_TOKENS_CAP", "700"))
    max_output_tokens = max(240, min(max_output_cap, 80 + (per_clip_tokens * len(texts))))

    for attempt in range(retries):
        try:
            _rate_limit_wait()
            _token_budget_wait(prompt, max_output_tokens=max_output_tokens)
            with _INFLIGHT:
                _track_metric("ai_calls_count", 1)
                kwargs: Dict[str, Any] = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": max_output_tokens,
                }
                if consistent_mode:
                    kwargs["seed"] = 42

                future = _EXECUTOR.submit(lambda: _client.chat.completions.create(**kwargs))
                response = future.result(timeout=timeout_seconds)

            raw = response.choices[0].message.content if response.choices else ""
            logger.info(json.dumps({"event": "ai_raw_response", "raw": raw or ""}, ensure_ascii=True))
            items = safe_extract_json(raw or "")
            if not items:
                raise ValueError("ai_items_empty")
            if len(items) < len(texts):
                logger.warning(
                    json.dumps(
                        {
                            "event": "ai_items_partial",
                            "expected": len(texts),
                            "received": len(items),
                        },
                        ensure_ascii=True,
                    )
                )
            return items[: len(texts)]
        except FuturesTimeoutError as exc:
            err = f"timeout_{timeout_seconds}s"
            logger.warning(json.dumps({"event": "ai_timeout", "attempt": attempt + 1, "error": err}, ensure_ascii=True))
            last_exc = exc
        except Exception as exc:
            logger.warning(json.dumps({"event": "ai_error", "attempt": attempt + 1, "error": str(exc)}, ensure_ascii=True))
            last_exc = exc

        if attempt < retries - 1:
            backoff = float(os.environ.get("AI_BACKOFF_BASE_SECONDS", "1.0")) * (2**attempt)
            logger.info(json.dumps({"event": "ai_backoff", "seconds": backoff, "attempt": attempt + 1}, ensure_ascii=True))
            time.sleep(backoff)

    raise RuntimeError(str(last_exc))


def _request_duration_batch_with_retry(clips: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if _client is None:
        raise RuntimeError("groq_not_configured")

    model = os.environ.get("GROQ_DURATION_MODEL", os.environ.get("GROQ_JUDGE_MODEL", os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")))
    timeout_seconds = float(os.environ.get("AI_DURATION_TIMEOUT_SECONDS", os.environ.get("AI_TIMEOUT_SECONDS", "18")))
    retries = max(1, int(os.environ.get("AI_DURATION_RETRIES", "2")))
    consistent_mode = _as_bool(os.environ.get("CONSISTENT_MODE", "1"), True)
    prompt = _build_duration_prompt(clips)
    max_output_tokens = max(220, min(int(os.environ.get("AI_MAX_OUTPUT_TOKENS_CAP", "700")), 90 + (85 * len(clips))))

    for attempt in range(retries):
        try:
            _rate_limit_wait()
            _token_budget_wait(prompt, max_output_tokens=max_output_tokens)
            with _INFLIGHT:
                _track_metric("ai_judge_calls_count", 1)
                kwargs: Dict[str, Any] = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": max_output_tokens,
                }
                if consistent_mode:
                    kwargs["seed"] = 42
                future = _EXECUTOR.submit(lambda: _client.chat.completions.create(**kwargs))
                response = future.result(timeout=timeout_seconds)

            raw = response.choices[0].message.content if response.choices else ""
            items = safe_extract_json(raw or "")
            if not items:
                raise ValueError("ai_duration_items_empty")
            return items[: len(clips)]
        except FuturesTimeoutError:
            logger.warning(
                json.dumps(
                    {"event": "ai_duration_timeout", "attempt": attempt + 1, "timeout_seconds": timeout_seconds},
                    ensure_ascii=True,
                )
            )
        except Exception as exc:
            logger.warning(json.dumps({"event": "ai_duration_error", "attempt": attempt + 1, "error": str(exc)}, ensure_ascii=True))
        if attempt < retries - 1:
            time.sleep(float(os.environ.get("AI_BACKOFF_BASE_SECONDS", "1.0")) * (2**attempt))
    raise RuntimeError("ai_duration_failed")


def _request_judge_batch_with_retry(texts: Sequence[str]) -> List[Dict[str, Any]]:
    if _client is None:
        raise RuntimeError("groq_not_configured")

    model = os.environ.get("GROQ_JUDGE_MODEL", os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"))
    timeout_seconds = float(os.environ.get("AI_TIMEOUT_SECONDS", "18"))
    retries = max(1, int(os.environ.get("AI_RETRIES", "2")))
    consistent_mode = _as_bool(os.environ.get("CONSISTENT_MODE", "1"), True)

    prompt = _build_judge_prompt(texts)
    per_clip_tokens = int(os.environ.get("AI_JUDGE_OUTPUT_TOKENS_PER_CLIP", "120"))
    max_output_cap = int(os.environ.get("AI_MAX_OUTPUT_TOKENS_CAP", "700"))
    max_output_tokens = max(260, min(max_output_cap, 120 + (per_clip_tokens * len(texts))))

    for attempt in range(retries):
        try:
            _rate_limit_wait()
            _token_budget_wait(prompt, max_output_tokens=max_output_tokens)
            with _INFLIGHT:
                _track_metric("ai_judge_calls_count", 1)
                kwargs: Dict[str, Any] = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": max_output_tokens,
                }
                if consistent_mode:
                    kwargs["seed"] = 42

                future = _EXECUTOR.submit(lambda: _client.chat.completions.create(**kwargs))
                response = future.result(timeout=timeout_seconds)

            raw = response.choices[0].message.content if response.choices else ""
            logger.info(json.dumps({"event": "ai_judge_raw_response", "raw": raw or ""}, ensure_ascii=True))
            items = safe_extract_json(raw or "")
            if not items:
                raise ValueError("ai_judge_items_empty")
            if len(items) < len(texts):
                logger.warning(
                    json.dumps(
                        {
                            "event": "ai_judge_items_partial",
                            "expected": len(texts),
                            "received": len(items),
                        },
                        ensure_ascii=True,
                    )
                )
            return items[: len(texts)]
        except FuturesTimeoutError as exc:
            err = f"timeout_{timeout_seconds}s"
            logger.warning(json.dumps({"event": "ai_judge_timeout", "attempt": attempt + 1, "error": err}, ensure_ascii=True))
            last_exc = exc
        except Exception as exc:
            logger.warning(json.dumps({"event": "ai_judge_error", "attempt": attempt + 1, "error": str(exc)}, ensure_ascii=True))
            last_exc = exc

        if attempt < retries - 1:
            backoff = float(os.environ.get("AI_BACKOFF_BASE_SECONDS", "1.0")) * (2**attempt)
            logger.info(json.dumps({"event": "ai_judge_backoff", "seconds": backoff, "attempt": attempt + 1}, ensure_ascii=True))
            time.sleep(backoff)

    raise RuntimeError(str(last_exc))


def batch_ai_request(texts: Sequence[str]) -> List[Dict[str, Any]]:
    normalized = [_normalize_text(text, max_chars=int(os.environ.get("AI_MAX_INPUT_CHARS", "340"))) for text in texts]
    if not normalized:
        return []

    results: List[Dict[str, Any]] = [None] * len(normalized)  # type: ignore[assignment]
    missing_indices: List[int] = []
    missing_texts: List[str] = []

    for idx, text in enumerate(normalized):
        cached = get_cached_ai_response(text)
        if cached:
            _track_metric("cache_hits", 1)
            results[idx] = dict(cached)
        else:
            _track_metric("cache_misses", 1)
            missing_indices.append(idx)
            missing_texts.append(text)

    if not missing_texts:
        return [dict(item) for item in results if isinstance(item, dict)]

    max_batch = max(1, int(os.environ.get("AI_BATCH_MAX_CLIPS", "4")))
    chunks = [missing_texts[i : i + max_batch] for i in range(0, len(missing_texts), max_batch)]

    pointer = 0
    for batch in chunks:
        try:
            raw_items = _request_batch_with_retry(batch)
            for local_idx, raw_item in enumerate(raw_items):
                text = batch[local_idx]
                validated = _validate_enrichment_item(raw_item, fallback_text=text)
                validated["raw_response"] = json.dumps(raw_item, ensure_ascii=True)
                set_cached_ai_response(text, validated)

                global_idx = missing_indices[pointer + local_idx]
                results[global_idx] = validated
        except Exception as exc:
            logger.warning(json.dumps({"event": "ai_batch_failed", "error": str(exc)}, ensure_ascii=True))
            for local_idx, text in enumerate(batch):
                fallback = fallback_enrichment(text, error=str(exc))
                set_cached_ai_response(text, fallback)
                global_idx = missing_indices[pointer + local_idx]
                results[global_idx] = fallback
        pointer += len(batch)

    final = []
    for idx, item in enumerate(results):
        if isinstance(item, dict):
            final.append(dict(item))
        else:
            final.append(fallback_enrichment(normalized[idx], error="missing_result"))
    return final


def batch_ai_judge_request(texts: Sequence[str]) -> List[Dict[str, Any]]:
    normalized = [_normalize_text(text, max_chars=int(os.environ.get("AI_MAX_INPUT_CHARS", "340"))) for text in texts]
    if not normalized:
        return []

    results: List[Dict[str, Any]] = [None] * len(normalized)  # type: ignore[assignment]
    missing_map: Dict[str, List[int]] = {}

    for idx, text in enumerate(normalized):
        cache_key = f"judge:v1::{text}"
        cached = get_cached_ai_response(cache_key)
        if cached and isinstance(cached.get("scores"), dict):
            _track_metric("cache_hits", 1)
            results[idx] = dict(cached)
        else:
            _track_metric("cache_misses", 1)
            missing_map.setdefault(text, []).append(idx)

    missing_texts = list(missing_map.keys())
    if not missing_texts:
        return [dict(item) for item in results if isinstance(item, dict)]

    max_batch = max(1, int(os.environ.get("AI_JUDGE_BATCH_MAX_CLIPS", "3")))
    chunks = [missing_texts[i : i + max_batch] for i in range(0, len(missing_texts), max_batch)]

    for batch in chunks:
        try:
            raw_items = _request_judge_batch_with_retry(batch)
            for local_idx, text in enumerate(batch):
                raw_item = raw_items[local_idx] if local_idx < len(raw_items) and isinstance(raw_items[local_idx], dict) else {}
                validated = _validate_judge_item(raw_item, fallback_text=text)
                validated["raw_response"] = json.dumps(raw_item, ensure_ascii=True)
                set_cached_ai_response(f"judge:v1::{text}", validated)
                for global_idx in missing_map.get(text, []):
                    results[global_idx] = dict(validated)
        except Exception as exc:
            logger.warning(json.dumps({"event": "ai_judge_batch_failed", "error": str(exc)}, ensure_ascii=True))
            for text in batch:
                fallback = fallback_judge(text, error=str(exc))
                set_cached_ai_response(f"judge:v1::{text}", fallback)
                for global_idx in missing_map.get(text, []):
                    results[global_idx] = dict(fallback)

    final = []
    for idx, item in enumerate(results):
        if isinstance(item, dict):
            final.append(dict(item))
        else:
            final.append(fallback_judge(normalized[idx], error="missing_judge_result"))
    return final


def analyze_clip_with_ai(text: str) -> Dict[str, Any]:
    results = batch_ai_request([text])
    if results:
        return results[0]
    return fallback_enrichment(text, error="empty_batch_result")


def batch_ai_duration_request(clips: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not clips:
        return []
    if not _as_bool(os.environ.get("AI_DURATION_ENHANCER_ENABLED", "1"), True):
        return []
    try:
        raw_items = _request_duration_batch_with_retry(clips)
    except Exception as exc:
        logger.warning(json.dumps({"event": "ai_duration_skipped", "error": str(exc)}, ensure_ascii=True))
        return []

    output: List[Dict[str, Any]] = []
    for idx in range(len(clips)):
        item = raw_items[idx] if idx < len(raw_items) else {}
        try:
            output.append(_validate_duration_item(item))
        except Exception:
            output.append(
                {
                    "quality_score": 0.0,
                    "suggested_duration_seconds": max(15.0, min(59.0, float(clips[idx].get("duration", 20.0)))),
                    "start_trim_seconds": 0.0,
                    "end_trim_seconds": 0.0,
                    "reason": "invalid_duration_item",
                    "source": "duration_fallback",
                }
            )
    return output

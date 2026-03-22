import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ab_testing import generate_variants
from ai import batch_ai_duration_request, batch_ai_judge_request, fallback_judge, get_ai_metrics_snapshot
from captions import create_ass_for_clip, create_static_ass_file
from chunk import create_chunks
from cutter import render_vertical_clip, subtitles_filter_available
from download import download_video
from feedback_store import load_all_feedback, save_clip_feedback
from hooks import pace_chunk
from learning_engine import (
    confidence_score as learning_confidence_score,
    load_clip_memory,
    load_filter_rules,
    load_selection_weights,
    memory_similarity,
    simulate_metric_bundle,
    update_learning_from_feedback,
)
from scoring import score_breakdown, smart_compress
from transcribe import extract_audio, flatten_words, transcribe_audio

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LAST_RUN_METADATA: Dict[str, Any] = {}
_RUN_REJECTION_LOG: List[Dict[str, Any]] = []
STRONG_SIGNAL_TERMS = {
    "secret",
    "mistake",
    "truth",
    "warning",
    "never",
    "why",
    "how",
    "crazy",
    "insane",
    "shocking",
    "unbelievable",
    "nobody",
    "hidden",
    "revealed",
}
WEAK_START_PHRASES = {"um", "uh", "you know", "like", "so yeah", "anyway"}
FILLER_TERMS = {
    "um",
    "uh",
    "like",
    "basically",
    "actually",
    "you know",
    "kind of",
    "sort of",
    "i mean",
}
RESOLUTION_TERMS = {
    "so",
    "therefore",
    "that's why",
    "this is why",
    "which means",
    "the point is",
    "in short",
    "finally",
    "bottom line",
}
PAYOFF_TERMS = {
    "result",
    "outcome",
    "solution",
    "answer",
    "here's why",
    "this works",
    "this failed",
    "you should",
    "you can",
    "now you know",
    "the fix",
    "the reason",
}
TRAILING_INCOMPLETE_TERMS = {
    "and",
    "but",
    "because",
    "so",
    "then",
    "if",
    "when",
    "which",
}
VISUAL_CUE_TERMS = {"look", "watch", "see", "here", "this", "show", "shown", "example"}
NARRATIVE_CONNECTORS = {"because", "so", "then", "therefore", "however", "but", "which means"}


def _as_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _resolve_path(path_like: str) -> Path:
    path = Path(str(path_like or "")).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _file_exists_and_nonempty(path_like: str, min_bytes: int = 1) -> bool:
    path = _resolve_path(path_like)
    try:
        return path.exists() and path.is_file() and path.stat().st_size >= max(1, int(min_bytes))
    except Exception:
        return False


def _validate_media_file(path_like: str, stream_selector: str) -> bool:
    path = _resolve_path(path_like)
    if not _file_exists_and_nonempty(str(path), min_bytes=1):
        return False
    ffprobe_bin = os.environ.get("FFPROBE_BIN", "ffprobe").strip() or "ffprobe"
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        stream_selector,
        "-show_entries",
        "stream=codec_type",
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
            timeout=max(3, int(float(os.environ.get("FFPROBE_TIMEOUT_SECONDS", "10")))),
            check=False,
        )
        return proc.returncode == 0 and bool((proc.stdout or "").strip())
    except FileNotFoundError:
        # If ffprobe is unavailable, keep the non-empty file check as a fallback.
        return _file_exists_and_nonempty(str(path), min_bytes=1)
    except Exception:
        return False


def _ensure_binary_available(default_binary: str, env_var: str, error_code: str) -> None:
    binary = (os.environ.get(env_var, default_binary) or default_binary).strip()
    if shutil.which(binary):
        return
    raise RuntimeError(error_code)


def _is_non_retryable_download_error(error_message: Optional[str]) -> bool:
    lowered = str(error_message or "").lower()
    if not lowered:
        return False
    return any(
        token in lowered
        for token in (
            "youtube_blocked",
            "youtube_private_video",
        )
    )


def _is_retryable_download_error(error_message: Optional[str]) -> bool:
    lowered = str(error_message or "").lower()
    if not lowered:
        return False
    return any(
        token in lowered
        for token in (
            "network_error",
            "transient_error",
            "download_timeout",
            "timed out",
            "temporary failure",
            "connection reset",
            "service unavailable",
        )
    )


def _clamp_score(value: Any, low: float = 0.0, high: float = 10.0) -> float:
    try:
        num = float(value)
    except Exception:
        num = 0.0
    return round(max(low, min(high, num)), 3)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", _normalize_text(text).lower())


def _token_set(text: str) -> set:
    return {tok for tok in _tokenize(text) if len(tok) > 2}


def _count_terms(text: str, terms: Sequence[str]) -> int:
    lowered = _normalize_text(text).lower()
    return sum(1 for term in terms if re.search(rf"\b{re.escape(term)}\b", lowered))


def _contains_term(text: str, terms: Sequence[str]) -> bool:
    lowered = _normalize_text(text).lower()
    return any(re.search(rf"\b{re.escape(term)}\b", lowered) for term in terms)


def _split_text_windows(text: str) -> Tuple[str, str, str]:
    words = _normalize_text(text).split()
    if not words:
        return "", "", ""
    n = len(words)
    i1 = max(1, int(round(n * 0.25)))
    i2 = max(i1 + 1, int(round(n * 0.75)))
    i2 = min(i2, n)
    return (" ".join(words[:i1]), " ".join(words[i1:i2]), " ".join(words[i2:]))


def _split_time_windows(start: float, end: float) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    duration = max(0.8, end - start)
    first_cut = start + (duration * 0.25)
    second_cut = start + (duration * 0.75)
    return ((start, first_cut), (first_cut, second_cut), (second_cut, end))


def _segments_in_window(segments: Sequence[Dict[str, Any]], start: float, end: float) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for seg in segments:
        ss = _safe_float(seg.get("start", 0.0))
        se = _safe_float(seg.get("end", ss))
        if se <= start or ss >= end:
            continue
        selected.append(seg)
    selected.sort(key=lambda x: _safe_float(x.get("start", 0.0)))
    return selected


def _text_ends_sentence(text: str) -> bool:
    return bool(re.search(r"[.!?][\"']?$", _normalize_text(text)))


def _snap_end_to_sentence(
    start: float,
    proposed_end: float,
    hard_end: float,
    transcript_segments: Sequence[Dict[str, Any]],
    min_duration: float,
) -> float:
    candidates: List[float] = []
    for seg in transcript_segments:
        seg_end = _safe_float(seg.get("end", 0.0))
        if seg_end < (start + min_duration) or seg_end > hard_end:
            continue
        if _text_ends_sentence(str(seg.get("text", ""))):
            candidates.append(seg_end)
    if not candidates:
        return max(start + min_duration, min(hard_end, proposed_end))
    return min(candidates, key=lambda value: abs(value - proposed_end))


def _apply_ai_duration_enhancements(
    selected: List[Dict[str, Any]],
    transcript_segments: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not selected:
        return selected

    max_video_end = max((_safe_float(seg.get("end", 0.0)) for seg in transcript_segments), default=0.0)
    payload: List[Dict[str, Any]] = []
    for clip in selected:
        comps = clip.get("score_components", {}) if isinstance(clip.get("score_components"), dict) else {}
        start = _safe_float(clip.get("start", 0.0))
        end = _safe_float(clip.get("end", start + 1.0))
        payload.append(
            {
                "text": _normalize_text(str(clip.get("text", ""))),
                "start": start,
                "end": end,
                "duration": max(0.8, end - start),
                "retention_score": _safe_float(comps.get("retention_score", 0.0)),
                "hook_score": _safe_float(comps.get("hook_score", 0.0)),
                "confidence_score": _safe_float(clip.get("confidence_score", comps.get("confidence_score", 0.0))),
            }
        )

    suggestions = batch_ai_duration_request(payload)
    if not suggestions:
        _log_event("ai_duration_skipped", reason="no_suggestions")
        return selected

    updated: List[Dict[str, Any]] = []
    adjusted = 0
    for idx, clip in enumerate(selected):
        suggestion = suggestions[idx] if idx < len(suggestions) else {}
        quality_score = _safe_float(suggestion.get("quality_score", 0.0))
        if quality_score <= 0:
            updated.append(clip)
            continue

        orig_start = _safe_float(clip.get("start", 0.0))
        orig_end = _safe_float(clip.get("end", orig_start + 1.0))
        hard_end = min(
            orig_start + float(os.environ.get("PACED_MAX_CLIP_SECONDS", "59.0")),
            max_video_end if max_video_end > 0 else orig_end,
        )
        min_duration = float(os.environ.get("PACED_MIN_CLIP_SECONDS", "15.0"))
        target_duration = max(15.0, min(59.0, _safe_float(suggestion.get("suggested_duration_seconds", orig_end - orig_start))))
        start_trim = max(0.0, min(6.0, _safe_float(suggestion.get("start_trim_seconds", 0.0))))
        end_trim = max(0.0, min(6.0, _safe_float(suggestion.get("end_trim_seconds", 0.0))))

        new_start = max(0.0, orig_start + start_trim)
        if new_start > (orig_end - min_duration):
            new_start = orig_start
        proposed_end = max(new_start + min_duration, orig_end - end_trim)
        current_duration = proposed_end - new_start
        if current_duration > (target_duration + 1.0):
            proposed_end = new_start + target_duration
        elif current_duration < (target_duration - 1.0):
            proposed_end = min(hard_end, new_start + target_duration)
        snapped_end = _snap_end_to_sentence(
            start=new_start,
            proposed_end=proposed_end,
            hard_end=hard_end,
            transcript_segments=transcript_segments,
            min_duration=min_duration,
        )
        if snapped_end <= new_start:
            snapped_end = min(hard_end, max(new_start + min_duration, proposed_end))
        if snapped_end - new_start < min_duration:
            snapped_end = min(hard_end, new_start + min_duration)
        if snapped_end - new_start > 59.0:
            snapped_end = new_start + 59.0

        changed = abs(new_start - orig_start) > 0.15 or abs(snapped_end - orig_end) > 0.35
        if changed:
            adjusted += 1
        updated.append(
            {
                **clip,
                "start": round(new_start, 3),
                "end": round(snapped_end, 3),
                "ai_duration": suggestion,
            }
        )

    _log_event(
        "ai_duration_enhancement",
        clips=len(selected),
        adjusted=adjusted,
        suggestions=len(suggestions),
    )
    return updated


def _window_audio_metrics(segments: Sequence[Dict[str, Any]], start: float, end: float) -> Dict[str, float]:
    duration = max(0.2, end - start)
    ordered = _segments_in_window(segments, start, end)
    if not ordered:
        return {
            "audio_score": 0.0,
            "speech_coverage": 0.0,
            "wps": 0.0,
            "long_pauses": 1.0,
            "pause_ratio": 1.0,
        }

    long_pause_threshold = float(os.environ.get("RETENTION_LONG_PAUSE_SEC", "0.9"))
    speech_seconds = 0.0
    total_pause = 0.0
    long_pauses = 0
    cursor = start
    word_count = 0

    for seg in ordered:
        ss = max(start, _safe_float(seg.get("start", start)))
        se = min(end, _safe_float(seg.get("end", ss)))
        if se <= ss:
            continue
        if ss > cursor:
            gap = ss - cursor
            total_pause += gap
            if gap >= long_pause_threshold:
                long_pauses += 1
        speech_seconds += max(0.0, se - ss)
        cursor = max(cursor, se)
        word_count += len(_normalize_text(str(seg.get("text", ""))).split())

    if cursor < end:
        tail = end - cursor
        total_pause += tail
        if tail >= long_pause_threshold:
            long_pauses += 1

    speech_coverage = max(0.0, min(1.0, speech_seconds / duration))
    pause_ratio = max(0.0, min(1.0, total_pause / duration))
    wps = word_count / max(0.15, speech_seconds)
    tempo_target = float(os.environ.get("RETENTION_TARGET_WPS", "2.9"))
    tempo_tolerance = float(os.environ.get("RETENTION_WPS_TOLERANCE", "2.2"))
    tempo_score = max(0.0, 1.0 - (abs(wps - tempo_target) / max(0.5, tempo_tolerance)))
    coverage_score = min(1.0, speech_coverage / 0.82)
    pause_penalty = min(1.0, (pause_ratio * 0.75) + (min(4.0, float(long_pauses)) * 0.12))
    audio_score = _clamp_score(((tempo_score * 0.48) + (coverage_score * 0.52) - (pause_penalty * 0.55)) * 10.0)
    return {
        "audio_score": audio_score,
        "speech_coverage": round(speech_coverage, 3),
        "wps": round(wps, 3),
        "long_pauses": float(long_pauses),
        "pause_ratio": round(pause_ratio, 3),
    }


def _retention_curve_analysis(
    text: str,
    start: float,
    end: float,
    transcript_segments: Sequence[Dict[str, Any]],
    base_breakdown: Dict[str, float],
) -> Dict[str, Any]:
    start_text, mid_text, end_text = _split_text_windows(text)
    (ts0, te0), (ts1, te1), (ts2, te2) = _split_time_windows(start, end)
    start_audio = _window_audio_metrics(transcript_segments, ts0, te0)
    mid_audio = _window_audio_metrics(transcript_segments, ts1, te1)
    end_audio = _window_audio_metrics(transcript_segments, ts2, te2)

    start_tokens = _token_set(start_text)
    mid_tokens = _token_set(mid_text)
    end_tokens = _token_set(end_text)
    mid_new_info_ratio = len(mid_tokens - start_tokens) / max(1, len(mid_tokens))
    mid_repeat_ratio = len(mid_tokens & start_tokens) / max(1, len(mid_tokens))
    end_novelty_ratio = len(end_tokens - (start_tokens | mid_tokens)) / max(1, len(end_tokens))

    start_emotional_hits = _count_terms(start_text, list(STRONG_SIGNAL_TERMS))
    mid_emotional_hits = _count_terms(mid_text, list(STRONG_SIGNAL_TERMS))
    tonal_variation = min(1.0, abs(mid_emotional_hits - start_emotional_hits) / 2.5)
    filler_hits_mid = _count_terms(mid_text, list(FILLER_TERMS))
    filler_hits_end = _count_terms(end_text, list(FILLER_TERMS))
    filler_hits_total = filler_hits_mid + filler_hits_end

    mid_engagement_score = _clamp_score(
        2.1
        + (mid_new_info_ratio * 3.2)
        + ((1.0 - mid_repeat_ratio) * 2.0)
        + (tonal_variation * 1.3)
        + ((mid_audio["audio_score"] / 10.0) * 2.2)
        - min(2.4, filler_hits_mid * 0.6)
    )
    if len(_normalize_text(mid_text).split()) < 8:
        mid_engagement_score = _clamp_score(mid_engagement_score - 1.2)

    resolution_hits = _count_terms(end_text, list(RESOLUTION_TERMS))
    payoff_hits = _count_terms(end_text, list(PAYOFF_TERMS))
    end_curiosity = "?" in end_text or _contains_term(end_text, ["next", "what if", "why"])
    complete_end = bool(re.search(r"[.!?][\"']?$", _normalize_text(end_text)))
    end_tokens_raw = _tokenize(end_text)
    tail_token = end_tokens_raw[-1] if end_tokens_raw else ""
    ends_incomplete = tail_token in TRAILING_INCOMPLETE_TERMS
    abrupt_end = len(_normalize_text(end_text).split()) < 6
    ending_score = _clamp_score(
        2.0
        + min(2.6, resolution_hits * 1.15)
        + min(2.6, payoff_hits * 1.0)
        + (1.2 if end_curiosity else 0.0)
        + (1.0 if complete_end else -0.5)
        + (end_novelty_ratio * 0.9)
        - (1.5 if abrupt_end else 0.0)
        - (1.6 if ends_incomplete else 0.0)
        - min(1.4, filler_hits_end * 0.45)
    )

    start_audio_score = start_audio["audio_score"]
    mid_audio_score = mid_audio["audio_score"]
    end_audio_score = end_audio["audio_score"]
    tail_audio = (mid_audio_score * 0.55) + (end_audio_score * 0.45)
    audio_energy_dip = max(0.0, start_audio_score - tail_audio)
    no_new_info = 1.0 - mid_new_info_ratio
    dropoff_risk_score = _clamp_score(
        (float(base_breakdown.get("dropoff_risk", 0.8)) * 3.2)
        + (audio_energy_dip * 0.85)
        + ((mid_audio["pause_ratio"] + end_audio["pause_ratio"]) * 2.8)
        + (no_new_info * 2.4)
        + (mid_repeat_ratio * 1.9)
        + min(2.0, filler_hits_total * 0.32)
        + (1.0 if ending_score < 4.5 else 0.0)
    )

    hook_norm = _clamp_score(float(base_breakdown.get("hook_score", 5.0))) / 10.0
    build_norm = min(
        1.0,
        (mid_new_info_ratio * 0.45)
        + ((mid_engagement_score / 10.0) * 0.4)
        + ((1.0 - mid_repeat_ratio) * 0.15),
    )
    payoff_norm = min(
        1.0,
        ((ending_score / 10.0) * 0.66)
        + min(1.0, payoff_hits / 2.0) * 0.2
        + (0.14 if end_curiosity else 0.0),
    )
    connector_hits = _count_terms(text, list(NARRATIVE_CONNECTORS))
    flow_norm = min(1.0, connector_hits / 3.0)
    story_structure_score = _clamp_score(((hook_norm * 0.3) + (build_norm * 0.35) + (payoff_norm * 0.3) + (flow_norm * 0.05)) * 10.0)
    if build_norm < 0.34:
        story_structure_score = _clamp_score(story_structure_score - 1.0)
    if payoff_norm < 0.34:
        story_structure_score = _clamp_score(story_structure_score - 1.2)

    retention_score = _clamp_score(
        (float(base_breakdown.get("hook_score", 5.0)) * 0.22)
        + (mid_engagement_score * 0.36)
        + (ending_score * 0.24)
        + (story_structure_score * 0.18)
        - (max(0.0, dropoff_risk_score - 4.0) * 0.42)
    )

    segs = _segments_in_window(transcript_segments, start, end)
    seg_count = len(segs)
    visual_term_hits = _count_terms(text, list(VISUAL_CUE_TERMS))
    pacing_variation = 0.0
    if seg_count > 1:
        durations = [max(0.05, _safe_float(seg.get("end", 0.0)) - _safe_float(seg.get("start", 0.0))) for seg in segs]
        mean_dur = sum(durations) / max(1, len(durations))
        variation = sum(abs(d - mean_dur) for d in durations) / max(1, len(durations))
        pacing_variation = min(1.0, variation / max(0.3, mean_dur))

    visual_score = _clamp_score(
        3.8
        + (min(1.0, visual_term_hits / 3.0) * 2.2)
        + (pacing_variation * 1.8)
        + (min(1.0, seg_count / 7.0) * 1.2)
        + (end_novelty_ratio * 1.0)
    )

    audio_score = _clamp_score((start_audio_score * 0.30) + (mid_audio_score * 0.45) + (end_audio_score * 0.25) - (audio_energy_dip * 0.35))

    return {
        "mid_engagement_score": mid_engagement_score,
        "ending_score": ending_score,
        "dropoff_risk_score": dropoff_risk_score,
        "story_structure_score": story_structure_score,
        "retention_score": retention_score,
        "audio_score": audio_score,
        "visual_score": visual_score,
        "mid_new_info_ratio": round(mid_new_info_ratio, 3),
        "mid_repeat_ratio": round(mid_repeat_ratio, 3),
        "end_curiosity": bool(end_curiosity),
        "end_payoff_hits": int(payoff_hits),
        "audio_energy_dip": round(audio_energy_dip, 3),
        "start_audio_score": round(start_audio_score, 3),
        "mid_audio_score": round(mid_audio_score, 3),
        "end_audio_score": round(end_audio_score, 3),
        "start_text": start_text,
        "mid_text": mid_text,
        "end_text": end_text,
    }


def _compose_final_score(
    heuristic_score: float,
    hook_score: float,
    audio_score: float,
    visual_score: float,
    retention_score: float,
    selection_weights: Optional[Dict[str, float]] = None,
) -> float:
    weights = selection_weights or {
        "heuristic_score": 0.30,
        "hook_score": 0.15,
        "audio_score": 0.20,
        "visual_score": 0.10,
        "retention_score": 0.25,
    }
    return _clamp_score(
        (heuristic_score * float(weights.get("heuristic_score", 0.30)))
        + (hook_score * float(weights.get("hook_score", 0.15)))
        + (audio_score * float(weights.get("audio_score", 0.20)))
        + (visual_score * float(weights.get("visual_score", 0.10)))
        + (retention_score * float(weights.get("retention_score", 0.25)))
    )


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except Exception:
        return str(path.resolve())


def _create_run_context() -> Dict[str, Any]:
    run_id = os.environ.get("RUN_ID", "").strip() or uuid.uuid4().hex[:12]
    runs_root = Path(os.environ.get("RUNS_DIR", "runs")).expanduser()
    run_dir = (runs_root / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_id": run_id,
        "runs_root": runs_root.resolve(),
        "runs_root_display": _display_path(runs_root.resolve()),
        "run_dir": run_dir,
        "run_dir_display": _display_path(run_dir),
    }


def _set_last_run_metadata(run_id: str, run_dir: Path, outputs: List[str]) -> None:
    global _LAST_RUN_METADATA
    _LAST_RUN_METADATA = {
        "run_id": str(run_id),
        "run_dir": _display_path(run_dir),
        "outputs": list(outputs),
    }


def get_last_run_metadata() -> Dict[str, Any]:
    return dict(_LAST_RUN_METADATA)


def run_pipeline_with_metadata(
    url: str,
    top_k: int = 3,
    whisper_model: Optional[str] = None,
    whisper_fast: Optional[bool] = None,
) -> Dict[str, Any]:
    clips = run_pipeline(url=url, top_k=top_k, whisper_model=whisper_model, whisper_fast=whisper_fast)
    meta = get_last_run_metadata()
    return {
        "run_id": str(meta.get("run_id", "")),
        "run_dir": str(meta.get("run_dir", "")),
        "clips": list(clips),
    }


def _log_event(event: str, **kwargs: Any) -> None:
    logger.info(json.dumps({"event": event, **kwargs}, ensure_ascii=True))


def _log_stage(stage: str, started_at: float, **details: Any) -> None:
    _log_event("stage_complete", stage=stage, seconds=round(time.perf_counter() - started_at, 3), **details)


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


def _run_stage_with_retry(
    stage: str,
    fn: Callable[[], Any],
    *,
    retries: int,
    retry_delay_seconds: float,
    timeout_seconds: float = 0.0,
) -> Tuple[Any, Optional[str]]:
    attempts = max(1, retries + 1)
    download_retry_cap = 2
    last_error: Optional[str] = None
    for attempt in range(1, attempts + 1):
        started_at = time.perf_counter()
        _log_event("stage_start", stage=stage, attempt=attempt, retries=retries)
        try:
            result = _call_with_timeout(fn, timeout_seconds=timeout_seconds)
            _log_stage(stage, started_at, attempt=attempt, retries=retries)
            return result, None
        except FutureTimeout:
            last_error = f"{stage}_timeout"
        except Exception as exc:
            last_error = str(exc)
        _log_event(
            "stage_failed",
            stage=stage,
            attempt=attempt,
            retries=retries,
            error=last_error,
        )
        if attempt < attempts:
            if stage == "download" and _is_non_retryable_download_error(last_error):
                _log_event(
                    "stage_abort",
                    stage=stage,
                    attempt=attempt,
                    retries=retries,
                    reason="non_retryable",
                    error=last_error,
                )
                break
            if stage == "download" and _is_retryable_download_error(last_error):
                max_attempts_for_retryable = min(attempts, download_retry_cap + 1)
                if attempt >= max_attempts_for_retryable:
                    _log_event(
                        "stage_abort",
                        stage=stage,
                        attempt=attempt,
                        retries=retries,
                        reason="retry_cap_reached",
                        error=last_error,
                    )
                    break
            _log_event(
                "stage_retry",
                stage=stage,
                next_attempt=attempt + 1,
                delay_seconds=round(retry_delay_seconds, 2),
            )
            time.sleep(max(0.1, retry_delay_seconds))
    return None, last_error or f"{stage}_failed"


def _clip_preview(text: str, max_chars: int = 160) -> str:
    normalized = _normalize_text(text)
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 1]}..."


def _log_clip_rejection(layer: str, reason: str, clip: Dict[str, Any], **details: Any) -> None:
    global _RUN_REJECTION_LOG
    rejection_record = {
        "layer": layer,
        "reason": reason,
        "start": round(_safe_float(clip.get("start", 0.0)), 3),
        "end": round(_safe_float(clip.get("end", 0.0)), 3),
        "text": _normalize_text(str(clip.get("text", "")))[:1800],
        "text_preview": _clip_preview(str(clip.get("text", ""))),
        "details": dict(details),
        "timestamp": int(time.time()),
    }
    _RUN_REJECTION_LOG.append(rejection_record)
    _log_event(
        "clip_rejected",
        layer=layer,
        reason=reason,
        start=round(_safe_float(clip.get("start", 0.0)), 3),
        end=round(_safe_float(clip.get("end", 0.0)), 3),
        text_preview=_clip_preview(str(clip.get("text", ""))),
        **details,
    )


def _is_near_duplicate(text_a: str, text_b: str, threshold: float = 0.78) -> bool:
    left = _token_set(text_a)
    right = _token_set(text_b)
    if not left or not right:
        return False
    overlap = len(left & right) / max(1, len(left | right))
    return overlap >= threshold


def _time_overlap_ratio(clip_a: Dict[str, Any], clip_b: Dict[str, Any]) -> float:
    a_start = _safe_float(clip_a.get("start", 0.0))
    a_end = _safe_float(clip_a.get("end", a_start))
    b_start = _safe_float(clip_b.get("start", 0.0))
    b_end = _safe_float(clip_b.get("end", b_start))
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    if inter <= 0:
        return 0.0
    a_len = max(0.2, a_end - a_start)
    b_len = max(0.2, b_end - b_start)
    return inter / max(a_len, b_len)


def _final_selection_rank_score(clip: Dict[str, Any]) -> float:
    comps = clip.get("score_components", {}) if isinstance(clip.get("score_components"), dict) else {}
    score = _safe_float(clip.get("score", 0.0))
    confidence = _safe_float(clip.get("confidence_score", comps.get("confidence_score", 0.0)))
    retention = _safe_float(comps.get("retention_score", 0.0))
    hook = _safe_float(comps.get("hook_score", 0.0))
    dropoff = _safe_float(comps.get("dropoff_risk", 0.0))

    start = _safe_float(clip.get("start", 0.0))
    end = _safe_float(clip.get("end", start))
    duration = max(0.2, end - start)

    duration_penalty = 0.0
    ideal_low = float(os.environ.get("IDEAL_CLIP_MIN_SECONDS", "15.0"))
    ideal_high = float(os.environ.get("IDEAL_CLIP_MAX_SECONDS", "52.0"))
    if duration < ideal_low:
        duration_penalty += min(0.7, (ideal_low - duration) * 0.06)
    elif duration > ideal_high:
        duration_penalty += min(0.7, (duration - ideal_high) * 0.03)

    confidence_penalty = max(0.0, 4.3 - confidence) * 0.18
    dropoff_penalty = max(0.0, dropoff - 5.4) * 0.12
    fallback_penalty = 0.18 if bool(clip.get("fallback", False)) else 0.0

    blended = (
        (score * 0.58)
        + (confidence * 0.20)
        + (retention * 0.14)
        + (hook * 0.08)
        - duration_penalty
        - confidence_penalty
        - dropoff_penalty
        - fallback_penalty
    )
    return round(blended, 4)


def _apply_diversity_filter(clips: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    near_duplicate_threshold = float(os.environ.get("TEXT_DUPLICATE_THRESHOLD", "0.78"))
    max_time_overlap = float(os.environ.get("MAX_CLIP_TIME_OVERLAP_RATIO", "0.55"))
    min_time_gap = max(0.0, float(os.environ.get("MIN_CLIP_TIME_GAP_SECONDS", "2.5")))
    min_rank_floor = float(os.environ.get("SELECTION_MIN_RANK_SCORE", "3.8"))

    selected: List[Dict[str, Any]] = []
    for clip in clips:
        text = str(clip.get("text", ""))
        rank_score = _safe_float(clip.get("selection_rank_score", _final_selection_rank_score(clip)))
        if selected and rank_score < min_rank_floor:
            continue

        if any(
            _is_near_duplicate(text, str(prev.get("text", "")), threshold=near_duplicate_threshold)
            for prev in selected
        ):
            continue

        overlap_rejected = False
        for prev in selected:
            overlap_ratio = _time_overlap_ratio(clip, prev)
            if overlap_ratio > max_time_overlap:
                overlap_rejected = True
                break

            clip_start = _safe_float(clip.get("start", 0.0))
            clip_end = _safe_float(clip.get("end", clip_start))
            prev_start = _safe_float(prev.get("start", 0.0))
            prev_end = _safe_float(prev.get("end", prev_start))
            # Ensure temporal spread so multiple clips are not near-identical moments.
            if min(abs(clip_start - prev_start), abs(clip_end - prev_end)) < min_time_gap:
                overlap_rejected = True
                break

        if overlap_rejected:
            continue

        selected.append(clip)
        if len(selected) >= limit:
            break
    return selected


def _safe_caption(text: str) -> str:
    words = _normalize_text(text).split()[:12]
    if len(words) < 3:
        words = ["WATCH", "THIS", "NOW"]
    return " ".join(words).upper()


def _merge_improved_text(original_text: str, improved_hook: str) -> str:
    clean_original = _normalize_text(original_text)
    clean_hook = _normalize_text(improved_hook)
    if not clean_hook:
        return clean_original
    if clean_hook.lower() in clean_original.lower():
        return clean_original
    return f"{clean_hook} {clean_original}".strip()


def _apply_safe_pacing(clip: Dict[str, Any]) -> Dict[str, Any]:
    ai = clip.get("ai", {}) if isinstance(clip.get("ai"), dict) else {}
    pacing = ai.get("pacing", {}) if isinstance(ai.get("pacing"), dict) else {}
    max_trim = float(os.environ.get("MAX_TRIM_SECONDS", "2.0"))
    min_duration = float(os.environ.get("MIN_CLIP_SECONDS", "15.0"))
    max_duration = float(os.environ.get("MAX_CLIP_SECONDS", "59.0"))

    start_raw = max(0.0, _safe_float(clip.get("start", 0.0)))
    end_raw = max(start_raw + 0.8, _safe_float(clip.get("end", start_raw + 1.0)))

    start_trim = min(max_trim, max(0.0, _safe_float(pacing.get("start_trim", 0.0))))
    end_trim = min(max_trim, max(0.0, _safe_float(pacing.get("end_trim", 0.0))))
    cut_style = str(pacing.get("cut_style", "normal")).strip().lower()
    if cut_style not in {"fast", "normal"}:
        cut_style = "normal"

    start = max(0.0, start_raw + start_trim)
    end = end_raw - end_trim

    if end <= start or (end - start) < min_duration:
        start = start_raw
        end = end_raw
    if (end - start) > max_duration:
        end = start + max_duration
    if end <= start:
        end = start + min_duration

    return {
        "start": round(start, 3),
        "end": round(end, 3),
        "preset": os.environ.get("FFMPEG_PRESET_FAST", "medium")
        if cut_style == "fast"
        else os.environ.get("FFMPEG_PRESET_NORMAL", "slow"),
        "cut_style": cut_style,
    }


def _words_in_window(words: List[Dict[str, object]], start: float, end: float) -> List[Dict[str, object]]:
    return [w for w in words if float(w.get("end", 0.0)) > start and float(w.get("start", 0.0)) < end]


def _render_clip_job(job: Dict[str, Any]) -> Dict[str, Any]:
    index = int(job["index"])
    clip = job["clip"]
    run_dir = Path(str(job["run_dir"])).expanduser().resolve()
    input_video = str(job["input_video"])
    transcript_segments = job["segments"]
    transcript_words = job["words"]

    input_video_path = Path(input_video).expanduser()
    if not input_video_path.exists() or not input_video_path.is_file():
        _log_event("render_skipped_missing_input", index=index, input_video=input_video)
        return {"index": index, "path": None}

    pacing = _apply_safe_pacing(clip)
    start = pacing["start"]
    end = pacing["end"]
    min_clip_duration = float(os.environ.get("MIN_CLIP_SECONDS", "15.0"))
    if end - start < min_clip_duration:
        _log_event("render_skipped_short_clip", index=index, duration=round(end - start, 3))
        return {"index": index, "path": None}

    output_path_abs = run_dir / f"clip_{index}.mp4"
    output_path_display = _display_path(output_path_abs)
    words = _words_in_window(transcript_words, start, end)
    speech_segments = [
        seg
        for seg in transcript_segments
        if float(seg.get("end", 0.0)) > start and float(seg.get("start", 0.0)) < end
    ]

    subtitle_path = None
    use_subtitles = subtitles_filter_available()
    caption_text = str(clip.get("caption", "")).strip() or _safe_caption(str(clip.get("text", "")))
    fallback_caption = _safe_caption(str(clip.get("original_text", "")))

    render_retries = max(0, int(os.environ.get("RENDER_RETRIES", "1")))
    render_retry_delay = max(0.2, float(os.environ.get("RENDER_RETRY_DELAY_SECONDS", "0.8")))

    try:
        if use_subtitles:
            try:
                subtitle_path, _ = create_ass_for_clip(
                    chunk_text=str(clip.get("text", "")),
                    hook_type=str(clip.get("hook_type", "statement")),
                    words=words,
                    clip_start=start,
                    clip_end=end,
                    tmp_dir=str(run_dir),
                    caption_override=caption_text,
                    inject_emoji=_as_bool(os.environ.get("CAPTION_INJECT_EMOJI", "0"), False),
                )
            except Exception as caption_exc:
                _log_event("caption_failed", index=index, error=str(caption_exc), fallback="static_ass")
                subtitle_path = create_static_ass_file(
                    text=fallback_caption,
                    duration=max(0.8, end - start),
                    tmp_dir=str(run_dir),
                )

        attempts = render_retries + 1
        rendered = False
        last_error = ""
        for attempt in range(1, attempts + 1):
            try:
                render_vertical_clip(
                    input_video=input_video,
                    start=start,
                    end=end,
                    output_video=str(output_path_abs),
                    subtitle_file=subtitle_path,
                    caption_text=caption_text,
                    speech_segments=speech_segments,
                    preset=pacing["preset"],
                    crf=int(os.environ.get("FFMPEG_CRF", "18")),
                )
                rendered = True
                break
            except Exception as ffmpeg_exc:
                last_error = str(ffmpeg_exc)
                _log_event(
                    "render_attempt_failed",
                    index=index,
                    attempt=attempt,
                    retries=render_retries,
                    error=last_error,
                )
                if attempt < attempts:
                    time.sleep(render_retry_delay)
        if not rendered:
            raise RuntimeError(last_error or "render_failed")
        _log_event("render_success", index=index, output=output_path_display)
        return {"index": index, "path": output_path_display}
    except Exception as ffmpeg_exc:
        _log_event("render_failed", index=index, error=str(ffmpeg_exc), start=start, end=end)
        return {"index": index, "path": None}
    finally:
        if subtitle_path and os.path.exists(subtitle_path):
            os.remove(subtitle_path)


def _pace_chunks(chunks: List[Dict[str, Any]], segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    paced: List[Dict[str, Any]] = []
    paced_min_duration = max(15.0, float(os.environ.get("PACED_MIN_CLIP_SECONDS", "15.0")))
    paced_max_duration = max(
        paced_min_duration + 1.0,
        float(os.environ.get("PACED_MAX_CLIP_SECONDS", "59.0")),
    )
    for chunk in chunks:
        try:
            paced.append(
                pace_chunk(
                    chunk,
                    segments,
                    min_duration=paced_min_duration,
                    max_duration=paced_max_duration,
                )
            )
        except Exception:
            paced.append(chunk)
    return paced


def filter_invalid_clips(
    chunks: List[Dict[str, Any]],
    min_duration: float = 15.0,
    min_words: int = 12,
    max_words: int = 140,
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for chunk in chunks:
        start = _safe_float(chunk.get("start", 0.0))
        end = _safe_float(chunk.get("end", start))
        text = _normalize_text(str(chunk.get("text", "")))
        duration = max(0.0, end - start)
        words = text.split()
        word_count = len(words)
        lower = text.lower()
        sentence_marks = len(re.findall(r"[.!?]", text))
        direct_address_hits = len(re.findall(r"\b(you|your|you'll|youre)\b", lower))
        strong_hits = sum(1 for w in STRONG_SIGNAL_TERMS if re.search(rf"\b{re.escape(w)}\b", lower))
        weak_start = any(lower.startswith(prefix) for prefix in WEAK_START_PHRASES)
        if duration < min_duration:
            _log_clip_rejection("cheap_filter", "too_short_duration", chunk, duration=round(duration, 3))
            continue
        if duration > float(os.environ.get("MAX_CLIP_SECONDS", "59.0")):
            _log_clip_rejection("cheap_filter", "too_long_duration", chunk, duration=round(duration, 3))
            continue
        if word_count < min_words:
            _log_clip_rejection("cheap_filter", "too_few_words", chunk, word_count=word_count)
            continue
        if word_count > max_words:
            _log_clip_rejection("cheap_filter", "too_many_words", chunk, word_count=word_count)
            continue
        if sentence_marks <= 0 and not re.search(r"\b(and|but|because|so|then|when|if|why|how)\b", lower):
            _log_clip_rejection("cheap_filter", "no_sentence_structure", chunk, word_count=word_count)
            continue
        if weak_start:
            _log_clip_rejection("cheap_filter", "weak_opening", chunk)
            continue
        if "?" not in text and direct_address_hits == 0 and strong_hits == 0:
            _log_clip_rejection(
                "cheap_filter",
                "low_signal_text",
                chunk,
                strong_hits=strong_hits,
                direct_hits=direct_address_hits,
            )
            continue
        filtered.append(
            {
                **chunk,
                "text": text,
                "start": start,
                "end": end,
                "cheap_signals": {
                    "word_count": word_count,
                    "sentence_marks": sentence_marks,
                    "strong_hits": strong_hits,
                    "direct_address_hits": direct_address_hits,
                },
            }
        )
    return filtered


def _heuristic_engagement_score(
    chunk: Dict[str, Any],
    transcript_segments: Sequence[Dict[str, Any]],
    selection_weights: Optional[Dict[str, float]] = None,
    clip_memory: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    text = _normalize_text(str(chunk.get("text", "")))
    lower = text.lower()
    words = text.split()
    word_count = len(words)
    start = _safe_float(chunk.get("start", 0.0))
    end = max(start + 0.5, _safe_float(chunk.get("end", start + 0.5)))
    duration = max(0.5, end - start)

    question_hits = text.count("?")
    direct_hits = len(re.findall(r"\b(you|your|you'll|youre)\b", lower))
    strong_hits = sum(1 for w in STRONG_SIGNAL_TERMS if re.search(rf"\b{re.escape(w)}\b", lower))
    sentence_count = max(1, len([s for s in re.split(r"[.!?]+", text) if s.strip()]))
    complete_ending = bool(re.search(r"[.!?][\"']?$", text))
    first_words = " ".join(words[:14]).lower()
    weak_lead = any(first_words.startswith(prefix) for prefix in WEAK_START_PHRASES)
    lead_power = 0.0
    if "?" in first_words:
        lead_power += 1.2
    if re.search(r"\b(you|your)\b", first_words):
        lead_power += 0.9
    if any(re.search(rf"\b{re.escape(w)}\b", first_words) for w in STRONG_SIGNAL_TERMS):
        lead_power += 0.9

    ideal_word_bonus = 2.2 if 28 <= word_count <= 82 else (1.0 if 18 <= word_count <= 100 else -0.8)
    ideal_duration_bonus = 2.0 if 18.0 <= duration <= 52.0 else (1.0 if 15.0 <= duration <= 59.0 else -1.0)
    completeness_bonus = 1.1 if complete_ending and sentence_count >= 2 else (0.3 if sentence_count >= 1 else -0.8)

    breakdown = score_breakdown(text)
    heuristic_raw = (
        (breakdown["first_3s_power"] * 0.95)
        + (breakdown["hook_score"] * 0.85)
        + (breakdown["retention_score"] * 0.55)
        + (breakdown["curiosity_score"] * 0.60)
        + (breakdown["clarity_score"] * 0.45)
        - (breakdown["dropoff_risk"] * 1.20)
        + min(2.2, question_hits * 1.0)
        + min(1.8, direct_hits * 0.55)
        + min(2.1, strong_hits * 0.5)
        + ideal_word_bonus
        + ideal_duration_bonus
        + completeness_bonus
        + lead_power
        - (0.9 if weak_lead else 0.0)
    )
    heuristic_score = _clamp_score(
        (float(breakdown["final_score"]) * 0.7)
        + (min(2.2, question_hits * 0.35 + direct_hits * 0.3 + strong_hits * 0.22))
        + (0.6 if complete_ending else 0.0)
        - (0.8 if weak_lead else 0.0)
    )
    retention = _retention_curve_analysis(
        text=text,
        start=start,
        end=end,
        transcript_segments=transcript_segments,
        base_breakdown=breakdown,
    )
    final_score = _compose_final_score(
        heuristic_score=heuristic_score,
        hook_score=float(breakdown["hook_score"]),
        audio_score=float(retention["audio_score"]),
        visual_score=float(retention["visual_score"]),
        retention_score=float(retention["retention_score"]),
        selection_weights=selection_weights,
    )
    memory = memory_similarity(text, clip_memory or {"best": [], "worst": []})
    adjusted_final_score = _clamp_score(final_score + float(memory.get("bias", 0.0)))
    confidence = learning_confidence_score(
        {
            "hook_score": float(breakdown["hook_score"]),
            "audio_score": float(retention["audio_score"]),
            "visual_score": float(retention["visual_score"]),
            "retention_score": float(retention["retention_score"]),
            "final_score": adjusted_final_score,
        },
        text=text,
        memory=clip_memory or {"best": [], "worst": []},
    )

    return {
        "heuristic_score": heuristic_score,
        "heuristic_raw": round(heuristic_raw, 3),
        "final_score": adjusted_final_score,
        "base_final_score": final_score,
        "score_components": {
            "hook_score": breakdown["hook_score"],
            "first_3s_power": breakdown["first_3s_power"],
            "retention_score": float(retention["retention_score"]),
            "curiosity_score": breakdown["curiosity_score"],
            "emotional_score": breakdown["emotional_score"],
            "clarity_score": breakdown["clarity_score"],
            "dropoff_risk": float(retention["dropoff_risk_score"]),
            "audio_score": float(retention["audio_score"]),
            "visual_score": float(retention["visual_score"]),
            "mid_engagement_score": float(retention["mid_engagement_score"]),
            "ending_score": float(retention["ending_score"]),
            "story_structure_score": float(retention["story_structure_score"]),
            "memory_good_similarity": float(memory.get("good_similarity", 0.0)),
            "memory_bad_similarity": float(memory.get("bad_similarity", 0.0)),
            "memory_bias": float(memory.get("bias", 0.0)),
            "confidence_score": float(confidence.get("confidence_score", 0.0)),
        },
        "heuristic_signals": {
            "word_count": word_count,
            "duration": round(duration, 3),
            "question_hits": question_hits,
            "direct_hits": direct_hits,
            "strong_hits": strong_hits,
            "sentence_count": sentence_count,
            "complete_ending": complete_ending,
            "lead_power": round(lead_power, 3),
            "mid_new_info_ratio": float(retention["mid_new_info_ratio"]),
            "mid_repeat_ratio": float(retention["mid_repeat_ratio"]),
            "end_curiosity": bool(retention["end_curiosity"]),
            "end_payoff_hits": int(retention["end_payoff_hits"]),
            "audio_energy_dip": float(retention["audio_energy_dip"]),
        },
        "retention_windows": {
            "start_text": str(retention["start_text"]),
            "mid_text": str(retention["mid_text"]),
            "end_text": str(retention["end_text"]),
            "start_audio_score": float(retention["start_audio_score"]),
            "mid_audio_score": float(retention["mid_audio_score"]),
            "end_audio_score": float(retention["end_audio_score"]),
        },
    }


def _rank_candidates_by_heuristic(
    chunks: List[Dict[str, Any]],
    transcript_segments: Sequence[Dict[str, Any]],
    keep_ratio: float,
    min_keep: int,
    max_keep: int,
    filter_rules: Optional[Dict[str, float]] = None,
    selection_weights: Optional[Dict[str, float]] = None,
    clip_memory: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    rules = filter_rules or {}
    hard_mid_min = float(rules.get("retention_mid_min", os.environ.get("RETENTION_MID_MIN_SCORE", "5.2")))
    hard_ending_min = float(rules.get("retention_ending_min", os.environ.get("RETENTION_ENDING_MIN_SCORE", "4.8")))
    hard_dropoff_max = float(rules.get("retention_dropoff_max", os.environ.get("RETENTION_DROPOFF_MAX_SCORE", "6.8")))
    hard_story_min = float(rules.get("retention_story_min", os.environ.get("RETENTION_STORY_MIN_SCORE", "4.5")))
    for chunk in chunks:
        scored_row = _heuristic_engagement_score(
            chunk,
            transcript_segments=transcript_segments,
            selection_weights=selection_weights,
            clip_memory=clip_memory,
        )
        comp = scored_row["score_components"]
        signals = scored_row["heuristic_signals"]
        hook_score = float(comp["hook_score"])
        mid_score = float(comp["mid_engagement_score"])
        ending_score = float(comp["ending_score"])
        dropoff = float(comp["dropoff_risk"])
        story_score = float(comp["story_structure_score"])
        if hook_score >= 7.0 and mid_score < hard_mid_min:
            _log_clip_rejection(
                "retention_hard_filter",
                "strong_start_weak_middle",
                chunk,
                hook_score=round(hook_score, 3),
                mid_engagement_score=round(mid_score, 3),
            )
            continue
        if ending_score < hard_ending_min and not bool(signals.get("end_curiosity", False)):
            _log_clip_rejection(
                "retention_hard_filter",
                "no_payoff_end",
                chunk,
                ending_score=round(ending_score, 3),
            )
            continue
        if dropoff > hard_dropoff_max:
            _log_clip_rejection(
                "retention_hard_filter",
                "high_dropoff_risk",
                chunk,
                dropoff_risk_score=round(dropoff, 3),
            )
            continue
        if story_score < hard_story_min:
            _log_clip_rejection(
                "retention_hard_filter",
                "weak_story_structure",
                chunk,
                story_structure_score=round(story_score, 3),
            )
            continue

        _log_event(
            "clip_score_breakdown",
            start=round(_safe_float(chunk.get("start", 0.0)), 3),
            end=round(_safe_float(chunk.get("end", 0.0)), 3),
            hook_score=round(hook_score, 3),
            audio_score=round(float(comp["audio_score"]), 3),
            visual_score=round(float(comp["visual_score"]), 3),
            mid_engagement_score=round(mid_score, 3),
            ending_score=round(ending_score, 3),
            dropoff_risk_score=round(dropoff, 3),
            retention_score=round(float(comp["retention_score"]), 3),
            confidence_score=round(float(comp.get("confidence_score", 0.0)), 3),
            final_score=round(float(scored_row["final_score"]), 3),
        )
        scored.append(
            {
                **chunk,
                "original_text": _normalize_text(str(chunk.get("text", ""))),
                "score": float(scored_row["final_score"]),
                "viral_score_v2": float(scored_row["final_score"]),
                "heuristic_score": float(scored_row["heuristic_score"]),
                "final_weighted_score": float(scored_row["final_score"]),
                "confidence_score": float(comp.get("confidence_score", 0.0)),
                "heuristic_signals": scored_row["heuristic_signals"],
                "score_components": scored_row["score_components"],
                "retention_windows": scored_row["retention_windows"],
            }
        )
    scored.sort(
        key=lambda item: (
            float(item.get("score", 0.0)),
            float(item.get("confidence_score", 0.0)),
            float((item.get("score_components") or {}).get("retention_score", 0.0)),
            float((item.get("score_components") or {}).get("hook_score", 0.0)),
        ),
        reverse=True,
    )
    keep_n = max(1, min(len(scored), max(min_keep, min(max_keep, int(round(len(scored) * keep_ratio))))))
    return scored[:keep_n]


def _build_relaxed_heuristic_fallback(
    chunks: List[Dict[str, Any]],
    limit: int,
    selection_weights: Optional[Dict[str, float]] = None,
    clip_memory: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    relaxed: List[Dict[str, Any]] = []
    for chunk in chunks:
        text = _normalize_text(str(chunk.get("text", "")))
        breakdown = score_breakdown(text)
        hook_score = _clamp_score(breakdown.get("hook_score", 5.0))
        retention_score = _clamp_score(breakdown.get("retention_score", 5.0))
        audio_score = _clamp_score(breakdown.get("audio_score", 5.0))
        visual_score = _clamp_score(breakdown.get("visual_score", 5.0))
        heuristic_score = _clamp_score(
            (retention_score * 0.45)
            + (hook_score * 0.30)
            + (float(breakdown.get("final_score", 5.0)) * 0.25)
        )
        final_score = _compose_final_score(
            heuristic_score=heuristic_score,
            hook_score=hook_score,
            audio_score=audio_score,
            visual_score=visual_score,
            retention_score=retention_score,
            selection_weights=selection_weights,
        )
        memory = memory_similarity(text, clip_memory or {"best": [], "worst": []})
        final_score = _clamp_score(final_score + float(memory.get("bias", 0.0)))
        conf = learning_confidence_score(
            {
                "hook_score": hook_score,
                "audio_score": audio_score,
                "visual_score": visual_score,
                "retention_score": retention_score,
                "final_score": final_score,
            },
            text=text,
            memory=clip_memory or {"best": [], "worst": []},
        )
        relaxed.append(
            {
                **chunk,
                "original_text": text,
                "score": final_score,
                "viral_score_v2": final_score,
                "heuristic_score": heuristic_score,
                "final_weighted_score": final_score,
                "confidence_score": float(conf.get("confidence_score", 0.0)),
                "score_components": {
                    "heuristic_score": heuristic_score,
                    "hook_score": hook_score,
                    "audio_score": audio_score,
                    "visual_score": visual_score,
                    "mid_engagement_score": _clamp_score(breakdown.get("retention_score", 5.0)),
                    "ending_score": _clamp_score(breakdown.get("retention_score", 5.0)),
                    "story_structure_score": _clamp_score(breakdown.get("retention_score", 5.0)),
                    "retention_score": retention_score,
                    "dropoff_risk": _clamp_score(float(breakdown.get("dropoff_risk", 1.5)) * 3.5),
                    "curiosity_score": _clamp_score(breakdown.get("curiosity_score", 5.0)),
                    "emotional_score": _clamp_score(breakdown.get("emotional_score", 5.0)),
                    "clarity_score": _clamp_score(breakdown.get("clarity_score", 5.0)),
                    "memory_good_similarity": float(memory.get("good_similarity", 0.0)),
                    "memory_bad_similarity": float(memory.get("bad_similarity", 0.0)),
                    "memory_bias": float(memory.get("bias", 0.0)),
                    "confidence_score": float(conf.get("confidence_score", 0.0)),
                },
                "fallback": True,
                "fallback_reason": "retention_hard_filter_exhausted",
            }
        )

    relaxed.sort(
        key=lambda item: (
            float(item.get("score", 0.0)),
            float(item.get("confidence_score", 0.0)),
            float((item.get("score_components") or {}).get("retention_score", 0.0)),
            float((item.get("score_components") or {}).get("hook_score", 0.0)),
        ),
        reverse=True,
    )
    keep_n = max(1, min(len(relaxed), limit))
    return relaxed[:keep_n]


def _judge_and_enrich_candidates(
    candidates: List[Dict[str, Any]],
    ai_top_n: int,
    min_overall: float,
    min_component: float,
    min_retention_component: float = 4.6,
    soft_accept_min: float = 4.8,
    fallback_keep: int = 1,
    selection_weights: Optional[Dict[str, float]] = None,
    clip_memory: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not candidates:
        return {
            "selected": [],
            "ranked_all": [],
            "ai_approved_count": 0,
            "fallback_count": 0,
            "fallback_reason": "",
        }

    top_for_ai = candidates[: max(1, min(ai_top_n, len(candidates)))]
    compressed_texts = [smart_compress(str(c.get("text", "")), max_words=42) for c in top_for_ai]
    try:
        judged = batch_ai_judge_request(compressed_texts)
    except Exception as exc:
        _log_event("ai_judge_failed", error=str(exc))
        judged = [fallback_judge(text, error=str(exc)) for text in compressed_texts]

    approved: List[Dict[str, Any]] = []
    ranked_all: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(top_for_ai):
        ai = judged[idx] if idx < len(judged) and isinstance(judged[idx], dict) else fallback_judge(
            compressed_texts[idx], error="missing_judge_item"
        )
        scores = ai.get("scores", {}) if isinstance(ai.get("scores"), dict) else {}
        hook_strength = _safe_float(scores.get("hook_strength", 0.0))
        emotional_impact = _safe_float(scores.get("emotional_impact", 0.0))
        standalone_clarity = _safe_float(scores.get("standalone_clarity", 0.0))
        curiosity_retention = _safe_float(scores.get("curiosity_retention", 0.0))
        retention_potential = _safe_float(scores.get("retention_potential", curiosity_retention))
        narrative_completeness = _safe_float(scores.get("narrative_completeness", standalone_clarity))
        payoff_satisfaction = _safe_float(scores.get("payoff_satisfaction", standalone_clarity))
        overall_score = _safe_float(ai.get("overall_score", 0.0))
        raw_pass = ai.get("pass", False)
        passes = raw_pass if isinstance(raw_pass, bool) else _as_bool(str(raw_pass), False)
        retention_hook_blend = _clamp_score(
            (retention_potential * 0.50)
            + (hook_strength * 0.30)
            + (curiosity_retention * 0.10)
            + (overall_score * 0.10)
        )
        overall_ok = overall_score >= min_overall
        hook_retention_ok = hook_strength >= min_component and retention_potential >= min_retention_component
        soft_accept_ok = retention_hook_blend >= soft_accept_min

        improved_hook = str(ai.get("hook", "")).strip()
        improved_text = _merge_improved_text(str(chunk.get("text", "")), improved_hook)
        caption = str(ai.get("caption", "")).strip() or _safe_caption(improved_text)
        base_components = chunk.get("score_components", {}) if isinstance(chunk.get("score_components"), dict) else {}
        text_breakdown = score_breakdown(improved_text)
        heuristic_score = _clamp_score((float(chunk.get("heuristic_score", text_breakdown["final_score"])) * 0.65) + (float(text_breakdown["final_score"]) * 0.35))
        hook_score = _clamp_score((float(text_breakdown["hook_score"]) * 0.65) + (hook_strength * 0.35))
        retention_score = _clamp_score(
            (float(base_components.get("retention_score", text_breakdown["retention_score"])) * 0.6)
            + (retention_potential * 0.25)
            + (narrative_completeness * 0.08)
            + (payoff_satisfaction * 0.07)
        )
        audio_score = _clamp_score(base_components.get("audio_score", 5.0))
        visual_score = _clamp_score(base_components.get("visual_score", 5.0))
        final_score = _compose_final_score(
            heuristic_score=heuristic_score,
            hook_score=hook_score,
            audio_score=audio_score,
            visual_score=visual_score,
            retention_score=retention_score,
            selection_weights=selection_weights,
        )
        memory = memory_similarity(improved_text, clip_memory or {"best": [], "worst": []})
        final_score = _clamp_score(final_score + float(memory.get("bias", 0.0)))
        conf = learning_confidence_score(
            {
                "hook_score": hook_score,
                "audio_score": audio_score,
                "visual_score": visual_score,
                "retention_score": retention_score,
                "final_score": final_score,
            },
            text=improved_text,
            memory=clip_memory or {"best": [], "worst": []},
        )

        updated_components = {
            **base_components,
            "hook_score": hook_score,
            "heuristic_score": heuristic_score,
            "retention_score": retention_score,
            "dropoff_risk": float(base_components.get("dropoff_risk", text_breakdown["dropoff_risk"])),
            "audio_score": audio_score,
            "visual_score": visual_score,
            "ai_retention_potential": retention_potential,
            "ai_narrative_completeness": narrative_completeness,
            "ai_payoff_satisfaction": payoff_satisfaction,
            "ai_retention_hook_blend": retention_hook_blend,
            "memory_good_similarity": float(memory.get("good_similarity", base_components.get("memory_good_similarity", 0.0))),
            "memory_bad_similarity": float(memory.get("bad_similarity", base_components.get("memory_bad_similarity", 0.0))),
            "memory_bias": float(memory.get("bias", base_components.get("memory_bias", 0.0))),
            "confidence_score": float(conf.get("confidence_score", base_components.get("confidence_score", 0.0))),
        }
        enriched_clip = {
            **chunk,
            "text": improved_text,
            "caption": caption,
            "ai": ai,
            "hook_type": str(chunk.get("hook_type", "statement")),
            "ai_judge_score": overall_score,
            "score_components": updated_components,
            "heuristic_score": heuristic_score,
            "final_weighted_score": final_score,
            "confidence_score": float(updated_components.get("confidence_score", 0.0)),
            "score": final_score,
            "ai_soft_accept_score": retention_hook_blend,
            "ai_soft_approved": bool(passes or overall_ok or hook_retention_ok or soft_accept_ok),
            "fallback": False,
            "fallback_reason": "",
        }
        enriched_clip["selection_rank_score"] = _final_selection_rank_score(enriched_clip)
        ranked_all.append(enriched_clip)
        if enriched_clip["ai_soft_approved"]:
            approved.append(enriched_clip)
        else:
            _log_clip_rejection(
                "ai_judge",
                str(ai.get("reason", "below_soft_threshold")) or "below_soft_threshold",
                chunk,
                overall=round(overall_score, 3),
                hook=round(hook_strength, 3),
                retention=round(retention_potential, 3),
                soft_score=round(retention_hook_blend, 3),
                narrative=round(narrative_completeness, 3),
                payoff=round(payoff_satisfaction, 3),
            )

    ranked_all.sort(
        key=lambda item: (
            float(item.get("selection_rank_score", _final_selection_rank_score(item))),
            float(item.get("confidence_score", 0.0)),
            float((item.get("score_components") or {}).get("retention_score", 0.0)),
            float((item.get("score_components") or {}).get("hook_score", 0.0)),
        ),
        reverse=True,
    )

    selected = approved
    fallback_count = 0
    fallback_reason = ""
    if not selected and ranked_all:
        keep_n = max(1, min(fallback_keep, len(ranked_all)))
        selected = []
        for clip in ranked_all[:keep_n]:
            selected.append(
                {
                    **clip,
                    "fallback": True,
                    "fallback_reason": "no_ai_soft_approved_clips",
                }
            )
        fallback_count = len(selected)
        fallback_reason = "no_ai_soft_approved_clips"

    return {
        "selected": selected,
        "ranked_all": ranked_all,
        "ai_approved_count": len(approved),
        "fallback_count": fallback_count,
        "fallback_reason": fallback_reason,
    }


def _expand_ab_variants(
    enriched: List[Dict[str, Any]],
    selection_weights: Optional[Dict[str, float]] = None,
    clip_memory: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    expanded: List[Dict[str, Any]] = []
    for idx, clip in enumerate(enriched):
        ai = clip.get("ai", {}) if isinstance(clip.get("ai"), dict) else {}
        variants = generate_variants(
            text=str(clip.get("original_text", clip.get("text", ""))),
            improved_hook=str(ai.get("hook", "")),
            caption=str(clip.get("caption", "")),
        )
        if not variants:
            variants = [{"variant_id": "A", "variant_type": "original", "text": str(clip.get("text", "")), "caption": str(clip.get("caption", ""))}]

        for variant in variants:
            text = _normalize_text(str(variant.get("text", clip.get("text", ""))))
            caption = str(variant.get("caption", clip.get("caption", ""))).strip() or _safe_caption(text)
            breakdown = score_breakdown(text)
            base_components = clip.get("score_components", {}) if isinstance(clip.get("score_components"), dict) else {}
            heuristic_score = _clamp_score((float(base_components.get("heuristic_score", breakdown["final_score"])) * 0.65) + (float(breakdown["final_score"]) * 0.35))
            hook_score = _clamp_score((float(base_components.get("hook_score", breakdown["hook_score"])) * 0.6) + (float(breakdown["hook_score"]) * 0.4))
            retention_score = _clamp_score((float(base_components.get("retention_score", breakdown["retention_score"])) * 0.7) + (float(breakdown["retention_score"]) * 0.3))
            audio_score = _clamp_score(base_components.get("audio_score", 5.0))
            visual_score = _clamp_score(base_components.get("visual_score", 5.0))
            mid_engagement_score = _clamp_score(base_components.get("mid_engagement_score", retention_score))
            ending_score = _clamp_score(base_components.get("ending_score", retention_score))
            story_structure_score = _clamp_score(base_components.get("story_structure_score", retention_score))
            dropoff_risk_score = _clamp_score(base_components.get("dropoff_risk", breakdown["dropoff_risk"] * 3.5))
            final_score = _compose_final_score(
                heuristic_score=heuristic_score,
                hook_score=hook_score,
                audio_score=audio_score,
                visual_score=visual_score,
                retention_score=retention_score,
                selection_weights=selection_weights,
            )
            memory = memory_similarity(text, clip_memory or {"best": [], "worst": []})
            final_score = _clamp_score(final_score + float(memory.get("bias", 0.0)))
            conf = learning_confidence_score(
                {
                    "hook_score": hook_score,
                    "audio_score": audio_score,
                    "visual_score": visual_score,
                    "retention_score": retention_score,
                    "final_score": final_score,
                },
                text=text,
                memory=clip_memory or {"best": [], "worst": []},
            )
            expanded_clip = {
                **clip,
                "text": text,
                "caption": caption,
                "variant_id": str(variant.get("variant_id", "A")),
                "variant_type": str(variant.get("variant_type", "original")),
                "score": final_score,
                "final_weighted_score": final_score,
                "heuristic_score": heuristic_score,
                "confidence_score": float(conf.get("confidence_score", 0.0)),
                "score_components": {
                    **base_components,
                    "heuristic_score": heuristic_score,
                    "hook_score": hook_score,
                    "audio_score": audio_score,
                    "visual_score": visual_score,
                    "mid_engagement_score": mid_engagement_score,
                    "ending_score": ending_score,
                    "story_structure_score": story_structure_score,
                    "retention_score": retention_score,
                    "dropoff_risk": dropoff_risk_score,
                    "memory_good_similarity": float(memory.get("good_similarity", base_components.get("memory_good_similarity", 0.0))),
                    "memory_bad_similarity": float(memory.get("bad_similarity", base_components.get("memory_bad_similarity", 0.0))),
                    "memory_bias": float(memory.get("bias", base_components.get("memory_bias", 0.0))),
                    "confidence_score": float(conf.get("confidence_score", base_components.get("confidence_score", 0.0))),
                    "curiosity_score": breakdown["curiosity_score"],
                    "emotional_score": breakdown["emotional_score"],
                    "clarity_score": breakdown["clarity_score"],
                },
                "clip_base_index": idx,
            }
            expanded_clip["selection_rank_score"] = _final_selection_rank_score(expanded_clip)
            expanded.append(expanded_clip)
    return expanded


def _infer_video_topic(segments: Sequence[Dict[str, Any]]) -> str:
    token_counter: Dict[str, int] = {}
    for seg in segments[:300]:
        for tok in _tokenize(str(seg.get("text", ""))):
            if len(tok) < 4 or tok in {
                "this",
                "that",
                "with",
                "from",
                "your",
                "you",
                "what",
                "when",
                "then",
                "they",
                "them",
                "have",
            }:
                continue
            token_counter[tok] = token_counter.get(tok, 0) + 1
    if not token_counter:
        return "general"
    ranked = sorted(token_counter.items(), key=lambda x: x[1], reverse=True)
    return str(ranked[0][0])


def _infer_video_category(topic: str) -> str:
    topic_lower = str(topic).lower()
    if any(k in topic_lower for k in {"ai", "tech", "code", "software", "python"}):
        return "technology"
    if any(k in topic_lower for k in {"money", "invest", "business", "market"}):
        return "business"
    if any(k in topic_lower for k in {"health", "fitness", "diet", "workout"}):
        return "health"
    if any(k in topic_lower for k in {"story", "drama", "movie", "show"}):
        return "entertainment"
    return "general"


def _build_clip_scores_payload(clip: Dict[str, Any]) -> Dict[str, float]:
    comps = clip.get("score_components", {}) if isinstance(clip.get("score_components"), dict) else {}
    return {
        "hook_score": float(comps.get("hook_score", 0.0)),
        "audio_score": float(comps.get("audio_score", 0.0)),
        "visual_score": float(comps.get("visual_score", 0.0)),
        "mid_engagement_score": float(comps.get("mid_engagement_score", 0.0)),
        "ending_score": float(comps.get("ending_score", 0.0)),
        "story_structure_score": float(comps.get("story_structure_score", 0.0)),
        "dropoff_risk_score": float(comps.get("dropoff_risk", 0.0)),
        "retention_score": float(comps.get("retention_score", 0.0)),
        "curiosity_score": float(comps.get("curiosity_score", 0.0)),
        "emotional_score": float(comps.get("emotional_score", 0.0)),
        "clarity_score": float(comps.get("clarity_score", 0.0)),
        "final_score": float(clip.get("score", 0.0)),
        "ai_judge_score": float(clip.get("ai_judge_score", 0.0)),
        "confidence_score": float(clip.get("confidence_score", comps.get("confidence_score", 0.0))),
        "memory_good_similarity": float(comps.get("memory_good_similarity", 0.0)),
        "memory_bad_similarity": float(comps.get("memory_bad_similarity", 0.0)),
        "memory_bias": float(comps.get("memory_bias", 0.0)),
    }


def _persist_feedback(
    selected: List[Dict[str, Any]],
    output_paths: List[str],
    source_video: str,
    video_topic: str,
    video_category: str,
    video_duration: float,
) -> None:
    for idx, clip in enumerate(selected):
        clip_id_base = hashlib.sha1(
            f"{source_video}:{clip.get('start')}:{clip.get('end')}:{clip.get('variant_id','A')}".encode("utf-8")
        ).hexdigest()[:14]
        clip_id = f"{clip_id_base}_variant_{clip.get('variant_id', 'A')}"
        start = _safe_float(clip.get("start", 0.0))
        end = _safe_float(clip.get("end", start))
        clip_len = max(0.0, end - start)
        score_payload = _build_clip_scores_payload(clip)
        metrics_payload = simulate_metric_bundle(score_payload)
        record = {
            "clip_id": clip_id,
            "status": "accepted",
            "text": str(clip.get("text", "")),
            "start": round(start, 3),
            "end": round(end, 3),
            "clip_length": round(clip_len, 3),
            "hook": str((clip.get("ai", {}) if isinstance(clip.get("ai"), dict) else {}).get("hook", "")),
            "caption": str(clip.get("caption", "")),
            "scores": score_payload,
            "final_score": float(clip.get("score", 0.0)),
            "metrics": metrics_payload,
            "metadata": {
                "video_topic": video_topic,
                "video_category": video_category,
                "video_duration": round(video_duration, 3),
                "clip_length": round(clip_len, 3),
                "source_video": source_video,
            },
            "output_path": output_paths[idx] if idx < len(output_paths) else "",
            "rejection_reasons": [],
            "ai_judge_score": float(clip.get("ai_judge_score", 0.0)),
        }
        save_clip_feedback(record)


def _persist_rejected_feedback(
    rejections: Sequence[Dict[str, Any]],
    source_video: str,
    video_topic: str,
    video_category: str,
    video_duration: float,
) -> None:
    for idx, row in enumerate(rejections):
        start = _safe_float(row.get("start", 0.0))
        end = _safe_float(row.get("end", start))
        clip_len = max(0.0, end - start)
        reason = str(row.get("reason", "rejected")).strip() or "rejected"
        details = row.get("details", {}) if isinstance(row.get("details"), dict) else {}
        scores = {
            "hook_score": _safe_float(details.get("hook_score", details.get("hook", 0.0))),
            "audio_score": _safe_float(details.get("audio_score", 0.0)),
            "visual_score": _safe_float(details.get("visual_score", 0.0)),
            "mid_engagement_score": _safe_float(details.get("mid_engagement_score", 0.0)),
            "ending_score": _safe_float(details.get("ending_score", 0.0)),
            "story_structure_score": _safe_float(details.get("story_structure_score", 0.0)),
            "dropoff_risk_score": _safe_float(details.get("dropoff_risk_score", details.get("dropoff", 0.0))),
            "retention_score": _safe_float(details.get("retention_score", 0.0)),
            "final_score": _safe_float(details.get("final_score", details.get("overall", 0.0))),
            "ai_judge_score": _safe_float(details.get("overall", 0.0)),
            "confidence_score": _safe_float(details.get("confidence_score", 0.0)),
        }
        metrics_payload = simulate_metric_bundle(scores)
        record = {
            "clip_id": f"rejected_{hashlib.sha1(f'{source_video}:{start}:{end}:{idx}'.encode('utf-8')).hexdigest()[:14]}",
            "status": "rejected",
            "text": str(row.get("text", row.get("text_preview", ""))),
            "start": round(start, 3),
            "end": round(end, 3),
            "clip_length": round(clip_len, 3),
            "hook": "",
            "caption": "",
            "scores": scores,
            "final_score": float(scores.get("final_score", 0.0)),
            "metrics": metrics_payload,
            "metadata": {
                "video_topic": video_topic,
                "video_category": video_category,
                "video_duration": round(video_duration, 3),
                "clip_length": round(clip_len, 3),
                "source_video": source_video,
            },
            "output_path": "",
            "rejection_reasons": [reason],
            "rejection_layer": str(row.get("layer", "filter")),
            "rejection_details": details,
            "ai_judge_score": float(scores.get("ai_judge_score", 0.0)),
        }
        save_clip_feedback(record)


def _persist_rejections_if_any(
    source_video: str,
    video_topic: str,
    video_category: str,
    video_duration: float,
) -> None:
    if not _RUN_REJECTION_LOG:
        return
    _persist_rejected_feedback(
        _RUN_REJECTION_LOG,
        source_video=source_video,
        video_topic=video_topic,
        video_category=video_category,
        video_duration=video_duration,
    )


def run_pipeline(
    url: str,
    top_k: int = 3,
    whisper_model: Optional[str] = None,
    whisper_fast: Optional[bool] = None,
) -> List[str]:
    global _RUN_REJECTION_LOG
    total_start = time.perf_counter()
    render_failures = 0
    run_ctx = _create_run_context()
    run_id = str(run_ctx["run_id"])
    run_dir = Path(run_ctx["run_dir"])
    run_dir_display = str(run_ctx["run_dir_display"])
    video = ""
    chunks: List[Dict[str, Any]] = []
    selected: List[Dict[str, Any]] = []
    _RUN_REJECTION_LOG = []
    _log_event("run_initialized", run_id=run_id, run_dir=run_dir_display)

    try:
        video_path = run_dir / "video.mp4"
        stage_retry_delay = max(0.2, float(os.environ.get("STAGE_RETRY_DELAY_SECONDS", "1.0")))
        download_retries = max(0, int(os.environ.get("DOWNLOAD_RETRIES", "2")))
        download_timeout = max(30.0, float(os.environ.get("DOWNLOAD_TIMEOUT_SECONDS", "900")))
        video_result, download_error = _run_stage_with_retry(
            "download",
            lambda: download_video(url, output_path=str(video_path)),
            retries=download_retries,
            retry_delay_seconds=stage_retry_delay,
            timeout_seconds=download_timeout,
        )
        if not video_result:
            raise RuntimeError(download_error or "download_failed")
        video = str(video_result)
        if not _file_exists_and_nonempty(video, min_bytes=1024):
            raise RuntimeError("downloaded_file_missing_or_empty")
        if not _validate_media_file(video, stream_selector="v"):
            raise RuntimeError("downloaded_video_invalid")

        _ensure_binary_available("ffmpeg", "FFMPEG_BIN", "ffmpeg_not_found")
        audio_path = run_dir / "audio.wav"
        if not _file_exists_and_nonempty(video, min_bytes=1024):
            raise RuntimeError("input_video_missing_or_empty")
        ffmpeg_retries = max(0, int(os.environ.get("FFMPEG_STAGE_RETRIES", "1")))
        ffmpeg_timeout = max(30.0, float(os.environ.get("FFMPEG_STAGE_TIMEOUT_SECONDS", "600")))
        audio_result, audio_error = _run_stage_with_retry(
            "extract_audio",
            lambda: extract_audio(video, output_path=str(audio_path)),
            retries=ffmpeg_retries,
            retry_delay_seconds=stage_retry_delay,
            timeout_seconds=ffmpeg_timeout,
        )

        segments: List[Dict[str, Any]] = []
        words: List[Dict[str, Any]] = []
        if (
            audio_result
            and _file_exists_and_nonempty(str(audio_result), min_bytes=1024)
            and _validate_media_file(str(audio_result), stream_selector="a")
        ):
            transcribe_timeout = max(30.0, float(os.environ.get("TRANSCRIBE_TIMEOUT_SECONDS", "900")))
            transcribe_result, transcribe_error = _run_stage_with_retry(
                "transcribe",
                lambda: transcribe_audio(
                    str(audio_result),
                    model_size=whisper_model,
                    fast_mode=whisper_fast,
                ),
                retries=0,
                retry_delay_seconds=stage_retry_delay,
                timeout_seconds=transcribe_timeout,
            )
            if isinstance(transcribe_result, list):
                segments = transcribe_result
                words = flatten_words(segments)
            else:
                _log_event("transcribe_fallback", error=transcribe_error or "transcribe_failed")
        else:
            _log_event(
                "extract_audio_fallback",
                error=audio_error or "audio_missing_or_invalid_after_extract",
            )

        stage_start = time.perf_counter()
        video_duration = max(0.0, max((_safe_float(seg.get("end", 0.0)) for seg in segments), default=0.0))
        video_topic = _infer_video_topic(segments)
        video_category = _infer_video_category(video_topic)
        selection_weights = load_selection_weights()
        filter_rules = load_filter_rules()
        clip_memory = load_clip_memory()
        _log_stage(
            "learning_context",
            stage_start,
            video_topic=video_topic,
            video_category=video_category,
            selection_weights=selection_weights,
            filter_rules=filter_rules,
            memory_best=len(clip_memory.get("best", [])) if isinstance(clip_memory, dict) else 0,
            memory_worst=len(clip_memory.get("worst", [])) if isinstance(clip_memory, dict) else 0,
        )

        stage_start = time.perf_counter()
        chunk_seconds = int(os.environ.get("CHUNK_SECONDS", "45"))
        max_chunks_before_ai = int(os.environ.get("MAX_CHUNKS_BEFORE_AI", "6"))
        chunks = create_chunks(segments, chunk_seconds=chunk_seconds)
        chunks = [{"start": c["start"], "end": c["end"], "text": _normalize_text(str(c.get("text", "")))} for c in chunks]
        chunks = _pace_chunks(chunks, segments)
        filtered_chunks = filter_invalid_clips(
            chunks,
            min_duration=float(os.environ.get("MIN_CLIP_SECONDS", "15.0")),
            min_words=int(os.environ.get("MIN_CLIP_WORDS", "12")),
            max_words=int(os.environ.get("MAX_CLIP_WORDS", "140")),
        )
        _log_stage("ingest_chunking", stage_start, chunks=len(chunks), chunk_seconds=chunk_seconds)
        _log_event("clip_filtering", before=len(chunks), after=len(filtered_chunks))
        if filtered_chunks:
            chunks = filtered_chunks

        if not chunks:
            fallback_duration = max(15.0, min(25.0, video_duration if video_duration > 0 else 25.0))
            chunks = [
                {
                    "start": 0.0,
                    "end": fallback_duration,
                    "text": "AUTO FALLBACK CLIP",
                }
            ]
            _log_event(
                "selection_fallback_triggered",
                reason="no_chunks_generated",
                ai_approved_clips=0,
                fallback_clips=1,
                threshold="chunking",
            )

        stage_start = time.perf_counter()
        requested_top_k = max(1, min(int(top_k), int(os.environ.get("MAX_TOP_K", "3"))))
        keep_ratio = float(os.environ.get("PRE_SCORE_KEEP_RATIO", "0.55"))
        heuristic_ranked = _rank_candidates_by_heuristic(
            chunks,
            transcript_segments=segments,
            keep_ratio=keep_ratio,
            min_keep=2,
            max_keep=max_chunks_before_ai,
            filter_rules=filter_rules,
            selection_weights=selection_weights,
            clip_memory=clip_memory,
        )
        _log_stage(
            "heuristic_ranking",
            stage_start,
            ranked=len(heuristic_ranked),
            reduction=max(0, len(chunks) - len(heuristic_ranked)),
        )
        if not heuristic_ranked:
            relaxed_limit = max(1, min(max_chunks_before_ai, len(chunks)))
            heuristic_ranked = _build_relaxed_heuristic_fallback(
                chunks,
                limit=relaxed_limit,
                selection_weights=selection_weights,
                clip_memory=clip_memory,
            )
            _log_event(
                "selection_fallback_triggered",
                reason="no_retention_approved_candidates",
                ai_approved_clips=0,
                fallback_clips=len(heuristic_ranked),
                threshold="pre_ai",
            )

        stage_start = time.perf_counter()
        ai_top_n = max(3, min(5, int(os.environ.get("AI_JUDGE_TOP_N", "4"))))
        ai_threshold = float(os.environ.get("AI_JUDGE_MIN_SCORE", "4.8"))
        ai_component_min = float(os.environ.get("AI_JUDGE_MIN_COMPONENT", "4.7"))
        ai_retention_min = float(filter_rules.get("ai_retention_component_min", os.environ.get("AI_JUDGE_RETENTION_MIN_COMPONENT", "4.6")))
        ai_soft_min = float(os.environ.get("AI_JUDGE_SOFT_MIN_SCORE", "4.8"))
        judge_outcome = _judge_and_enrich_candidates(
            heuristic_ranked,
            ai_top_n=ai_top_n,
            min_overall=ai_threshold,
            min_component=ai_component_min,
            min_retention_component=ai_retention_min,
            soft_accept_min=ai_soft_min,
            fallback_keep=max(1, min(2, requested_top_k)),
            selection_weights=selection_weights,
            clip_memory=clip_memory,
        )
        enriched = list(judge_outcome.get("selected", []))
        _log_stage(
            "ai_judge",
            stage_start,
            considered=min(len(heuristic_ranked), ai_top_n),
            approved=int(judge_outcome.get("ai_approved_count", 0)),
            fallback_clips=int(judge_outcome.get("fallback_count", 0)),
            threshold=ai_threshold,
            soft_threshold=ai_soft_min,
        )
        if int(judge_outcome.get("fallback_count", 0)) > 0:
            _log_event(
                "selection_fallback_triggered",
                reason=str(judge_outcome.get("fallback_reason", "ai_soft_rejection")),
                ai_approved_clips=int(judge_outcome.get("ai_approved_count", 0)),
                fallback_clips=int(judge_outcome.get("fallback_count", 0)),
                threshold=ai_threshold,
            )
        if not enriched:
            ranked_all = list(judge_outcome.get("ranked_all", []))
            if ranked_all:
                enriched = [{**ranked_all[0], "fallback": True, "fallback_reason": "emergency_ranked_all"}]
                _log_event(
                    "selection_fallback_triggered",
                    reason="emergency_ranked_all",
                    ai_approved_clips=0,
                    fallback_clips=1,
                    threshold=ai_threshold,
                )
            elif heuristic_ranked:
                enriched = [{**heuristic_ranked[0], "fallback": True, "fallback_reason": "emergency_heuristic"}]
                _log_event(
                    "selection_fallback_triggered",
                    reason="emergency_heuristic",
                    ai_approved_clips=0,
                    fallback_clips=1,
                    threshold=ai_threshold,
                )

        stage_start = time.perf_counter()
        candidates = _expand_ab_variants(enriched, selection_weights=selection_weights, clip_memory=clip_memory)
        for candidate in candidates:
            candidate["selection_rank_score"] = _safe_float(
                candidate.get("selection_rank_score"),
                _final_selection_rank_score(candidate),
            )
        ranked = sorted(
            candidates,
            key=lambda item: (
                float(item.get("selection_rank_score", _final_selection_rank_score(item))),
                float(item.get("score", 0.0)),
                float(item.get("confidence_score", 0.0)),
                float((item.get("score_components") or {}).get("retention_score", 0.0)),
                float((item.get("score_components") or {}).get("audio_score", 0.0)),
            ),
            reverse=True,
        )[: max(requested_top_k * 4, requested_top_k + 2)]
        candidate_limit = max(requested_top_k * 4, requested_top_k + 3)
        selected = _apply_diversity_filter(ranked, limit=candidate_limit)
        if not selected:
            selected = ranked[:candidate_limit]
        if not selected:
            selected = [{**enriched[0], "fallback": True, "fallback_reason": "no_selected_candidates"}] if enriched else []
            _log_event(
                "selection_fallback_triggered",
                reason="no_selected_candidates",
                ai_approved_clips=0,
                fallback_clips=len(selected),
                threshold=ai_threshold,
            )
        if not selected and chunks:
            selected = [{**chunks[0], "fallback": True, "fallback_reason": "emergency_chunk_selection"}]
            _log_event(
                "selection_fallback_triggered",
                reason="emergency_chunk_selection",
                ai_approved_clips=0,
                fallback_clips=1,
                threshold=ai_threshold,
            )
        if not selected:
            _persist_rejections_if_any(
                source_video=video,
                video_topic=video_topic,
                video_category=video_category,
                video_duration=video_duration,
            )
            return []
        selected = _apply_ai_duration_enhancements(selected, segments)
        _log_stage(
            "final_selection",
            stage_start,
            selected_candidates=len(selected),
            requested_top_k=requested_top_k,
            candidate_variants=len(candidates),
        )

        if not _file_exists_and_nonempty(video, min_bytes=1024):
            raise RuntimeError("input_video_missing_before_render")
        _ensure_binary_available("ffmpeg", "FFMPEG_BIN", "ffmpeg_not_found")
        stage_start = time.perf_counter()
        outputs_by_index: Dict[int, str] = {}
        max_workers = max(1, min(int(os.environ.get("RENDER_CONCURRENCY", "2")), len(selected)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _render_clip_job,
                    {
                        "index": idx,
                        "clip": clip,
                        "input_video": video,
                        "segments": segments,
                        "words": words,
                        "run_dir": str(run_dir),
                    },
                ): idx
                for idx, clip in enumerate(selected)
            }
            for future in as_completed(futures):
                idx = int(futures[future])
                try:
                    result = future.result()
                    idx = int(result.get("index", futures[future]))
                    path = result.get("path")
                    if path and _file_exists_and_nonempty(str(path), min_bytes=1024) and _validate_media_file(str(path), stream_selector="v"):
                        outputs_by_index[idx] = str(path)
                    else:
                        render_failures += 1
                        _log_event(
                            "render_output_invalid",
                            index=idx,
                            path=str(path or ""),
                        )
                except Exception as render_exc:
                    render_failures += 1
                    _log_event(
                        "render_future_failed",
                        index=idx,
                        error=str(render_exc),
                    )

        successful_indices = sorted(outputs_by_index)
        final_indices = successful_indices[:requested_top_k]
        outputs = [outputs_by_index[idx] for idx in final_indices]
        _log_stage(
            "render",
            stage_start,
            rendered_total=len(successful_indices),
            rendered_returned=len(outputs),
            render_failures=render_failures,
        )
        _log_event("render_outputs", run_id=run_id, outputs=outputs)

        ai_metrics = get_ai_metrics_snapshot()
        _log_event(
            "pipeline_metrics",
            ai_calls_count=ai_metrics.get("ai_calls_count", 0),
            ai_judge_calls_count=ai_metrics.get("ai_judge_calls_count", 0),
            cache_hits=ai_metrics.get("cache_hits", 0),
            fallback_usage=ai_metrics.get("fallback_usage", 0),
            rejected_clips=len(_RUN_REJECTION_LOG),
            render_failures=render_failures,
        )

        try:
            selected_for_feedback = [selected[idx] for idx in final_indices if idx < len(selected)]
            _persist_feedback(
                selected_for_feedback,
                outputs,
                source_video=video,
                video_topic=video_topic,
                video_category=video_category,
                video_duration=video_duration,
            )
            if _RUN_REJECTION_LOG:
                _persist_rejected_feedback(
                    _RUN_REJECTION_LOG,
                    source_video=video,
                    video_topic=video_topic,
                    video_category=video_category,
                    video_duration=video_duration,
                )
            if _as_bool(os.environ.get("LEARNING_UPDATE_ON_RUN", "1"), True):
                learning_state = update_learning_from_feedback(load_all_feedback())
                _log_event(
                    "learning_update",
                    feedback_count=learning_state.get("feedback_count", 0),
                    selection_weights=learning_state.get("selection_weights", {}),
                    filter_rules=learning_state.get("filters", {}),
                    top_patterns=(learning_state.get("patterns", {}) if isinstance(learning_state.get("patterns", {}), dict) else {}).get("top_patterns", [])[:5],
                    memory_best=len((learning_state.get("memory", {}) if isinstance(learning_state.get("memory", {}), dict) else {}).get("best", [])),
                    memory_worst=len((learning_state.get("memory", {}) if isinstance(learning_state.get("memory", {}), dict) else {}).get("worst", [])),
                )
        except Exception as learn_exc:
            _log_event("learning_update_failed", error=str(learn_exc))

        _log_stage("pipeline_total", total_start, output_count=len(outputs))
        if outputs:
            _set_last_run_metadata(run_id=run_id, run_dir=run_dir, outputs=outputs)
            _log_event("pipeline_result", run_id=run_id, run_dir=run_dir_display, outputs=outputs)
            return outputs
    except Exception as exc:
        _log_event("pipeline_error", error=str(exc))
        if not video:
            raise

    # Non-crashing final fallback: attempt one clip.
    try:
        if not chunks:
            return []
        fallback_clip = chunks[0]
        fallback_result = _render_clip_job(
            {
                "index": 0,
                "clip": fallback_clip,
                "input_video": video,
                "segments": [],
                "words": [],
                "run_dir": str(run_dir),
            }
        )
        path = fallback_result.get("path")
        outputs = [str(path)] if path else []
        _set_last_run_metadata(run_id=run_id, run_dir=run_dir, outputs=outputs)
        _log_event("pipeline_result", run_id=run_id, run_dir=run_dir_display, outputs=outputs)
        return outputs
    except Exception:
        _set_last_run_metadata(run_id=run_id, run_dir=run_dir, outputs=[])
        _log_event("pipeline_result", run_id=run_id, run_dir=run_dir_display, outputs=[])
        return []

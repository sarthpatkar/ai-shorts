from __future__ import annotations

import os
import shutil
import subprocess
import json
import logging
import re
import tempfile
import textwrap
import statistics
import math
from threading import Lock
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

FFMPEG_BIN = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
FFPROBE_BIN = os.environ.get("FFPROBE_BIN") or shutil.which("ffprobe") or FFMPEG_BIN.replace("ffmpeg", "ffprobe")
logger = logging.getLogger(__name__)

VERTICAL_OUTPUT_WIDTH = int(os.environ.get("VERTICAL_WIDTH", "1080"))
VERTICAL_OUTPUT_HEIGHT = int(os.environ.get("VERTICAL_HEIGHT", "1920"))
_SUBTITLE_FILTER_AVAILABLE: Optional[bool] = None
_SUBTITLE_FILTER_LOCK = Lock()
_DRAWTEXT_FILTER_AVAILABLE: Optional[bool] = None
_DRAWTEXT_FILTER_LOCK = Lock()


def _escape_filter_path(path: str) -> str:
    escaped = path.replace("\\", "\\\\")
    escaped = escaped.replace(":", "\\:")
    escaped = escaped.replace("'", "\\'")
    escaped = escaped.replace(",", "\\,")
    return escaped


def _quality_finish_filters() -> List[str]:
    contrast = max(0.90, min(1.20, float(os.environ.get("VIDEO_CONTRAST", "1.03"))))
    saturation = max(0.85, min(1.35, float(os.environ.get("VIDEO_SATURATION", "1.06"))))
    sharpen = max(0.0, min(1.0, float(os.environ.get("VIDEO_SHARPEN", "0.34"))))
    # Force full-frame 9:16 output. We intentionally upscale and then center-crop
    # so clips never render as a small video padded inside a black canvas.
    scale_filter = (
        f"scale={VERTICAL_OUTPUT_WIDTH}:{VERTICAL_OUTPUT_HEIGHT}:"
        "force_original_aspect_ratio=increase:"
        "flags=lanczos+accurate_rnd+full_chroma_int"
    )
    filters = [
        scale_filter,
        f"crop={VERTICAL_OUTPUT_WIDTH}:{VERTICAL_OUTPUT_HEIGHT}:(iw-{VERTICAL_OUTPUT_WIDTH})/2:(ih-{VERTICAL_OUTPUT_HEIGHT})/2",
        "setsar=1",
        f"eq=contrast={contrast:.3f}:saturation={saturation:.3f}",
    ]
    if sharpen > 0:
        filters.append(f"unsharp=5:5:{sharpen:.3f}:5:5:0.0")
    return filters


def _speech_boundaries(start: float, end: float, speech_segments: Optional[Sequence[dict]]) -> List[float]:
    points = {start, end}
    if not speech_segments:
        return sorted(points)

    for seg in speech_segments:
        ss = float(seg.get("start", 0.0))
        se = float(seg.get("end", 0.0))
        if se <= start or ss >= end:
            continue
        points.add(max(start, min(end, ss)))
        points.add(max(start, min(end, se)))

    return sorted(points)


def _build_jump_intervals(
    start: float,
    end: float,
    speech_segments: Optional[Sequence[dict]],
    min_len: float = 2.2,
    max_len: float = 4.2,
) -> List[Tuple[float, float]]:
    if end <= start:
        return []

    min_len = max(1.4, float(os.environ.get("CAMERA_UPDATE_MIN_SEC", str(min_len))))
    max_len = max(min_len + 0.2, float(os.environ.get("CAMERA_UPDATE_MAX_SEC", str(max_len))))
    bounds = _speech_boundaries(start, end, speech_segments)
    intervals: List[Tuple[float, float]] = []
    cursor = start

    while cursor < end - 0.15:
        target = cursor + 3.0
        left = cursor + min_len
        right = min(end, cursor + max_len)

        candidates = [b for b in bounds if left <= b <= right and b > cursor]
        if candidates:
            nxt = min(candidates, key=lambda x: abs(x - target))
        else:
            nxt = min(end, target)

        if nxt <= cursor + 0.05:
            nxt = min(end, cursor + min_len)

        intervals.append((cursor, nxt))
        cursor = nxt

    if not intervals:
        return [(start, end)]

    last_s, last_e = intervals[-1]
    if end - last_e > 0.5:
        intervals[-1] = (last_s, end)

    return intervals


def _probe_video_dimensions(input_video: str) -> Tuple[int, int]:
    try:
        probe = subprocess.run(
            [
                FFPROBE_BIN,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=p=0:s=x",
                str(input_video),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        raw = (probe.stdout or "").strip().splitlines()
        if raw and "x" in raw[0]:
            width_s, height_s = raw[0].strip().split("x", 1)
            return max(1, int(width_s)), max(1, int(height_s))
    except Exception:
        pass
    return 1920, 1080


def _collect_motion_samples(input_video: str, start_ts: float, end_ts: float) -> List[Dict[str, float]]:
    sample_fps = max(1.0, float(os.environ.get("CAMERA_SAMPLE_FPS", "2.0")))
    analysis_width = max(320, int(os.environ.get("CAMERA_ANALYSIS_WIDTH", "640")))
    dur = max(0.5, end_ts - start_ts)
    cmd = [
        FFMPEG_BIN,
        "-hide_banner",
        "-loglevel",
        "verbose",
        "-ss",
        f"{start_ts:.3f}",
        "-to",
        f"{end_ts:.3f}",
        "-i",
        str(input_video),
        "-vf",
        f"fps={sample_fps:.3f},scale={analysis_width}:-2,format=gray,tblend=all_mode=difference,bbox",
        "-an",
        "-f",
        "null",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        output = f"{proc.stdout}\n{proc.stderr}"
    except Exception:
        return []

    pattern = re.compile(
        r"pts_time:(?P<pts>[0-9.]+)\s+"
        r"x1:(?P<x1>\d+)\s+x2:(?P<x2>\d+)\s+"
        r"y1:(?P<y1>\d+)\s+y2:(?P<y2>\d+)\s+"
        r"w:(?P<w>\d+)\s+h:(?P<h>\d+)"
    )
    samples: List[Dict[str, float]] = []
    for m in pattern.finditer(output):
        pts = float(m.group("pts"))
        x1 = float(m.group("x1"))
        x2 = float(m.group("x2"))
        w = float(m.group("w"))
        h = float(m.group("h"))
        if w <= 1 or h <= 1:
            continue
        is_full_frame = w >= analysis_width * 0.93
        center_ratio = max(0.0, min(1.0, ((x1 + x2) / 2.0) / float(analysis_width)))
        weight = max(1.0, w * h)
        if is_full_frame:
            # Keep as weak prior instead of dropping entirely.
            weight *= 0.08
        samples.append(
            {
                "time": start_ts + pts,
                "center_ratio": center_ratio,
                "weight": weight,
            }
        )
    return samples


def _weighted_ratio(samples: List[Dict[str, float]], default_ratio: float) -> float:
    if not samples:
        return default_ratio
    weighted_sum = sum(s["center_ratio"] * s["weight"] for s in samples)
    weight_sum = sum(s["weight"] for s in samples)
    if weight_sum <= 0:
        values = [s["center_ratio"] for s in samples]
        return float(statistics.median(values)) if values else default_ratio
    return max(0.0, min(1.0, weighted_sum / weight_sum))


def _smooth_ratios(raw_ratios: List[float], intervals: List[Tuple[float, float]]) -> List[float]:
    if not raw_ratios:
        return []
    dead_zone = max(0.01, min(0.20, float(os.environ.get("CAMERA_DEAD_ZONE_RATIO", "0.05"))))
    max_speed = max(0.02, min(0.60, float(os.environ.get("CAMERA_MAX_SPEED_RATIO_PER_SEC", "0.10"))))
    smooth_alpha = max(0.05, min(1.0, float(os.environ.get("CAMERA_SMOOTH_ALPHA", "0.45"))))
    edge_padding = max(0.0, min(0.35, float(os.environ.get("CAMERA_EDGE_PADDING_RATIO", "0.14"))))

    smoothed: List[float] = [max(edge_padding, min(1.0 - edge_padding, raw_ratios[0]))]
    for idx in range(1, len(raw_ratios)):
        prev = smoothed[-1]
        target = max(edge_padding, min(1.0 - edge_padding, raw_ratios[idx]))
        if abs(target - prev) <= dead_zone:
            target = prev

        seg_start, seg_end = intervals[min(idx, len(intervals) - 1)]
        seg_dur = max(0.4, seg_end - seg_start)
        max_delta = max_speed * seg_dur
        delta = target - prev
        if delta > max_delta:
            delta = max_delta
        elif delta < -max_delta:
            delta = -max_delta

        constrained = prev + delta
        eased = prev + (constrained - prev) * smooth_alpha
        smoothed.append(max(edge_padding, min(1.0 - edge_padding, eased)))
    return smoothed


def _clip_speech_segments(
    speech_segments: Optional[Sequence[dict]],
    start_ts: float,
    end_ts: float,
) -> List[Dict[str, float]]:
    if not speech_segments:
        return []
    clipped: List[Dict[str, float]] = []
    for seg in speech_segments:
        seg_start = max(start_ts, float(seg.get("start", start_ts)))
        seg_end = min(end_ts, float(seg.get("end", seg_start)))
        if seg_end <= seg_start:
            continue
        clipped.append({"start": seg_start, "end": seg_end})
    clipped.sort(key=lambda x: x["start"])
    return clipped


def _speech_overlap_seconds(seg_start: float, seg_end: float, speech_events: List[Dict[str, float]]) -> float:
    overlap = 0.0
    for event in speech_events:
        left = max(seg_start, float(event.get("start", 0.0)))
        right = min(seg_end, float(event.get("end", 0.0)))
        if right > left:
            overlap += right - left
    return overlap


def _extract_audio_samples(
    input_video: str,
    start_ts: float,
    end_ts: float,
    sample_rate: int = 16000,
) -> Tuple[np.ndarray, int]:
    if np is None:
        return [], sample_rate  # type: ignore[return-value]
    cmd = [
        FFMPEG_BIN,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-ss",
        f"{start_ts:.3f}",
        "-to",
        f"{end_ts:.3f}",
        "-i",
        str(input_video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True)
        if proc.returncode != 0 or not proc.stdout:
            return np.zeros(0, dtype=np.float32), sample_rate
        audio_i16 = np.frombuffer(proc.stdout, dtype=np.int16)
        if audio_i16.size == 0:
            return np.zeros(0, dtype=np.float32), sample_rate
        return (audio_i16.astype(np.float32) / 32768.0), sample_rate
    except Exception:
        return np.zeros(0, dtype=np.float32), sample_rate


def _segment_audio_features(
    audio: np.ndarray,
    sample_rate: int,
    clip_start: float,
    seg_start: float,
    seg_end: float,
) -> Optional[np.ndarray]:
    if np is None:
        return None
    if audio.size == 0:
        return None
    i0 = max(0, int((seg_start - clip_start) * sample_rate))
    i1 = min(audio.size, int((seg_end - clip_start) * sample_rate))
    if i1 <= i0:
        return None
    window = audio[i0:i1]
    if window.size < max(256, int(sample_rate * 0.25)):
        return None

    rms = float(np.sqrt(np.mean(np.square(window)) + 1e-8))
    zcr = float(np.mean((window[:-1] * window[1:]) < 0)) if window.size > 1 else 0.0
    tapered = window * np.hanning(window.size)
    spectrum = np.abs(np.fft.rfft(tapered))
    if spectrum.size <= 1 or float(np.sum(spectrum)) <= 1e-8:
        centroid_norm = 0.0
    else:
        freqs = np.fft.rfftfreq(tapered.size, d=1.0 / sample_rate)
        centroid = float(np.sum(freqs * spectrum) / np.sum(spectrum))
        centroid_norm = max(0.0, min(1.0, centroid / (sample_rate * 0.5)))
    return np.array([rms, zcr, centroid_norm], dtype=np.float32)


def _simple_kmeans(features: np.ndarray, k: int = 2, max_iter: int = 12) -> np.ndarray:
    if np is None:
        return []
    if features.size == 0:
        return np.zeros((0,), dtype=np.int32)
    n = features.shape[0]
    k = max(1, min(k, n))
    if k == 1:
        return np.zeros((n,), dtype=np.int32)

    # Deterministic initialization: lowest and highest energy points.
    first_idx = int(np.argmin(features[:, 0]))
    second_idx = int(np.argmax(features[:, 0]))
    centroids = np.stack([features[first_idx], features[second_idx]], axis=0).astype(np.float32)
    labels = np.zeros((n,), dtype=np.int32)

    for _ in range(max_iter):
        dist = np.sum((features[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dist, axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            cluster = features[labels == c]
            if cluster.size > 0:
                centroids[c] = np.mean(cluster, axis=0)
    return labels


def _smooth_speaker_timeline(events: List[Dict[str, object]]) -> List[Dict[str, object]]:
    if len(events) < 3:
        return events
    min_flip_dur = max(0.4, float(os.environ.get("SPEAKER_MIN_FLIP_SECONDS", "1.1")))
    smoothed = [dict(e) for e in events]
    for idx in range(1, len(smoothed) - 1):
        prev_spk = str(smoothed[idx - 1].get("speaker_id", "A"))
        curr_spk = str(smoothed[idx].get("speaker_id", "A"))
        next_spk = str(smoothed[idx + 1].get("speaker_id", "A"))
        curr_dur = float(smoothed[idx].get("end", 0.0)) - float(smoothed[idx].get("start", 0.0))
        if curr_dur <= min_flip_dur and prev_spk == next_spk and curr_spk != prev_spk:
            smoothed[idx]["speaker_id"] = prev_spk
    return smoothed


def _voice_only_speaker_timeline(
    input_video: str,
    start_ts: float,
    end_ts: float,
    speech_events: List[Dict[str, float]],
) -> List[Dict[str, object]]:
    if not speech_events:
        return []
    if np is None:
        logger.warning(json.dumps({"event": "voice_diarization_disabled", "reason": "numpy_unavailable"}, ensure_ascii=True))
        return [{"start": e["start"], "end": e["end"], "speaker_id": "A"} for e in speech_events]

    if os.environ.get("DISABLE_VOICE_DIARIZATION", "").strip().lower() in {"1", "true", "yes", "on"}:
        return [{"start": e["start"], "end": e["end"], "speaker_id": "A"} for e in speech_events]

    audio, sample_rate = _extract_audio_samples(input_video=input_video, start_ts=start_ts, end_ts=end_ts)
    if audio.size == 0:
        return [{"start": e["start"], "end": e["end"], "speaker_id": "A"} for e in speech_events]

    features: List[np.ndarray] = []
    valid_indices: List[int] = []
    for idx, event in enumerate(speech_events):
        feat = _segment_audio_features(
            audio=audio,
            sample_rate=sample_rate,
            clip_start=start_ts,
            seg_start=float(event["start"]),
            seg_end=float(event["end"]),
        )
        if feat is None:
            continue
        features.append(feat)
        valid_indices.append(idx)

    if not features:
        return [{"start": e["start"], "end": e["end"], "speaker_id": "A"} for e in speech_events]

    feat_mat = np.stack(features, axis=0)
    # Normalize per feature for stable clustering.
    mu = np.mean(feat_mat, axis=0)
    sigma = np.std(feat_mat, axis=0) + 1e-6
    norm_feat = (feat_mat - mu) / sigma

    enough_variance = float(np.std(norm_feat[:, 0])) > 0.15 or float(np.std(norm_feat[:, 2])) > 0.15
    num_speakers = 2 if (len(features) >= 3 and enough_variance) else 1
    labels = _simple_kmeans(norm_feat, k=num_speakers)

    result: List[Dict[str, object]] = []
    label_to_speaker = {0: "A", 1: "B", 2: "C"}
    label_lookup: Dict[int, int] = {}
    if num_speakers > 1:
        # Keep IDs stable by sorting cluster centroids on normalized centroid feature.
        cluster_mean: List[Tuple[int, float]] = []
        for lbl in range(num_speakers):
            idxs = np.where(labels == lbl)[0]
            if idxs.size == 0:
                continue
            cluster_mean.append((lbl, float(np.mean(norm_feat[idxs, 2]))))
        cluster_mean.sort(key=lambda x: x[1])
        label_lookup = {orig_lbl: new_idx for new_idx, (orig_lbl, _) in enumerate(cluster_mean)}

    for idx, event in enumerate(speech_events):
        speaker = "A"
        if idx in valid_indices:
            local = valid_indices.index(idx)
            raw_label = int(labels[local])
            mapped_label = label_lookup.get(raw_label, raw_label)
            speaker = label_to_speaker.get(mapped_label, "A")
        result.append({"start": event["start"], "end": event["end"], "speaker_id": speaker})

    result = _smooth_speaker_timeline(result)
    unique_speakers = sorted({str(r.get("speaker_id", "A")) for r in result})
    logger.info(
        json.dumps(
            {
                "event": "voice_speaker_timeline",
                "segments": len(result),
                "speakers": unique_speakers,
                "speaker_count": len(unique_speakers),
            },
            ensure_ascii=True,
        )
    )
    return result


def _speaker_for_interval(
    seg_start: float,
    seg_end: float,
    speaker_timeline: List[Dict[str, object]],
) -> Tuple[Optional[str], float]:
    if not speaker_timeline:
        return None, 0.0
    overlap_by_speaker: Dict[str, float] = {}
    total_overlap = 0.0
    for event in speaker_timeline:
        left = max(seg_start, float(event.get("start", 0.0)))
        right = min(seg_end, float(event.get("end", 0.0)))
        if right <= left:
            continue
        overlap = right - left
        speaker = str(event.get("speaker_id", "A"))
        overlap_by_speaker[speaker] = overlap_by_speaker.get(speaker, 0.0) + overlap
        total_overlap += overlap
    if not overlap_by_speaker:
        return None, 0.0
    dominant_speaker, dominant_overlap = max(overlap_by_speaker.items(), key=lambda x: x[1])
    dominance_ratio = dominant_overlap / max(1e-6, total_overlap)
    return dominant_speaker, max(0.0, min(1.0, dominance_ratio))


def _build_speaker_region_map(
    speaker_timeline: List[Dict[str, object]],
    motion_samples: List[Dict[str, float]],
) -> Dict[str, float]:
    if not speaker_timeline:
        return {}
    mapping: Dict[str, float] = {}
    speakers = sorted({str(e.get("speaker_id", "A")) for e in speaker_timeline})
    for idx, spk in enumerate(speakers):
        relevant_events = [e for e in speaker_timeline if str(e.get("speaker_id", "A")) == spk]
        relevant_samples: List[Dict[str, float]] = []
        for event in relevant_events:
            es = float(event.get("start", 0.0))
            ee = float(event.get("end", es))
            relevant_samples.extend([s for s in motion_samples if es <= s["time"] <= ee])
        if relevant_samples:
            mapping[spk] = _weighted_ratio(relevant_samples, default_ratio=0.5)
        else:
            if len(speakers) == 1:
                mapping[spk] = 0.5
            elif len(speakers) == 2:
                mapping[spk] = 0.38 if idx == 0 else 0.62
            else:
                mapping[spk] = max(0.1, min(0.9, 0.2 + (0.6 * idx / max(1, len(speakers) - 1))))
    return mapping


def _motion_metrics(interval_samples: List[Dict[str, float]], default_ratio: float) -> Dict[str, float]:
    if not interval_samples:
        return {"ratio": default_ratio, "strength": 0.0, "stability": 1.0, "scene_change": 0.0}

    centers = [float(s["center_ratio"]) for s in interval_samples]
    weights = [float(max(1.0, s["weight"])) for s in interval_samples]
    mean_ratio = _weighted_ratio(interval_samples, default_ratio=default_ratio)
    std_center = float(statistics.pstdev(centers)) if len(centers) > 1 else 0.0
    if len(centers) > 1:
        diffs = [abs(centers[i] - centers[i - 1]) for i in range(1, len(centers))]
        mean_delta = float(sum(diffs) / max(1, len(diffs)))
    else:
        mean_delta = 0.0
    weight_mean = float(sum(weights) / max(1, len(weights)))
    weight_std = float(statistics.pstdev(weights)) if len(weights) > 1 else 0.0
    scene_change = min(1.0, weight_std / max(1e-6, weight_mean))
    strength = min(1.0, (std_center * 4.0) + (mean_delta * 3.2))
    stability = max(0.0, 1.0 - strength)
    return {"ratio": mean_ratio, "strength": strength, "stability": stability, "scene_change": scene_change}


def _motion_camera_plan_for_intervals(
    input_video: str,
    start_ts: float,
    end_ts: float,
    intervals: List[Tuple[float, float]],
) -> List[Dict[str, float]]:
    if not intervals:
        return []
    samples = _collect_motion_samples(input_video=input_video, start_ts=start_ts, end_ts=end_ts)
    raw_ratios: List[float] = []
    prev_ratio = 0.5
    for seg_start, seg_end in intervals:
        seg_samples = [s for s in samples if seg_start <= s["time"] <= seg_end]
        seg_ratio = _weighted_ratio(seg_samples, default_ratio=prev_ratio)
        raw_ratios.append(seg_ratio)
        prev_ratio = seg_ratio
    smoothed_ratios = _smooth_ratios(raw_ratios, intervals)
    total_dur = max(0.5, end_ts - start_ts)
    zoom_start = max(1.0, float(os.environ.get("CAMERA_ZOOM_START", "1.03")))
    zoom_end = max(zoom_start, float(os.environ.get("CAMERA_ZOOM_END", "1.12")))
    plan: List[Dict[str, float]] = []
    for idx, (seg_start, seg_end) in enumerate(intervals):
        p0 = max(0.0, min(1.0, (seg_start - start_ts) / total_dur))
        p1 = max(0.0, min(1.0, (seg_end - start_ts) / total_dur))
        z0 = zoom_start + ((zoom_end - zoom_start) * p0)
        z1 = zoom_start + ((zoom_end - zoom_start) * p1)
        ratio_end = smoothed_ratios[idx]
        ratio_start = smoothed_ratios[idx - 1] if idx > 0 else ratio_end
        plan.append(
            {
                "ratio_start": ratio_start,
                "ratio_end": ratio_end,
                "zoom_start": z0,
                "zoom_end": z1,
                "mode": "motion",
                "source": "motion_only",
                "speaker_conf": 0.0,
                "motion_conf": 1.0,
            }
        )
    return plan


def _adaptive_camera_plan_for_intervals(
    input_video: str,
    start_ts: float,
    end_ts: float,
    intervals: List[Tuple[float, float]],
    speech_segments: Optional[Sequence[dict]],
) -> List[Dict[str, float]]:
    if not intervals:
        return []

    motion_samples = _collect_motion_samples(input_video=input_video, start_ts=start_ts, end_ts=end_ts)
    speech_events = _clip_speech_segments(speech_segments=speech_segments, start_ts=start_ts, end_ts=end_ts)
    speaker_timeline = _voice_only_speaker_timeline(
        input_video=input_video,
        start_ts=start_ts,
        end_ts=end_ts,
        speech_events=speech_events,
    ) if speech_events else []
    speaker_region_map = _build_speaker_region_map(speaker_timeline, motion_samples)

    switch_margin = max(0.03, float(os.environ.get("MODE_SWITCH_MARGIN", "0.08")))
    strong_margin = max(switch_margin + 0.02, float(os.environ.get("MODE_STRONG_SWITCH_MARGIN", "0.16")))
    min_mode_hold_sec = max(2.0, float(os.environ.get("MODE_MIN_HOLD_SECONDS", "3.2")))

    raw_ratios: List[float] = []
    mode_list: List[str] = []
    conf_list: List[Tuple[float, float]] = []
    source_list: List[str] = []
    active_speaker_list: List[Optional[str]] = []
    low_confidence_center_fallbacks = 0

    prev_ratio = 0.5
    prev_mode: Optional[str] = None
    hold_intervals_remaining = 0
    speaker_dominance_min = max(0.45, min(0.9, float(os.environ.get("SPEAKER_DOMINANCE_MIN", "0.56"))))

    for seg_start, seg_end in intervals:
        seg_dur = max(0.4, seg_end - seg_start)
        interval_motion = [s for s in motion_samples if seg_start <= s["time"] <= seg_end]
        motion = _motion_metrics(interval_motion, default_ratio=prev_ratio)

        speech_overlap = _speech_overlap_seconds(seg_start, seg_end, speech_events)
        speech_coverage = max(0.0, min(1.0, speech_overlap / seg_dur))
        dominant_speaker, dominance_ratio = _speaker_for_interval(seg_start, seg_end, speaker_timeline)
        has_voice_map = dominant_speaker in speaker_region_map if dominant_speaker else False
        # Reserved for future face detector integration; keeps scoring API stable.
        face_conf = max(0.0, min(1.0, float(os.environ.get("FACE_SIGNAL_CONFIDENCE", "0.0"))))

        # Continuity favors stable speaker turns over fragmented/noisy activity.
        if speech_overlap <= 1e-6:
            continuity = 0.0
        else:
            dominant_overlap = 0.0
            if dominant_speaker:
                for evt in speaker_timeline:
                    if str(evt.get("speaker_id", "A")) != dominant_speaker:
                        continue
                    left = max(seg_start, float(evt.get("start", 0.0)))
                    right = min(seg_end, float(evt.get("end", left)))
                    if right > left:
                        dominant_overlap += right - left
            continuity = max(0.0, min(1.0, dominant_overlap / max(1e-6, speech_overlap)))

        speaker_conf = (
            0.38 * speech_coverage
            + 0.18 * continuity
            + 0.16 * (1.0 - motion["strength"])
            + 0.10 * motion["stability"]
            + 0.10 * (1.0 if has_voice_map else 0.0)
            + 0.08 * face_conf
        )
        speaker_conf *= (0.85 + (0.15 * dominance_ratio))
        motion_conf = (
            0.50 * motion["strength"]
            + 0.20 * motion["scene_change"]
            + 0.20 * (1.0 - speech_coverage)
            + 0.10 * (1.0 - continuity)
        )
        speaker_conf = max(0.0, min(1.0, speaker_conf))
        motion_conf = max(0.0, min(1.0, motion_conf))
        speaker_low_conf = dominance_ratio < speaker_dominance_min

        proposed_mode = "speaker" if speaker_conf >= (motion_conf + switch_margin) else "motion"
        if prev_mode and hold_intervals_remaining > 0 and proposed_mode != prev_mode:
            if abs(speaker_conf - motion_conf) < strong_margin:
                proposed_mode = prev_mode
        if prev_mode == "speaker" and proposed_mode == "motion" and motion_conf < (speaker_conf + switch_margin * 0.75):
            proposed_mode = "speaker"
        if prev_mode == "motion" and proposed_mode == "speaker" and speaker_conf < (motion_conf + switch_margin * 0.75):
            proposed_mode = "motion"

        if prev_mode != proposed_mode:
            hold_intervals_remaining = max(1, int(math.ceil(min_mode_hold_sec / seg_dur)))
        else:
            hold_intervals_remaining = max(0, hold_intervals_remaining - 1)
        prev_mode = proposed_mode

        motion_ratio = motion["ratio"]
        speaker_ratio = (
            speaker_region_map.get(dominant_speaker, motion_ratio)
            if dominant_speaker
            else motion_ratio
        )
        if proposed_mode == "speaker" and dominant_speaker is not None:
            speaker_weight = (
                0.25
                + (0.35 * continuity)
                + (0.25 * speech_coverage)
                + (0.15 if has_voice_map else 0.0)
            )
            speaker_weight = max(0.35, min(0.90, speaker_weight))
            target_ratio = (speaker_ratio * speaker_weight) + (
                motion_ratio * (1.0 - speaker_weight)
            )
            if speaker_low_conf:
                target_ratio = (motion_ratio * 0.35) + 0.325
                low_confidence_center_fallbacks += 1
        else:
            target_ratio = motion_ratio
        target_ratio = max(0.08, min(0.92, float(target_ratio)))

        if proposed_mode == "speaker" and speaker_low_conf:
            source = "speaker_low_conf_center"
        elif proposed_mode == "speaker" and has_voice_map and face_conf >= 0.45:
            source = "face_voice"
        elif proposed_mode == "speaker" and has_voice_map:
            source = "voice_motion"
        elif proposed_mode == "speaker":
            source = "speaker_fallback"
        else:
            source = "motion_only"

        raw_ratios.append(target_ratio)
        mode_list.append(proposed_mode)
        conf_list.append((speaker_conf, motion_conf))
        source_list.append(source)
        active_speaker_list.append(dominant_speaker)
        prev_ratio = target_ratio

    smoothed = _smooth_ratios(raw_ratios, intervals)
    total_dur = max(0.5, end_ts - start_ts)
    speaker_zoom_start = max(1.0, float(os.environ.get("SPEAKER_ZOOM_START", "1.00")))
    speaker_zoom_end = max(speaker_zoom_start, float(os.environ.get("SPEAKER_ZOOM_END", "1.05")))
    motion_zoom_start = max(1.0, float(os.environ.get("MOTION_ZOOM_START", "1.01")))
    motion_zoom_end = max(motion_zoom_start, float(os.environ.get("MOTION_ZOOM_END", "1.08")))

    plan: List[Dict[str, float]] = []
    for idx, (seg_start, seg_end) in enumerate(intervals):
        mode = mode_list[idx]
        p0 = max(0.0, min(1.0, (seg_start - start_ts) / total_dur))
        p1 = max(0.0, min(1.0, (seg_end - start_ts) / total_dur))
        if mode == "speaker":
            z0 = speaker_zoom_start + ((speaker_zoom_end - speaker_zoom_start) * p0)
            z1 = speaker_zoom_start + ((speaker_zoom_end - speaker_zoom_start) * p1)
        else:
            z0 = motion_zoom_start + ((motion_zoom_end - motion_zoom_start) * p0)
            z1 = motion_zoom_start + ((motion_zoom_end - motion_zoom_start) * p1)
        ratio_end = smoothed[idx]
        ratio_start = smoothed[idx - 1] if idx > 0 else ratio_end
        spk_conf, mov_conf = conf_list[idx]
        plan.append(
            {
                "ratio_start": ratio_start,
                "ratio_end": ratio_end,
                "zoom_start": z0,
                "zoom_end": z1,
                "mode": mode,
                "source": source_list[idx],
                "speaker_conf": spk_conf,
                "motion_conf": mov_conf,
                "active_speaker": active_speaker_list[idx] or "",
            }
        )

    speaker_count = sum(1 for m in mode_list if m == "speaker")
    motion_count = sum(1 for m in mode_list if m == "motion")
    logger.info(
        json.dumps(
            {
                "event": "adaptive_framing_plan",
                "segments": len(intervals),
                "speaker_mode_segments": speaker_count,
                "motion_mode_segments": motion_count,
                "speech_events": len(speech_events),
                "voice_speakers": len({str(x.get('speaker_id', 'A')) for x in speaker_timeline}) if speaker_timeline else 0,
                "motion_samples": len(motion_samples),
                "speaker_focus_low_conf_fallbacks": low_confidence_center_fallbacks,
            },
            ensure_ascii=True,
        )
    )
    return plan


def _ease_expr(start_value: float, end_value: float, duration: float) -> str:
    safe_duration = max(0.25, duration)
    p_expr = f"if(lt(t,0),0,if(gt(t,{safe_duration:.3f}),1,t/{safe_duration:.3f}))"
    return (
        f"({start_value:.6f})+(({end_value:.6f})-({start_value:.6f}))*"
        f"(3*pow(({p_expr}),2)-2*pow(({p_expr}),3))"
    )


def _video_chain_for_interval(
    seg_start: float,
    seg_end: float,
    ratio_start: float,
    ratio_end: float,
    zoom_start: float,
    zoom_end: float,
) -> str:
    seg_dur = max(0.25, seg_end - seg_start)
    ratio_expr = _ease_expr(ratio_start, ratio_end, seg_dur)
    # FFmpeg 8+ removed crop eval=frame. Keep crop size static per segment and
    # animate position (x) with t-based easing for smooth motion and stability.
    segment_zoom = max(1.0, (float(zoom_start) + float(zoom_end)) / 2.0)
    crop_w_expr = f"trunc(((ih*9/16)/{segment_zoom:.6f})/2)*2"
    crop_h_expr = f"trunc((ih/{segment_zoom:.6f})/2)*2"
    x_expr = f"max(0,min(iw-({crop_w_expr}),({ratio_expr})*iw-({crop_w_expr})/2))"
    y_expr = f"max(0,(ih-({crop_h_expr}))/2)"
    filters = [
        f"trim=start={seg_start:.3f}:end={seg_end:.3f}",
        "setpts=PTS-STARTPTS",
        f"crop='{crop_w_expr}':'{crop_h_expr}':'{x_expr}':'{y_expr}'",
        *_quality_finish_filters(),
    ]
    return ",".join(filters)


def _default_thread_count() -> int:
    configured = os.environ.get("FFMPEG_THREADS", "").strip()
    if configured:
        try:
            return max(1, int(configured))
        except ValueError:
            pass
    return max(1, os.cpu_count() or 2)


def _ffmpeg_supports_subtitles() -> bool:
    global _SUBTITLE_FILTER_AVAILABLE

    if os.environ.get("DISABLE_SUBTITLES", "").strip().lower() in {"1", "true", "yes", "on"}:
        _SUBTITLE_FILTER_AVAILABLE = False
        return False

    with _SUBTITLE_FILTER_LOCK:
        if _SUBTITLE_FILTER_AVAILABLE is not None:
            return _SUBTITLE_FILTER_AVAILABLE

        try:
            proc = subprocess.run(
                [FFMPEG_BIN, "-hide_banner", "-filters"],
                check=False,
                capture_output=True,
                text=True,
            )
            filter_dump = f"{proc.stdout}\n{proc.stderr}".lower()
            _SUBTITLE_FILTER_AVAILABLE = bool(re.search(r"\bsubtitles\b", filter_dump))
        except Exception:
            _SUBTITLE_FILTER_AVAILABLE = False

        if not _SUBTITLE_FILTER_AVAILABLE:
            logger.warning(
                json.dumps(
                    {
                        "event": "ffmpeg_subtitles_filter_unavailable",
                        "hint": "Install ffmpeg with libass support for ASS subtitles.",
                    },
                    ensure_ascii=True,
                )
            )

        return _SUBTITLE_FILTER_AVAILABLE


def subtitles_filter_available() -> bool:
    return _ffmpeg_supports_subtitles()


def _ffmpeg_supports_drawtext() -> bool:
    global _DRAWTEXT_FILTER_AVAILABLE
    with _DRAWTEXT_FILTER_LOCK:
        if _DRAWTEXT_FILTER_AVAILABLE is not None:
            return _DRAWTEXT_FILTER_AVAILABLE
        try:
            proc = subprocess.run(
                [FFMPEG_BIN, "-hide_banner", "-filters"],
                check=False,
                capture_output=True,
                text=True,
            )
            filter_dump = f"{proc.stdout}\n{proc.stderr}".lower()
            _DRAWTEXT_FILTER_AVAILABLE = bool(re.search(r"\bdrawtext\b", filter_dump))
        except Exception:
            _DRAWTEXT_FILTER_AVAILABLE = False
        return _DRAWTEXT_FILTER_AVAILABLE


def _escape_drawtext_text(text: str) -> str:
    escaped = (text or "").replace("\\", "\\\\")
    escaped = escaped.replace(":", "\\:")
    escaped = escaped.replace("'", "\\'")
    escaped = escaped.replace("%", "\\%")
    escaped = escaped.replace("\n", " ")
    return escaped


def _simple_render_cmd(
    input_video: str,
    start_ts: float,
    end_ts: float,
    output_path: Path,
    preset: str,
    crf: int,
    thread_count: int,
    caption_text: Optional[str] = None,
    caption_image: Optional[Path] = None,
) -> List[str]:
    simple_zoom = max(1.0, min(1.25, float(os.environ.get("SIMPLE_RENDER_ZOOM", "1.04"))))
    simple_ratio = max(0.08, min(0.92, float(os.environ.get("SIMPLE_RENDER_RATIO", "0.50"))))
    simple_pan_amplitude = max(0.0, min(0.08, float(os.environ.get("SIMPLE_RENDER_PAN_AMPLITUDE", "0.015"))))
    crop_w_expr = f"trunc(((ih*9/16)/{simple_zoom:.4f})/2)*2"
    crop_h_expr = f"trunc((ih/{simple_zoom:.4f})/2)*2"
    base_ratio_expr = f"({simple_ratio:.4f})"
    if simple_pan_amplitude > 0:
        ratio_expr = f"{base_ratio_expr}+({simple_pan_amplitude:.4f})*sin(t*1.25)"
    else:
        ratio_expr = base_ratio_expr
    x_expr = f"max(0,min(iw-({crop_w_expr}),({ratio_expr})*iw-({crop_w_expr})/2))"
    y_expr = f"max(0,(ih-({crop_h_expr}))/2)"
    vf_chain = [
        f"crop='{crop_w_expr}':'{crop_h_expr}':'{x_expr}':'{y_expr}'",
        *_quality_finish_filters(),
    ]
    if caption_text and _ffmpeg_supports_drawtext():
        draw_text = _escape_drawtext_text(caption_text.strip()[:120])
        vf_chain.append(
            "drawtext=text='{}':fontsize=52:fontcolor=white:borderw=2:bordercolor=black:"
            "x=(w-text_w)/2:y=h-220:box=1:boxcolor=black@0.45:boxborderw=18".format(draw_text)
        )
    base_cmd = [
        FFMPEG_BIN,
        "-y",
        "-loglevel",
        "error",
        "-nostdin",
        "-ss",
        f"{start_ts:.3f}",
        "-to",
        f"{end_ts:.3f}",
        "-i",
        str(input_video),
    ]

    if caption_image:
        filter_complex = (
            f"[0:v]{','.join(vf_chain)}[vbase];"
            "[1:v]format=rgba[vcap];"
            "[vbase][vcap]overlay=x=(W-w)/2:y=H-h-160:format=auto[vout]"
        )
        base_cmd.extend(
            [
                "-i",
                str(caption_image),
                "-filter_complex",
                filter_complex,
                "-map",
                "[vout]",
                "-map",
                "0:a:0?",
            ]
        )
    else:
        base_cmd.extend(
            [
                "-map",
                "0:v:0",
                "-map",
                "0:a:0?",
                "-vf",
                ",".join(vf_chain),
            ]
        )

    base_cmd.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-profile:v",
            "high",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            os.environ.get("FFMPEG_AUDIO_BITRATE", "192k"),
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            "-threads",
            str(thread_count),
            str(output_path),
        ]
    )
    return base_cmd


def _create_caption_overlay_image(caption_text: str, tmp_dir: Optional[str] = None) -> Optional[Path]:
    text = (caption_text or "").strip()
    if not text:
        return None

    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        logger.warning(json.dumps({"event": "caption_overlay_image_unavailable", "reason": "pillow_missing"}, ensure_ascii=True))
        return None

    safe_text = re.sub(r"\s+", " ", text).strip().upper()
    if not safe_text:
        return None

    width = 1080
    height = 320
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    try:
        font_candidates = [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/SFNS.ttf",
        ]
        font = None
        for candidate in font_candidates:
            if os.path.exists(candidate):
                font = ImageFont.truetype(candidate, 66)
                break
        if font is None:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    lines = textwrap.wrap(safe_text, width=28)[:2]
    if not lines:
        lines = [safe_text[:48]]

    line_boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    line_heights = [(box[3] - box[1]) for box in line_boxes]
    text_block_height = sum(line_heights) + (16 * (len(lines) - 1))
    top = max(20, (height - text_block_height) // 2)

    box_pad_h = 42
    box_pad_v = 28
    draw.rounded_rectangle(
        (40, max(8, top - box_pad_v), width - 40, min(height - 8, top + text_block_height + box_pad_v)),
        radius=30,
        fill=(0, 0, 0, 170),
    )

    cursor_y = top
    for idx, line in enumerate(lines):
        line_w = line_boxes[idx][2] - line_boxes[idx][0]
        draw.text(((width - line_w) / 2, cursor_y), line, font=font, fill=(255, 255, 255, 255))
        cursor_y += line_heights[idx] + 16

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=tmp_dir)
    tmp.close()
    out_path = Path(tmp.name)
    image.save(out_path, format="PNG")
    return out_path


def render_vertical_short_safe(
    video_path: str,
    start_time: float,
    end_time: float,
    output_path: str,
    caption_text: Optional[str] = None,
) -> str:
    """
    Reliable 9:16 render path for FFmpeg 8+.

    - No deprecated crop eval option
    - Trim + center crop + scale + pad
    - Optional drawtext caption with safe fallback to no caption
    """
    start_ts = max(0.0, float(start_time))
    end_ts = max(start_ts + 0.5, float(end_time))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    safe_crop_w = "if(gte(iw/ih,9/16),trunc((ih*9/16)/2)*2,trunc(iw/2)*2)"
    safe_crop_h = "if(gte(iw/ih,9/16),trunc(ih/2)*2,trunc((iw*16/9)/2)*2)"
    vf_chain = [
        f"crop='{safe_crop_w}':'{safe_crop_h}':'(iw-ow)/2':'(ih-oh)/2'",
        *_quality_finish_filters(),
    ]
    if caption_text and _ffmpeg_supports_drawtext():
        draw_text = _escape_drawtext_text(caption_text.strip()[:120])
        vf_chain.append(
            "drawtext=text='{}':fontsize=52:fontcolor=white:borderw=2:bordercolor=black:"
            "x=(w-text_w)/2:y=h-220:box=1:boxcolor=black@0.45:boxborderw=18".format(draw_text)
        )

    cmd = [
        FFMPEG_BIN,
        "-y",
        "-loglevel",
        "error",
        "-nostdin",
        "-ss",
        f"{start_ts:.3f}",
        "-to",
        f"{end_ts:.3f}",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-vf",
        ",".join(vf_chain),
        "-c:v",
        "libx264",
        "-preset",
        os.environ.get("FFMPEG_PRESET", "medium"),
        "-crf",
        os.environ.get("FFMPEG_CRF", "20"),
        "-profile:v",
        "high",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        os.environ.get("FFMPEG_AUDIO_BITRATE", "192k"),
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        str(out),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        if caption_text:
            fallback_chain = [
                f"crop='{safe_crop_w}':'{safe_crop_h}':'(iw-ow)/2':'(ih-oh)/2'",
                *_quality_finish_filters(),
            ]
            fallback_cmd = [
                FFMPEG_BIN,
                "-y",
                "-loglevel",
                "error",
                "-nostdin",
                "-ss",
                f"{start_ts:.3f}",
                "-to",
                f"{end_ts:.3f}",
                "-i",
                str(video_path),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0?",
                "-vf",
                ",".join(fallback_chain),
                "-c:v",
                "libx264",
                "-preset",
                os.environ.get("FFMPEG_PRESET", "medium"),
                "-crf",
                os.environ.get("FFMPEG_CRF", "20"),
                "-profile:v",
                "high",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                os.environ.get("FFMPEG_AUDIO_BITRATE", "192k"),
                "-ac",
                "2",
                "-movflags",
                "+faststart",
                str(out),
            ]
            subprocess.run(fallback_cmd, check=True, capture_output=True, text=True)
        else:
            raise

    return str(out)


def _repair_audio_track(input_video: str, start_ts: float, end_ts: float, output_path: Path) -> bool:
    fixed_path = output_path.with_name(f"{output_path.stem}.audiofix{output_path.suffix}")
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-loglevel",
        "error",
        "-nostdin",
        "-ss",
        f"{start_ts:.3f}",
        "-to",
        f"{end_ts:.3f}",
        "-i",
        str(input_video),
        "-i",
        str(output_path),
        "-map",
        "1:v:0",
        "-map",
        "0:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        os.environ.get("FFMPEG_AUDIO_BITRATE", "192k"),
        "-ac",
        "2",
        "-shortest",
        str(fixed_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        fixed_path.replace(output_path)
        logger.info(json.dumps({"event": "render_audio_repaired", "output_video": str(output_path)}, ensure_ascii=True))
        return True
    except Exception as exc:
        logger.warning(
            json.dumps(
                {
                    "event": "render_audio_repair_failed",
                    "output_video": str(output_path),
                    "error": str(exc),
                },
                ensure_ascii=True,
            )
        )
        if fixed_path.exists():
            fixed_path.unlink(missing_ok=True)
        return False


def _has_audio_stream(video_path: Path) -> bool:
    if not video_path.exists():
        return False
    try:
        probe = subprocess.run(
            [
                FFPROBE_BIN,
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        output = f"{probe.stdout}\n{probe.stderr}".lower()
        return "audio" in output
    except Exception:
        return True


def render_vertical_clip(
    input_video: str,
    start: float,
    end: float,
    output_video: str,
    subtitle_file: Optional[str] = None,
    caption_text: Optional[str] = None,
    speech_segments: Optional[Sequence[dict]] = None,
    preset: str = "medium",
    crf: int = 18,
    threads: Optional[int] = None,
) -> str:
    start_ts = max(0.0, float(start))
    end_ts = max(start_ts + 0.5, float(end))

    intervals = _build_jump_intervals(start_ts, end_ts, speech_segments=speech_segments)
    try:
        camera_plan = _adaptive_camera_plan_for_intervals(
            input_video=input_video,
            start_ts=start_ts,
            end_ts=end_ts,
            intervals=intervals,
            speech_segments=speech_segments,
        )
    except Exception as adaptive_exc:
        logger.warning(
            json.dumps(
                {
                    "event": "adaptive_framing_failed_fallback_motion",
                    "error": str(adaptive_exc),
                },
                ensure_ascii=True,
            )
        )
        camera_plan = _motion_camera_plan_for_intervals(
            input_video=input_video,
            start_ts=start_ts,
            end_ts=end_ts,
            intervals=intervals,
        )
    output_path = Path(output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subtitles_supported = _ffmpeg_supports_subtitles() if subtitle_file else False
    drawtext_supported = _ffmpeg_supports_drawtext()

    if caption_text and not drawtext_supported and not subtitles_supported:
        caption_image: Optional[Path] = _create_caption_overlay_image(caption_text, tmp_dir=str(output_path.parent))
        try:
            if caption_image:
                logger.warning(
                    json.dumps(
                        {
                            "event": "caption_image_overlay_fallback",
                            "reason": "ffmpeg_text_filters_unavailable",
                        },
                        ensure_ascii=True,
                    )
                )
                effective_preset = preset or os.environ.get("FFMPEG_PRESET", "medium")
                thread_count = threads if threads is not None else _default_thread_count()
                image_cmd = _simple_render_cmd(
                    input_video=input_video,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    output_path=output_path,
                    preset=effective_preset,
                    crf=crf,
                    thread_count=thread_count,
                    caption_text=None,
                    caption_image=caption_image,
                )
                try:
                    subprocess.run(image_cmd, check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError:
                    fallback_cmd = _simple_render_cmd(
                        input_video=input_video,
                        start_ts=start_ts,
                        end_ts=end_ts,
                        output_path=output_path,
                        preset=effective_preset,
                        crf=crf,
                        thread_count=thread_count,
                        caption_text=None,
                        caption_image=None,
                    )
                    subprocess.run(fallback_cmd, check=True, capture_output=True, text=True)

                if not _has_audio_stream(output_path):
                    _repair_audio_track(input_video=input_video, start_ts=start_ts, end_ts=end_ts, output_path=output_path)
                return str(output_path)
            logger.warning(
                json.dumps(
                    {
                        "event": "caption_disabled_no_supported_filter",
                        "reason": "subtitles_and_drawtext_unavailable",
                    },
                    ensure_ascii=True,
                )
            )
        finally:
            if caption_image and caption_image.exists():
                caption_image.unlink(missing_ok=True)

    graph_parts: List[str] = []

    for idx, (seg_start, seg_end) in enumerate(intervals):
        cam = camera_plan[idx] if idx < len(camera_plan) else {
            "ratio_start": 0.5,
            "ratio_end": 0.5,
            "zoom_start": 1.03,
            "zoom_end": 1.08,
        }
        graph_parts.append(
            f"[0:v]{_video_chain_for_interval(seg_start, seg_end, cam['ratio_start'], cam['ratio_end'], cam['zoom_start'], cam['zoom_end'])}[v{idx}]"
        )
        graph_parts.append(
            f"[0:a]atrim=start={seg_start:.3f}:end={seg_end:.3f},"
            f"asetpts=PTS-STARTPTS,aresample=async=1:first_pts=0[a{idx}]"
        )

    if len(intervals) > 1:
        concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(len(intervals)))
        graph_parts.append(f"{concat_inputs}concat=n={len(intervals)}:v=1:a=1[vcat][acat]")
        video_label = "vcat"
        audio_label = "acat"
    else:
        video_label = "v0"
        audio_label = "a0"

    if subtitle_file and subtitles_supported:
        subtitle_abs = os.path.abspath(subtitle_file)
        subtitle_path = _escape_filter_path(subtitle_abs)
        graph_parts.append(f"[{video_label}]subtitles=filename='{subtitle_path}'[vout]")
        out_video_label = "vout"
    elif subtitle_file:
        logger.warning(json.dumps({"event": "subtitle_disabled_for_render"}, ensure_ascii=True))
        if caption_text and _ffmpeg_supports_drawtext():
            draw_text = _escape_drawtext_text(caption_text.strip()[:120])
            graph_parts.append(
                f"[{video_label}]drawtext=text='{draw_text}':fontsize=52:fontcolor=white:borderw=2:"
                "bordercolor=black:x=(w-text_w)/2:y=h-220:box=1:boxcolor=black@0.45:boxborderw=18[vtxt]"
            )
            out_video_label = "vtxt"
        else:
            out_video_label = video_label
    else:
        if caption_text and _ffmpeg_supports_drawtext():
            draw_text = _escape_drawtext_text(caption_text.strip()[:120])
            graph_parts.append(
                f"[{video_label}]drawtext=text='{draw_text}':fontsize=52:fontcolor=white:borderw=2:"
                "bordercolor=black:x=(w-text_w)/2:y=h-220:box=1:boxcolor=black@0.45:boxborderw=18[vtxt]"
            )
            out_video_label = "vtxt"
        else:
            out_video_label = video_label

    filter_complex = ";".join(graph_parts)
    effective_preset = preset or os.environ.get("FFMPEG_PRESET", "medium")
    thread_count = threads if threads is not None else _default_thread_count()

    cmd = [
        FFMPEG_BIN,
        "-y",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(input_video),
        "-filter_complex",
        filter_complex,
        "-map",
        f"[{out_video_label}]",
        "-map",
        f"[{audio_label}]",
        "-c:v",
        "libx264",
        "-preset",
        effective_preset,
        "-crf",
        str(crf),
        "-profile:v",
        "high",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        os.environ.get("FFMPEG_AUDIO_BITRATE", "192k"),
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        "-threads",
        str(thread_count),
        str(output_path),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr_tail = (exc.stderr or "").strip()[-300:]
        if subtitle_file:
            logger.warning(
                json.dumps(
                    {
                        "event": "ffmpeg_subtitle_retry_without_subtitles",
                        "error": str(exc),
                        "stderr": stderr_tail,
                        "subtitle_file": os.path.abspath(subtitle_file),
                    },
                    ensure_ascii=True,
                )
            )
            return render_vertical_clip(
                input_video=input_video,
                start=start,
                end=end,
                output_video=output_video,
                subtitle_file=None,
                caption_text=caption_text,
                speech_segments=speech_segments,
                preset=preset,
                crf=crf,
                threads=threads,
            )
        logger.warning(
            json.dumps(
                {
                    "event": "ffmpeg_complex_retry_simple",
                    "error": str(exc),
                    "stderr": stderr_tail,
                },
                ensure_ascii=True,
            )
        )
        simple_cmd = _simple_render_cmd(
            input_video=input_video,
            start_ts=start_ts,
            end_ts=end_ts,
            output_path=output_path,
            preset=effective_preset,
            crf=crf,
            thread_count=thread_count,
            caption_text=caption_text,
        )
        try:
            subprocess.run(simple_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as simple_exc:
            if caption_text:
                logger.warning(
                    json.dumps(
                        {
                            "event": "ffmpeg_simple_retry_without_caption",
                            "error": str(simple_exc),
                        },
                        ensure_ascii=True,
                    )
                )
                fallback_simple_cmd = _simple_render_cmd(
                    input_video=input_video,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    output_path=output_path,
                    preset=effective_preset,
                    crf=crf,
                    thread_count=thread_count,
                    caption_text=None,
                )
                subprocess.run(fallback_simple_cmd, check=True, capture_output=True, text=True)
            else:
                raise

    if not _has_audio_stream(output_path):
        logger.warning(
            json.dumps(
                {"event": "render_missing_audio_retry_simple", "output_video": str(output_path)},
                ensure_ascii=True,
            )
        )
        simple_cmd = _simple_render_cmd(
            input_video=input_video,
            start_ts=start_ts,
            end_ts=end_ts,
            output_path=output_path,
            preset=effective_preset,
            crf=crf,
            thread_count=thread_count,
            caption_text=caption_text if drawtext_supported else None,
        )
        subprocess.run(simple_cmd, check=True, capture_output=True, text=True)
        if not _has_audio_stream(output_path):
            _repair_audio_track(input_video=input_video, start_ts=start_ts, end_ts=end_ts, output_path=output_path)
    return str(output_path)


def cut_vertical_clip(
    input_video,
    start,
    end,
    output_video,
    preset="medium",
    crf=18,
):
    return render_vertical_clip(
        input_video=input_video,
        start=start,
        end=end,
        output_video=output_video,
        subtitle_file=None,
        caption_text=None,
        speech_segments=None,
        preset=preset,
        crf=crf,
    )

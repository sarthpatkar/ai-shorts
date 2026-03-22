import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from feedback_store import load_all_feedback

DEFAULT_WEIGHTS = {
    "hook_score": 0.30,
    "retention_score": 0.25,
    "curiosity_score": 0.20,
    "emotional_score": 0.15,
    "clarity_score": 0.10,
}

DEFAULT_SELECTION_WEIGHTS = {
    "heuristic_score": 0.30,
    "hook_score": 0.15,
    "audio_score": 0.20,
    "visual_score": 0.10,
    "retention_score": 0.25,
}

DEFAULT_FILTER_RULES = {
    "retention_mid_min": 5.2,
    "retention_ending_min": 4.8,
    "retention_dropoff_max": 6.8,
    "retention_story_min": 4.5,
    "ai_retention_component_min": 5.8,
}

DEFAULT_MEMORY = {
    "best": [],
    "worst": [],
    "updated_at": 0,
}

_STOPWORDS = {
    "this",
    "that",
    "with",
    "from",
    "your",
    "you",
    "what",
    "when",
    "then",
    "have",
    "just",
    "about",
    "because",
    "there",
    "they",
    "them",
    "would",
    "could",
    "should",
    "into",
    "also",
    "really",
    "very",
    "over",
    "under",
}


def _weights_path() -> Path:
    configured = os.environ.get("WEIGHTS_PATH", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path("data") / "weights.json"


def _selection_weights_path() -> Path:
    configured = os.environ.get("SELECTION_WEIGHTS_PATH", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path("data") / "selection_weights.json"


def _patterns_path() -> Path:
    configured = os.environ.get("PATTERNS_PATH", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path("data") / "patterns.json"


def _filters_path() -> Path:
    configured = os.environ.get("FILTER_RULES_PATH", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path("data") / "filter_rules.json"


def _memory_path() -> Path:
    configured = os.environ.get("CLIP_MEMORY_PATH", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path("data") / "clip_memory.json"


def _learning_log_path() -> Path:
    configured = os.environ.get("LEARNING_LOG_PATH", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path("data") / "learning_events.jsonl"


def _safe_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    os.replace(tmp, path)


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _clamp(value: Any, low: float, high: float, default: float) -> float:
    try:
        num = float(value)
    except Exception:
        num = default
    return max(low, min(high, num))


def _normalize_rate(value: float) -> float:
    value = float(value)
    if value > 1.0:
        value = value / 100.0
    return max(0.0, min(1.0, value))


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    cleaned: Dict[str, float] = {}
    for key, default in DEFAULT_WEIGHTS.items():
        cleaned[key] = _clamp(weights.get(key, default), 0.05, 0.50, default)
    total = sum(cleaned.values())
    if total <= 0:
        return dict(DEFAULT_WEIGHTS)
    return {k: v / total for k, v in cleaned.items()}


def _normalize_selection_weights(weights: Dict[str, float]) -> Dict[str, float]:
    cleaned: Dict[str, float] = {}
    for key, default in DEFAULT_SELECTION_WEIGHTS.items():
        cleaned[key] = _clamp(weights.get(key, default), 0.05, 0.60, default)
    total = sum(cleaned.values())
    if total <= 0:
        return dict(DEFAULT_SELECTION_WEIGHTS)
    return {k: v / total for k, v in cleaned.items()}


def _pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x <= 1e-12 or den_y <= 1e-12:
        return 0.0
    return num / (den_x * den_y)


def _percentile(values: Sequence[float], q: float, default: float = 0.0) -> float:
    if not values:
        return default
    data = sorted(float(v) for v in values)
    if len(data) == 1:
        return data[0]
    q = max(0.0, min(1.0, float(q)))
    idx = q * (len(data) - 1)
    low = int(math.floor(idx))
    high = int(math.ceil(idx))
    if low == high:
        return data[low]
    frac = idx - low
    return data[low] * (1.0 - frac) + data[high] * frac


def _safe_score_from_row(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    scores = row.get("scores", {})
    if not isinstance(scores, dict):
        return default
    return _clamp(scores.get(key, default), 0.0, 10.0, default)


def _token_set(text: str) -> set:
    return {tok for tok in re.findall(r"[a-z0-9']+", str(text).lower()) if len(tok) > 2 and tok not in _STOPWORDS}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def _first_line(text: str) -> str:
    clean = re.sub(r"\s+", " ", str(text).strip())
    if not clean:
        return ""
    return re.split(r"(?<=[.!?])\s+", clean)[0]


def _derive_pattern_label(text: str) -> str:
    t = str(text).lower()
    if "?" in t:
        return "question_hook"
    if re.search(r"\b(shocking|insane|crazy|unbelievable)\b", t):
        return "shock_hook"
    if re.search(r"\b(why|how|secret|nobody)\b", t):
        return "curiosity_hook"
    if re.search(r"\b(so|therefore|finally|bottom line)\b", t):
        return "payoff_ending"
    return "statement_hook"


def _row_target_performance(row: Dict[str, Any]) -> float:
    metrics = row.get("metrics", {})
    if isinstance(metrics, dict) and any(float(metrics.get(k, 0.0) or 0.0) > 0.0 for k in ("watch_time", "completion_rate", "likes", "shares", "comments", "watch_time_percentage")):
        return performance_score(metrics)
    scores = row.get("scores", {}) if isinstance(row.get("scores"), dict) else {}
    return simulate_performance_from_scores(scores)


def load_weights() -> Dict[str, float]:
    path = _weights_path()
    if not path.exists():
        return dict(DEFAULT_WEIGHTS)
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            return dict(DEFAULT_WEIGHTS)
        merged = dict(DEFAULT_WEIGHTS)
        for key in merged:
            merged[key] = float(payload.get(key, merged[key]))
        return _normalize_weights(merged)
    except Exception:
        return dict(DEFAULT_WEIGHTS)


def save_weights(weights: Dict[str, float]) -> Dict[str, float]:
    normalized = _normalize_weights(weights)
    _safe_write_json(_weights_path(), normalized)
    return normalized


def load_selection_weights() -> Dict[str, float]:
    path = _selection_weights_path()
    if not path.exists():
        return dict(DEFAULT_SELECTION_WEIGHTS)
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            return dict(DEFAULT_SELECTION_WEIGHTS)
        merged = dict(DEFAULT_SELECTION_WEIGHTS)
        for key in merged:
            merged[key] = float(payload.get(key, merged[key]))
        return _normalize_selection_weights(merged)
    except Exception:
        return dict(DEFAULT_SELECTION_WEIGHTS)


def save_selection_weights(weights: Dict[str, float]) -> Dict[str, float]:
    normalized = _normalize_selection_weights(weights)
    _safe_write_json(_selection_weights_path(), normalized)
    return normalized


def load_filter_rules() -> Dict[str, float]:
    path = _filters_path()
    if not path.exists():
        return dict(DEFAULT_FILTER_RULES)
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            return dict(DEFAULT_FILTER_RULES)
        merged = dict(DEFAULT_FILTER_RULES)
        for key, default in merged.items():
            merged[key] = float(payload.get(key, default))
        return merged
    except Exception:
        return dict(DEFAULT_FILTER_RULES)


def save_filter_rules(rules: Dict[str, float]) -> Dict[str, float]:
    merged = dict(DEFAULT_FILTER_RULES)
    for key, default in merged.items():
        merged[key] = float(rules.get(key, default))
    _safe_write_json(_filters_path(), merged)
    return merged


def load_clip_memory() -> Dict[str, Any]:
    path = _memory_path()
    if not path.exists():
        return dict(DEFAULT_MEMORY)
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            return dict(DEFAULT_MEMORY)
        best = payload.get("best", [])
        worst = payload.get("worst", [])
        return {
            "best": [dict(x) for x in best if isinstance(x, dict)][:200],
            "worst": [dict(x) for x in worst if isinstance(x, dict)][:200],
            "updated_at": int(payload.get("updated_at", 0) or 0),
        }
    except Exception:
        return dict(DEFAULT_MEMORY)


def save_clip_memory(memory: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "best": [dict(x) for x in memory.get("best", []) if isinstance(x, dict)][:200],
        "worst": [dict(x) for x in memory.get("worst", []) if isinstance(x, dict)][:200],
        "updated_at": int(memory.get("updated_at", int(time.time()))),
    }
    _safe_write_json(_memory_path(), payload)
    return payload


def performance_score(metrics: Dict[str, Any]) -> float:
    completion = _normalize_rate(float(metrics.get("completion_rate", metrics.get("watch_time_percentage", 0.0)) or 0.0))
    watch_time = _normalize_rate(float(metrics.get("watch_time", metrics.get("watch_time_percentage", 0.0)) or 0.0))
    shares = min(1.0, float(metrics.get("shares", 0.0) or 0.0) / 1000.0)
    likes = min(1.0, float(metrics.get("likes", 0.0) or 0.0) / 2000.0)
    comments = min(1.0, float(metrics.get("comments", 0.0) or 0.0) / 500.0)
    score = (completion * 0.50) + (watch_time * 0.20) + (shares * 0.15) + (likes * 0.10) + (comments * 0.05)
    return max(0.0, min(1.0, float(score)))


def simulate_performance_from_scores(scores: Dict[str, Any]) -> float:
    hook = _clamp(scores.get("hook_score", 5.0), 0.0, 10.0, 5.0)
    audio = _clamp(scores.get("audio_score", 5.0), 0.0, 10.0, 5.0)
    visual = _clamp(scores.get("visual_score", 5.0), 0.0, 10.0, 5.0)
    retention = _clamp(scores.get("retention_score", 5.0), 0.0, 10.0, 5.0)
    final = _clamp(scores.get("final_score", scores.get("score", 5.0)), 0.0, 10.0, 5.0)
    sim = ((final * 0.45) + (retention * 0.35) + (hook * 0.10) + (audio * 0.05) + (visual * 0.05)) / 10.0
    return max(0.0, min(1.0, float(sim)))


def simulate_metric_bundle(scores: Dict[str, Any]) -> Dict[str, Any]:
    perf = simulate_performance_from_scores(scores)
    watch_pct = round(perf, 4)
    watch_time = round(max(0.0, perf * 0.95), 4)
    likes = int(round(perf * 1200))
    shares = int(round(perf * 210))
    comments = int(round(perf * 95))
    return {
        "views": 0,
        "watch_time": watch_time,
        "watch_time_percentage": watch_pct,
        "completion_rate": watch_pct,
        "likes": likes,
        "shares": shares,
        "comments": comments,
        "simulated": True,
    }


def memory_similarity(text: str, memory: Dict[str, Any]) -> Dict[str, float]:
    tokens = _token_set(text)
    best = [row for row in memory.get("best", []) if isinstance(row, dict)]
    worst = [row for row in memory.get("worst", []) if isinstance(row, dict)]

    def _max_sim(rows: Sequence[Dict[str, Any]]) -> float:
        sims: List[float] = []
        for row in rows:
            row_tokens = _token_set(str(row.get("text", "")))
            if not row_tokens:
                continue
            sims.append(_jaccard(tokens, row_tokens))
        return max(sims) if sims else 0.0

    good = _max_sim(best)
    bad = _max_sim(worst)
    bias = max(-2.0, min(2.0, (good - bad) * 1.8))
    return {
        "good_similarity": round(good, 4),
        "bad_similarity": round(bad, 4),
        "bias": round(bias, 4),
    }


def confidence_score(scores: Dict[str, Any], text: str, memory: Dict[str, Any]) -> Dict[str, float]:
    retention = _clamp(scores.get("retention_score", 5.0), 0.0, 10.0, 5.0)
    hook = _clamp(scores.get("hook_score", 5.0), 0.0, 10.0, 5.0)
    audio = _clamp(scores.get("audio_score", 5.0), 0.0, 10.0, 5.0)
    visual = _clamp(scores.get("visual_score", 5.0), 0.0, 10.0, 5.0)
    final_score = _clamp(scores.get("final_score", scores.get("score", 5.0)), 0.0, 10.0, 5.0)
    base_strength = (
        (retention * 0.35)
        + (hook * 0.20)
        + (audio * 0.20)
        + (visual * 0.10)
        + (final_score * 0.15)
    ) / 10.0
    sim = memory_similarity(text, memory)
    memory_strength = max(0.0, min(1.0, (sim["good_similarity"] * 0.65) + ((1.0 - sim["bad_similarity"]) * 0.35)))
    confidence = max(0.0, min(10.0, ((base_strength * 0.75) + (memory_strength * 0.25)) * 10.0))
    return {
        "confidence_score": round(confidence, 3),
        "memory_good_similarity": float(sim["good_similarity"]),
        "memory_bad_similarity": float(sim["bad_similarity"]),
        "memory_bias": float(sim["bias"]),
    }


def update_weights_from_feedback(
    feedback_records: Sequence[Dict[str, Any]],
    learning_rate: float = 0.05,
    min_samples: int = 12,
) -> Dict[str, float]:
    valid = [row for row in feedback_records if isinstance(row, dict) and isinstance(row.get("scores"), dict)]
    if len(valid) < min_samples:
        return load_weights()

    perf = [_row_target_performance(row) for row in valid]
    current = load_weights()
    updated = dict(current)
    for key in DEFAULT_WEIGHTS:
        xs = [_safe_score_from_row(row, key, 0.0) for row in valid]
        updated[key] = current.get(key, DEFAULT_WEIGHTS[key]) + (_pearson_correlation(xs, perf) * learning_rate)
    normalized = _normalize_weights(updated)
    return save_weights(normalized)


def update_selection_weights_from_feedback(
    feedback_records: Sequence[Dict[str, Any]],
    learning_rate: float = 0.06,
    min_samples: int = 14,
) -> Dict[str, float]:
    valid = [row for row in feedback_records if isinstance(row, dict) and isinstance(row.get("scores"), dict)]
    if len(valid) < min_samples:
        return load_selection_weights()

    perf = [_row_target_performance(row) for row in valid]
    current = load_selection_weights()
    updated = dict(current)
    for key in DEFAULT_SELECTION_WEIGHTS:
        xs = [_safe_score_from_row(row, key, 0.0) for row in valid]
        updated[key] = current.get(key, DEFAULT_SELECTION_WEIGHTS[key]) + (_pearson_correlation(xs, perf) * learning_rate)
    normalized = _normalize_selection_weights(updated)
    return save_selection_weights(normalized)


def _structure_effects(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    if not rows:
        return {}
    perf = [_row_target_performance(row) for row in rows]
    hi = _percentile(perf, 0.70, default=0.65)
    lo = _percentile(perf, 0.30, default=0.35)
    top = [row for row, p in zip(rows, perf) if p >= hi]
    bottom = [row for row, p in zip(rows, perf) if p <= lo]
    if not top or not bottom:
        return {}

    def _rate(records: Sequence[Dict[str, Any]], predicate) -> float:
        if not records:
            return 0.0
        hits = sum(1 for row in records if predicate(row))
        return hits / max(1, len(records))

    def _first_question(row: Dict[str, Any]) -> bool:
        return "?" in _first_line(str(row.get("text", "")))

    def _strong_ending(row: Dict[str, Any]) -> bool:
        ending = _safe_score_from_row(row, "ending_score", 0.0)
        return ending >= 6.8

    def _low_audio(row: Dict[str, Any]) -> bool:
        audio = _safe_score_from_row(row, "audio_score", 10.0)
        return audio <= 4.4

    def _high_dropoff(row: Dict[str, Any]) -> bool:
        dropoff = _safe_score_from_row(row, "dropoff_risk_score", _safe_score_from_row(row, "dropoff_risk", 0.0))
        return dropoff >= 6.6

    predicates = {
        "question_first_line": _first_question,
        "strong_ending": _strong_ending,
        "low_audio_energy": _low_audio,
        "high_dropoff_risk": _high_dropoff,
    }
    effects: Dict[str, Dict[str, float]] = {}
    for key, pred in predicates.items():
        top_rate = _rate(top, pred)
        bottom_rate = _rate(bottom, pred)
        effects[key] = {
            "top_rate": round(top_rate, 4),
            "bottom_rate": round(bottom_rate, 4),
            "lift": round(top_rate - bottom_rate, 4),
        }
    return effects


def extract_top_patterns(feedback_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [row for row in feedback_records if isinstance(row, dict)]
    if not rows:
        return {"top_hooks": [], "top_words": [], "top_patterns": [], "keyword_lift": [], "structure_effects": {}, "score_correlations": {}}

    scored_rows = sorted(((_row_target_performance(row), row) for row in rows), key=lambda x: x[0], reverse=True)
    top_n = max(1, int(len(scored_rows) * 0.25))
    bottom_n = max(1, int(len(scored_rows) * 0.25))
    top_rows = [row for _, row in scored_rows[:top_n]]
    bottom_rows = [row for _, row in scored_rows[-bottom_n:]]

    hook_counter: Counter[str] = Counter()
    top_word_counter: Counter[str] = Counter()
    bottom_word_counter: Counter[str] = Counter()
    pattern_counter: Counter[str] = Counter()

    for row in top_rows:
        hook = str(row.get("hook", "")).strip()
        text = str(row.get("text", "")).strip()
        caption = str(row.get("caption", "")).strip()
        if hook:
            hook_counter[hook] += 1
        for tok in _token_set(f"{text} {hook} {caption}"):
            top_word_counter[tok] += 1
        pattern_counter[_derive_pattern_label(hook or text)] += 1

    for row in bottom_rows:
        text = str(row.get("text", "")).strip()
        hook = str(row.get("hook", "")).strip()
        caption = str(row.get("caption", "")).strip()
        for tok in _token_set(f"{text} {hook} {caption}"):
            bottom_word_counter[tok] += 1

    keyword_lift: List[Dict[str, Any]] = []
    for tok, top_count in top_word_counter.most_common(120):
        top_rate = top_count / max(1, len(top_rows))
        bottom_rate = bottom_word_counter.get(tok, 0) / max(1, len(bottom_rows))
        lift = top_rate - bottom_rate
        if lift <= 0.04:
            continue
        keyword_lift.append(
            {
                "keyword": tok,
                "lift": round(lift, 4),
                "top_rate": round(top_rate, 4),
                "bottom_rate": round(bottom_rate, 4),
            }
        )

    score_keys = set(DEFAULT_SELECTION_WEIGHTS) | {
        "mid_engagement_score",
        "ending_score",
        "dropoff_risk_score",
        "story_structure_score",
    }
    perf = [_row_target_performance(row) for row in rows]
    correlations: Dict[str, float] = {}
    for key in sorted(score_keys):
        xs = [_safe_score_from_row(row, key, _safe_score_from_row(row, key.replace("_score", ""), 0.0)) for row in rows]
        correlations[key] = round(_pearson_correlation(xs, perf), 4)

    return {
        "top_hooks": [k for k, _ in hook_counter.most_common(20)],
        "top_words": [k for k, _ in top_word_counter.most_common(30)],
        "top_patterns": [k for k, _ in pattern_counter.most_common(20)],
        "keyword_lift": keyword_lift[:30],
        "structure_effects": _structure_effects(rows),
        "score_correlations": correlations,
    }


def load_patterns() -> Dict[str, Any]:
    path = _patterns_path()
    if not path.exists():
        return {"top_hooks": [], "top_words": [], "top_patterns": [], "keyword_lift": [], "structure_effects": {}, "score_correlations": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            return {"top_hooks": [], "top_words": [], "top_patterns": [], "keyword_lift": [], "structure_effects": {}, "score_correlations": {}}
        return {
            "top_hooks": [str(x) for x in payload.get("top_hooks", [])[:20]],
            "top_words": [str(x) for x in payload.get("top_words", [])[:30]],
            "top_patterns": [str(x) for x in payload.get("top_patterns", [])[:20]],
            "keyword_lift": [dict(x) for x in payload.get("keyword_lift", [])[:30] if isinstance(x, dict)],
            "structure_effects": dict(payload.get("structure_effects", {})) if isinstance(payload.get("structure_effects", {}), dict) else {},
            "score_correlations": dict(payload.get("score_correlations", {})) if isinstance(payload.get("score_correlations", {}), dict) else {},
        }
    except Exception:
        return {"top_hooks": [], "top_words": [], "top_patterns": [], "keyword_lift": [], "structure_effects": {}, "score_correlations": {}}


def save_patterns(patterns: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "top_hooks": [str(x) for x in patterns.get("top_hooks", [])[:20]],
        "top_words": [str(x) for x in patterns.get("top_words", [])[:30]],
        "top_patterns": [str(x) for x in patterns.get("top_patterns", [])[:20]],
        "keyword_lift": [dict(x) for x in patterns.get("keyword_lift", [])[:30] if isinstance(x, dict)],
        "structure_effects": dict(patterns.get("structure_effects", {})) if isinstance(patterns.get("structure_effects", {}), dict) else {},
        "score_correlations": dict(patterns.get("score_correlations", {})) if isinstance(patterns.get("score_correlations", {}), dict) else {},
    }
    _safe_write_json(_patterns_path(), payload)
    return payload


def update_filter_rules_from_feedback(feedback_records: Sequence[Dict[str, Any]], min_samples: int = 16) -> Dict[str, float]:
    rows = [row for row in feedback_records if isinstance(row, dict) and isinstance(row.get("scores"), dict)]
    if len(rows) < min_samples:
        return load_filter_rules()

    perf = [_row_target_performance(row) for row in rows]
    hi = _percentile(perf, 0.70, default=0.65)
    lo = _percentile(perf, 0.30, default=0.35)
    top = [row for row, p in zip(rows, perf) if p >= hi]
    bottom = [row for row, p in zip(rows, perf) if p <= lo]
    if not top:
        return load_filter_rules()

    top_mid = [_safe_score_from_row(row, "mid_engagement_score", 5.0) for row in top]
    top_ending = [_safe_score_from_row(row, "ending_score", 5.0) for row in top]
    top_story = [_safe_score_from_row(row, "story_structure_score", 5.0) for row in top]
    top_dropoff = [
        _safe_score_from_row(row, "dropoff_risk_score", _safe_score_from_row(row, "dropoff_risk", 5.0))
        for row in top
    ]
    bottom_dropoff = [
        _safe_score_from_row(row, "dropoff_risk_score", _safe_score_from_row(row, "dropoff_risk", 5.0))
        for row in bottom
    ]

    current = load_filter_rules()
    updated = dict(current)
    updated["retention_mid_min"] = _clamp(_percentile(top_mid, 0.20, 5.2) - 0.15, 3.8, 8.6, current["retention_mid_min"])
    updated["retention_ending_min"] = _clamp(_percentile(top_ending, 0.20, 4.8) - 0.10, 3.6, 8.6, current["retention_ending_min"])
    updated["retention_story_min"] = _clamp(_percentile(top_story, 0.20, 4.5) - 0.10, 3.5, 8.4, current["retention_story_min"])

    top_dropoff_hi = _percentile(top_dropoff, 0.82, 6.8)
    bottom_dropoff_mid = _percentile(bottom_dropoff, 0.50, 7.4) if bottom_dropoff else top_dropoff_hi + 0.5
    blended_dropoff = (top_dropoff_hi * 0.75) + (bottom_dropoff_mid * 0.25)
    updated["retention_dropoff_max"] = _clamp(blended_dropoff, 4.8, 9.2, current["retention_dropoff_max"])

    updated["ai_retention_component_min"] = _clamp(
        min(updated["retention_mid_min"], updated["retention_ending_min"], updated["retention_story_min"]) - 0.2,
        4.5,
        8.0,
        current["ai_retention_component_min"],
    )

    return save_filter_rules(updated)


def update_clip_memory_from_feedback(feedback_records: Sequence[Dict[str, Any]], max_items: int = 120) -> Dict[str, Any]:
    rows = [row for row in feedback_records if isinstance(row, dict) and str(row.get("text", "")).strip()]
    if not rows:
        return load_clip_memory()
    scored = sorted(((_row_target_performance(row), row) for row in rows), key=lambda x: x[0], reverse=True)
    keep = max(12, min(max_items, int(len(scored) * 0.35)))

    def _compact_row(row: Dict[str, Any], perf: float) -> Dict[str, Any]:
        return {
            "clip_id": str(row.get("clip_id", "")),
            "text": str(row.get("text", ""))[:1200],
            "performance": round(float(perf), 5),
            "retention_score": _safe_score_from_row(row, "retention_score", 0.0),
            "final_score": _safe_score_from_row(row, "final_score", _safe_score_from_row(row, "score", 0.0)),
            "timestamp": int(row.get("timestamp", int(time.time()))),
        }

    best_rows = [_compact_row(row, perf) for perf, row in scored[:keep]]
    worst_rows = [_compact_row(row, perf) for perf, row in scored[-keep:]]
    memory = {
        "best": best_rows,
        "worst": worst_rows,
        "updated_at": int(time.time()),
    }
    return save_clip_memory(memory)


def update_learning_from_feedback(feedback_records: Sequence[Dict[str, Any]] = ()) -> Dict[str, Any]:
    rows = list(feedback_records) if feedback_records else load_all_feedback()
    before_selection = load_selection_weights()
    before_filters = load_filter_rules()

    legacy_weights = update_weights_from_feedback(rows)
    selection_weights = update_selection_weights_from_feedback(rows)
    patterns = save_patterns(extract_top_patterns(rows))
    filters = update_filter_rules_from_feedback(rows)
    memory = update_clip_memory_from_feedback(rows)

    decision_payload = {
        "event": "learning_update",
        "timestamp": int(time.time()),
        "feedback_count": len(rows),
        "selection_weight_delta": {
            key: round(selection_weights.get(key, 0.0) - before_selection.get(key, 0.0), 6)
            for key in DEFAULT_SELECTION_WEIGHTS
        },
        "filter_delta": {
            key: round(filters.get(key, 0.0) - before_filters.get(key, 0.0), 6)
            for key in DEFAULT_FILTER_RULES
        },
        "top_patterns": patterns.get("top_patterns", [])[:5],
    }
    _append_jsonl(_learning_log_path(), decision_payload)

    return {
        "weights": legacy_weights,
        "selection_weights": selection_weights,
        "patterns": patterns,
        "filters": filters,
        "memory": memory,
        "feedback_count": len(rows),
    }


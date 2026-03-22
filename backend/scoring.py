import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

EMOTIONAL_WORDS = {
    "shocking",
    "secret",
    "mistake",
    "never",
    "truth",
    "warning",
    "crazy",
    "insane",
    "danger",
    "unbelievable",
    "massive",
    "disaster",
}
PATTERN_INTERRUPTS = {"wait", "stop", "listen", "hold on", "but", "however", "nobody"}
CURIOSITY_TERMS = {"why", "how", "secret", "nobody", "what", "hidden", "revealed"}
CONNECTORS = {"but", "however", "because", "so", "then", "now", "therefore", "meanwhile"}

DEFAULT_WEIGHTS = {
    "hook_score": 0.30,
    "retention_score": 0.25,
    "curiosity_score": 0.20,
    "emotional_score": 0.15,
    "clarity_score": 0.10,
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\n", " ").strip())


def _words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+", (text or "").lower())


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", normalize_text(text)) if s.strip()]


def _clamp(value: float, low: float = 0.0, high: float = 10.0) -> float:
    return round(max(low, min(high, value)), 3)


def _safe_score(default: float = 5.0) -> float:
    return _clamp(default)


def hook_score(text: str) -> float:
    payload = normalize_text(text)
    if not payload:
        return _safe_score(2.0)

    first_words = payload.split()[:12]
    lead = " ".join(first_words)
    lead_lower = lead.lower()
    score = 2.0

    if "?" in lead:
        score += 2.2
    if re.search(r"\b(you|your)\b", lead_lower):
        score += 1.5

    emo_hits = sum(1 for w in EMOTIONAL_WORDS if re.search(rf"\b{re.escape(w)}\b", lead_lower))
    score += min(2.5, emo_hits * 0.9)

    interrupt_hits = sum(1 for w in PATTERN_INTERRUPTS if w in lead_lower)
    score += min(2.0, interrupt_hits * 0.8)

    if re.search(r"\d", lead):
        score += 0.8

    if len(first_words) < 6:
        score -= 0.5

    return _clamp(score)


def first_3s_power(text: str) -> float:
    payload = normalize_text(text)
    if not payload:
        return _safe_score(2.0)

    first_words = payload.split()[:8]
    lead = " ".join(first_words)
    lead_lower = lead.lower()
    score = 2.0

    if "?" in lead:
        score += 2.2
    if re.search(r"\b(you|your)\b", lead_lower):
        score += 1.6
    if re.search(r"\d", lead):
        score += 0.8

    emo_hits = sum(1 for w in EMOTIONAL_WORDS if re.search(rf"\b{re.escape(w)}\b", lead_lower))
    score += min(2.2, emo_hits * 0.9)

    interrupt_hits = sum(1 for w in PATTERN_INTERRUPTS if w in lead_lower)
    score += min(2.0, interrupt_hits * 0.8)

    return _clamp(score)


def retention_score(text: str) -> float:
    payload = normalize_text(text)
    words = payload.split()
    if not words:
        return _safe_score(2.0)

    word_count = len(words)
    score = 3.0

    if 40 <= word_count <= 90:
        score += 3.0
    elif 25 <= word_count < 40 or 90 < word_count <= 110:
        score += 1.6
    elif word_count < 20 or word_count > 140:
        score -= 1.8

    sents = _sentences(payload)
    sent_lens = [len(s.split()) for s in sents] if sents else []
    if sent_lens:
        variation = max(sent_lens) - min(sent_lens)
        if variation >= 8:
            score += 1.4
        elif variation <= 2:
            score -= 0.9

    interrupts = sum(1 for w in PATTERN_INTERRUPTS if re.search(rf"\b{re.escape(w)}\b", payload.lower()))
    score += min(1.8, interrupts * 0.6)

    return _clamp(score)


def curiosity_score(text: str) -> float:
    payload = normalize_text(text).lower()
    if not payload:
        return _safe_score(2.0)

    score = 2.0
    if "?" in payload:
        score += 2.0

    term_hits = sum(1 for w in CURIOSITY_TERMS if re.search(rf"\b{re.escape(w)}\b", payload))
    score += min(3.0, term_hits * 0.8)

    open_loop_hits = sum(1 for w in {"but", "however"} if re.search(rf"\b{re.escape(w)}\b", payload))
    score += min(1.6, open_loop_hits * 0.8)

    return _clamp(score)


def emotional_score(text: str) -> float:
    payload = normalize_text(text).lower()
    if not payload:
        return _safe_score(2.0)

    hits = sum(1 for w in EMOTIONAL_WORDS if re.search(rf"\b{re.escape(w)}\b", payload))
    score = 2.0 + min(6.0, hits * 1.1)
    if "!" in payload:
        score += 0.6
    return _clamp(score)


def clarity_score(text: str) -> float:
    payload = normalize_text(text)
    tokens = _words(payload)
    if not tokens:
        return _safe_score(2.0)

    avg_word_len = sum(len(t) for t in tokens) / max(1, len(tokens))
    score = 7.0

    if avg_word_len > 6.0:
        score -= 2.2
    elif avg_word_len > 5.2:
        score -= 1.2
    elif avg_word_len < 4.0:
        score += 0.8

    sents = _sentences(payload)
    if sents:
        avg_sent_len = sum(len(_words(s)) for s in sents) / max(1, len(sents))
        if avg_sent_len > 22:
            score -= 1.8
        elif avg_sent_len > 16:
            score -= 0.9
        elif avg_sent_len < 8:
            score += 0.5

    return _clamp(score)


def dropoff_risk(text: str) -> float:
    payload = normalize_text(text).lower()
    if not payload:
        return 1.2

    words = payload.split()
    sents = _sentences(payload)
    risk = 0.2

    if len(words) > 110:
        risk += 1.1
    elif len(words) > 90:
        risk += 0.6

    connector_hits = sum(1 for c in CONNECTORS if re.search(rf"\b{re.escape(c)}\b", payload))
    if connector_hits == 0:
        risk += 0.7

    if sents:
        sent_lens = [len(s.split()) for s in sents]
        variation = max(sent_lens) - min(sent_lens)
        if variation <= 2:
            risk += 0.8

    if payload.count("?") == 0 and payload.count("!") == 0:
        risk += 0.3

    return round(max(0.0, min(2.0, risk)), 3)


def _load_weights() -> Dict[str, float]:
    try:
        from learning_engine import load_weights

        loaded = load_weights()
        if not isinstance(loaded, dict):
            return dict(DEFAULT_WEIGHTS)
        weights = dict(DEFAULT_WEIGHTS)
        for key in weights:
            raw = loaded.get(key, weights[key])
            try:
                weights[key] = float(raw)
            except Exception:
                pass
        total = sum(max(0.0, v) for v in weights.values())
        if total <= 0:
            return dict(DEFAULT_WEIGHTS)
        return {k: max(0.0, v) / total for k, v in weights.items()}
    except Exception:
        return dict(DEFAULT_WEIGHTS)


def score_breakdown(text: str, weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    try:
        hs = hook_score(text)
        fps = first_3s_power(text)
        rs = retention_score(text)
        cs = curiosity_score(text)
        es = emotional_score(text)
        cls = clarity_score(text)
        dr = dropoff_risk(text)
    except Exception:
        hs, fps, rs, cs, es, cls, dr = 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 0.7

    w = weights or _load_weights()
    base_weighted = (
        hs * float(w.get("hook_score", DEFAULT_WEIGHTS["hook_score"]))
        + rs * float(w.get("retention_score", DEFAULT_WEIGHTS["retention_score"]))
        + cs * float(w.get("curiosity_score", DEFAULT_WEIGHTS["curiosity_score"]))
        + es * float(w.get("emotional_score", DEFAULT_WEIGHTS["emotional_score"]))
        + cls * float(w.get("clarity_score", DEFAULT_WEIGHTS["clarity_score"]))
    )
    final = (fps * 0.5) + (base_weighted * 0.5) - dr

    return {
        "hook_score": _clamp(hs),
        "first_3s_power": _clamp(fps),
        "retention_score": _clamp(rs),
        "curiosity_score": _clamp(cs),
        "emotional_score": _clamp(es),
        "clarity_score": _clamp(cls),
        "dropoff_risk": round(max(0.0, dr), 3),
        "final_score": _clamp(final),
    }


def viral_score_v2(text: str, weights: Optional[Dict[str, float]] = None) -> float:
    return score_breakdown(text, weights=weights).get("final_score", 5.0)


def _sentence_priority(sentence: str) -> float:
    payload = normalize_text(sentence)
    lower = payload.lower()
    if not payload:
        return 0.0

    score = 0.0
    score += sum(1.0 for w in EMOTIONAL_WORDS if re.search(rf"\b{re.escape(w)}\b", lower))
    score += sum(0.9 for w in CURIOSITY_TERMS if re.search(rf"\b{re.escape(w)}\b", lower))
    if re.search(r"\b(you|your)\b", lower):
        score += 1.0
    if "?" in payload:
        score += 1.2
    return score


def smart_compress(text: str, max_words: int = 35) -> str:
    payload = normalize_text(text)
    if not payload:
        return payload

    sentences = _sentences(payload)
    if not sentences:
        return " ".join(payload.split()[:max_words])

    scored = sorted(((s, _sentence_priority(s)) for s in sentences), key=lambda item: item[1], reverse=True)
    picked = [scored[0][0]]
    if len(scored) > 1 and scored[1][1] > 0:
        picked.append(scored[1][0])

    compressed = " ".join(picked).strip()
    words = compressed.split()
    if len(words) > max_words:
        compressed = " ".join(words[:max_words])
    if not compressed:
        compressed = " ".join(payload.split()[:max_words])
    return compressed


def _clamp_keep_count(total: int, keep_ratio: float, min_keep: int, max_keep: int) -> int:
    if total <= 0:
        return 0
    by_ratio = int(round(total * keep_ratio))
    target = max(min_keep, by_ratio)
    target = min(max_keep, target)
    return max(1, min(total, target))


def pre_score_chunks(
    chunks: Sequence[Dict[str, Any]],
    keep_ratio: float = 0.4,
    min_keep: int = 2,
    max_keep: int = 6,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    for chunk in chunks:
        text = normalize_text(str(chunk.get("text", "")))
        breakdown = score_breakdown(text)
        row = {
            **chunk,
            "text": text,
            "original_text": text,
            "score": breakdown["final_score"],
            "viral_score_v2": breakdown["final_score"],
            "score_components": {
                "hook_score": breakdown["hook_score"],
                "first_3s_power": breakdown["first_3s_power"],
                "retention_score": breakdown["retention_score"],
                "curiosity_score": breakdown["curiosity_score"],
                "emotional_score": breakdown["emotional_score"],
                "clarity_score": breakdown["clarity_score"],
                "dropoff_risk": breakdown["dropoff_risk"],
            },
        }
        if debug:
            row["score_debug"] = breakdown
        scored.append(row)

    scored.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    keep_n = _clamp_keep_count(len(scored), keep_ratio=keep_ratio, min_keep=min_keep, max_keep=max_keep)
    return scored[:keep_n]


def rank_final_clips(chunks: Sequence[Dict[str, Any]], top_k: int = 3, debug: bool = False) -> List[Dict[str, Any]]:
    ranked: List[Tuple[float, Dict[str, Any]]] = []
    for clip in chunks:
        text = normalize_text(str(clip.get("text", "")))
        breakdown = score_breakdown(text)
        merged = {**clip, "score": breakdown["final_score"], "viral_score_v2": breakdown["final_score"]}
        merged["score_components"] = {
            "hook_score": breakdown["hook_score"],
            "first_3s_power": breakdown["first_3s_power"],
            "retention_score": breakdown["retention_score"],
            "curiosity_score": breakdown["curiosity_score"],
            "emotional_score": breakdown["emotional_score"],
            "clarity_score": breakdown["clarity_score"],
            "dropoff_risk": breakdown["dropoff_risk"],
        }
        if debug:
            merged["score_debug"] = breakdown
        ranked.append((breakdown["final_score"], merged))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in ranked[: max(1, int(top_k))]]


def score_chunks(chunks: List[dict], top_k: int = 3) -> List[dict]:
    pre = pre_score_chunks(chunks, keep_ratio=1.0, min_keep=1, max_keep=max(1, len(chunks)))
    return rank_final_clips(pre, top_k=top_k)

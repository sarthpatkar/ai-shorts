import re
from typing import Dict, List, Optional, Sequence

CURIOSITY_TERMS = {
    "secret",
    "nobody",
    "why",
    "how",
    "what",
    "revealed",
    "truth",
    "behind",
    "hidden",
    "never",
}
SHOCK_TERMS = {
    "insane",
    "crazy",
    "shocking",
    "unbelievable",
    "wild",
    "massive",
    "huge",
    "disaster",
    "mistake",
}
WARNING_TERMS = {"danger", "warning", "risk", "avoid", "alert"}
LOW_ENERGY_TERMS = {
    "um",
    "uh",
    "like",
    "basically",
    "actually",
    "kind of",
    "sort of",
}
STORY_LONG_TERMS = {
    "because",
    "therefore",
    "which means",
    "so the",
    "the reason",
    "the result",
    "finally",
    "in the end",
}
INCOMPLETE_TAIL_TERMS = {
    "and",
    "but",
    "because",
    "so",
    "then",
    "if",
    "when",
    "which",
}
SENTENCE_END_RE = re.compile(r"[.!?][\"']?$")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def first_line(text: str, max_words: int = 14) -> str:
    clean = _normalize(text)
    if not clean:
        return ""

    sentence = re.split(r"(?<=[.!?])\s+", clean)[0]
    words = sentence.split()
    if len(words) <= max_words:
        return sentence
    return " ".join(words[:max_words])


def _classify_hook_type(text: str) -> str:
    lowered = text.lower()
    if "?" in text:
        return "question"
    if any(term in lowered for term in SHOCK_TERMS):
        return "shock"
    if any(term in lowered for term in CURIOSITY_TERMS):
        return "curiosity"
    if any(term in lowered for term in WARNING_TERMS):
        return "warning"
    return "statement"


def _hook_score_text(text: str) -> float:
    lead = first_line(text)
    lowered = lead.lower()

    question_score = 3.0 if "?" in lead else 0.0
    numeric_score = 2.4 if re.search(r"\d", lead) else 0.0
    you_score = 1.7 if re.search(r"\b(you|your)\b", lowered) else 0.0
    curiosity_score = min(3.0, sum(1.0 for term in CURIOSITY_TERMS if term in lowered))
    shock_score = min(3.0, sum(1.0 for term in SHOCK_TERMS if term in lowered))
    warning_score = min(2.0, sum(1.0 for term in WARNING_TERMS if term in lowered))
    brevity_boost = 0.8 if len(lead.split()) <= 12 else 0.0

    return 1.0 + question_score + numeric_score + you_score + curiosity_score + shock_score + warning_score + brevity_boost


def segment_energy(text: str) -> float:
    clean = _normalize(text)
    if not clean:
        return 0.0

    words = clean.split()
    unique_ratio = len(set(w.lower() for w in words)) / max(1, len(words))
    exclaim_bonus = 0.4 if "!" in clean else 0.0
    question_bonus = 0.55 if "?" in clean else 0.0
    numeric_bonus = 0.5 if re.search(r"\d", clean) else 0.0
    low_energy_penalty = sum(0.2 for term in LOW_ENERGY_TERMS if term in clean.lower())

    score = 1.4 + unique_ratio + exclaim_bonus + question_bonus + numeric_bonus - low_energy_penalty
    return max(0.0, min(score, 5.0))


def detect_hook(text: str) -> Dict[str, object]:
    lead = first_line(text)
    raw_score = _hook_score_text(lead)
    return {
        "hook_type": _classify_hook_type(lead),
        "hook_line": lead,
        "hook_score": round(max(1.0, min(raw_score, 10.0)), 2),
    }


def _ends_with_sentence(text: str) -> bool:
    clean = _normalize(text)
    if not clean:
        return False
    return bool(SENTENCE_END_RE.search(clean))


def _ends_with_incomplete_tail(text: str) -> bool:
    tokens = re.findall(r"[a-z0-9']+", _normalize(text).lower())
    if not tokens:
        return False
    return tokens[-1] in INCOMPLETE_TAIL_TERMS


def _target_duration_seconds(
    text: str,
    hook_score: float,
    window_segments: Sequence[dict],
    min_duration: float,
    max_duration: float,
) -> float:
    clean = _normalize(text)
    lowered = clean.lower()
    word_count = len(clean.split())
    segment_count = len(window_segments)
    story_terms = sum(1 for term in STORY_LONG_TERMS if term in lowered)
    punctuation_count = len(re.findall(r"[.!?]", clean))
    question_bonus = 1 if "?" in clean else 0

    long_form_signal = (
        (1.0 if hook_score >= 6.0 else 0.0)
        + min(2.0, story_terms * 0.7)
        + min(1.5, punctuation_count * 0.25)
        + min(1.5, max(0, segment_count - 2) * 0.22)
        + min(1.8, max(0, word_count - 40) / 40.0)
        + question_bonus * 0.4
    )
    short_form_signal = (
        (1.0 if word_count <= 34 else 0.0)
        + (0.8 if punctuation_count <= 1 else 0.0)
        + (0.9 if segment_count <= 2 else 0.0)
    )

    baseline = min_duration + 7.0
    if long_form_signal >= 3.2:
        baseline = min_duration + 28.0
    elif long_form_signal >= 2.2:
        baseline = min_duration + 18.0
    elif short_form_signal >= 2.0:
        baseline = min_duration + 5.0

    return max(min_duration, min(max_duration, baseline))


def strongest_hook_moment(segments: Sequence[dict], start: float, end: float) -> Optional[Dict[str, object]]:
    candidates = []
    for seg in segments:
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", seg_start + 0.5))
        if seg_end <= start or seg_start >= end:
            continue

        text = _normalize(seg.get("text", ""))
        if not text:
            continue

        hook = detect_hook(text)
        energy = segment_energy(text)
        score = float(hook["hook_score"]) * 0.72 + energy * 0.28
        candidates.append(
            {
                **hook,
                "time": seg_start,
                "score": round(score, 3),
                "segment_start": seg_start,
                "segment_end": seg_end,
            }
        )

    if not candidates:
        return None

    return sorted(candidates, key=lambda item: item["score"], reverse=True)[0]


def rank_by_hook(chunks: List[dict], segments: Optional[Sequence[dict]] = None) -> List[dict]:
    enriched = []
    for chunk in chunks:
        start = float(chunk.get("start", 0.0))
        end = float(chunk.get("end", start + 1.0))

        hook_info = strongest_hook_moment(segments or [], start, end) if segments else None
        if hook_info is None:
            base = detect_hook(chunk.get("text", ""))
            hook_info = {
                **base,
                "time": start,
                "score": float(base["hook_score"]),
            }

        merged = {
            **chunk,
            "hook_type": hook_info["hook_type"],
            "hook_line": hook_info["hook_line"],
            "hook_score": round(float(hook_info["score"]), 2),
            "hook_time": float(hook_info["time"]),
        }
        enriched.append(merged)

    return sorted(
        enriched,
        key=lambda item: (item.get("hook_score", 0), item.get("score", 0)),
        reverse=True,
    )


def pace_chunk(chunk: dict, segments: Sequence[dict], min_duration: float = 15.0, max_duration: float = 59.0) -> dict:
    orig_start = max(0.0, float(chunk.get("start", 0.0)))
    orig_end = max(orig_start + 1.0, float(chunk.get("end", orig_start + 1.0)))

    hook = strongest_hook_moment(segments, orig_start, orig_end)
    start = float(hook["time"]) if hook else orig_start
    start = max(orig_start, start)

    in_window = [
        seg
        for seg in segments
        if float(seg.get("end", 0.0)) >= start and float(seg.get("start", 0.0)) <= orig_end
    ]

    first_spoken = None
    for seg in in_window:
        words = _normalize(seg.get("text", "")).split()
        if len(words) >= 3:
            first_spoken = seg
            break
    if first_spoken is not None:
        start = max(start, float(first_spoken.get("start", start)))

    hard_end = min(orig_end, start + max_duration)
    target_duration = _target_duration_seconds(
        text=str(chunk.get("text", "")),
        hook_score=float(hook.get("score", 5.0)) if hook else 5.0,
        window_segments=in_window,
        min_duration=min_duration,
        max_duration=max_duration,
    )
    target_end = min(hard_end, start + target_duration)

    candidates = []
    for seg in in_window:
        seg_end = float(seg.get("end", 0.0))
        if seg_end <= start:
            continue
        if seg_end < (start + min_duration):
            continue
        if seg_end > hard_end:
            continue
        seg_text = _normalize(str(seg.get("text", "")))
        if not seg_text:
            continue
        candidates.append((seg_end, seg_text))

    sentence_ends = [
        seg_end
        for seg_end, seg_text in candidates
        if _ends_with_sentence(seg_text) and not _ends_with_incomplete_tail(seg_text)
    ]
    energetic_end = None
    for seg in reversed(in_window):
        seg_end = float(seg.get("end", hard_end))
        if seg_end <= (start + min_duration):
            continue
        if seg_end > hard_end:
            continue
        if segment_energy(seg.get("text", "")) >= 1.7:
            energetic_end = seg_end
            break

    end = hard_end
    if sentence_ends:
        end = min(sentence_ends, key=lambda value: abs(value - target_end))
    elif energetic_end is not None:
        end = min(hard_end, energetic_end)
    else:
        end = target_end

    if end - start < min_duration:
        end = min(hard_end, start + min_duration)

    if end < hard_end:
        for seg_end, seg_text in candidates:
            if seg_end <= end:
                continue
            if seg_end > min(hard_end, end + 6.0):
                continue
            if _ends_with_sentence(seg_text) and not _ends_with_incomplete_tail(seg_text):
                end = seg_end
                break

    if end <= start:
        end = start + 1.0

    paced = {
        **chunk,
        "start": round(start, 3),
        "end": round(end, 3),
    }

    if hook:
        paced.update(
            {
                "hook_type": hook.get("hook_type", "statement"),
                "hook_line": hook.get("hook_line", ""),
                "hook_score": round(float(hook.get("score", 0.0)), 2),
                "hook_time": round(float(hook.get("time", start)), 3),
            }
        )

    return paced

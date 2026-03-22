import re
from typing import Any, Dict, List, Sequence


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _safe_caption(text: str) -> str:
    words = _normalize(text).split()[:12]
    if len(words) < 3:
        words = ["WATCH", "THIS", "NOW"]
    return " ".join(words).upper()


def generate_variants(text: str, improved_hook: str = "", caption: str = "") -> List[Dict[str, Any]]:
    base_text = _normalize(text)
    hook = _normalize(improved_hook)
    base_caption = _safe_caption(caption or base_text)

    candidates: List[Dict[str, Any]] = [
        {
            "variant_id": "A",
            "variant_type": "original",
            "text": base_text,
            "hook": "",
            "caption": base_caption,
        }
    ]

    if hook and hook.lower() not in base_text.lower():
        candidates.append(
            {
                "variant_id": "B",
                "variant_type": "improved_hook",
                "text": f"{hook} {base_text}".strip(),
                "hook": hook,
                "caption": base_caption,
            }
        )

    curiosity_text = base_text
    if "?" not in curiosity_text:
        curiosity_text = f"WHY THIS MATTERS: {base_text}"
    candidates.append(
        {
            "variant_id": "C",
            "variant_type": "curiosity",
            "text": curiosity_text,
            "hook": hook or "",
            "caption": _safe_caption(curiosity_text),
        }
    )

    deduped: List[Dict[str, Any]] = []
    seen = set()
    for row in candidates:
        key = _normalize(str(row.get("text", ""))).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(row)
        if len(deduped) >= 3:
            break

    return deduped


def select_winner_variant(
    clip_id: str,
    variants: Sequence[Dict[str, Any]],
    feedback_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    perf_by_variant = {}
    for row in feedback_rows:
        if not isinstance(row, dict):
            continue
        row_id = str(row.get("clip_id", ""))
        if not row_id.startswith(f"{clip_id}_variant_"):
            continue
        metrics = row.get("metrics", {}) if isinstance(row.get("metrics"), dict) else {}
        completion = float(metrics.get("completion_rate", 0.0))
        watch_time = float(metrics.get("watch_time", 0.0))
        shares = float(metrics.get("shares", 0.0))
        likes = float(metrics.get("likes", 0.0))
        perf = completion * 0.4 + watch_time * 0.3 + shares * 0.2 + likes * 0.1
        perf_by_variant[row_id] = perf

    winner = None
    winner_perf = float("-inf")
    for variant in variants:
        var_id = str(variant.get("variant_id", "A"))
        key = f"{clip_id}_variant_{var_id}"
        perf = perf_by_variant.get(key, float("-inf"))
        if perf > winner_perf:
            winner = variant
            winner_perf = perf

    return dict(winner) if isinstance(winner, dict) else (dict(variants[0]) if variants else {})

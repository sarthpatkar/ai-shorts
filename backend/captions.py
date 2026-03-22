import os
import re
import shutil
import subprocess
import tempfile
from typing import Dict, Iterable, List, Optional, Tuple

FFMPEG_BIN = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"

EMOTION_WORDS = {
    "danger",
    "secret",
    "shocking",
    "insane",
    "mistake",
    "warning",
    "truth",
    "crazy",
    "urgent",
    "money",
    "million",
    "never",
    "always",
    "wrong",
}
MONEY_WORDS = {"money", "cash", "profit", "income", "dollar", "revenue", "sales"}
WARNING_WORDS = {"warning", "danger", "risk", "avoid", "mistake", "alert"}
SHOCK_WORDS = {"shocking", "insane", "crazy", "unbelievable", "wild"}

ASS_HIGHLIGHT = "&H0000FFFF&"
ASS_PRIMARY = "&H00FFFFFF&"


def _clean_text(text: str) -> str:
    cleaned = (text or "").replace("\n", " ").replace("\r", " ").replace("\t", " ")
    cleaned = cleaned.replace('"', "").replace("'", "")
    cleaned = re.sub(r"[^0-9A-Za-z\s.,!?%:$&@()\-+/]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _limit_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text

    sentences = re.findall(r"[^.!?]+[.!?]?", text)
    kept: List[str] = []
    for sentence in sentences:
        sentence_words = sentence.strip().split()
        if not sentence_words:
            continue
        if len(kept) + len(sentence_words) > max_words:
            break
        kept.extend(sentence_words)

    if kept:
        return " ".join(kept)

    trimmed = words[:max_words]
    if trimmed and trimmed[-1][-1] not in ".!?":
        trimmed[-1] = f"{trimmed[-1]}..."
    return " ".join(trimmed)


def _detect_tone(text: str) -> str:
    lowered = (text or "").lower()
    if any(w in lowered for w in WARNING_WORDS):
        return "warning"
    if any(w in lowered for w in MONEY_WORDS):
        return "money"
    if any(w in lowered for w in SHOCK_WORDS):
        return "shock"
    return "neutral"


def _inject_emoji(text: str) -> str:
    tone = _detect_tone(text)
    emoji = {
        "warning": "⚠️",
        "money": "💰",
        "shock": "😱",
    }.get(tone)
    if not emoji:
        return text
    return f"{text} {emoji}"


def _is_highlight_word(word: str) -> bool:
    token = re.sub(r"[^A-Za-z0-9]", "", word or "")
    if not token:
        return False
    if token.isupper() and len(token) > 1:
        return True
    if any(ch.isdigit() for ch in token):
        return True
    if token.lower() in EMOTION_WORDS:
        return True
    return False


def _escape_ass_text(text: str) -> str:
    safe = (text or "").replace("\n", " ").replace("\r", " ").replace("\t", " ")
    safe = re.sub(r"\s+", " ", safe).strip()
    safe = safe.replace("{", "(").replace("}", ")")
    safe = safe.replace("\\", "\\\\")
    return safe


def _ass_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    cs = int(round(seconds * 100))
    sec = (cs // 100) % 60
    mins = (cs // 6000) % 60
    hrs = cs // 360000
    cents = cs % 100
    return f"{hrs}:{mins:02d}:{sec:02d}.{cents:02d}"


def _hook_prefix(hook_type: Optional[str]) -> str:
    mapping = {
        "question": "NOBODY TELLS YOU THIS",
        "shock": "THIS CHANGES EVERYTHING",
        "curiosity": "YOURE DOING THIS WRONG",
    }
    return mapping.get((hook_type or "").lower(), "NOBODY TELLS YOU THIS")


def generate_caption(
    text: str,
    hook_type: Optional[str] = None,
    max_words: int = 12,
    inject_emoji: bool = True,
) -> str:
    cleaned = _clean_text(text)
    trimmed = _limit_words(cleaned, max_words=max_words)
    prefix = _hook_prefix(hook_type)
    caption = f"{prefix}: {trimmed}" if trimmed else prefix
    caption = caption.upper()
    return _inject_emoji(caption) if inject_emoji else caption


def _format_caption_words(words: List[str]) -> str:
    rendered = []
    for raw in words:
        token = _escape_ass_text(raw).upper()
        if not token:
            continue
        if _is_highlight_word(token):
            rendered.append(f"{{\\c{ASS_HIGHLIGHT}\\b1}}{token}{{\\r}}")
        else:
            rendered.append(token)
    return " ".join(rendered)


def _build_word_events(
    words: List[Dict[str, object]],
    clip_start: float,
    clip_end: float,
    window_size: int = 3,
) -> List[Dict[str, object]]:
    if not words:
        return []

    usable = []
    for w in words:
        ws = float(w.get("start", 0.0))
        we = float(w.get("end", ws + 0.2))
        if we <= clip_start or ws >= clip_end:
            continue
        token = _clean_text(str(w.get("word", "")).strip())
        if not token:
            continue
        usable.append({"start": ws, "end": we, "word": token})

    events = []
    idx = 0
    while idx < len(usable):
        group = usable[idx : idx + window_size]
        if not group:
            break

        rel_start = max(0.0, float(group[0]["start"]) - clip_start)
        rel_end = min(clip_end - clip_start, float(group[-1]["end"]) - clip_start)
        rel_end = max(rel_start + 0.30, rel_end)

        group_text = _format_caption_words([str(x["word"]) for x in group])
        if group_text:
            events.append({"start": rel_start, "end": rel_end, "text": group_text})

        idx += window_size

    return events


def _ass_header(play_res_x: int = 1080, play_res_y: int = 1920) -> str:
    return "\n".join(
        [
            "[Script Info]",
            "ScriptType: v4.00+",
            "WrapStyle: 2",
            "ScaledBorderAndShadow: yes",
            f"PlayResX: {play_res_x}",
            f"PlayResY: {play_res_y}",
            "",
            "[V4+ Styles]",
            "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding",
            "Style: Default,Arial,74,&H00FFFFFF,&H000000FF,&H00000000,&H66000000,-1,0,0,0,100,100,0,0,3,3,0,2,60,60,120,1",
            "",
            "[Events]",
            "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
        ]
    )


def write_ass_subtitle_file(
    events: Iterable[Dict[str, object]],
    output_path: str,
    play_res_x: int = 1080,
    play_res_y: int = 1920,
) -> str:
    lines = [_ass_header(play_res_x=play_res_x, play_res_y=play_res_y)]

    for event in events:
        start = _ass_time(float(event.get("start", 0.0)))
        end = _ass_time(float(event.get("end", 0.0)))
        if float(event.get("end", 0.0)) <= float(event.get("start", 0.0)):
            continue
        text = str(event.get("text", "")).strip()
        if not text:
            continue
        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return output_path


def create_ass_for_clip(
    chunk_text: str,
    hook_type: Optional[str],
    words: List[Dict[str, object]],
    clip_start: float,
    clip_end: float,
    tmp_dir: Optional[str] = None,
    caption_override: Optional[str] = None,
    inject_emoji: bool = False,
) -> Tuple[str, str]:
    clip_duration = max(0.5, clip_end - clip_start)
    if caption_override:
        cleaned_override = _limit_words(_clean_text(caption_override), max_words=12).upper()
        hook_caption = _inject_emoji(cleaned_override) if inject_emoji else cleaned_override
    else:
        hook_caption = generate_caption(chunk_text, hook_type=hook_type, max_words=12, inject_emoji=inject_emoji)

    hook_end = min(1.6, clip_duration * 0.35)
    events: List[Dict[str, object]] = [
        {
            "start": 0.0,
            "end": max(0.9, hook_end),
            "text": _format_caption_words(hook_caption.split()),
        }
    ]

    events.extend(_build_word_events(words, clip_start=clip_start, clip_end=clip_end, window_size=3))

    ass_file = tempfile.NamedTemporaryFile(suffix=".ass", delete=False, dir=tmp_dir)
    ass_file.close()
    write_ass_subtitle_file(events, ass_file.name)
    return ass_file.name, hook_caption


def create_static_ass_file(
    text: str,
    duration: float,
    tmp_dir: Optional[str] = None,
) -> str:
    safe_text = _format_caption_words(_inject_emoji(_clean_text(text)).split())
    events = [{"start": 0.0, "end": max(0.6, duration), "text": safe_text}]

    ass_file = tempfile.NamedTemporaryFile(suffix=".ass", delete=False, dir=tmp_dir)
    ass_file.close()
    write_ass_subtitle_file(events, ass_file.name)
    return ass_file.name


def _escape_filter_path(path: str) -> str:
    escaped = path.replace("\\", "\\\\")
    escaped = escaped.replace(":", "\\:")
    escaped = escaped.replace("'", "\\'")
    escaped = escaped.replace(",", "\\,")
    return escaped


def add_caption(
    input_video: str,
    text: str,
    output_video: str,
    caption_segments: Optional[List[Dict[str, object]]] = None,
    caption_duration: float = 3.5,
    preset: str = "veryfast",
):
    ass_path = None
    try:
        if caption_segments:
            events = []
            for seg in caption_segments:
                seg_text = _format_caption_words(_clean_text(str(seg.get("text", ""))).split())
                seg_start = max(0.0, float(seg.get("start", 0.0)))
                seg_end = max(seg_start + 0.3, float(seg.get("end", seg_start + caption_duration)))
                if seg_text:
                    events.append({"start": seg_start, "end": seg_end, "text": seg_text})

            temp = tempfile.NamedTemporaryFile(suffix=".ass", delete=False)
            temp.close()
            ass_path = write_ass_subtitle_file(events, temp.name)
        else:
            ass_path = create_static_ass_file(text=text, duration=caption_duration)

        escaped = _escape_filter_path(ass_path)
        cmd = [
            FFMPEG_BIN,
            "-y",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(input_video),
            "-vf",
            f"subtitles='{escaped}'",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            "21",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(output_video),
        ]
        subprocess.run(cmd, check=True)
        return output_video
    finally:
        if ass_path and os.path.exists(ass_path):
            os.remove(ass_path)

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

FFMPEG_BIN = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"


def extract_audio(video_path: str, output_path: Optional[str] = None) -> str:
    if output_path is None:
        output_path = str(Path(video_path).with_name("audio.wav"))

    subprocess.run(
        [
            FFMPEG_BIN,
            "-y",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            video_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            output_path,
        ],
        check=True,
    )
    return output_path


def _approximate_words(segment: Dict[str, object]) -> List[Dict[str, object]]:
    text = str(segment.get("text", "")).strip()
    start = float(segment.get("start", 0.0))
    end = float(segment.get("end", start + 0.5))

    tokens = re.findall(r"\S+", text)
    if not tokens:
        return []

    duration = max(0.1, end - start)
    step = duration / max(1, len(tokens))

    words = []
    for i, token in enumerate(tokens):
        ws = start + i * step
        we = min(end, ws + step * 0.95)
        words.append({"start": round(ws, 3), "end": round(we, 3), "word": token})
    return words


def _as_bool(value: Optional[str], default: bool = True) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _transcribe_with_faster_whisper(
    audio_path: str,
    model_size: str = "tiny",
    fast_mode: bool = True,
) -> List[Dict[str, object]]:
    from faster_whisper import WhisperModel

    compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
    model = WhisperModel(model_size, device="cpu", compute_type=compute_type)
    beam_size = 1 if fast_mode else 3
    segments_iter, _ = model.transcribe(
        audio_path,
        beam_size=beam_size,
        word_timestamps=True,
        vad_filter=True,
        condition_on_previous_text=False,
        temperature=0,
    )

    segments: List[Dict[str, object]] = []
    for seg in segments_iter:
        words = []
        for word in seg.words or []:
            if word.start is None or word.end is None:
                continue
            token = str(word.word or "").strip()
            if not token:
                continue
            words.append(
                {
                    "start": round(float(word.start), 3),
                    "end": round(float(word.end), 3),
                    "word": token,
                }
            )

        segment = {
            "start": round(float(seg.start), 3),
            "end": round(float(seg.end), 3),
            "text": str(seg.text or "").strip(),
            "words": words,
        }
        if not segment["words"]:
            segment["words"] = _approximate_words(segment)

        segments.append(segment)

    return segments


def _transcribe_with_openai_whisper(
    audio_path: str,
    model_size: str = "tiny",
    fast_mode: bool = True,
) -> List[Dict[str, object]]:
    import whisper

    model = whisper.load_model(model_size)

    try:
        kwargs = {"word_timestamps": True, "fp16": False, "temperature": 0}
        if fast_mode:
            kwargs.update({"best_of": 1, "beam_size": 1})
        result = model.transcribe(audio_path, **kwargs)
    except TypeError:
        result = model.transcribe(audio_path, fp16=False)

    segments: List[Dict[str, object]] = []
    for seg in result.get("segments", []):
        words = []
        for word in seg.get("words", []) or []:
            ws = word.get("start")
            we = word.get("end")
            token = str(word.get("word", "")).strip()
            if ws is None or we is None or not token:
                continue
            words.append(
                {
                    "start": round(float(ws), 3),
                    "end": round(float(we), 3),
                    "word": token,
                }
            )

        segment = {
            "start": round(float(seg.get("start", 0.0)), 3),
            "end": round(float(seg.get("end", 0.0)), 3),
            "text": str(seg.get("text", "")).strip(),
            "words": words,
        }
        if not segment["words"]:
            segment["words"] = _approximate_words(segment)

        segments.append(segment)

    return segments


def transcribe_audio(
    audio_path: str,
    model_size: Optional[str] = None,
    fast_mode: Optional[bool] = None,
) -> List[Dict[str, object]]:
    selected_model = model_size or os.environ.get("WHISPER_MODEL", "tiny")
    selected_fast_mode = fast_mode if fast_mode is not None else _as_bool(os.environ.get("WHISPER_FAST", "1"), True)

    try:
        return _transcribe_with_faster_whisper(audio_path, model_size=selected_model, fast_mode=selected_fast_mode)
    except Exception:
        return _transcribe_with_openai_whisper(audio_path, model_size=selected_model, fast_mode=selected_fast_mode)


def flatten_words(segments: List[Dict[str, object]]) -> List[Dict[str, object]]:
    words: List[Dict[str, object]] = []
    for seg in segments:
        for word in seg.get("words", []) or []:
            token = str(word.get("word", "")).strip()
            if not token:
                continue
            words.append(
                {
                    "start": float(word.get("start", seg.get("start", 0.0))),
                    "end": float(word.get("end", seg.get("end", seg.get("start", 0.0)))),
                    "word": token,
                }
            )
    return words

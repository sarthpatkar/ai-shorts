import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

_LOCK = threading.Lock()
_CACHE_DATA: Optional[Dict[str, Dict[str, Any]]] = None


def _cache_path() -> Path:
    configured = os.environ.get("AI_CACHE_PATH", "")
    if configured.strip():
        return Path(configured).expanduser()
    return Path("cache") / "ai_clip_cache.json"


def _ensure_loaded() -> Dict[str, Dict[str, Any]]:
    global _CACHE_DATA

    with _LOCK:
        if _CACHE_DATA is not None:
            return _CACHE_DATA

        path = _cache_path()
        if not path.exists():
            _CACHE_DATA = {}
            return _CACHE_DATA

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                _CACHE_DATA = {str(k): v for k, v in raw.items() if isinstance(v, dict)}
            else:
                _CACHE_DATA = {}
        except Exception:
            _CACHE_DATA = {}

        return _CACHE_DATA


def _persist_cache(data: Dict[str, Dict[str, Any]]) -> None:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, sort_keys=True)
    os.replace(temp_path, path)


def text_hash(text: str) -> str:
    payload = (text or "").strip().encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def get_cached_ai_response(text: str) -> Optional[Dict[str, Any]]:
    key = text_hash(text)
    data = _ensure_loaded()
    with _LOCK:
        item = data.get(key)
        return dict(item) if isinstance(item, dict) else None


def set_cached_ai_response(text: str, response: Dict[str, Any]) -> None:
    key = text_hash(text)
    data = _ensure_loaded()
    with _LOCK:
        data[key] = dict(response)
        _persist_cache(data)

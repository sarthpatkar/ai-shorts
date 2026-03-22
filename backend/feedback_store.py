import json
import os
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List

_LOCK = Lock()


def _store_path() -> Path:
    configured = os.environ.get("FEEDBACK_STORE_PATH", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path("data") / "feedback_store.json"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_all_feedback() -> List[Dict[str, Any]]:
    with _LOCK:
        return _read_feedback_unlocked()


def _read_feedback_unlocked() -> List[Dict[str, Any]]:
    path = _store_path()
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        return []
    except Exception:
        return []


def _write_feedback(records: List[Dict[str, Any]]) -> None:
    path = _store_path()
    _ensure_parent(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=True, indent=2)
    os.replace(tmp, path)


def save_clip_feedback(data: Dict[str, Any]) -> Dict[str, Any]:
    record = dict(data or {})
    record.setdefault("clip_id", f"clip_{uuid.uuid4().hex[:12]}")
    record.setdefault("timestamp", int(time.time()))
    record.setdefault("metrics", {"views": 0, "watch_time": 0.0, "completion_rate": 0.0, "likes": 0, "shares": 0})

    with _LOCK:
        existing = _read_feedback_unlocked()
        existing.append(record)
        _write_feedback(existing)
    return record

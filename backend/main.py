from __future__ import annotations

from typing import Any, Dict, List

from pipeline import run_pipeline_with_metadata


def run_pipeline(url: str) -> Dict[str, Any]:
    result = run_pipeline_with_metadata(url=url, top_k=3)
    clips = result.get("clips", [])
    run_dir = result.get("run_dir", "")
    return {
        "clips": list(clips) if isinstance(clips, list) else [],
        "run_dir": str(run_dir),
    }

import asyncio
from typing import Any, Dict, List, Sequence

from ai import batch_ai_request, fallback_enrichment


async def analyze_chunks_with_ai_async(chunks: Sequence[Dict[str, Any]], text_key: str = "text") -> List[Dict[str, Any]]:
    texts = [str(chunk.get(text_key, "")) for chunk in chunks]
    if not texts:
        return []
    try:
        return await asyncio.to_thread(batch_ai_request, texts)
    except Exception as exc:
        return [fallback_enrichment(text, error=str(exc)) for text in texts]

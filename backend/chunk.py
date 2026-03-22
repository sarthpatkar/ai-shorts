def create_chunks(segments, chunk_seconds=30):
    if not segments:
        return []

    chunks = []
    current_text = []
    start = float(segments[0]["start"])
    last_end = start

    for seg in segments:
        text = (seg.get("text") or "").strip()
        if text:
            current_text.append(text)

        last_end = float(seg.get("end", last_end))
        if last_end - start >= chunk_seconds:
            chunks.append(
                {
                    "start": start,
                    "end": last_end,
                    "text": " ".join(current_text).strip(),
                }
            )
            current_text = []
            start = last_end

    if current_text:
        chunks.append(
            {
                "start": start,
                "end": last_end,
                "text": " ".join(current_text).strip(),
            }
        )

    return [chunk for chunk in chunks if chunk["end"] > chunk["start"] and chunk["text"]]

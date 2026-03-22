import os

from captions import create_static_ass_file
from cutter import render_vertical_clip


def cut_clip(video, start, end, output, caption):
    duration = max(0.8, min(4.0, float(end) - float(start)))
    subtitle_path = create_static_ass_file(caption, duration=duration)

    try:
        render_vertical_clip(
            input_video=video,
            start=float(start),
            end=float(end),
            output_video=output,
            subtitle_file=subtitle_path,
            speech_segments=None,
            preset="medium",
            crf=18,
        )
    finally:
        if os.path.exists(subtitle_path):
            os.remove(subtitle_path)

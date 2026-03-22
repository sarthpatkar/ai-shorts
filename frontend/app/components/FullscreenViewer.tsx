"use client";

import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import type { Clip } from "@/lib/types";

export type TrimWindow = {
  start: number;
  end: number;
};

type FullscreenViewerProps = {
  open: boolean;
  clips: Clip[];
  activeIndex: number;
  favoriteMap: Record<string, boolean>;
  captionEnabledMap: Record<string, boolean>;
  trimWindows: Record<string, TrimWindow>;
  getTags: (clipId: string) => string[];
  onClose: () => void;
  onNavigate: (nextIndex: number) => void;
  onToggleFavorite: (clipId: string) => void;
  onToggleCaption: (clipId: string) => void;
  onTrimChange: (clipId: string, trim: TrimWindow) => void;
  onDownload: (clipId: string) => void;
  onRegenerate: (clipId: string) => void;
  onPlaybackError: (clipId: string) => Promise<string | null>;
};

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function formatTimestamp(value: number): string {
  if (!Number.isFinite(value) || value <= 0) {
    return "00:00";
  }
  const whole = Math.floor(value);
  const minutes = Math.floor(whole / 60);
  const seconds = whole % 60;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

export default function FullscreenViewer({
  open,
  clips,
  activeIndex,
  favoriteMap,
  captionEnabledMap,
  trimWindows,
  getTags,
  onClose,
  onNavigate,
  onToggleFavorite,
  onToggleCaption,
  onTrimChange,
  onDownload,
  onRegenerate,
  onPlaybackError,
}: FullscreenViewerProps) {
  const clip = clips[activeIndex] ?? null;
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const touchStartY = useRef<number | null>(null);
  const [durationSeconds, setDurationSeconds] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [videoSrc, setVideoSrc] = useState(clip?.videoUrl ?? "");
  const [playbackHint, setPlaybackHint] = useState<string | null>(null);
  const [recovering, setRecovering] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    setVideoSrc(clip?.videoUrl ?? "");
    setPlaybackHint(null);
    setRecovering(false);
    setCurrentTime(0);
    setDurationSeconds(0);
    setPlaybackRate(1);
    setIsPlaying(false);
  }, [clip?.id, clip?.videoUrl]);

  useEffect(() => {
    if (!videoRef.current) {
      return;
    }
    videoRef.current.playbackRate = playbackRate;
  }, [playbackRate, clip?.id]);

  useEffect(() => {
    if (!open) {
      return;
    }
    const handler = (event: KeyboardEvent) => {
      if (!clip) {
        return;
      }
      if (event.key === "Escape") {
        onClose();
      } else if (event.key === "ArrowRight" || event.key === "ArrowDown") {
        onNavigate(activeIndex + 1);
      } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
        onNavigate(activeIndex - 1);
      }
    };

    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [activeIndex, clip, onClose, onNavigate, open]);

  const trim = clip
    ? trimWindows[clip.id] ?? {
        start: 0,
        end: 1,
      }
    : { start: 0, end: 1 };
  const startSeconds = durationSeconds * trim.start;
  const endSeconds = durationSeconds * trim.end;
  const tags = useMemo(() => (clip ? getTags(clip.id) : []), [clip, getTags]);

  const handleTrimStartChange = (nextValue: number) => {
    if (!clip) {
      return;
    }
    const nextStart = clamp(nextValue, 0, trim.end - 0.05);
    onTrimChange(clip.id, { start: nextStart, end: trim.end });
  };

  const handleTrimEndChange = (nextValue: number) => {
    if (!clip) {
      return;
    }
    const nextEnd = clamp(nextValue, trim.start + 0.05, 1);
    onTrimChange(clip.id, { start: trim.start, end: nextEnd });
  };

  const recoverPlayback = async () => {
    if (!clip || recovering) {
      return;
    }
    setRecovering(true);
    setPlaybackHint("Refreshing secure playback URL...");
    try {
      const refreshed = await onPlaybackError(clip.id);
      if (refreshed) {
        setVideoSrc(refreshed);
        setPlaybackHint(null);
      } else {
        setPlaybackHint("Playback unavailable. Try regenerate.");
      }
    } finally {
      setRecovering(false);
    }
  };

  const seekBySeconds = (deltaSeconds: number) => {
    const video = videoRef.current;
    if (!video) {
      return;
    }
    const minBound = startSeconds;
    const maxBound = endSeconds > minBound ? endSeconds : durationSeconds;
    const next = clamp(video.currentTime + deltaSeconds, minBound, Math.max(minBound, maxBound));
    video.currentTime = next;
    setCurrentTime(next);
  };

  if (!open || !clip) {
    return null;
  }

  return (
    <div className="viewer-backdrop" role="dialog" aria-modal="true">
      <div
        className="viewer-shell"
        onTouchStart={(event) => {
          touchStartY.current = event.changedTouches[0]?.clientY ?? null;
        }}
        onTouchEnd={(event) => {
          const start = touchStartY.current;
          const end = event.changedTouches[0]?.clientY ?? null;
          if (start === null || end === null) {
            return;
          }
          const delta = start - end;
          if (delta > 40) {
            onNavigate(activeIndex + 1);
          } else if (delta < -40) {
            onNavigate(activeIndex - 1);
          }
        }}
      >
        <div className="viewer-topbar">
          <button type="button" className="btn btn-secondary" onClick={onClose}>
            Close
          </button>
          <span className="mono">
            Clip {activeIndex + 1} / {clips.length}
          </span>
        </div>

        <div className="viewer-main">
          <button
            type="button"
            className="viewer-nav"
            onClick={() => onNavigate(activeIndex - 1)}
            aria-label="Previous clip"
          >
            ↑
          </button>

          <div className="viewer-video-col">
            <video
              key={clip.id}
              ref={videoRef}
              src={videoSrc}
              className="viewer-video"
              playsInline
              autoPlay
              controls={false}
              onLoadedMetadata={(event) => {
                const duration = event.currentTarget.duration;
                if (Number.isFinite(duration) && duration > 0) {
                  setDurationSeconds(duration);
                  const boundedStart = duration * trim.start;
                  event.currentTarget.currentTime = boundedStart;
                  event.currentTarget.playbackRate = playbackRate;
                }
              }}
              onTimeUpdate={(event) => {
                const next = event.currentTarget.currentTime;
                setCurrentTime(next);
                if (durationSeconds > 0 && next >= endSeconds) {
                  event.currentTarget.currentTime = startSeconds;
                }
              }}
              onError={() => {
                void recoverPlayback();
              }}
              onPlay={() => setIsPlaying(true)}
              onPause={() => setIsPlaying(false)}
            />
            {captionEnabledMap[clip.id] && (
              <div className="viewer-caption">{clip.caption || "AI caption enabled"}</div>
            )}
            <div className="viewer-controls">
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => seekBySeconds(-10)}
              >
                -10s
              </button>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => {
                  if (!videoRef.current) {
                    return;
                  }
                  if (videoRef.current.paused) {
                    void videoRef.current.play();
                  } else {
                    videoRef.current.pause();
                  }
                }}
              >
                {isPlaying ? "Pause" : "Play"}
              </button>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => seekBySeconds(10)}
              >
                +10s
              </button>
              <label className="viewer-speed">
                <span className="mono">Speed</span>
                <select
                  value={String(playbackRate)}
                  onChange={(event) => setPlaybackRate(Number(event.target.value))}
                >
                  {[0.5, 0.75, 1, 1.25, 1.5, 1.75, 2].map((rate) => (
                    <option key={rate} value={rate}>
                      {rate}x
                    </option>
                  ))}
                </select>
              </label>
            </div>
          </div>

          <button
            type="button"
            className="viewer-nav"
            onClick={() => onNavigate(activeIndex + 1)}
            aria-label="Next clip"
          >
            ↓
          </button>
        </div>

        {playbackHint && <p className="status-note">{playbackHint}</p>}

        <div className="viewer-meta">
          <div className="chips-row">
            {tags.map((tag) => (
              <span key={tag} className="chip">
                {tag}
              </span>
            ))}
          </div>
          <p className="viewer-caption-preview">{clip.caption}</p>
        </div>

        <section className="editor-panel">
          <div className="editor-head">
            <p className="mono">Creator Controls</p>
            <span className="chip">CapCut-style trim</span>
          </div>

          <div className="timeline-wrap">
            <div
              className="timeline-bar"
              style={
                {
                  "--trim-start": `${trim.start * 100}%`,
                  "--trim-end": `${trim.end * 100}%`,
                } as CSSProperties
              }
            >
              <div className="timeline-wave" />
            </div>
            <div className="trim-inputs">
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={trim.start}
                onChange={(event) => handleTrimStartChange(Number(event.target.value))}
                aria-label="Trim start"
              />
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={trim.end}
                onChange={(event) => handleTrimEndChange(Number(event.target.value))}
                aria-label="Trim end"
              />
            </div>
            <p className="editor-time">
              {formatTimestamp(startSeconds)} - {formatTimestamp(endSeconds)} /{" "}
              {formatTimestamp(durationSeconds)} · Live {formatTimestamp(currentTime)}
            </p>
          </div>

          <div className="editor-actions">
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => onToggleCaption(clip.id)}
            >
              {captionEnabledMap[clip.id] ? "Hide Captions" : "Show Captions"}
            </button>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => onToggleFavorite(clip.id)}
            >
              {favoriteMap[clip.id] ? "★ Favorited" : "☆ Favorite"}
            </button>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => onDownload(clip.id)}
            >
              Download
            </button>
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => onRegenerate(clip.id)}
            >
              Regenerate
            </button>
          </div>
        </section>
      </div>
    </div>
  );
}

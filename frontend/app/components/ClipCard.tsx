"use client";

import { memo, useEffect, useMemo, useRef, useState } from "react";
import type { Clip, ClipInteractionState } from "@/lib/types";

type FeedbackAction = "good" | "improve" | "regenerate";

type ClipCardProps = {
  clip: Clip;
  order: number;
  tags: string[];
  interaction: ClipInteractionState;
  feedbackState: "idle" | "sending" | "sent" | "error";
  onPlay: (clipId: string) => void;
  onProgress: (clipId: string, currentTime: number) => void;
  onPause: (clipId: string, currentTime: number) => void;
  onDownload: (clipId: string) => void;
  onOpenStudio: (clipId: string) => void;
  onFeedback: (clipId: string, action: FeedbackAction) => void;
  onPlaybackError: (clipId: string) => Promise<string | null>;
};

const domains = [
  { key: "h", label: "Prime Pick" },
  { key: "e", label: "Momentum" },
  { key: "a", label: "Story Arc" },
  { key: "f", label: "Hook Focus" },
  { key: "g", label: "Audience Fit" },
] as const;

function pickDomain(clipId: string) {
  let hash = 0;
  for (let i = 0; i < clipId.length; i += 1) {
    hash = (hash + clipId.charCodeAt(i) * (i + 1)) % domains.length;
  }
  return domains[Math.abs(hash) % domains.length];
}

function formatDuration(value: number | null): string {
  if (!value || Number.isNaN(value) || value <= 0) {
    return "00:00";
  }
  const total = Math.floor(value);
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function ClipCard({
  clip,
  order,
  tags,
  interaction,
  feedbackState,
  onPlay,
  onProgress,
  onPause,
  onDownload,
  onOpenStudio,
  onFeedback,
  onPlaybackError,
}: ClipCardProps) {
  const domain = useMemo(() => pickDomain(clip.id), [clip.id]);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [durationSeconds, setDurationSeconds] = useState<number | null>(null);
  const [videoSrc, setVideoSrc] = useState(clip.videoUrl);
  const [isRecovering, setIsRecovering] = useState(false);
  const [playbackError, setPlaybackError] = useState<string | null>(null);

  useEffect(() => {
    setVideoSrc(clip.videoUrl);
    setPlaybackError(null);
    setIsRecovering(false);
  }, [clip.videoUrl]);

  const handlePlaybackError = async () => {
    if (isRecovering) {
      return;
    }
    setIsRecovering(true);
    setPlaybackError("Refreshing secure playback URL...");
    try {
      const refreshed = await onPlaybackError(clip.id);
      if (refreshed) {
        setVideoSrc(refreshed);
        setPlaybackError(null);
      } else {
        setPlaybackError("Playback unavailable right now.");
      }
    } finally {
      setIsRecovering(false);
    }
  };

  const feedbackLabel =
    feedbackState === "sending"
      ? "Saving feedback..."
      : feedbackState === "sent"
      ? "Feedback captured"
      : feedbackState === "error"
      ? "Feedback failed"
      : "Ready for feedback";

  return (
    <article className="clip-card" data-domain={domain.key}>
      <div className="card-top">
        <span className={`pill pill-${domain.key}`}>{domain.label}</span>
        <span className="card-num">#{String(order).padStart(2, "0")}</span>
      </div>

      <h3 className="clip-title">Clip {order}</h3>

      <button
        type="button"
        className="clip-video-wrap"
        onClick={() => onOpenStudio(clip.id)}
        aria-label={`Open clip ${order} in fullscreen studio`}
      >
        <video
          ref={videoRef}
          src={videoSrc}
          preload="metadata"
          playsInline
          muted
          className="clip-video"
          onPlay={() => onPlay(clip.id)}
          onTimeUpdate={(event) => {
            onProgress(clip.id, event.currentTarget.currentTime);
          }}
          onPause={(event) => {
            onPause(clip.id, event.currentTarget.currentTime);
          }}
          onLoadedMetadata={(event) => {
            const loaded = event.currentTarget.duration;
            setDurationSeconds(Number.isFinite(loaded) ? loaded : null);
          }}
          onError={() => {
            void handlePlaybackError();
          }}
        />
        <div className="clip-overlay">
          <span className="chip">Open Studio</span>
          <span className="chip">{formatDuration(durationSeconds)}</span>
        </div>
      </button>

      {playbackError && <p className="status-note">{playbackError}</p>}

      <div className="chips-row ai-chip-row">
        {tags.map((tag) => (
          <span key={tag} className="chip">
            {tag}
          </span>
        ))}
      </div>

      <div className="detail-row">
        <div className="detail-lbl">Caption</div>
        <p className="detail-val">{clip.caption || "No caption available."}</p>
      </div>

      <div className="detail-row">
        <div className="detail-lbl">Engagement</div>
        <p className="detail-val">
          {interaction.playCount} plays · {Math.floor(interaction.watchTimeSeconds)}s
          watched
        </p>
      </div>

      <div className="chips-row">
        <span className="chip">{feedbackLabel}</span>
      </div>

      <div className="clip-actions">
        <button
          onClick={() => onFeedback(clip.id, "good")}
          className="btn btn-secondary"
          type="button"
          disabled={feedbackState === "sending"}
        >
          👍 Good
        </button>
        <button
          onClick={() => onFeedback(clip.id, "improve")}
          className="btn btn-secondary"
          type="button"
          disabled={feedbackState === "sending"}
        >
          👎 Improve
        </button>
        <button
          onClick={() => onFeedback(clip.id, "regenerate")}
          className="btn btn-secondary"
          type="button"
          disabled={feedbackState === "sending"}
        >
          🔁 Regenerate
        </button>
      </div>

      <div className="clip-actions">
        <button
          type="button"
          className="btn btn-primary"
          onClick={() => onOpenStudio(clip.id)}
        >
          Open Fullscreen
        </button>
        <button
          type="button"
          onClick={() => onDownload(clip.id)}
          className="btn btn-secondary"
        >
          Download
        </button>
      </div>
    </article>
  );
}

export default memo(ClipCard);

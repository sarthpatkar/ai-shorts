"use client";

import { useEffect, useState } from "react";

const THINKING_MESSAGES = [
  "Ingesting source video...",
  "Analyzing speech and pacing...",
  "Ranking high-retention moments...",
  "Running clip generation engine...",
];

const PIPELINE_STAGES = [
  "Source Intake",
  "Audio + Transcript Analysis",
  "Moment Scoring",
  "Clip Generation Engine",
  "Render + Delivery",
];

type LoadingStateProps = {
  revealedCount?: number;
};

export default function LoadingState({ revealedCount = 0 }: LoadingStateProps) {
  const [messageIndex, setMessageIndex] = useState(0);
  const [stageIndex, setStageIndex] = useState(0);

  useEffect(() => {
    const messageInterval = setInterval(() => {
      setMessageIndex((prev) => (prev + 1) % THINKING_MESSAGES.length);
    }, 2800);

    const stageInterval = setInterval(() => {
      setStageIndex((prev) => Math.min(PIPELINE_STAGES.length - 1, prev + 1));
    }, 3600);

    return () => {
      clearInterval(messageInterval);
      clearInterval(stageInterval);
    };
  }, []);

  return (
    <section className="thinking-panel" role="status" aria-live="polite">
      <div className="thinking-head">
        <p className="mono">Processing Pipeline</p>
        <span className="badge-dark">Live</span>
      </div>

      <p className="thinking-line">{THINKING_MESSAGES[messageIndex]}</p>

      <div className="pipeline-track" aria-label="Pipeline progress">
        {PIPELINE_STAGES.map((stage, index) => {
          const state =
            index < stageIndex ? "done" : index === stageIndex ? "active" : "idle";
          return (
            <div key={stage} className={`pipeline-item is-${state}`}>
              <span className="pipeline-dot" aria-hidden />
              <span>{stage}</span>
            </div>
          );
        })}
      </div>

      <div className="skeleton-grid">
        {Array.from({ length: 3 }).map((_, index) => {
          const alreadyRevealed = index < revealedCount;
          return (
            <article
              key={index}
              className={`skeleton-card ${alreadyRevealed ? "is-revealed" : ""}`}
              aria-hidden
            >
              <div className="skeleton-frame shimmer" />
              <div className="skeleton-bar shimmer" />
              <div className="skeleton-bar shimmer short" />
            </article>
          );
        })}
      </div>
    </section>
  );
}

"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import InputSection from "./components/InputSection";
import LoadingState from "./components/LoadingState";
import ClipCard from "./components/ClipCard";
import FullscreenViewer, { type TrimWindow } from "./components/FullscreenViewer";
import FeedbackModal from "./components/FeedbackModal";
import AuthModal from "./components/AuthModal";
import DownloadOptionsModal from "./components/DownloadOptionsModal";
import {
  ApiError,
  DEMO_VIDEO_META,
  createCheckout,
  fetchDownloadOptions,
  generateClips,
  generateFromDemo,
  refreshClipUrls,
  requestDownloadUrl,
  submitClipFeedback,
} from "@/lib/api-client";
import { clearSession, loadSession, signInWithPassword, signUpWithPassword, type AuthSession } from "@/lib/auth-client";
import type {
  Clip,
  ClipInteractionEvent,
  ClipInteractionState,
  DownloadOption,
  DownloadQuality,
  FeedbackTriggerType,
  GenerateClipsResult,
} from "@/lib/types";

const WATCH_THRESHOLD_SECONDS = 5;
const MAX_TRACKED_EVENTS = 30;

type FeedbackState = "idle" | "sending" | "sent" | "error";
type FeedbackAction = "good" | "improve" | "regenerate";
type GenerateInput = {
  youtubeUrl?: string;
  videoFile?: File | null;
  userConfirmedRights: boolean;
};

type FeedbackModalState = {
  open: boolean;
  clipId: string;
  action: "good" | "improve";
};

type DownloadModalState = {
  open: boolean;
  clipId: string;
  options: DownloadOption[];
  loading: boolean;
  submitting: boolean;
  error: string | null;
  userAuthenticated: boolean;
};

const AI_TAGS = [
  ["High retention", "Strong hook", "Story-driven"],
  ["Audience magnet", "Hook first", "Short-form fit"],
  ["Replay potential", "Fast pacing", "Narrative arc"],
  ["Scroll stopper", "Emotion-led", "Creator ready"],
] as const;

function createInitialClipInteraction(): ClipInteractionState {
  return {
    watchTimeSeconds: 0,
    playCount: 0,
    downloadClicks: 0,
    rateClicks: 0,
    interactionEvents: [],
    lastKnownVideoTimeSeconds: null,
    feedbackPrompted: false,
    feedbackSubmitted: false,
    lastFeedbackTrigger: null,
  };
}

function appendEvent(
  existing: ClipInteractionEvent[],
  type: ClipInteractionEvent["type"]
): ClipInteractionEvent[] {
  const next = [...existing, { type, at: new Date().toISOString() }];
  if (next.length <= MAX_TRACKED_EVENTS) {
    return next;
  }
  return next.slice(next.length - MAX_TRACKED_EVENTS);
}

function buildInteractionMap(
  clips: Clip[],
  previous: Record<string, ClipInteractionState>
): Record<string, ClipInteractionState> {
  const map: Record<string, ClipInteractionState> = {};
  for (const clip of clips) {
    map[clip.id] = previous[clip.id] ?? createInitialClipInteraction();
  }
  return map;
}

function deriveTags(clipId: string): string[] {
  let hash = 0;
  for (let index = 0; index < clipId.length; index += 1) {
    hash = (hash + clipId.charCodeAt(index) * (index + 1)) % AI_TAGS.length;
  }
  return [...AI_TAGS[Math.abs(hash) % AI_TAGS.length]];
}

function toFriendlyError(error: unknown): string {
  const fallback = "We couldn't extract clips from this source. Try another video.";
  if (!(error instanceof ApiError)) {
    return fallback;
  }

  if (error.reason === "youtube_blocked") {
    return "⚠️ This video could not be processed due to YouTube restrictions.";
  }

  const raw = error.message.toLowerCase();
  if (raw.includes("restricted") || raw.includes("private")) {
    return "This video may be restricted. Try another link.";
  }
  if (raw.includes("timeout") || raw.includes("too long")) {
    return "Crafting clips is taking longer than expected. Try again in a moment.";
  }
  if (raw.includes("valid youtube") || raw.includes("valid video source")) {
    return "Please provide a valid source video.";
  }
  if (raw.includes("confirm rights")) {
    return "Please confirm rights before processing.";
  }
  return fallback;
}

function triggerBrowserDownload(url: string, fileName: string): void {
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = fileName;
  anchor.rel = "noreferrer";
  anchor.target = "_blank";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
}

export default function Home() {
  const [inputUrl, setInputUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [clips, setClips] = useState<Clip[]>([]);
  const [revealedCount, setRevealedCount] = useState(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [failureReason, setFailureReason] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [isDemoModeActive, setIsDemoModeActive] = useState(false);
  const [openFilePickerSignal, setOpenFilePickerSignal] = useState(0);
  const [jobId, setJobId] = useState<string | null>(null);
  const [viewerIndex, setViewerIndex] = useState<number | null>(null);
  const [feedbackStateMap, setFeedbackStateMap] = useState<Record<string, FeedbackState>>({});
  const [favoriteMap, setFavoriteMap] = useState<Record<string, boolean>>({});
  const [captionEnabledMap, setCaptionEnabledMap] = useState<Record<string, boolean>>({});
  const [trimWindows, setTrimWindows] = useState<Record<string, TrimWindow>>({});
  const [clipInteractions, setClipInteractions] = useState<Record<string, ClipInteractionState>>({});

  const [authSession, setAuthSession] = useState<AuthSession | null>(null);
  const [authModalOpen, setAuthModalOpen] = useState(false);
  const [authLoading, setAuthLoading] = useState(false);
  const [authError, setAuthError] = useState<string | null>(null);
  const [pendingDownload, setPendingDownload] = useState<{ clipId: string; quality: DownloadQuality } | null>(null);

  const [feedbackModal, setFeedbackModal] = useState<FeedbackModalState | null>(null);
  const [feedbackModalError, setFeedbackModalError] = useState<string | null>(null);

  const [downloadModal, setDownloadModal] = useState<DownloadModalState>({
    open: false,
    clipId: "",
    options: [],
    loading: false,
    submitting: false,
    error: null,
    userAuthenticated: false,
  });

  const generationRequestIdRef = useRef(0);
  const interactionsRef = useRef<Record<string, ClipInteractionState>>({});

  useEffect(() => {
    interactionsRef.current = clipInteractions;
  }, [clipInteractions]);

  useEffect(() => {
    setAuthSession(loadSession());
  }, []);

  useEffect(() => {
    if (clips.length === 0) {
      setRevealedCount(0);
      return;
    }
    setRevealedCount(0);
    const timer = window.setInterval(() => {
      setRevealedCount((previous) => {
        const next = Math.min(clips.length, previous + 1);
        if (next >= clips.length) {
          window.clearInterval(timer);
        }
        return next;
      });
    }, 340);

    return () => window.clearInterval(timer);
  }, [clips]);

  const updateClipInteraction = useCallback(
    (clipId: string, updater: (current: ClipInteractionState) => ClipInteractionState) => {
      setClipInteractions((previous) => {
        const current = previous[clipId] ?? createInitialClipInteraction();
        const next = updater(current);
        return {
          ...previous,
          [clipId]: next,
        };
      });
    },
    []
  );

  const submitInlineFeedback = useCallback(
    async (
      clipId: string,
      input: {
        action: FeedbackAction;
        liked: boolean;
        reasons: string[];
        note?: string;
        triggerType?: FeedbackTriggerType;
      }
    ) => {
      const interaction = interactionsRef.current[clipId] ?? createInitialClipInteraction();
      setFeedbackStateMap((previous) => ({ ...previous, [clipId]: "sending" }));

      try {
        await submitClipFeedback({
          clipId,
          action: input.action,
          liked: input.liked,
          reasons: input.reasons,
          note: input.note ?? "",
          triggerType: input.triggerType ?? "rate_button",
          implicit: {
            watchTimeSeconds: interaction.watchTimeSeconds,
            playCount: interaction.playCount,
            downloadClicks: interaction.downloadClicks,
            rateClicks: interaction.rateClicks,
            interactionEvents: interaction.interactionEvents,
          },
        });

        updateClipInteraction(clipId, (current) => ({
          ...current,
          feedbackSubmitted: true,
          feedbackPrompted: true,
          lastFeedbackTrigger: input.triggerType ?? "rate_button",
        }));
        setFeedbackStateMap((previous) => ({ ...previous, [clipId]: "sent" }));
      } catch {
        setFeedbackStateMap((previous) => ({ ...previous, [clipId]: "error" }));
        throw new Error("Feedback submit failed");
      }
    },
    [updateClipInteraction]
  );

  const applyGeneratedClips = useCallback((generated: GenerateClipsResult) => {
    setJobId(generated.jobId);
    setClips(generated.clips);
    setClipInteractions((previous) => buildInteractionMap(generated.clips, previous));
    setFavoriteMap((previous) => {
      const next: Record<string, boolean> = {};
      for (const clip of generated.clips) {
        next[clip.id] = previous[clip.id] ?? false;
      }
      return next;
    });
    setCaptionEnabledMap((previous) => {
      const next: Record<string, boolean> = {};
      for (const clip of generated.clips) {
        next[clip.id] = previous[clip.id] ?? true;
      }
      return next;
    });
    setTrimWindows((previous) => {
      const next: Record<string, TrimWindow> = {};
      for (const clip of generated.clips) {
        next[clip.id] = previous[clip.id] ?? { start: 0, end: 1 };
      }
      return next;
    });
  }, []);

  const runGenerationRequest = useCallback(
    async (
      request: () => Promise<GenerateClipsResult>,
      options?: { statusMessage?: string; demoMode?: boolean }
    ) => {
      const requestId = generationRequestIdRef.current + 1;
      generationRequestIdRef.current = requestId;

      setLoading(true);
      setErrorMessage(null);
      setFailureReason(null);
      setStatusMessage(options?.statusMessage ?? null);
      setIsDemoModeActive(Boolean(options?.demoMode));
      setViewerIndex(null);
      setJobId(null);
      setClips([]);
      setRevealedCount(0);
      setFeedbackStateMap({});
      setDownloadModal((previous) => ({ ...previous, open: false }));

      try {
        const generated = await request();
        if (generationRequestIdRef.current !== requestId) {
          return;
        }
        setStatusMessage(null);
        applyGeneratedClips(generated);
      } catch (error) {
        if (generationRequestIdRef.current !== requestId) {
          return;
        }
        setStatusMessage(null);
        setClips([]);
        setFailureReason(error instanceof ApiError ? error.reason ?? null : null);
        setErrorMessage(toFriendlyError(error));
      } finally {
        if (generationRequestIdRef.current === requestId) {
          setLoading(false);
          setStatusMessage(null);
          setIsDemoModeActive(false);
        }
      }
    },
    [applyGeneratedClips]
  );

  const handleGenerate = useCallback(
    async (input: GenerateInput) => {
      const youtubeUrl = (input.youtubeUrl ?? "").trim();
      const videoFile = input.videoFile ?? null;
      if (!youtubeUrl && !videoFile) {
        setStatusMessage(null);
        setIsDemoModeActive(false);
        setFailureReason(null);
        setErrorMessage("Provide a YouTube URL or upload a video file.");
        return;
      }

      if (!input.userConfirmedRights) {
        setStatusMessage(null);
        setIsDemoModeActive(false);
        setFailureReason(null);
        setErrorMessage("Please confirm rights before processing.");
        return;
      }

      await runGenerationRequest(() =>
        generateClips({
          youtubeUrl,
          videoFile,
          userConfirmedRights: input.userConfirmedRights,
        })
      );
    },
    [runGenerationRequest]
  );

  const handleDemoClick = useCallback(async () => {
    await runGenerationRequest(() => generateFromDemo(), {
      statusMessage: "Running demo using preloaded video",
      demoMode: true,
    });
  }, [runGenerationRequest]
  );

  const handlePlay = useCallback(
    (clipId: string) => {
      updateClipInteraction(clipId, (current) => ({
        ...current,
        playCount: current.playCount + 1,
        lastKnownVideoTimeSeconds: null,
        interactionEvents: appendEvent(current.interactionEvents, "play"),
      }));
    },
    [updateClipInteraction]
  );

  const handleProgress = useCallback(
    (clipId: string, currentTime: number) => {
      const current = interactionsRef.current[clipId] ?? createInitialClipInteraction();
      const last = current.lastKnownVideoTimeSeconds;
      const delta = typeof last === "number" ? currentTime - last : 0;
      const safeDelta = delta > 0 && delta < 5 ? delta : 0;
      const nextWatchTime = current.watchTimeSeconds + safeDelta;

      updateClipInteraction(clipId, (existing) => ({
        ...existing,
        watchTimeSeconds: nextWatchTime,
        lastKnownVideoTimeSeconds: currentTime,
      }));

      if (
        nextWatchTime >= WATCH_THRESHOLD_SECONDS &&
        !current.feedbackSubmitted &&
        !current.feedbackPrompted
      ) {
        updateClipInteraction(clipId, (existing) => ({
          ...existing,
          feedbackPrompted: true,
          lastFeedbackTrigger: "watch_threshold",
        }));
      }
    },
    [updateClipInteraction]
  );

  const handlePause = useCallback(
    (clipId: string, currentTime: number) => {
      updateClipInteraction(clipId, (existing) => ({
        ...existing,
        lastKnownVideoTimeSeconds: currentTime,
      }));
    },
    [updateClipInteraction]
  );

  const openDownloadModal = useCallback(
    async (clipId: string) => {
      if (!jobId) {
        setFailureReason(null);
        setErrorMessage("Missing job context for download.");
        return;
      }

      setDownloadModal({
        open: true,
        clipId,
        options: [],
        loading: true,
        submitting: false,
        error: null,
        userAuthenticated: Boolean(authSession?.accessToken),
      });

      try {
        const payload = await fetchDownloadOptions({
          jobId,
          clipId,
          accessToken: authSession?.accessToken,
        });
        setDownloadModal((previous) => ({
          ...previous,
          loading: false,
          options: Array.isArray(payload.options) ? payload.options : [],
          userAuthenticated: Boolean(payload.userAuthenticated),
          error: null,
        }));
      } catch (error) {
        setDownloadModal((previous) => ({
          ...previous,
          loading: false,
          error: error instanceof Error ? error.message : "Failed to load download options.",
        }));
      }
    },
    [authSession?.accessToken, jobId]
  );

  const performDownload = useCallback(
    async (clipId: string, quality: DownloadQuality) => {
      if (!jobId) {
        setDownloadModal((previous) => ({
          ...previous,
          error: "Missing job context for download.",
        }));
        return;
      }

      if (!authSession?.accessToken) {
        setPendingDownload({ clipId, quality });
        setAuthModalOpen(true);
        return;
      }

      setDownloadModal((previous) => ({ ...previous, submitting: true, error: null }));
      try {
        const payload = await requestDownloadUrl({
          jobId,
          clipId,
          quality,
          accessToken: authSession.accessToken,
        });
        triggerBrowserDownload(payload.downloadUrl, `${clipId}_${quality}.mp4`);
        updateClipInteraction(clipId, (current) => ({
          ...current,
          downloadClicks: current.downloadClicks + 1,
          interactionEvents: appendEvent(current.interactionEvents, "download"),
        }));
        setDownloadModal((previous) => ({ ...previous, submitting: false, open: false }));
      } catch (error) {
        if (error instanceof ApiError && error.status === 401) {
          setPendingDownload({ clipId, quality });
          setAuthModalOpen(true);
          setDownloadModal((previous) => ({ ...previous, submitting: false }));
          return;
        }
        setDownloadModal((previous) => ({
          ...previous,
          submitting: false,
          error: error instanceof Error ? error.message : "Download failed.",
        }));
      }
    },
    [authSession?.accessToken, jobId, updateClipInteraction]
  );

  const checkout = useCallback(
    async (purchaseType: "monthly_subscription" | "job_unlock") => {
      if (!jobId && purchaseType === "job_unlock") {
        setDownloadModal((previous) => ({
          ...previous,
          error: "Missing job context for checkout.",
        }));
        return;
      }
      if (!authSession?.accessToken) {
        setAuthModalOpen(true);
        return;
      }

      setDownloadModal((previous) => ({ ...previous, submitting: true, error: null }));
      try {
        const payload = await createCheckout({
          purchaseType,
          jobId: purchaseType === "job_unlock" ? jobId ?? "" : undefined,
          accessToken: authSession.accessToken,
        });
        window.location.href = payload.checkoutUrl;
      } catch (error) {
        setDownloadModal((previous) => ({
          ...previous,
          submitting: false,
          error: error instanceof Error ? error.message : "Checkout failed.",
        }));
      }
    },
    [authSession?.accessToken, jobId]
  );

  const handleFeedbackAction = useCallback(
    async (clipId: string, action: FeedbackAction) => {
      updateClipInteraction(clipId, (current) => ({
        ...current,
        rateClicks: current.rateClicks + 1,
        interactionEvents: appendEvent(current.interactionEvents, "rate"),
      }));

      if (action === "regenerate") {
        try {
          await submitInlineFeedback(clipId, {
            action,
            liked: false,
            reasons: ["Regenerate requested"],
          });
        } catch {
          // feedback submission failure should not block regenerate request
        }
        if (!loading && inputUrl.trim()) {
          void handleGenerate({
            youtubeUrl: inputUrl.trim(),
            userConfirmedRights: true,
          });
        }
        return;
      }

      setFeedbackModal({
        open: true,
        clipId,
        action,
      });
      setFeedbackModalError(null);
    },
    [handleGenerate, inputUrl, loading, submitInlineFeedback, updateClipInteraction]
  );

  const submitDetailedFeedback = useCallback(
    async (payload: { reasons: string[]; note: string }) => {
      if (!feedbackModal) {
        return;
      }
      try {
        await submitInlineFeedback(feedbackModal.clipId, {
          action: feedbackModal.action,
          liked: feedbackModal.action === "good",
          reasons: payload.reasons,
          note: payload.note,
        });
        setFeedbackModal(null);
        setFeedbackModalError(null);
      } catch {
        setFeedbackModalError("Feedback failed. Please try again.");
      }
    },
    [feedbackModal, submitInlineFeedback]
  );

  const handlePlaybackError = useCallback(
    async (clipId: string): Promise<string | null> => {
      if (!jobId) {
        return null;
      }

      try {
        const refreshed = await refreshClipUrls(jobId);
        if (refreshed.length === 0) {
          return null;
        }

        const refreshedById = new Map(refreshed.map((clip) => [clip.id, clip]));
        let nextUrl: string | null = null;
        setClips((previous) =>
          previous.map((item, index) => {
            const picked = refreshedById.get(item.id) ?? refreshed[index] ?? item;
            if (picked.id === clipId) {
              nextUrl = picked.videoUrl;
            }
            return picked;
          })
        );
        return nextUrl;
      } catch {
        return null;
      }
    },
    [jobId]
  );

  const handleAuthSubmit = useCallback(
    async (payload: { mode: "login" | "signup"; email: string; password: string }) => {
      setAuthLoading(true);
      setAuthError(null);
      try {
        const session =
          payload.mode === "login"
            ? await signInWithPassword({ email: payload.email, password: payload.password })
            : await signUpWithPassword({ email: payload.email, password: payload.password });
        setAuthSession(session);
        setAuthModalOpen(false);
        if (pendingDownload) {
          const pending = pendingDownload;
          setPendingDownload(null);
          void performDownload(pending.clipId, pending.quality);
        } else if (downloadModal.open && downloadModal.clipId) {
          void openDownloadModal(downloadModal.clipId);
        }
      } catch (error) {
        setAuthError(error instanceof Error ? error.message : "Authentication failed.");
      } finally {
        setAuthLoading(false);
      }
    },
    [downloadModal.clipId, downloadModal.open, openDownloadModal, pendingDownload, performDownload]
  );

  const displayedClips = useMemo(() => clips.slice(0, revealedCount), [clips, revealedCount]);

  const viewerOpen = viewerIndex !== null && viewerIndex >= 0 && viewerIndex < clips.length;

  return (
    <main className="app-shell">
      <header className="app-header">
        <div className="header-top">
          <span className="label-outlined">AI Shorts Engine</span>
          <div className="auth-header-actions">
            <span className="badge-dark">
              {clips.length > 0 ? `${clips.length} Clips` : "Ready"}
            </span>
            {authSession ? (
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => {
                  clearSession();
                  setAuthSession(null);
                }}
              >
                Logout
              </button>
            ) : (
              <button type="button" className="btn btn-secondary" onClick={() => setAuthModalOpen(true)}>
                Login
              </button>
            )}
          </div>
        </div>
        <h1 className="hero-title">
          AI Shorts <span className="muted">Creator Studio</span>
        </h1>
        <p className="hero-sub">
          AI video processing with a creator-first clip generation engine.
        </p>
      </header>

      <section className="section-card">
        <InputSection
          value={inputUrl}
          onChange={setInputUrl}
          onGenerate={handleGenerate}
          onDemoClick={handleDemoClick}
          loading={loading}
          openFilePickerSignal={openFilePickerSignal}
        />

        {errorMessage && (
          <div className="status-note status-error">
            <p>{errorMessage}</p>
            {failureReason === "youtube_blocked" && (
              <>
                <p>👉 Try another video or upload your own file.</p>
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => setOpenFilePickerSignal((previous) => previous + 1)}
                  disabled={loading}
                >
                  Upload Video Instead
                </button>
              </>
            )}
          </div>
        )}
        {statusMessage && !errorMessage && (
          <p className="status-note">{statusMessage}</p>
        )}
        {loading && isDemoModeActive && (
          <div className="status-note">
            <p>🎬 Demo Mode Activated</p>
            <p>• Name: {DEMO_VIDEO_META.name}</p>
            <p>• Duration: {DEMO_VIDEO_META.duration}</p>
            <p>• Size: {DEMO_VIDEO_META.size}</p>
            <p>• Source: {DEMO_VIDEO_META.source}</p>
          </div>
        )}
        {loading && <LoadingState revealedCount={displayedClips.length} />}

        {!loading && !errorMessage && clips.length === 0 && (
          <p className="status-note">
            Add a YouTube link or upload a video file to start the pipeline.
          </p>
        )}
      </section>

      {clips.length > 0 && (
        <section>
          <div className="results-head">
            <p className="mono">Generated Clips</p>
            <span className="label-outlined">
              {displayedClips.length}/{clips.length} Ready
            </span>
          </div>

          <div className="cards-grid">
            {displayedClips.map((clip, index) => (
              <ClipCard
                key={clip.id}
                clip={clip}
                order={index + 1}
                tags={deriveTags(clip.id)}
                interaction={clipInteractions[clip.id] ?? createInitialClipInteraction()}
                feedbackState={feedbackStateMap[clip.id] ?? "idle"}
                onPlay={handlePlay}
                onProgress={handleProgress}
                onPause={handlePause}
                onDownload={(clipId) => {
                  void openDownloadModal(clipId);
                }}
                onOpenStudio={(clipId) => {
                  const nextIndex = clips.findIndex((item) => item.id === clipId);
                  if (nextIndex >= 0) {
                    setViewerIndex(nextIndex);
                  }
                }}
                onFeedback={handleFeedbackAction}
                onPlaybackError={handlePlaybackError}
              />
            ))}

            {!loading &&
              displayedClips.length < clips.length &&
              Array.from({ length: clips.length - displayedClips.length }).map((_, index) => (
                <article className="skeleton-card" key={`reveal-${index}`}>
                  <div className="skeleton-frame shimmer" />
                  <div className="skeleton-bar shimmer" />
                  <div className="skeleton-bar shimmer short" />
                </article>
              ))}
          </div>
        </section>
      )}

      <FullscreenViewer
        open={viewerOpen}
        clips={clips}
        activeIndex={viewerIndex ?? 0}
        favoriteMap={favoriteMap}
        captionEnabledMap={captionEnabledMap}
        trimWindows={trimWindows}
        getTags={deriveTags}
        onClose={() => setViewerIndex(null)}
        onNavigate={(nextIndex) => {
          if (clips.length === 0) {
            return;
          }
          const wrapped = (nextIndex + clips.length) % clips.length;
          setViewerIndex(wrapped);
        }}
        onToggleFavorite={(clipId) => {
          setFavoriteMap((previous) => ({
            ...previous,
            [clipId]: !previous[clipId],
          }));
        }}
        onToggleCaption={(clipId) => {
          setCaptionEnabledMap((previous) => ({
            ...previous,
            [clipId]: !(previous[clipId] ?? true),
          }));
        }}
        onTrimChange={(clipId, trim) => {
          setTrimWindows((previous) => ({
            ...previous,
            [clipId]: trim,
          }));
        }}
        onDownload={(clipId) => {
          void openDownloadModal(clipId);
        }}
        onRegenerate={(clipId) => {
          void handleFeedbackAction(clipId, "regenerate");
        }}
        onPlaybackError={handlePlaybackError}
      />

      <FeedbackModal
        key={`${feedbackModal?.clipId ?? "none"}_${feedbackModal?.action ?? "improve"}_${feedbackModal?.open ? "open" : "closed"}`}
        open={Boolean(feedbackModal?.open)}
        clipId={feedbackModal?.clipId ?? ""}
        action={feedbackModal?.action ?? "improve"}
        submitting={feedbackModal ? feedbackStateMap[feedbackModal.clipId] === "sending" : false}
        errorMessage={feedbackModalError}
        onSubmit={(payload) => {
          void submitDetailedFeedback(payload);
        }}
        onClose={() => {
          setFeedbackModal(null);
          setFeedbackModalError(null);
        }}
      />

      <DownloadOptionsModal
        open={downloadModal.open}
        clipId={downloadModal.clipId}
        options={downloadModal.options}
        loading={downloadModal.loading}
        submitting={downloadModal.submitting}
        error={downloadModal.error}
        userAuthenticated={downloadModal.userAuthenticated}
        onRefresh={() => {
          void openDownloadModal(downloadModal.clipId);
        }}
        onDownloadQuality={(quality) => {
          void performDownload(downloadModal.clipId, quality);
        }}
        onBuyJobUnlock={() => {
          void checkout("job_unlock");
        }}
        onBuyMonthly={() => {
          void checkout("monthly_subscription");
        }}
        onClose={() =>
          setDownloadModal((previous) => ({
            ...previous,
            open: false,
          }))
        }
      />

      <AuthModal
        open={authModalOpen}
        loading={authLoading}
        error={authError}
        onSubmit={(payload) => {
          void handleAuthSubmit(payload);
        }}
        onClose={() => {
          setAuthModalOpen(false);
          setAuthError(null);
        }}
      />
    </main>
  );
}

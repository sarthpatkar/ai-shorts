export type Clip = {
  id: string;
  videoUrl: string;
  caption: string;
};

export type GenerateClipsResult = {
  clips: Clip[];
  runId: string | null;
  jobId: string | null;
};

export type InteractionEventType = "play" | "download" | "rate";

export type ClipInteractionEvent = {
  type: InteractionEventType;
  at: string;
};

export type FeedbackTriggerType =
  | "watch_threshold"
  | "download_click"
  | "rate_button";

export type ClipImplicitFeedback = {
  watchTimeSeconds: number;
  playCount: number;
  downloadClicks: number;
  rateClicks: number;
  interactionEvents: ClipInteractionEvent[];
};

export type ClipInteractionState = ClipImplicitFeedback & {
  lastKnownVideoTimeSeconds: number | null;
  feedbackPrompted: boolean;
  feedbackSubmitted: boolean;
  lastFeedbackTrigger: FeedbackTriggerType | null;
};

export type FeedbackSubmissionInput = {
  clipId: string;
  action: "good" | "improve" | "regenerate";
  liked: boolean;
  reasons: string[];
  note?: string;
  triggerType: FeedbackTriggerType | null;
  implicit: ClipImplicitFeedback;
};

export type DownloadQuality = "240p" | "360p" | "480p" | "720p" | "1080p";

export type DownloadOption = {
  quality: DownloadQuality;
  available: boolean;
  locked: boolean;
  reason: string;
  height: number | null;
};

"use client";

import type { DownloadOption, DownloadQuality } from "@/lib/types";

type DownloadOptionsModalProps = {
  open: boolean;
  clipId: string;
  options: DownloadOption[];
  loading: boolean;
  submitting: boolean;
  error: string | null;
  userAuthenticated: boolean;
  onRefresh: () => void;
  onDownloadQuality: (quality: DownloadQuality) => void;
  onBuyJobUnlock: () => void;
  onBuyMonthly: () => void;
  onClose: () => void;
};

function labelForReason(reason: string): string {
  if (reason === "needs_auth") {
    return "Login required";
  }
  if (reason === "unavailable") {
    return "Not available";
  }
  if (reason === "free_480_already_used") {
    return "One-time 480 used";
  }
  if (reason === "premium_unlock_required") {
    return "Premium required";
  }
  if (reason === "needs_purchase") {
    return "Purchase required";
  }
  return reason.replace(/_/g, " ");
}

export default function DownloadOptionsModal({
  open,
  clipId,
  options,
  loading,
  submitting,
  error,
  userAuthenticated,
  onRefresh,
  onDownloadQuality,
  onBuyJobUnlock,
  onBuyMonthly,
  onClose,
}: DownloadOptionsModalProps) {
  if (!open) {
    return null;
  }

  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true">
      <section className="modal-shell" aria-label={`Download options for ${clipId}`}>
        <div className="feedback-head">
          <span className="mono">Download Options</span>
          <span className="chip">Clip {clipId.slice(0, 8)}</span>
        </div>

        {!userAuthenticated && (
          <p className="status-note">Login is required to download at any quality.</p>
        )}

        <div className="quality-grid">
          {options.map((option) => (
            <div key={option.quality} className="quality-row">
              <div>
                <p className="quality-title">{option.quality}</p>
                <p className="quality-note">{labelForReason(option.reason)}</p>
              </div>
              <button
                type="button"
                className="btn btn-secondary"
                disabled={loading || submitting || option.locked || !option.available}
                onClick={() => onDownloadQuality(option.quality)}
              >
                Download
              </button>
            </div>
          ))}
        </div>

        {error && <p className="status-note status-error">{error}</p>}

        <div className="modal-actions">
          <button type="button" className="btn btn-secondary" onClick={onRefresh} disabled={loading || submitting}>
            Refresh
          </button>
          <button type="button" className="btn btn-secondary" onClick={onBuyJobUnlock} disabled={loading || submitting}>
            Unlock This Video
          </button>
          <button type="button" className="btn btn-primary" onClick={onBuyMonthly} disabled={loading || submitting}>
            Go Premium
          </button>
        </div>

        <div className="modal-actions">
          <button type="button" className="btn btn-secondary" onClick={onClose} disabled={loading || submitting}>
            Close
          </button>
        </div>
      </section>
    </div>
  );
}

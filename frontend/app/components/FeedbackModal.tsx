"use client";

import { useMemo, useState } from "react";

type FeedbackAction = "good" | "improve";

type FeedbackModalProps = {
  open: boolean;
  clipId: string;
  action: FeedbackAction;
  submitting: boolean;
  errorMessage: string | null;
  onSubmit: (payload: { reasons: string[]; note: string }) => void;
  onClose: () => void;
};

const positiveReasons = [
  "Strong hook",
  "Great pacing",
  "Clear message",
  "High retention",
  "Good caption",
];

const improveReasons = [
  "Weak opening",
  "Slow pacing",
  "Confusing context",
  "Caption mismatch",
  "Needs stronger ending",
];

export default function FeedbackModal({
  open,
  clipId,
  action,
  submitting,
  errorMessage,
  onSubmit,
  onClose,
}: FeedbackModalProps) {
  const [selected, setSelected] = useState<string[]>([]);
  const [note, setNote] = useState("");

  const reasons = useMemo(
    () => (action === "good" ? positiveReasons : improveReasons),
    [action]
  );

  if (!open) {
    return null;
  }

  const toggleReason = (reason: string) => {
    setSelected((prev) =>
      prev.includes(reason) ? prev.filter((item) => item !== reason) : [...prev, reason]
    );
  };

  const submit = () => {
    if (submitting) {
      return;
    }
    onSubmit({ reasons: selected, note: note.trim() });
  };

  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true">
      <section className="modal-shell" aria-label={`Detailed feedback for clip ${clipId}`}>
        <div className="feedback-head">
          <span className="mono">Detailed Feedback</span>
          <span className="chip">ID {clipId.slice(0, 8)}</span>
        </div>

        <p className="feedback-copy">
          {action === "good"
            ? "What worked in this clip?"
            : "What should be improved in this clip?"}
        </p>

        <div className="reason-grid">
          {reasons.map((reason) => {
            const selectedState = selected.includes(reason);
            return (
              <button
                key={reason}
                type="button"
                disabled={submitting}
                onClick={() => toggleReason(reason)}
                className={`reason-chip ${selectedState ? "is-selected" : ""}`}
              >
                {reason}
              </button>
            );
          })}
        </div>

        <label className="mono" htmlFor="feedback-note">
          Optional note
        </label>
        <textarea
          id="feedback-note"
          className="ui-input"
          rows={4}
          value={note}
          onChange={(event) => setNote(event.target.value)}
          placeholder="Add details to improve the learning loop..."
          disabled={submitting}
        />

        {errorMessage && <p className="status-note status-error">{errorMessage}</p>}

        <div className="modal-actions">
          <button
            type="button"
            onClick={onClose}
            disabled={submitting}
            className="btn btn-secondary"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={submit}
            disabled={submitting}
            className="btn btn-primary"
          >
            {submitting ? "Submitting..." : "Submit feedback"}
          </button>
        </div>
      </section>
    </div>
  );
}

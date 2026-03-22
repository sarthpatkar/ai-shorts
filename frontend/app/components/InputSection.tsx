"use client";

import { ChangeEvent, FormEvent, useEffect, useMemo, useRef, useState } from "react";

type GenerateInput = {
  youtubeUrl?: string;
  videoFile?: File | null;
  userConfirmedRights: boolean;
};

type InputSectionProps = {
  value: string;
  onChange: (value: string) => void;
  onGenerate: (input: GenerateInput) => void | Promise<void>;
  loading: boolean;
  openFilePickerSignal?: number;
};

export default function InputSection({
  value,
  onChange,
  onGenerate,
  loading,
  openFilePickerSignal = 0,
}: InputSectionProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [userConfirmedRights, setUserConfirmedRights] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const trimmedUrl = useMemo(() => value.trim(), [value]);
  const canSubmit =
    !loading &&
    userConfirmedRights &&
    (trimmedUrl.length > 0 || Boolean(selectedFile));

  useEffect(() => {
    if (loading) {
      return;
    }
    if (openFilePickerSignal <= 0) {
      return;
    }
    fileInputRef.current?.click();
  }, [loading, openFilePickerSignal]);

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!canSubmit) {
      return;
    }

    onGenerate({
      youtubeUrl: trimmedUrl || undefined,
      videoFile: selectedFile,
      userConfirmedRights,
    });
  };

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null;
    setSelectedFile(file);
  };

  return (
    <form onSubmit={handleSubmit} className="input-panel" aria-label="Generate clips">
      <div>
        <h2 className="section-title">AI Video Processing</h2>
        <p className="section-sub">
          Submit a YouTube link or local file and run it through the clip generation
          engine.
        </p>
      </div>

      <div className="source-grid">
        <label htmlFor="youtube-url" className="mono">
          YouTube Source (Optional)
        </label>
        <input
          id="youtube-url"
          type="url"
          placeholder="Paste a YouTube URL..."
          value={value}
          onChange={(event) => onChange(event.target.value)}
          className="ui-input"
          autoComplete="off"
          disabled={loading}
        />

        <label htmlFor="video-file" className="mono">
          Video Upload (Optional)
        </label>
        <div className="file-input-wrap">
          <input
            ref={fileInputRef}
            id="video-file"
            type="file"
            accept="video/*,.mp4,.mov,.m4v,.webm,.mkv,.avi"
            onChange={handleFileChange}
            className="file-input"
            disabled={loading}
          />
          {selectedFile && <span className="file-meta">{selectedFile.name}</span>}
        </div>
      </div>

      <p className="rights-disclaimer">
        Use only videos you own or have permission to process
      </p>

      <label className="rights-row">
        <input
          type="checkbox"
          checked={userConfirmedRights}
          onChange={(event) => setUserConfirmedRights(event.target.checked)}
          disabled={loading}
        />
        <span>I confirm I have rights to process this content.</span>
      </label>

      <div className="input-row">
        <button type="submit" disabled={!canSubmit} className="btn btn-primary">
          {loading ? "Processing..." : "Start Processing"}
        </button>
      </div>

      <p className="input-hint">
        Longer videos can take a few minutes. The processing pipeline runs in stages.
      </p>
    </form>
  );
}

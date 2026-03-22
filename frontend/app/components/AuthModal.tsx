"use client";

import { useState } from "react";

type AuthMode = "login" | "signup";

type AuthModalProps = {
  open: boolean;
  loading: boolean;
  error: string | null;
  onSubmit: (payload: { mode: AuthMode; email: string; password: string }) => void;
  onClose: () => void;
};

export default function AuthModal({
  open,
  loading,
  error,
  onSubmit,
  onClose,
}: AuthModalProps) {
  const [mode, setMode] = useState<AuthMode>("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  if (!open) {
    return null;
  }

  const submit = () => {
    const cleanEmail = email.trim();
    if (!cleanEmail || !password.trim() || loading) {
      return;
    }
    onSubmit({ mode, email: cleanEmail, password });
  };

  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true">
      <section className="modal-shell" aria-label="Authentication required">
        <div className="feedback-head">
          <span className="mono">Account Required</span>
          <span className="chip">Download Access</span>
        </div>

        <p className="feedback-copy">
          Sign in or create an account to download generated clips.
        </p>

        <div className="auth-tabs">
          <button
            type="button"
            className={`btn btn-secondary ${mode === "login" ? "is-active" : ""}`}
            onClick={() => setMode("login")}
            disabled={loading}
          >
            Login
          </button>
          <button
            type="button"
            className={`btn btn-secondary ${mode === "signup" ? "is-active" : ""}`}
            onClick={() => setMode("signup")}
            disabled={loading}
          >
            Sign Up
          </button>
        </div>

        <label className="mono" htmlFor="auth-email">
          Email
        </label>
        <input
          id="auth-email"
          className="ui-input"
          type="email"
          value={email}
          onChange={(event) => setEmail(event.target.value)}
          placeholder="you@example.com"
          disabled={loading}
        />

        <label className="mono" htmlFor="auth-password">
          Password
        </label>
        <input
          id="auth-password"
          className="ui-input"
          type="password"
          value={password}
          onChange={(event) => setPassword(event.target.value)}
          placeholder="Enter password"
          disabled={loading}
        />

        {error && <p className="status-note status-error">{error}</p>}

        <div className="modal-actions">
          <button
            type="button"
            onClick={onClose}
            disabled={loading}
            className="btn btn-secondary"
          >
            Close
          </button>
          <button type="button" onClick={submit} disabled={loading} className="btn btn-primary">
            {loading ? "Please wait..." : mode === "login" ? "Login" : "Create account"}
          </button>
        </div>
      </section>
    </div>
  );
}

"use client";

import { useCallback, useEffect, useRef, useState } from "react";

const CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789".split("");

// Must match the words your LSTM was trained on
const WORDS = ["book", "drink", "go", "clothes", "who", "before", "yes", "no", "help", "chair", "all"];

const MIN_CONFIDENCE = 0.5;
const MAX_CONFIDENCE = 1;
const MIN_POLL_MS = 300;
const MAX_POLL_MS = 2000;
const DEFAULT_CONFIDENCE = 0.9;
const DEFAULT_POLL_MS = 800;

type Mode = "letters" | "words";

export default function Home() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Session ID: unique per question, sent to backend so it can track the frame buffer
  const sessionIdRef = useRef<string>(crypto.randomUUID());

  const [mode, setMode] = useState<Mode>("letters");
  const [target, setTarget] = useState<string | null>(null);
  const [score, setScore] = useState<number | null>(null);
  const [predictedLabel, setPredictedLabel] = useState<string | null>(null);
  const [status, setStatus] = useState<string>("");
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [hasMatch, setHasMatch] = useState(false);
  // Progress 0-100 shown while word buffer is filling
  const [captureProgress, setCaptureProgress] = useState<number>(0);
  const [confidenceThreshold, setConfidenceThreshold] = useState(DEFAULT_CONFIDENCE);
  const [pollIntervalMs, setPollIntervalMs] = useState(DEFAULT_POLL_MS);
  const [settingsOpen, setSettingsOpen] = useState(false);

  const isPredictingRef = useRef(false);

  useEffect(() => {
    nextQuestion();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const startCamera = async () => {
    if (isCameraOn || !navigator.mediaDevices?.getUserMedia) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setIsCameraOn(true);
      setStatus("");
    } catch (err) {
      console.error(err);
      setStatus("Could not access camera. Please check permissions.");
    }
  };

  const stopCamera = () => {
    const video = videoRef.current;
    if (video && video.srcObject instanceof MediaStream) {
      video.srcObject.getTracks().forEach((t: MediaStreamTrack) => t.stop());
      video.srcObject = null;
    }
    setIsCameraOn(false);
  };

  const clearWordBuffer = async () => {
    try {
      await fetch("http://127.0.0.1:5000/api/clear-buffer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionIdRef.current }),
      });
    } catch (_) {
      // Non-critical
    }
  };

  const nextQuestion = (currentMode: Mode = mode) => {
    // New session ID = fresh frame buffer on the backend
    sessionIdRef.current = crypto.randomUUID();
    void clearWordBuffer();

    const pool = currentMode === "letters" ? CHARACTERS : WORDS;
    const randomItem = pool[Math.floor(Math.random() * pool.length)];
    setTarget(randomItem);
    setScore(null);
    setPredictedLabel(null);
    setStatus("");
    setHasMatch(false);
    setCaptureProgress(0);
  };

  const switchMode = (newMode: Mode) => {
    setMode(newMode);
    nextQuestion(newMode);
  };

  const captureAndPredict = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || !target || hasMatch) return;
    if (isPredictingRef.current) return;
    isPredictingRef.current = true;

    try {
      const video = videoRef.current;
      if (video.videoWidth === 0 || video.videoHeight === 0) {
        setStatus("Waiting for camera to initialize...");
        return;
      }

      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL("image/jpeg");

      if (mode === "letters") {
        // ── Letters/numbers: single-frame prediction, unchanged ──────────────
        const res = await fetch("http://127.0.0.1:5000/api/predict-sign", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image: dataUrl }),
        });

        if (!res.ok) {
          const error = await res.json().catch(() => ({}));
          setStatus(error.error || "Backend error while scoring sign.");
          return;
        }

        const json = await res.json();
        const confidence = typeof json.confidence === "number" ? json.confidence : null;
        const predicted = typeof json.predicted_label === "string" ? json.predicted_label : null;
        if (!predicted || confidence == null) return;

        if (!hasMatch) {
          setPredictedLabel(predicted);
          setScore(confidence);
        }

        if (predicted === target && confidence >= confidenceThreshold) {
          setHasMatch(true);
          setStatus("Great job! We detected your sign.");
        } else if (!hasMatch) {
          setStatus(`Listening... (model sees "${predicted}" at ${Math.round(confidence * 100)}% confidence)`);
        }

      } else {
        // ── Words: snapshot mode — collect frames until buffer full ───────────
        const res = await fetch("http://127.0.0.1:5000/api/predict-word-snapshot", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image: dataUrl, session_id: sessionIdRef.current }),
        });

        if (!res.ok) {
          const error = await res.json().catch(() => ({}));
          setStatus(error.error || "Backend error while scoring sign.");
          return;
        }

        const json = await res.json();

        if (json.locked) {
          // Buffer already full and result already shown — do nothing
          return;
        }

        if (!json.ready) {
          // Still collecting frames — show a progress bar in the status
          const pct = Math.round((json.frames_collected / json.frames_needed) * 100);
          setCaptureProgress(pct);
          setStatus(`Capturing... ${pct}%`);
          return;
        }

        // Buffer just filled — result is in
        setCaptureProgress(100);
        const confidence = typeof json.confidence === "number" ? json.confidence : null;
        const predicted = typeof json.predicted_label === "string" ? json.predicted_label : null;
        if (!predicted || confidence == null) return;

        setPredictedLabel(predicted);
        setScore(confidence);

        if (predicted === target && confidence >= confidenceThreshold) {
          setHasMatch(true);
          setStatus("Great job! We detected your sign.");
        } else {
          setStatus(`Model saw "${predicted}" at ${Math.round(confidence * 100)}% confidence.`);
        }
      }
    } catch (err) {
      console.error(err);
      setStatus("Network error talking to the backend.");
    } finally {
      isPredictingRef.current = false;
    }
  }, [hasMatch, target, confidenceThreshold, mode]);

  useEffect(() => {
    if (!isCameraOn || !target || hasMatch) return;

    if (!status) {
      setStatus(
        mode === "words"
          ? "Sign the word — we'll capture automatically."
          : "Show the sign for the prompt towards the camera. We'll detect it automatically."
      );
    }

    const intervalId = window.setInterval(() => {
      void captureAndPredict();
    }, pollIntervalMs);

    return () => window.clearInterval(intervalId);
  }, [captureAndPredict, hasMatch, isCameraOn, pollIntervalMs, status, target, mode]);

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-50 font-sans flex flex-col">
      {/* Header */}
      <header className="absolute top-0 left-0 right-0 z-10 flex items-center justify-between px-4 py-3 bg-gradient-to-b from-zinc-950/80 to-transparent">
        <h1 className="text-xl font-semibold tracking-tight text-white/95">
          ASL Quiz Mode
        </h1>
        <div className="flex items-center gap-2">

          {/* Mode toggle */}
          <div className="flex items-center rounded-full bg-white/10 p-1 backdrop-blur-sm">
            <button
              type="button"
              onClick={() => switchMode("letters")}
              className={`rounded-full px-3 py-1 text-sm font-medium transition-colors ${
                mode === "letters" ? "bg-white text-zinc-900" : "text-white hover:bg-white/10"
              }`}
            >
              Letters & Numbers
            </button>
            <button
              type="button"
              onClick={() => switchMode("words")}
              className={`rounded-full px-3 py-1 text-sm font-medium transition-colors ${
                mode === "words" ? "bg-white text-zinc-900" : "text-white hover:bg-white/10"
              }`}
            >
              Words
            </button>
          </div>

          <button
            type="button"
            onClick={() => nextQuestion()}
            className="rounded-full bg-white/10 px-4 py-2 text-sm font-medium text-white hover:bg-white/20 transition-colors backdrop-blur-sm"
          >
            New Prompt
          </button>
          {!isCameraOn ? (
            <button
              type="button"
              onClick={startCamera}
              className="rounded-full bg-emerald-500 text-emerald-950 px-4 py-2 text-sm font-medium hover:bg-emerald-400 transition-colors"
            >
              Enable Camera
            </button>
          ) : (
            <button
              type="button"
              onClick={stopCamera}
              className="rounded-full bg-zinc-700 text-zinc-100 px-4 py-2 text-sm font-medium hover:bg-zinc-600 transition-colors"
            >
              Turn Off Camera
            </button>
          )}
        </div>
      </header>

      {/* Main area */}
      <main className="flex-1 flex flex-col min-h-0 relative">
        <div className="flex-1 flex items-center justify-center min-h-0 bg-zinc-900 p-2 md:p-4">
          <div className="relative w-full max-w-5xl aspect-video max-h-[calc(100vh-8rem)] rounded-2xl overflow-hidden border border-zinc-700/50 bg-black shadow-2xl">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              className="absolute inset-0 w-full h-full object-cover"
            />

            {/* Prompt badge */}
            <div className="absolute top-3 left-3 flex items-center gap-2 pointer-events-none">
              <span className="inline-flex items-center justify-center rounded-md bg-black/60 px-3 py-1 text-xs font-medium uppercase tracking-wider text-zinc-300">
                {mode === "letters" ? "Letter / Number" : "Word"}
              </span>
              {mode === "letters" ? (
                <span className="inline-flex items-center justify-center rounded-md bg-white/90 text-zinc-900 text-2xl font-semibold w-10 h-10">
                  {target ?? "?"}
                </span>
              ) : (
                <span className="inline-flex items-center justify-center rounded-md bg-white/90 text-zinc-900 text-xl font-semibold px-4 h-10">
                  {target ?? "?"}
                </span>
              )}
            </div>

            {/* Capture progress bar — only shown in word mode while collecting */}
            {mode === "words" && captureProgress > 0 && captureProgress < 100 && (
              <div className="absolute bottom-0 left-0 right-0 h-1 bg-zinc-800/60">
                <div
                  className="h-full bg-emerald-400 transition-all duration-200"
                  style={{ width: `${captureProgress}%` }}
                />
              </div>
            )}

            <div className="absolute top-3 right-3 px-2 py-1 rounded-md bg-black/50 text-xs uppercase tracking-wider text-zinc-400">
              {isCameraOn ? "Live" : "Off"}
            </div>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="shrink-0 flex flex-wrap items-center justify-between gap-4 px-4 py-3 bg-zinc-900/95 border-t border-zinc-800">
          <p className="text-sm text-zinc-400 min-w-0 truncate" aria-live="polite">
            {status || (isCameraOn ? "Show the sign for the prompt." : "Enable the camera to start.")}
          </p>
          <div className="flex items-center gap-6">
            <div>
              <span className="text-xs uppercase tracking-wider text-zinc-500">Score</span>
              <span className="ml-2 text-lg font-semibold tabular-nums">
                {score != null ? `${Math.round(score * 100)}%` : "—"}
              </span>
            </div>
            <div>
              <span className="text-xs uppercase tracking-wider text-zinc-500">Prediction</span>
              <span className="ml-2 text-lg font-medium tabular-nums">
                {predictedLabel ?? "—"}
              </span>
            </div>
          </div>
        </div>
      </main>

      {/* Click-away overlay */}
      {settingsOpen && (
        <div className="fixed inset-0 z-20 bg-black/20" onClick={() => setSettingsOpen(false)} />
      )}

      {/* Settings panel */}
      <div
        className={`fixed top-0 right-0 h-full w-72 bg-zinc-900/98 border-l border-zinc-700 shadow-xl overflow-y-auto transition-transform duration-300 ease-out z-30 ${
          settingsOpen ? "translate-x-0" : "translate-x-full"
        }`}
      >
        <div className="p-4 space-y-6">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-zinc-400">
            Detection settings
          </h2>
          <div>
            <label className="block text-sm text-zinc-300 mb-1">
              Match threshold: {Math.round(confidenceThreshold * 100)}%
            </label>
            <p className="text-xs text-zinc-500 mb-2">
              Only count as correct when confidence ≥ this
            </p>
            <input
              type="range"
              min={MIN_CONFIDENCE}
              max={MAX_CONFIDENCE}
              step={0.05}
              value={confidenceThreshold}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                setConfidenceThreshold(Number(e.target.value))
              }
              className="w-full h-2 rounded-full appearance-none bg-zinc-700 accent-emerald-500"
            />
          </div>
          <div>
            <label className="block text-sm text-zinc-300 mb-1">
              Check every {pollIntervalMs} ms
            </label>
            <p className="text-xs text-zinc-500 mb-2">
              How often we send a frame to the model
            </p>
            <input
              type="range"
              min={MIN_POLL_MS}
              max={MAX_POLL_MS}
              step={100}
              value={pollIntervalMs}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                setPollIntervalMs(Number(e.target.value))
              }
              className="w-full h-2 rounded-full appearance-none bg-zinc-700 accent-emerald-500"
            />
          </div>
        </div>
      </div>

      {/* Settings tab button */}
      <button
        type="button"
        onClick={() => setSettingsOpen((o) => !o)}
        className="fixed right-0 top-0 h-full w-10 flex items-center justify-center bg-transparent hover:bg-black/40 z-40 transition-colors duration-200 ease-out group"
        aria-label={settingsOpen ? "Close settings" : "Open settings"}
      >
        <span
          className={`inline-block text-3xl text-zinc-200/85 group-hover:text-zinc-50 drop-shadow-md tracking-wide transition-all duration-200 ${
            settingsOpen ? "rotate-180" : ""
          }`}
          aria-hidden
        >
          ‹
        </span>
      </button>

      <canvas ref={canvasRef} className="hidden" />
    </div>
  );
}
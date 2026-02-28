 "use client";

import { useCallback, useEffect, useRef, useState } from "react";

const CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789".split("");
const MIN_CONFIDENCE = 0.5;
const MAX_CONFIDENCE = 1;
const MIN_POLL_MS = 300;
const MAX_POLL_MS = 2000;
const DEFAULT_CONFIDENCE = 0.9;
const DEFAULT_POLL_MS = 800;

export default function Home() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [target, setTarget] = useState<string | null>(null);
  const [score, setScore] = useState<number | null>(null);
  const [predictedLabel, setPredictedLabel] = useState<string | null>(null);
  const [status, setStatus] = useState<string>("");
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [hasMatch, setHasMatch] = useState(false);
  const [confidenceThreshold, setConfidenceThreshold] = useState(DEFAULT_CONFIDENCE);
  const [pollIntervalMs, setPollIntervalMs] = useState(DEFAULT_POLL_MS);

  const isPredictingRef = useRef(false);

  useEffect(() => {
    // Pick an initial random character on load
    nextQuestion();
    // We intentionally don't add dependencies here to only run once
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

  const nextQuestion = () => {
    const randomChar =
      CHARACTERS[Math.floor(Math.random() * CHARACTERS.length)];
    setTarget(randomChar);
    setScore(null);
    setPredictedLabel(null);
    setStatus("");
    setHasMatch(false);
  };

  const captureAndPredict = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || !target || hasMatch) {
      return;
    }

    if (isPredictingRef.current) return;
    isPredictingRef.current = true;

    try {
      const video = videoRef.current;
      if (!video) return;

      if (video.videoWidth === 0 || video.videoHeight === 0) {
        setStatus("Waiting for camera to initialize...");
        return;
      }

      const canvas = canvasRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext("2d");
      if (!ctx) {
        setStatus("Could not get canvas context.");
        return;
      }

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      const dataUrl = canvas.toDataURL("image/jpeg");

      const res = await fetch("http://127.0.0.1:5000/api/predict-sign", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          image: dataUrl,
        }),
      });

      if (!res.ok) {
        const error = await res.json().catch(() => ({}));
        console.error("Backend error", error);
        setStatus(error.error || "Backend error while scoring sign.");
        return;
      }

      const json = await res.json();
      const confidence =
        typeof json.confidence === "number" ? json.confidence : null;
      const predicted =
        typeof json.predicted_label === "string"
          ? (json.predicted_label as string)
          : null;

      if (!predicted || confidence == null) {
        return;
      }

      if (!hasMatch) {
        setPredictedLabel(predicted);
        setScore(confidence);
      }

      if (predicted === target && confidence >= confidenceThreshold) {
        setHasMatch(true);
        setStatus("Great job! We detected your sign.");
      } else if (!hasMatch) {
        setStatus(
          `Listening... (model sees ${predicted} at ${Math.round(
            confidence * 100
          )}% confidence)`
        );
      }
    } catch (err) {
      console.error(err);
      setStatus("Network error talking to the backend.");
    } finally {
      isPredictingRef.current = false;
    }
  }, [hasMatch, target, confidenceThreshold]);

  useEffect(() => {
    if (!isCameraOn || !target || hasMatch) return;

    if (!status) {
      setStatus(
        "Show the sign for the prompt towards the camera. We'll detect it automatically."
      );
    }

    const intervalId = window.setInterval(() => {
      void captureAndPredict();
    }, pollIntervalMs);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [captureAndPredict, hasMatch, isCameraOn, pollIntervalMs, status, target]);

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-50 flex items-center justify-center px-4 py-10 font-sans">
      <main className="w-full max-w-5xl bg-zinc-900/60 border border-zinc-800 rounded-2xl shadow-xl p-6 md:p-10 flex flex-col gap-8">
        <header className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h1 className="text-3xl md:text-4xl font-semibold tracking-tight">
              ASL Practice Quiz
            </h1>
            <p className="mt-2 text-sm md:text-base text-zinc-400 max-w-xl">
              Show the sign for the highlighted letter or number in front of
              your camera. We&apos;ll capture a snapshot and score how closely
              it matches.
            </p>
          </div>
          <button
            type="button"
            onClick={nextQuestion}
            className="inline-flex items-center justify-center rounded-full bg-zinc-50 text-zinc-900 px-5 py-2 text-sm font-medium hover:bg-white transition-colors"
          >
            New Prompt
          </button>
        </header>

        <section className="grid grid-cols-1 lg:grid-cols-[minmax(0,2fr)_minmax(0,1.3fr)] gap-8 items-start">
          <div className="flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <div className="text-sm uppercase tracking-[0.2em] text-zinc-500">
                Current prompt
              </div>
            </div>
            <div className="flex items-center justify-center rounded-2xl border border-zinc-800 bg-zinc-900/60 py-10">
              <span className="text-7xl md:text-8xl font-semibold tracking-tight">
                {target ?? "?"}
              </span>
            </div>

            <div className="mt-4 flex flex-wrap gap-3">
              {!isCameraOn ? (
                <button
                  type="button"
                  onClick={startCamera}
                  className="inline-flex items-center justify-center rounded-full bg-emerald-500 text-emerald-950 px-5 py-2 text-sm font-medium hover:bg-emerald-400 transition-colors"
                >
                  Enable Camera
                </button>
              ) : (
                <button
                  type="button"
                  onClick={stopCamera}
                  className="inline-flex items-center justify-center rounded-full bg-zinc-800 text-zinc-100 px-5 py-2 text-sm font-medium hover:bg-zinc-700 transition-colors"
                >
                  Turn Off Camera
                </button>
              )}
            </div>

            <div className="mt-6 rounded-xl border border-zinc-800 bg-zinc-900/50 p-4 space-y-4">
              <div className="text-xs uppercase tracking-[0.2em] text-zinc-500">
                Detection settings
              </div>
              <div>
                <label className="flex items-center justify-between gap-2 text-sm text-zinc-300">
                  <span>Match threshold: {Math.round(confidenceThreshold * 100)}%</span>
                  <span className="text-zinc-500 text-xs">Only count as correct when confidence ≥ this</span>
                </label>
                <input
                  type="range"
                  min={MIN_CONFIDENCE}
                  max={MAX_CONFIDENCE}
                  step={0.05}
                  value={confidenceThreshold}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => setConfidenceThreshold(Number(e.target.value))}
                  className="mt-1 w-full h-2 rounded-full appearance-none bg-zinc-700 accent-emerald-500"
                />
              </div>
              <div>
                <label className="flex items-center justify-between gap-2 text-sm text-zinc-300">
                  <span>Check every {pollIntervalMs} ms</span>
                  <span className="text-zinc-500 text-xs">How often we send a frame to the model</span>
                </label>
                <input
                  type="range"
                  min={MIN_POLL_MS}
                  max={MAX_POLL_MS}
                  step={100}
                  value={pollIntervalMs}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => setPollIntervalMs(Number(e.target.value))}
                  className="mt-1 w-full h-2 rounded-full appearance-none bg-zinc-700 accent-emerald-500"
                />
              </div>
            </div>

            {status && (
              <p className="mt-2 text-sm text-zinc-400" aria-live="polite">
                {status}
              </p>
            )}
          </div>

          <div className="flex flex-col gap-4">
            <div className="rounded-2xl border border-zinc-800 bg-black/40 overflow-hidden">
              <div className="px-4 py-2 flex items-center justify-between border-b border-zinc-800 bg-zinc-900/70">
                <span className="text-xs uppercase tracking-[0.2em] text-zinc-500">
                  Live camera
                </span>
                <span className="text-xs text-zinc-500">
                  {isCameraOn ? "On" : "Off"}
                </span>
              </div>
              <div className="aspect-video bg-zinc-950 flex items-center justify-center">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  className="h-full w-full object-cover rounded-b-2xl"
                />
              </div>
            </div>

            <div className="rounded-2xl border border-zinc-800 bg-zinc-900/70 p-4 flex flex-col gap-3">
              <div className="flex items-baseline justify-between">
                <div>
                  <div className="text-xs uppercase tracking-[0.2em] text-zinc-500">
                    Your score
                  </div>
                  <div className="mt-1 text-3xl font-semibold tracking-tight">
                    {score != null ? `${Math.round(score * 100)}%` : "—"}
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-xs uppercase tracking-[0.2em] text-zinc-500">
                    Model prediction
                  </div>
                  <div className="mt-1 text-lg font-medium">
                    {predictedLabel ?? "—"}
                  </div>
                </div>
              </div>
              <p className="text-xs text-zinc-500">
                Score is the model&apos;s confidence (MediaPipe Hands + classifier).
                Adjust &quot;Match threshold&quot; and &quot;Check every&quot; above to tune behavior.
              </p>
            </div>
          </div>
        </section>

        {/* Hidden canvas used for capturing a frame from the video */}
        <canvas ref={canvasRef} className="hidden" />
      </main>
    </div>
  );
}

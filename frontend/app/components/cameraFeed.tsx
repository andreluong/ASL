"use client";

import { useRef, useState, useEffect, ReactNode } from "react";

interface CameraFeedProps {
  onStreamReady?: (video: HTMLVideoElement) => void;
  onStreamStopped?: () => void;
  children?: ReactNode;
}

export default function CameraFeed({
  onStreamReady,
  onStreamStopped,
  children
}: CameraFeedProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        onStreamReady?.(videoRef.current);
      }
      setIsCameraOn(true);
      setError(null);
    } catch (err) {
      // console.error(err);
      // setError("Could not access camera. Please check permissions.");
    }
  } ;

  const stopCamera = () => {
    const video = videoRef.current;
    if (video && video.srcObject instanceof MediaStream) {
      video.srcObject.getTracks().forEach((t: MediaStreamTrack) => t.stop());
      video.srcObject = null;
    }
    setIsCameraOn(false);
    onStreamStopped?.();
  };

  // Clean up on unmount
  useEffect(() => {
    void startCamera();
  }, []);

  return (
    <div className={`flex flex-col items-center gap-3`}>
      {/* Video container */}
      <div className="relative w-full rounded-2xl overflow-hidden border border-zinc-700/50 bg-black shadow-2xl aspect-video h-[calc(100vh-6rem)]">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          className="absolute inset-0 w-full h-full object-cover"
        />

        {/* Offline placeholder */}
        {!isCameraOn && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-zinc-900">
            <svg
              className="w-10 h-10 text-zinc-600"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={1.5}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9A2.25 2.25 0 0013.5 5.25h-9A2.25 2.25 0 002.25 7.5v9A2.25 2.25 0 004.5 18.75z"
              />
            </svg>
            <p className="text-sm text-zinc-500">Camera is off</p>
          </div>
        )}

        {/* Controls */}
        {!isCameraOn ? (
          <button
            type="button"
            onClick={startCamera}
            className="absolute top-3 right-3 flex items-center rounded-full bg-emerald-500 text-emerald-950 px-5 py-2 text-sm font-medium hover:bg-emerald-400 transition-colors"
          >
            Enable Camera
          </button>
        ) : (
          <button
            type="button"
            onClick={stopCamera}
            className="absolute top-3 right-3 flex items-center rounded-full bg-zinc-700 text-zinc-100 px-5 py-2 text-sm font-medium hover:bg-zinc-600 transition-colors"
          >
            Turn Off Camera
          </button>
        )}

        {children}
      </div>

      {/* Error message */}
      {/* {error && (
        <p className="text-sm text-red-400">{error}</p>
      )} */}

    </div>
  );
}
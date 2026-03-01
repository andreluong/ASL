"use client";

import { RefObject } from "react";

interface CameraProps {
  videoRef: RefObject<HTMLVideoElement | null>;
  isCameraOn: boolean;
  target: string | null;
}

export default function Camera({ videoRef, isCameraOn, target }: CameraProps) {
  return (
    <div className="relative w-full max-w-5xl aspect-video max-h-[calc(100vh-8rem)] rounded-2xl overflow-hidden border border-zinc-700/50 bg-black shadow-2xl">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        className="absolute inset-0 w-full h-full object-cover"
      />
      <div className="absolute top-3 left-3 flex items-center gap-2 pointer-events-none">
        <span className="inline-flex items-center justify-center rounded-md bg-black/60 px-3 py-1 text-xs font-medium uppercase tracking-wider text-zinc-300">
          Prompt
        </span>
        <span className="inline-flex items-center justify-center rounded-md bg-white/90 text-zinc-900 text-2xl font-semibold w-10 h-10">
          {target ?? "?"}
        </span>
      </div>
      <div className="absolute top-3 right-3 px-2 py-1 rounded-md bg-black/50 text-xs uppercase tracking-wider text-zinc-400">
        {isCameraOn ? "Live" : "Off"}
      </div>
    </div>
  );
}
"use client";

import React, { useEffect, useRef, useState } from 'react'
import CameraFeed from './cameraFeed';
import { GoogleGenerativeAI } from '@google/generative-ai'

const genAI = new GoogleGenerativeAI(process.env.NEXT_PUBLIC_GEMINI_API_KEY!)

interface Sign {
  label: string
	media: string // Video or image URL
	hasMotion?: boolean
	learned?: boolean
}

export default function LearnMode() {
	const INFO = "info"
	const PRACTICE = "practice"
	
	const [selected, setSelected] = React.useState<Sign | null>(null)
	const [modalStep, setModalStep] = React.useState<'info' | 'practice'>(INFO)
	const [instructions, setInstructions] = useState<string | null>(null)
	const [loadingInstructions, setLoadingInstructions] = useState(false)
	const canvasRef = useRef<HTMLCanvasElement | null>(null)
	const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
	const [predictedSign, setPredictedSign] = useState<string | null>(null)
	const [confidence, setConfidence] = useState<number | null>(null)

	const [signs, setSigns] = useState<Sign[]>(() => {
		const initialSigns: Sign[] = [
			{ label: 'A', media: 'assets/A.png' },
			{ label: 'B', media: 'assets/B.png' },
			{ label: 'C', media: 'assets/C.png' },
			{ label: 'D', media: 'assets/D.png' },
			{ label: 'E', media: 'assets/E.png' },
			{ label: 'F', media: 'assets/F.png' },
			{ label: 'G', media: 'assets/G.png' },
			{ label: 'H', media: 'assets/H.png' },
			{ label: 'I', media: 'assets/I.png' },
			{ label: 'hello', media: 'assets/hello.mp4', hasMotion: true },
		]
		return initialSigns
	})


  const closeModal = () => {
		if (intervalRef.current) {
			clearInterval(intervalRef.current)
			intervalRef.current = null
		}
		setSelected(null)
		setModalStep(INFO)
	}

	const getPromptForSign = (sign: Sign) => {
		return `This is an ASL sign for the label "${sign.label}". 
						In 2 short sentences, describe how to form this hand sign. 
						Be specific about finger and thumb positions.`
	}

	const handlePrediction = (json: { prediction: string; confidence: number }) => {
		setPredictedSign(json.prediction)
		setConfidence(json.confidence)

		if (json.confidence > 0.8 && json.prediction === selected?.label) {
			setSigns((prev) =>
				prev.map((s) => s.label === selected.label ? { ...s, learned: true } : s)
			)
			setSelected((prev) => prev ? { ...prev, learned: true } : prev)
			localStorage.setItem(`asl_learned_${selected?.label}`, 'true')
		}
	}

	const handleStreamReady = (video: HTMLVideoElement) => {
		if (!canvasRef.current) {
			canvasRef.current = document.createElement('canvas')
		}

		const canvas = canvasRef.current
		const frameBufferRef: string[] = []
		const BUFFER_SIZE = 20

		intervalRef.current = setInterval(async () => {
			if (video.videoWidth === 0) return

			canvas.width = video.videoWidth
			canvas.height = video.videoHeight
			canvas.getContext('2d')!.drawImage(video, 0, 0)

			if (selected?.hasMotion) {
      // Buffer frames and send when full
      frameBufferRef.push(canvas.toDataURL('image/jpeg'))
      if (frameBufferRef.length < BUFFER_SIZE) return

      const frames = [...frameBufferRef]
      frameBufferRef.length = 0 // clear buffer

      try {
        const res = await fetch('http://127.0.0.1:5000/api/predict', {
          method: 'POST',
          mode: 'cors',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ frames }),
        })
        const json = await res.json()
        handlePrediction(json)
      } catch (err) {
        console.error('Prediction error:', err)
      }

    } else {
      // Single frame
      try {
        const res = await fetch('http://127.0.0.1:5000/api/predict', {
          method: 'POST',
          mode: 'cors',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: canvas.toDataURL('image/jpeg') }),
        })
        const json = await res.json()
        handlePrediction(json)
      } catch (err) {
        console.error('Prediction error:', err)
      }
    }
		}, 100)
	}

	useEffect(() => {
		if (!selected) return

		// Return cached result if available
		const cached = localStorage.getItem(`asl_storage_${selected.label}`)
		if (cached) { 
			setInstructions(cached)
			return 
		}

		const fetchInstructions = async () => {
			setLoadingInstructions(true)
			setInstructions(null)

			try {
				// Check if image exists first
				const check = await fetch(selected.media, { method: 'HEAD' })
				if (!check.ok) {
					setInstructions('No image available for this sign.')
					return
				}

				// Fetch the image and convert to base64
				const res = await fetch(selected.media)
				const blob = await res.blob()
				const base64 = await new Promise<string>((resolve) => {
					const reader = new FileReader()
					reader.onloadend = () => resolve((reader.result as string).split(',')[1])
					reader.readAsDataURL(blob)
				})
				
				// Generate instructions from Gemini
				const model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash-lite' })
				const result = await model.generateContent([
					{
						inlineData: {
							mimeType: blob.type as 'image/png' | 'image/jpeg',
							data: base64,
						},
					},
					getPromptForSign(selected)
				])

				const text = result.response.text()
				localStorage.setItem(`asl_storage_${selected.label}`, text)
				setInstructions(text)
			} catch (err) {
				console.error(err)
				setInstructions('Could not load instructions.')
			} finally {
				setLoadingInstructions(false)
			}
		}

  	void fetchInstructions()

	}, [selected])

  return (
		<>
			<div className="grid grid-cols-3 gap-4 p-4">
				{signs.map((sign) => (
					<div
						key={sign.label}
						onClick={() => setSelected(sign)}
						className="relative border border-zinc-700 bg-zinc-800 rounded-lg overflow-hidden px-2 py-10 text-center hover:bg-zinc-600 transition-colors cursor-pointer"
					>
						{sign.learned && (
							<span className="absolute top-2 right-2 w-5 h-5 flex items-center justify-center rounded-full bg-emerald-500 text-emerald-950">
								<svg xmlns="http://www.w3.org/2000/svg" className="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={3} strokeLinecap="round" strokeLinejoin="round">
									<path d="M20 6L9 17l-5-5" />
								</svg>
							</span>
						)}
						{sign.label}
					</div>
				))}
			</div>

			{selected && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
          onClick={closeModal}
        >
          <div
            className="relative bg-zinc-900 border border-zinc-700 rounded-2xl p-6 flex flex-col items-center gap-4 shadow-2xl max-w-sm w-full mx-4"
            onClick={(e) => e.stopPropagation()}
          >
  
            {modalStep === INFO ? (
              <>
								<button
									onClick={closeModal}
									className="absolute top-3 right-3 text-zinc-400 hover:text-white transition-colors"
									aria-label="Close"
								>
									<svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
										<path d="M18 6L6 18M6 6l12 12" />
									</svg>
								</button>

                <h2 className="text-2xl font-bold text-white">{selected.label}</h2>
								
								{selected.media.endsWith('.mp4') ? (
									<video src={selected.media} autoPlay loop className="w-48 h-48 object-contain rounded-lg" />
								) : (
									<img src={selected.media} alt={`ASL sign for ${selected.label}`} className="w-48 h-48 object-contain rounded-lg" />
								)}

								{/* Gemini instructions for sign */}
								<div className="text-sm text-zinc-400 mt-2">
									{loadingInstructions ? (
										<span className="animate-pulse text-zinc-500">Analyzing sign...</span>
									) : (
										instructions
									)}
								</div>

                <button
                  onClick={() => setModalStep(PRACTICE)}
                  className="mt-2 rounded-full bg-emerald-500 text-emerald-950 px-5 py-2 text-sm font-medium hover:bg-emerald-400 transition-colors"
                >
                  Continue
                </button>
              </>
            ) : (
              <>
                <CameraFeed onStreamReady={handleStreamReady}>
									{/* Back */}
									<button
										onClick={() => setModalStep(INFO)}
										className="absolute top-3 left-3 z-10 rounded-full bg-black/50 backdrop-blur-sm text-zinc-100 px-3 py-1.5 text-sm font-medium hover:bg-black/70 transition-colors"
									>
										← Back
									</button>

									{/* Message */}
									<div className={`absolute bottom-3 left-1/2 -translate-x-1/2 z-10 px-4 py-2 rounded-full backdrop-blur-sm text-sm whitespace-nowrap ${selected.learned ? 'bg-emerald-500 text-emerald-900' : 'bg-black/60 text-zinc-300'}`}>
										{selected.learned ? (
											<span className="flex items-center gap-1">
												<span>Perform the sign for <span className="text-white font-semibold">{selected.label}</span></span>
												<svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={3} strokeLinecap="round" strokeLinejoin="round">
													<path d="M20 6L9 17l-5-5" />
												</svg>
											</span>
										) : (
											<span>Perform the sign for <span className="text-white font-semibold">{selected.label}</span></span>
										)}
									</div>
									
									{/* AI predicting labels and score */}
									<div className="absolute bottom-3 right-3 z-10 px-4 py-2 rounded-full bg-black/60 backdrop-blur-sm text-sm whitespace-nowrap">
										<span className="text-zinc-400">AI Prediction:</span>{" "}
										<span className="text-white font-semibold">{predictedSign}</span>{" "}
										<span className="text-green-400">({confidence ? (confidence * 100).toFixed(1) + '%' : '0%'})</span>
									</div>
																		
								</CameraFeed>
              </>
            )}
          </div>
        </div>
      )}
		</>
  )
}

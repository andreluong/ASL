import React from 'react'

interface signCardProp {
  letter: string;
  isCompleted?: boolean;
  onClick?: (letter: string) => void;
}

export default function page() {
	const signs = [
		{ name: "A", video: "/videos/A.mp4" },
		{ name: "B", video: "/videos/B.mp4" },
		{ name: "C", video: "/videos/C.mp4" },
		// Add more signs as needed
	]

	const signGrid = signs.map((sign) => (
		<div key={sign.name} className="sign-card">
			<video src={sign.video} controls className="sign-video" />
			<p className="sign-name">{sign.name}</p>
		</div>
	))

  return (
    <div className="w-full px-4 py-4">
        {/* Grid of all supported signs */}
        <div className="sign-grid">
          {signGrid}
        </div>

        
    </div>
  )
}

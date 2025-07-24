"use client"

import { useState, useEffect, useRef } from "react"

interface RotatingPromptsProps {
    prompts: string[]
    className?: string
}

export function RotatingPrompts({ prompts, className = "" }: RotatingPromptsProps) {
    const [currentPrompt, setCurrentPrompt] = useState("")
    const hasInitialized = useRef(false)

    useEffect(() => {
        // Only initialize once when component first mounts
        if (prompts.length > 0 && !hasInitialized.current) {
            // Randomly select a prompt on mount
            const randomIndex = Math.floor(Math.random() * prompts.length)
            setCurrentPrompt(prompts[randomIndex])
            hasInitialized.current = true
        }
    }, [prompts])

    if (!currentPrompt) {
        return null
    }

    return (
        <div className={`text-center ${className}`}>
            <p className="text-xl text-gray-800 mb-4 font-light leading-relaxed max-w-md mx-auto">
                {currentPrompt}
            </p>
        </div>
    )
}

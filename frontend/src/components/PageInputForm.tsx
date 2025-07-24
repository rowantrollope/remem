"use client"

import { GroundingToggle } from "@/components/GroundingToggle"
import { useRef, useEffect } from "react"

interface PageInputFormProps {
    input: string
    setInput: (value: string) => void
    isLoading: boolean
    onSubmit: (e: React.FormEvent) => void
    pageType: 'chat' | 'search' | 'context' | 'save'
    // Optional props
    placeholder?: string
    chatMode?: 'ask' | 'save'
    groundingEnabled?: boolean
    onGroundingToggle?: (enabled: boolean) => void
}

export function PageInputForm({
    input,
    setInput,
    isLoading,
    onSubmit,
    pageType,
    placeholder,
    chatMode,
    groundingEnabled,
    onGroundingToggle
}: PageInputFormProps) {
    const inputRef = useRef<HTMLInputElement>(null)

    useEffect(() => {
        // Focus the input field when component mounts
        if (inputRef.current) {
            inputRef.current.focus()
        }
    }, [])
    const getPlaceholder = () => {
        // Use custom placeholder if provided
        if (placeholder) {
            return placeholder
        }

        switch (pageType) {
            case "chat":
                return chatMode === 'save'
                    ? "What do you want to remember today?"
                    : "Ask about your memories..."
            case "search":
                return "Search memories..."
            case "context":
                return "Update context..."
            case "save":
                return "What would you like to remember?"
            default:
                return "Type something..."
        }
    }



    return (
        <div className="space-y-3 border p-4 rounded-xl shadow-md">
            {/* Input Form */}
            <form onSubmit={onSubmit} className="flex gap-2">
                <input
                    ref={inputRef}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder={getPlaceholder()}
                    className="flex-1 border-none shadow-none text-lg focus:border-none focus:ring-0 focus:outline-none"
                    disabled={isLoading}
                />
                {/* <Button
                    type="submit"
                    disabled={!input.trim() || isLoading}
                    className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700"
                >
                    {isLoading ? (
                        <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    ) : (
                        getIcon()
                    )}
                </Button> */}
            </form>
            {/* Grounding Toggle for Chat and Save Pages */}
            {((pageType === 'chat' && chatMode === 'save') || pageType === 'save') && groundingEnabled !== undefined && onGroundingToggle && (
                <div className="p-3">
                    <div className="flex justify-end">
                        <GroundingToggle
                            enabled={groundingEnabled}
                            onChange={onGroundingToggle}
                        />
                    </div>
                </div>
            )}

            
            
        </div>
    )
}

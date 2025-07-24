"use client"

import { useState, useRef, useEffect } from "react"

// Components
import { PageLayout } from "@/components/PageLayout"
import { PageInputForm } from "@/components/PageInputForm"
import { RotatingPrompts } from "@/components/RotatingPrompts"

// Hooks
import { useConfiguredAPI } from "@/hooks/useConfiguredAPI"
import { useMemoryAPI } from "@/hooks"

// Types
import type { ChatResponse } from "@/lib/api"

interface ChatMessage {
    id: string
    type: 'user' | 'assistant'
    content: string
    created_at: string
}

const chatPrompts = [
    "Tell me about all my restaurant experiences",
    "What patterns do you see in my memories?",
    "Help me understand my food preferences",
    "What have I learned about travel?",
    "Summarize my recent activities"
]

export default function ChatDemoPage() {
    const [input, setInput] = useState("")
    const [chatHistory, setChatHistory] = useState<ChatMessage[]>([])
    const [isLoading, setIsLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const messagesEndRef = useRef<HTMLDivElement>(null)
    const scrollContainerRef = useRef<HTMLDivElement>(null)

    const { api: memoryAPI } = useConfiguredAPI()
    const { apiStatus } = useMemoryAPI()

    // Auto-scroll to bottom when new messages are added
    useEffect(() => {
        if (messagesEndRef.current && scrollContainerRef.current) {
            // Scroll the container to the bottom smoothly
            scrollContainerRef.current.scrollTo({
                top: scrollContainerRef.current.scrollHeight,
                behavior: 'smooth'
            })
        }
    }, [chatHistory])

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        if (!input.trim() || isLoading) return

        const userMessage: ChatMessage = {
            id: `user-${Date.now()}`,
            type: 'user',
            content: input.trim(),
            created_at: new Date().toISOString()
        }

        // Add user message immediately
        setChatHistory(prev => [...prev, userMessage])
        setInput("")
        setIsLoading(true)
        setError(null)

        try {
            const response: ChatResponse = await memoryAPI.chat(userMessage.content)
            
            if (response.success) {
                const assistantMessage: ChatMessage = {
                    id: `assistant-${Date.now()}`,
                    type: 'assistant',
                    content: response.response,
                    created_at: new Date().toISOString()
                }
                setChatHistory(prev => [...prev, assistantMessage])
            } else {
                setError('Failed to get response from chat API')
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to send message')
        } finally {
            setIsLoading(false)
        }
    }

    const clearError = () => setError(null)

    // Check if there are any chat messages
    const hasMessages = chatHistory.length > 0

    return (
        <PageLayout
            error={error}
            apiStatus={apiStatus}
            onClearError={clearError}
        >
            {/* Chat Demo Content */}
            <div className="h-full flex flex-col">
                {hasMessages ? (
                    // Layout when there are messages - input at bottom
                    <>
                        <div ref={scrollContainerRef} className="flex-1 min-h-0 overflow-y-auto p-4 bg-white">
                            <div className="space-y-4">
                                {chatHistory.map((message) => (
                                    <div
                                        key={message.id}
                                        className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                                    >
                                        <div
                                            className={`max-w-[80%] rounded-lg px-4 py-2 ${
                                                message.type === 'user'
                                                    ? 'bg-blue-500 text-white'
                                                    : 'bg-gray-100 text-gray-900'
                                            }`}
                                        >
                                            <div className="whitespace-pre-wrap">{message.content}</div>
                                            <div className={`text-xs mt-1 ${
                                                message.type === 'user' ? 'text-blue-100' : 'text-gray-500'
                                            }`}>
                                                {new Date(message.created_at).toLocaleTimeString()}
                                            </div>
                                        </div>
                                    </div>
                                ))}
                                {isLoading && (
                                    <div className="flex justify-start">
                                        <div className="bg-gray-100 text-gray-900 rounded-lg px-4 py-2">
                                            <div className="flex items-center space-x-2">
                                                <div className="animate-pulse">Thinking...</div>
                                            </div>
                                        </div>
                                    </div>
                                )}
                                {/* Scroll target */}
                                <div ref={messagesEndRef} />
                            </div>
                        </div>

                        {/* Input Form at bottom */}
                        <div className="flex-shrink-0 rounded">
                            <PageInputForm
                                input={input}
                                setInput={setInput}
                                pageType="chat"
                                isLoading={isLoading}
                                onSubmit={handleSubmit}
                                placeholder="Ask me anything about your memories..."
                            />
                        </div>
                    </>
                ) : (
                    // Layout when no messages - input centered vertically with prompt
                    <div className="flex-1 flex items-center justify-center -mt-40 bg-white">
                        <div className="w-full">
                            <div className="text-center mb-8">
                                <h1 className="text-3xl font-bold text-gray-900 mb-2">/api/agent/chat</h1>
                                <p className="text-gray-600">
                                    Have a conversation with your memories using the LangGraph workflow
                                </p>
                            </div>
                            <RotatingPrompts prompts={chatPrompts} />
                            <PageInputForm
                                input={input}
                                setInput={setInput}
                                pageType="chat"
                                isLoading={isLoading}
                                onSubmit={handleSubmit}
                                placeholder="Ask me anything about your memories..."
                            />
                        </div>
                    </div>
                )}
            </div>
        </PageLayout>
    )
}

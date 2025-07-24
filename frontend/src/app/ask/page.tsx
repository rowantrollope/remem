"use client"

import React, { useState, useEffect } from "react"

// Components
import { PageLayout } from "@/components/PageLayout"
import { PageInputForm } from "@/components/PageInputForm"
import { RotatingPrompts } from "@/components/RotatingPrompts"
import { ChatBox, type UnifiedChatMessage } from "@/components/ChatBox"
import { ApiPageHeader } from "@/components/ApiPageHeader"

// Hooks and types
import { useMemoryAPI } from "@/hooks"
import { usePersistentChat } from "@/hooks/usePersistentChat"
import { useSettings } from "@/hooks/useSettings"

// Utils
import { conversationToMessages, createThinkingMessage, updateThinkingMessage } from "@/lib/chatMessageUtils"



const chatPrompts = [
    "Example: What restaurants have I been to?",
    "Example: What did I eat yesterday?",
    "Example: Tell me about my recent travels",
    "Example: What books have I read?",
    "Example: What movies did I enjoy?"
]

export default function AskPage() {
    const [input, setInput] = useState("")
    const [messages, setMessages] = useState<UnifiedChatMessage[]>([])
    const [copiedId, setCopiedId] = useState<string | null>(null)

    // Function to copy ID to clipboard with visual feedback
    const copyIdToClipboard = async (fullId: string) => {
        try {
            await navigator.clipboard.writeText(fullId)
            setCopiedId(fullId)
            setTimeout(() => setCopiedId(null), 2000) // Clear feedback after 2 seconds
        } catch (err) {
            console.error('Failed to copy ID to clipboard:', err)
        }
    }

    const {
        isLoading,
        error,
        apiStatus,
        groundingEnabled,
        askQuestion,
        setGroundingEnabled,
        clearError,
    } = useMemoryAPI()

    const { settings, updateSetting } = useSettings()

    // Use persistent chat hook for conversations (for persistence)
    const {
        conversations: persistentConversations,
        addConversation,
        updateConversations,
    } = usePersistentChat(settings.vectorSetName)

    // Handle vectorset change
    const handleVectorStoreChange = (newVectorStoreName: string) => {
        updateSetting('vectorSetName', newVectorStoreName)
    }

    // Convert persistent conversations to messages when vectorset changes
    useEffect(() => {
        if (persistentConversations.length > 0) {
            const convertedMessages = conversationToMessages(persistentConversations)
            setMessages(convertedMessages)
        } else {
            // Clear messages when switching to vectorset with no history
            setMessages([])
        }
    }, [persistentConversations, settings.vectorSetName])

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        if (!input.trim()) return

        const currentInput = input
        setInput("")

        // Add user message
        const userMessage: UnifiedChatMessage = {
            id: `user-${Date.now()}`,
            type: 'user',
            content: currentInput,
            created_at: new Date().toISOString()
        }

        // Add thinking message
        const thinkingMessage = createThinkingMessage(Date.now().toString())

        setMessages(prev => [...prev, userMessage, thinkingMessage])

        try {
            console.log('About to ask question with vectorset:', settings.vectorSetName) // Debug logging
            const result = await askQuestion(currentInput, settings.questionTopK, settings.minSimilarity)
            console.log('Ask question result:', result) // Debug logging

            if (typeof result === 'object' && result.success) {
                // Create real response message
                const responseMessage: UnifiedChatMessage = {
                    id: `response-${Date.now()}`,
                    type: 'assistant',
                    content: result.conversation.answer,
                    created_at: new Date().toISOString(),
                    confidence: result.conversation.confidence,
                    reasoning: result.conversation.reasoning,
                    supporting_memories: result.conversation.supporting_memories,
                    excluded_memories: result.conversation.excluded_memories,
                    filtering_info: result.conversation.filtering_info
                }

                // Replace thinking message with real response
                setMessages(prev => updateThinkingMessage(prev, thinkingMessage.id, responseMessage))

                // Also update persistent conversations for persistence
                addConversation(result.conversation)
            } else {
                console.error('Ask question failed:', result) // Debug logging
                // Remove thinking message on error
                setMessages(prev => prev.filter(msg => msg.id !== thinkingMessage.id))
            }
        } catch (error) {
            console.error('Ask question error:', error) // Debug logging
            // Remove thinking message on error
            setMessages(prev => prev.filter(msg => msg.id !== thinkingMessage.id))
        }
    }

    const clearChat = () => {
        setMessages([])
        // Also clear persistent conversations
        updateConversations([])
    }

    // Check if there are any chat messages
    const hasMessages = messages.length > 0

    return (
        <PageLayout
            error={error}
            apiStatus={apiStatus}
            onClearError={clearError}
        >
            {/* Ask API Content */}
            <div className="h-full flex flex-col">
                <ApiPageHeader
                    endpoint={`(POST) /api/memory/${settings.vectorSetName}/ask`}
                    hasMessages={hasMessages}
                    onClearChat={clearChat}
                    isLoading={isLoading}
                    title="Ask Question"
                    showSettingsButton={true}
                    showVectorStoreSelector={true}
                    vectorSetName={settings.vectorSetName}
                    onVectorStoreChange={handleVectorStoreChange}
                />

                {hasMessages ? (
                    // Layout when there are messages - ChatBox + input at bottom
                    <>
                        <div className="flex-1 min-h-0 px-4 bg-white">
                            <ChatBox
                                messages={messages}
                                isLoading={isLoading}
                                loadingText="Searching..."
                                showMemoryIndicators={true}
                                enableMemoryExpansion={true}
                                copiedId={copiedId}
                                onCopyId={copyIdToClipboard}
                            />
                        </div>

                        {/* Input Form at bottom */}
                        <div className="flex-shrink-0 rounded">
                            <PageInputForm
                                input={input}
                                setInput={setInput}
                                pageType="chat"
                                isLoading={isLoading}
                                onSubmit={handleSubmit}
                                groundingEnabled={groundingEnabled}
                                onGroundingToggle={setGroundingEnabled}
                                placeholder="Ask a question about your memories..."
                            />
                        </div>
                    </>
                ) : (
                    // Layout when no messages - input centered vertically with prompt
                    <div className="flex-1 flex items-center justify-center bg-white">
                        <div className="w-full">
                            <div className="text-center mb-8">
                                <h1 className="text-3xl font-bold text-gray-900 mb-2">
                                    Ask Memory a Question
                                </h1>
                                <p className="text-gray-600">
                                    Get structured answers with confidence analysis and supporting evidence
                                </p>
                            </div>
                            <RotatingPrompts prompts={chatPrompts} />
                            <PageInputForm
                                input={input}
                                setInput={setInput}
                                pageType="chat"
                                isLoading={isLoading}
                                onSubmit={handleSubmit}
                                groundingEnabled={groundingEnabled}
                                onGroundingToggle={setGroundingEnabled}
                                placeholder="Ask a question about your memories..."
                            />
                        </div>
                    </div>
                )}
            </div>
        </PageLayout>
    )
}

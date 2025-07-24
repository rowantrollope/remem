"use client"

import React, { useState, useEffect } from "react"

// Components
import { PageLayout } from "@/components/PageLayout"
import { PageInputForm } from "@/components/PageInputForm"
import { RotatingPrompts } from "@/components/RotatingPrompts"
import { ChatBox, type UnifiedChatMessage } from "@/components/ChatBox"
import { ApiPageHeader } from "@/components/ApiPageHeader"

// Hooks
import { useMemoryAPI } from "@/hooks"
import { usePersistentChat } from "@/hooks/usePersistentChat"
import { useSettings } from "@/hooks/useSettings"

// Types
import type { MemorySaveResponse } from "@/hooks/usePersistentChat"
import type { RememberResponse } from "@/lib/api"

// Utils
import { memorySaveResponsesToMessages, createThinkingMessage, updateThinkingMessage } from "@/lib/chatMessageUtils"

const savePrompts = [
    "Example: I had lunch at a great Italian restaurant",
    "Example: Finished reading an amazing book about AI",
    "Example: Met an interesting person at the conference",
    "Example: Learned a new programming technique today",
    "Example: Discovered a beautiful hiking trail"
]

export default function SavePage() {
    const [input, setInput] = useState("")
    const [messages, setMessages] = useState<UnifiedChatMessage[]>([])

    const {
        isLoading,
        error,
        apiStatus,
        groundingEnabled,
        saveMemory,
        setGroundingEnabled,
        clearError,
    } = useMemoryAPI()

    const { settings, updateSetting } = useSettings()

    // Use persistent chat hook for memory save responses (for persistence)
    const {
        memorySaveResponses,
        addMemorySaveResponse,
        clearChatHistory,
    } = usePersistentChat(settings.vectorSetName)

    // Handle vectorset change
    const handleVectorStoreChange = (newVectorStoreName: string) => {
        updateSetting('vectorSetName', newVectorStoreName)
    }

    // Convert persistent memory save responses to messages when vectorset changes
    useEffect(() => {
        if (memorySaveResponses.length > 0) {
            const convertedMessages = memorySaveResponsesToMessages(memorySaveResponses)
            setMessages(convertedMessages)
        } else {
            // Clear messages when switching to vectorset with no history
            setMessages([])
        }
    }, [memorySaveResponses, settings.vectorSetName])

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        if (!input.trim()) return

        const currentInput = input
        setInput("")

        // Add user message
        const userMessage: UnifiedChatMessage = {
            id: `user-${Date.now()}`,
            type: 'user',
            content: `Add memory: ${currentInput}`,
            created_at: new Date().toISOString()
        }

        // Add thinking message
        const thinkingMessage = createThinkingMessage(Date.now().toString())

        setMessages(prev => [...prev, userMessage, thinkingMessage])

        try {
            const result = await saveMemory(currentInput)
            if (result.success && result.response) {
                // Create a nicely formatted success message
                const formatSuccessMessage = (response: RememberResponse) => {
                    const memoryId = response.memory_id
                    const shortId = memoryId ? memoryId.split('-').pop() || memoryId : 'unknown'

                    let message = `Memory "${shortId}" saved successfully!\n\n`

                    // Add grounding information if available
                    if (response.grounding_applied) {
                        message += `🔗 Contextual grounding applied\n`

                        if (response.grounding_info?.changes_made && response.grounding_info.changes_made.length > 0) {
                            message += `📝 Enhanced with context:\n`
                            response.grounding_info.changes_made.forEach((change) => {
                                message += `   • "${change.original}" → "${change.replacement}"\n`
                            })
                        }

                        if (response.grounding_info?.dependencies_found) {
                            const deps = response.grounding_info.dependencies_found
                            if (deps.temporal && deps.temporal.length > 0) {
                                message += `⏰ Temporal context: ${deps.temporal.join(', ')}\n`
                            }
                            if (deps.spatial && deps.spatial.length > 0) {
                                message += `📍 Location context: ${deps.spatial.join(', ')}\n`
                            }
                            if (deps.social && deps.social.length > 0) {
                                message += `👥 Social context: ${deps.social.join(', ')}\n`
                            }
                            if (deps.environmental && deps.environmental.length > 0) {
                                message += `🌍 Environmental context: ${deps.environmental.join(', ')}\n`
                            }
                        }
                    } else {
                        message += `📝 Stored as provided (no contextual grounding)\n`
                    }

                    // Add context snapshot information if available
                    if (response.context_snapshot) {
                        const context = response.context_snapshot
                        message += `\n📊 Context captured:\n`

                        if (context.temporal?.date) {
                            message += `   📅 Date: ${context.temporal.date}\n`
                        }
                        if (context.spatial?.location) {
                            message += `   📍 Location: ${context.spatial.location}\n`
                        }
                        if (context.spatial?.activity) {
                            message += `   🎯 Activity: ${context.spatial.activity}\n`
                        }
                        if (context.environmental?.mood) {
                            message += `   😊 Mood: ${context.environmental.mood}\n`
                        }
                    }

                    message += `\n🆔 Full Memory ID: ${memoryId}`

                    return message
                }

                // Create success message with grounding information
                const successMessage: UnifiedChatMessage = {
                    id: `save-${Date.now()}`,
                    type: 'memory_save',
                    content: formatSuccessMessage(result.response),
                    created_at: new Date().toISOString(),
                    memory_id: result.response.memory_id,
                    save_success: true,
                    grounding_applied: result.response.grounding_applied,
                    grounding_info: result.response.grounding_info,
                    context_snapshot: result.response.context_snapshot,
                    original_text: result.response.original_text,
                    grounded_text: result.response.grounded_text
                }

                // Replace thinking message with success message
                setMessages(prev => updateThinkingMessage(prev, thinkingMessage.id, successMessage))

                // Also add to persistent storage
                const saveResponse: MemorySaveResponse = {
                    success: true,
                    response: result.response,
                    originalText: currentInput,
                    timestamp: new Date().toISOString()
                }
                addMemorySaveResponse(saveResponse)
            } else {
                // Remove thinking message on error
                setMessages(prev => prev.filter(msg => msg.id !== thinkingMessage.id))
            }
        } catch {
            // Remove thinking message on error
            setMessages(prev => prev.filter(msg => msg.id !== thinkingMessage.id))
        }
    }

    const clearChat = () => {
        setMessages([])
        clearChatHistory()
    }

    // Check if there are any memory save responses
    const hasMessages = messages.length > 0

    return (
        <PageLayout
            error={error}
            apiStatus={apiStatus}
            onClearError={clearError}
        >
            {/* Memory Save Content */}
            <div className="h-full flex flex-col">
                <ApiPageHeader
                    endpoint={`(POST) /api/memory/${settings.vectorSetName}`}
                    hasMessages={hasMessages}
                    onClearChat={clearChat}
                    isLoading={isLoading}
                    title="Add memory"
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
                                loadingText="Saving memory..."
                                showMemoryIndicators={true}
                                enableMemoryExpansion={true}
                            />
                        </div>

                        {/* Input Form at bottom */}
                        <div className="flex-shrink-0 rounded">
                            <PageInputForm
                                input={input}
                                setInput={setInput}
                                pageType="save"
                                isLoading={isLoading}
                                onSubmit={handleSubmit}
                                groundingEnabled={groundingEnabled}
                                onGroundingToggle={setGroundingEnabled}
                                placeholder="Enter a memory to add..."
                            />
                        </div>
                    </>
                ) : (
                    // Layout when no messages - input centered vertically with prompt
                    <div className="flex-1 flex items-center justify-center bg-white">
                        
                        <div className="w-full">
                            <div className="text-center mb-8">
                                <h1 className="text-3xl font-bold text-gray-900 mb-2">
                                    Save a new Memory:
                                </h1>
                                <p className="text-gray-600">
                                    Store important moments and information with contextual grounding
                                </p>
                            </div>
                            <RotatingPrompts prompts={savePrompts} />
                            <PageInputForm
                                input={input}
                                setInput={setInput}
                                pageType="save"
                                isLoading={isLoading}
                                onSubmit={handleSubmit}
                                groundingEnabled={groundingEnabled}
                                onGroundingToggle={setGroundingEnabled}
                                placeholder="Enter a memory to add..."
                            />
                        </div>
                    </div>
                )}
            </div>
        </PageLayout>
    )
}

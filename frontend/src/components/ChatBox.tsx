"use client"

import React, { useRef, useEffect } from 'react'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Badge } from '@/components/ui/badge'
import { Brain, MapPin } from 'lucide-react'
import { ConfidencePill } from '@/components/ConfidencePill'
import { SupportingMemoriesDialog } from '@/components/SupportingMemoriesDialog'
import { SessionMemoriesDialog } from '@/components/SessionMemoriesDialog'
import { GroundingInfo } from '@/components/GroundingInfo'
import type { Memory } from '@/types'
import type { ApiMemory } from '@/lib/api'

export interface SessionMemory {
    memory_id?: string
    text: string
    created_at?: string
    grounded_text?: string
    similarity_score?: number
}

// Enhanced message interface that can handle different API response types
export interface UnifiedChatMessage {
    id: string
    type: 'user' | 'assistant' | 'system' | 'memory_save' | 'recall_result'
    content: string
    created_at: string | Date

    // Optional fields for different message types
    user_question?: string

    // For API responses with supporting memories
    confidence?: 'high' | 'medium' | 'low' | number
    reasoning?: string
    supporting_memories?: Memory[] | ApiMemory[]
    excluded_memories?: Memory[] | ApiMemory[]
    filtering_info?: {
        min_similarity_threshold?: number
        total_candidates?: number
        excluded_count?: number
        included_count?: number
    }

    // For recall/K-line responses
    mental_state?: string
    memory_count?: number
    answer?: string
    coherence_score?: number
    vectorset_name?: string

    // For memory save responses
    memory_id?: string
    save_success?: boolean
    grounding_applied?: boolean
    grounding_info?: {
        dependencies_found?: {
            spatial?: string[]
            environmental?: string[]
            temporal?: string[]
            social?: string[]
        }
        changes_made?: Array<{
            original: string
            replacement: string
            type: string
        }>
        unresolved_references?: string[]
    }
    context_snapshot?: {
        temporal?: {
            date?: string
            time?: string
            iso_date?: string
            day_of_week?: string
            month?: string
            year?: number
        }
        spatial?: {
            location?: string
            activity?: string
        }
        social?: {
            people_present?: string[]
        }
        environmental?: {
            weather?: string
            temperature?: string
            mood?: string
            [key: string]: any
        }
    }
    original_text?: string
    grounded_text?: string

    // For session-based chat
    hasMemory?: boolean
    session_memories?: SessionMemory[]
}

// Legacy interface for backward compatibility
export interface ChatMessage {
    id: string
    question: string
    answer: string
    created_at: Date
    hasMemory: boolean
    confidence?: 'high' | 'medium' | 'low' | number
    reasoning?: string
    supporting_memories?: unknown[]
    excluded_memories?: unknown[]
    filtering_info?: {
        min_similarity_threshold?: number
        total_candidates?: number
        excluded_count?: number
        included_count?: number
    }
    session_memories?: SessionMemory[]
}

// Simplified props interface focused on chat content only
export interface ChatBoxProps {
    messages: UnifiedChatMessage[] | ChatMessage[]
    isLoading?: boolean
    loadingText?: string
    showMemoryIndicators?: boolean
    className?: string

    // New props for enhanced functionality
    showTimestamps?: boolean
    enableMemoryExpansion?: boolean
    copiedId?: string | null
    onCopyId?: (id: string) => void
}

export function ChatBox({
    messages,
    isLoading = false,
    loadingText = "Thinking...",
    showMemoryIndicators = false,
    className = "",
    showTimestamps = true,
    enableMemoryExpansion = true,
    copiedId,
    onCopyId
}: ChatBoxProps) {
    const messagesEndRef = useRef<HTMLDivElement>(null)
    const scrollAreaRef = useRef<HTMLDivElement>(null)

    // Auto-scroll to bottom when new messages are added
    useEffect(() => {
        if (messagesEndRef.current && scrollAreaRef.current) {
            // Find the scrollable viewport within the ScrollArea
            const viewport = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]') as HTMLElement
            if (viewport) {
                // Scroll the viewport to the bottom smoothly
                viewport.scrollTo({
                    top: viewport.scrollHeight,
                    behavior: 'smooth'
                })
            }
        }
    }, [messages])

    // Helper function to get the last component of an ID after the final dash
    const getShortId = (id: string) => {
        const parts = id.split('-')
        return parts[parts.length - 1]
    }

    // Function to copy ID to clipboard with visual feedback
    const copyIdToClipboard = async (fullId: string) => {
        if (onCopyId) {
            onCopyId(fullId)
        } else {
            try {
                await navigator.clipboard.writeText(fullId)
            } catch (err) {
                console.error('Failed to copy ID to clipboard:', err)
            }
        }
    }

    // Helper function to determine if message is legacy format
    const isLegacyMessage = (msg: UnifiedChatMessage | ChatMessage): msg is ChatMessage => {
        return 'question' in msg && 'answer' in msg
    }

    // Helper function to get memory ID safely
    const getMemoryId = (memory: Memory | ApiMemory | unknown): string => {
        if (memory && typeof memory === 'object' && 'id' in memory) {
            return (memory as { id: string }).id
        }
        return ''
    }

    // Helper function to render memory indicators for legacy messages
    const renderMemoryIndicators = (message: ChatMessage) => {
        if (!showMemoryIndicators) return null

        return (
            <div className="flex flex-wrap items-center gap-2 mt-2">
                {message.hasMemory && message.session_memories && message.session_memories.length > 0 ? (
                    <SessionMemoriesDialog
                        memories={message.session_memories}
                        excludedMemories={message.excluded_memories as SessionMemory[]}
                        filteringInfo={message.filtering_info}
                        className="text-xs"
                    />
                ) : message.hasMemory ? (
                    <Badge variant="secondary" className="text-xs">
                        <Brain className="w-3 h-3 mr-1" />
                        Memory Enhanced
                    </Badge>
                ) : null}
                {message.confidence && (
                    <ConfidencePill confidence={message.confidence} className="text-xs" />
                )}
                {message.supporting_memories && message.supporting_memories.length > 0 && (
                    <SupportingMemoriesDialog
                        memories={message.supporting_memories as Memory[]}
                        excludedMemories={message.excluded_memories as Memory[]}
                        filteringInfo={message.filtering_info}
                        className="text-xs"
                    />
                )}
                {message.reasoning && (
                    <Badge variant="outline" className="text-xs">
                        <MapPin className="w-3 h-3 mr-1" />
                        Reasoning available
                    </Badge>
                )}
            </div>
        )
    }

    // Helper function to render assistant messages in unified format
    const renderAssistantMessage = (message: UnifiedChatMessage) => {
        const bgColor = message.type === 'memory_save' ? 'bg-green-100 text-green-800' :
                       message.type === 'system' ? 'bg-yellow-100 text-yellow-800' :
                       'bg-gray-100 text-gray-800'

        return (
            <div className={`${bgColor} rounded-lg px-4 py-2`}>
                {/* Confidence Badge */}
                {message.confidence && (
                    <div className="mb-2">
                        <span className={`inline-block px-2 py-1 rounded-full text-xs font-medium ${
                            (typeof message.confidence === 'string' && message.confidence === 'high') ||
                            (typeof message.confidence === 'number' && message.confidence >= 0.8) ? 'bg-green-100 text-green-800' :
                            (typeof message.confidence === 'string' && message.confidence === 'medium') ||
                            (typeof message.confidence === 'number' && message.confidence >= 0.6) ? 'bg-yellow-100 text-yellow-800' :
                            'bg-red-100 text-red-800'
                        }`}>
                            {typeof message.confidence === 'number' ?
                                `${(message.confidence * 100).toFixed(1)}% confidence` :
                                `${message.confidence} confidence`
                            }
                        </span>
                    </div>
                )}

                {/* Main Content */}
                <div className="whitespace-pre-wrap mb-2">{message.content}</div>

                {/* Memory Save Grounding Information */}
                {message.type === 'memory_save' && message.grounding_applied && message.grounding_info && (
                    <div className="mt-2">
                        <GroundingInfo
                            memory={{
                                id: message.memory_id || 'temp-id',
                                content: message.content,
                                text: message.grounded_text || message.content,
                                original_text: message.original_text,
                                grounded_text: message.grounded_text,
                                created_at: typeof message.created_at === 'string' ? message.created_at : message.created_at.toISOString(),
                                grounding_applied: message.grounding_applied,
                                grounding_info: message.grounding_info,
                                context_snapshot: message.context_snapshot
                            } as Memory}
                            className="text-xs"
                        />
                    </div>
                )}

                {/* Recall Result Information */}
                {message.type === 'recall_result' && (
                    <div className="mt-2 space-y-2">
                        {/* Answer */}
                        {message.answer && (
                            <div className="bg-blue-50 border border-blue-200 rounded p-3">
                                <div className="text-sm font-medium text-blue-800 mb-1">Answer:</div>
                                <div className="text-blue-700 whitespace-pre-wrap">{message.answer}</div>
                            </div>
                        )}

                        {/* Metrics */}
                        <div className="flex flex-wrap gap-2 text-xs">
                            {typeof message.confidence === 'number' && (
                                <span className={`px-2 py-1 rounded-full font-medium ${
                                    message.confidence >= 0.8 ? 'bg-green-100 text-green-800' :
                                    message.confidence >= 0.6 ? 'bg-yellow-100 text-yellow-800' :
                                    'bg-red-100 text-red-800'
                                }`}>
                                    Confidence: {(message.confidence * 100).toFixed(1)}%
                                </span>
                            )}
                            {typeof message.coherence_score === 'number' && (
                                <span className={`px-2 py-1 rounded-full font-medium ${
                                    message.coherence_score >= 0.8 ? 'bg-green-100 text-green-800' :
                                    message.coherence_score >= 0.6 ? 'bg-yellow-100 text-yellow-800' :
                                    'bg-red-100 text-red-800'
                                }`}>
                                    Coherence: {(message.coherence_score * 100).toFixed(1)}%
                                </span>
                            )}
                            {message.memory_count && (
                                <span className="px-2 py-1 rounded-full bg-gray-100 text-gray-800 font-medium">
                                    {message.memory_count} memories
                                </span>
                            )}
                            {message.vectorset_name && (
                                <span className="px-2 py-1 rounded-full bg-purple-100 text-purple-800 font-medium">
                                    Store: {message.vectorset_name}
                                </span>
                            )}
                        </div>
                    </div>
                )}

                {/* Reasoning */}
                {message.reasoning && (
                    <div className="text-sm text-gray-600 italic mb-2">
                        Reasoning: {message.reasoning}
                    </div>
                )}

                {/* Supporting Memories with expandable link */}
                {enableMemoryExpansion && message.supporting_memories && message.supporting_memories.length > 0 && (
                    <details className="mt-2">
                        <summary className="text-sm text-blue-600 cursor-pointer hover:text-blue-800">
                            See {message.supporting_memories.length} supporting memories
                        </summary>
                        <div className="mt-2 space-y-2">
                            {message.supporting_memories.map((memory, index) => (
                                <div key={getMemoryId(memory) || `${message.id}-memory-${index}`}
                                     className="bg-white border rounded p-2 text-sm">
                                    {renderMemoryDetails(memory as Memory, false)}
                                </div>
                            ))}
                        </div>
                    </details>
                )}

                {/* Excluded Memories */}
                {enableMemoryExpansion && message.excluded_memories && message.excluded_memories.length > 0 && (
                    <details className="mt-2">
                        <summary className="text-sm text-orange-600 cursor-pointer hover:text-orange-800">
                            See {message.excluded_memories.length} excluded memories
                        </summary>
                        <div className="mt-2 space-y-2">
                            <div className="text-xs text-orange-700 bg-orange-50 p-2 rounded">
                                These memories were retrieved but excluded due to similarity threshold filtering.
                                {message.filtering_info?.min_similarity_threshold && (
                                    <span> Threshold: {(message.filtering_info.min_similarity_threshold * 100).toFixed(1)}%</span>
                                )}
                            </div>
                            {message.excluded_memories.map((memory, index) => (
                                <div key={getMemoryId(memory) || `${message.id}-excluded-memory-${index}`}
                                     className="bg-orange-50 border border-orange-200 rounded p-2 text-sm">
                                    {renderMemoryDetails(memory as Memory, true)}
                                </div>
                            ))}
                        </div>
                    </details>
                )}

                {showTimestamps && (
                    <div className="text-xs text-gray-500 mt-1">
                        {new Date(message.created_at).toLocaleTimeString()}
                    </div>
                )}
            </div>
        )
    }

    // Helper function to render memory details
    const renderMemoryDetails = (memory: Memory, isExcluded: boolean = false) => {
        const badgeStyle = isExcluded ?
            'border-orange-300 text-orange-700 hover:bg-orange-100' :
            'hover:bg-gray-50'

        return (
            <>
                {/* Memory ID Badge */}
                {memory.id && (
                    <div className="mb-2">
                        <Badge
                            variant="outline"
                            className={`font-mono text-xs cursor-pointer transition-colors ${
                                copiedId === memory.id
                                    ? 'bg-green-50 text-green-700 border-green-200'
                                    : badgeStyle
                            }`}
                            onClick={() => copyIdToClipboard(memory.id!)}
                            title={`Click to copy full ID: ${memory.id}`}
                        >
                            {copiedId === memory.id ? '✓ Copied!' : `Memory ID: ${memory.id}`}
                        </Badge>
                    </div>
                )}
                <div className="text-gray-700">{memory.content || memory.text}</div>
                <div className="text-xs text-gray-500 mt-1 flex justify-between">
                    <span>{new Date(memory.created_at).toLocaleString()}</span>
                    {memory.metadata?.relevance_score && (
                        <span className={isExcluded ? 'text-orange-600' : ''}>
                            {memory.metadata.relevance_score}% relevant
                        </span>
                    )}
                </div>
            </>
        )
    }

    // Helper function to render message content based on type
    const renderMessageContent = (message: UnifiedChatMessage | ChatMessage) => {
        if (isLegacyMessage(message)) {
            // Legacy format - render as question/answer pair
            return (
                <div className="space-y-2">
                    <div className="bg-blue-100 p-3 rounded-lg ml-8">
                        <p className="text-sm font-medium text-blue-800">You:</p>
                        <p className="text-blue-700">{message.question}</p>
                    </div>
                    <div className="bg-gray-100 text-gray-800 p-3 rounded-lg mr-8">
                        <p className="text-sm font-medium">Assistant:</p>
                        <p className="whitespace-pre-wrap">{message.answer}</p>
                        {renderMemoryIndicators(message)}
                    </div>
                </div>
            )
        } else {
            // New unified format
            const unifiedMsg = message as UnifiedChatMessage

            if (unifiedMsg.type === 'user') {
                return (
                    <div className="flex justify-end">
                        <div className="max-w-[80%] bg-blue-500 text-white rounded-lg px-4 py-2">
                            <div className="whitespace-pre-wrap">{unifiedMsg.content}</div>
                            {showTimestamps && (
                                <div className="text-xs text-blue-100 mt-1">
                                    {new Date(unifiedMsg.created_at).toLocaleTimeString()}
                                </div>
                            )}
                        </div>
                    </div>
                )
            } else {
                return (
                    <div className="flex justify-start">
                        <div className="max-w-[90%] space-y-2">
                            {renderAssistantMessage(unifiedMsg)}
                        </div>
                    </div>
                )
            }
        }
    }

    return (
        <div className={`h-full ${className}`}>
            <ScrollArea ref={scrollAreaRef} className="h-full">
                <div className="space-y-4">
                    {messages.map((message) => (
                        <div key={message.id} className="space-y-2">
                            {renderMessageContent(message)}
                        </div>
                    ))}
                    {isLoading && (
                        <div className="flex justify-start">
                            <div className="max-w-[90%] bg-gray-100 text-gray-800 rounded-lg px-4 py-2">
                                <div className="flex items-center gap-2">
                                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-600"></div>
                                    <span>{loadingText}</span>
                                </div>
                            </div>
                        </div>
                    )}
                    {/* Scroll target */}
                    <div ref={messagesEndRef} />
                </div>
            </ScrollArea>
        </div>
    )
}

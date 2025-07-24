import { useState, useEffect, useCallback } from "react"
import { useConfiguredAPI } from "./useConfiguredAPI"
import type { Memory, Conversation, ApiStatus, ContextInfo } from "@/types"

export function useMemoryAPI() {
    const [memories, setMemories] = useState<Memory[]>([])
    const [searchResults, setSearchResults] = useState<Memory[]>([])
    const [excludedMemories, setExcludedMemories] = useState<Memory[]>([])
    const [filteringInfo, setFilteringInfo] = useState<{
        min_similarity_threshold?: number
        total_candidates?: number
        excluded_count?: number
        included_count?: number
    } | null>(null)
    const [isLoading, setIsLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [apiStatus, setApiStatus] = useState<ApiStatus>('unknown')
    const [currentContext, setCurrentContext] = useState<ContextInfo | null>(null)
    const [groundingEnabled, setGroundingEnabled] = useState(true)

    // Get the configured API client
    const { api: memoryAPI, isLoaded } = useConfiguredAPI()

    // Check API status on mount and when API configuration changes
    useEffect(() => {
        if (!isLoaded) {
            return
        }

        const initializeAPI = async () => {
            try {
                setApiStatus('unknown') // Reset to unknown while checking
                // Try to fetch memory info to check if API is working
                await memoryAPI.getMemoryInfo()
                setApiStatus('ready')
                setError(null) // Clear any previous errors
            } catch (error) {
                console.error('API status check failed:', error)
                setApiStatus('not_initialized')
                setError(`Failed to connect to Memory Agent API: ${error instanceof Error ? error.message : 'Unknown error'}`)
            }
        }
        initializeAPI()
    }, [isLoaded, memoryAPI])

    const saveMemory = async (content: string) => {
        setIsLoading(true)
        setError(null)

        try {
            const response = await memoryAPI.remember(content, groundingEnabled)

            // Debug logging to see what the API returns
            console.log('Save memory API response:', response)
            console.log('Grounding enabled:', groundingEnabled)

            if (response.success) {
                const newMemory: Memory = {
                    id: response.memory_id,
                    content,
                    text: response.grounded_text || content,
                    original_text: response.original_text,
                    grounded_text: response.grounded_text,
                    created_at: new Date().toISOString(),
                    grounding_applied: response.grounding_applied,
                    grounding_info: response.grounding_info,
                    context_snapshot: response.context_snapshot,
                    metadata: {
                        tags: ["user-input"],
                        confidence: 0.9,
                    },
                }

                setMemories((prev) => [newMemory, ...prev])
                return { success: true, response }
            } else {
                setError('Failed to save memory')
                return { success: false, response: null }
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to save memory')
            return { success: false, response: null }
        } finally {
            setIsLoading(false)
        }
    }

    const askQuestion = async (question: string, topK: number = 5, minSimilarity?: number) => {
        setIsLoading(true)
        setError(null)

        try {
            console.log('useMemoryAPI: askQuestion called with vectorset:', memoryAPI.getVectorStoreName())
            const response = await memoryAPI.ask(question, topK, minSimilarity)

            if (response.success) {
                // Convert supporting memories to the Memory format
                const supportingMemories: Memory[] = response.supporting_memories?.map((mem, index) => {
                    const content = mem.text || 'No content available'
                    const created_at = mem.timestamp || new Date().toISOString()
                    const id = mem.id || `supporting-memory-${Date.now()}-${Math.random().toString(36).substr(2, 9)}-${index}`

                    return {
                        id,
                        content,
                        text: mem.text,
                        original_text: mem.text,
                        grounded_text: mem.text,
                        created_at,
                        last_accessed_at: created_at,
                        grounding_applied: false,
                        grounding_info: {},
                        context_snapshot: {},
                        metadata: {
                            score: mem.relevance_score / 100, // Convert from 0-100 to 0-1
                            relevance_score: mem.relevance_score,
                            tags: mem.tags,
                            relevance_reasoning: mem.relevance_reasoning
                        }
                    }
                }) || []

                // Convert excluded memories to the Memory format
                const excludedMemories: Memory[] = response.excluded_memories?.map((mem, index) => {
                    const content = mem.content || mem.text || mem.memory || 'No content available'
                    const created_at = mem.created_at || new Date().toISOString()
                    const id = mem.id || `excluded-memory-${Date.now()}-${Math.random().toString(36).substr(2, 9)}-${index}`

                    return {
                        id,
                        content,
                        text: mem.text,
                        original_text: mem.original_text,
                        grounded_text: mem.grounded_text,
                        created_at,
                        last_accessed_at: mem.last_accessed_at,
                        grounding_applied: mem.grounding_applied,
                        grounding_info: mem.grounding_info,
                        context_snapshot: mem.context_snapshot,
                        metadata: {
                            ...mem.metadata,
                            score: mem.score,
                            relevance_score: mem.relevance_score
                        }
                    }
                }) || []

                const newConversation: Conversation = {
                    id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
                    question,
                    answer: response.answer,
                    created_at: new Date().toISOString(),
                    confidence: response.confidence,
                    reasoning: response.reasoning,
                    supporting_memories: supportingMemories,
                    excluded_memories: excludedMemories,
                    filtering_info: response.filtering_info,
                    // K-line specific fields
                    kline: response.kline,
                    total_memories_searched: response.total_memories_searched,
                    relevant_memories_used: response.relevant_memories_used,
                    type: response.type,
                }

                // Don't store in local state - let the calling component handle persistence
                return { success: true, conversation: newConversation }
            } else {
                setError('Failed to get answer from Memory Agent')
                return false
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to ask question')
            return false
        } finally {
            setIsLoading(false)
        }
    }

    const searchMemories = async (query: string, topK: number = 5, minSimilarity?: number) => {
        setIsLoading(true)
        setError(null)

        try {
            const response = await memoryAPI.recall(query, topK, minSimilarity)

            // Debug logging to see what the search returns
            console.log('Search memories API response:', response)
            console.log('Number of memories found:', response.memories?.length || 0)
            if (response.memories?.length > 0) {
                console.log('First memory grounding info:', {
                    grounding_applied: response.memories[0].grounding_applied,
                    grounding_info: response.memories[0].grounding_info,
                    original_text: response.memories[0].original_text,
                    grounded_text: response.memories[0].grounded_text
                })
            }

            if (response.success) {
                const formattedMemories: Memory[] = response.memories.map((mem, index) => {
                    const content = mem.content || mem.text || mem.memory || 'No content available'
                    const created_at = mem.created_at || new Date().toISOString()
                    const id = mem.id || `memory-${Date.now()}-${Math.random().toString(36).substr(2, 9)}-${index}`

                    return {
                        id,
                        content,
                        text: mem.text,
                        original_text: mem.original_text,
                        grounded_text: mem.grounded_text,
                        created_at,
                        last_accessed_at: mem.last_accessed_at,
                        grounding_applied: mem.grounding_applied,
                        grounding_info: mem.grounding_info,
                        context_snapshot: mem.context_snapshot,
                        metadata: {
                            ...mem.metadata,
                            score: mem.score,
                            relevance_score: mem.relevance_score
                        }
                    }
                })

                // Debug logging for formatted memories
                console.log('Formatted memories for display:', formattedMemories.map(mem => ({
                    id: mem.id,
                    content: mem.content,
                    grounding_applied: mem.grounding_applied,
                    has_grounding_info: !!mem.grounding_info
                })))

                setSearchResults(formattedMemories)

                // Handle excluded memories
                if (response.excluded_memories) {
                    const formattedExcludedMemories = response.excluded_memories.map(mem => ({
                        id: mem.id || '',
                        content: mem.content || mem.text || mem.memory || '',
                        text: mem.text || mem.content || mem.memory || '',
                        original_text: mem.original_text,
                        grounded_text: mem.grounded_text,
                        created_at: mem.created_at || new Date().toISOString(),
                        last_accessed_at: mem.last_accessed_at,
                        grounding_applied: mem.grounding_applied,
                        grounding_info: mem.grounding_info,
                        context_snapshot: mem.context_snapshot,
                        metadata: {
                            ...mem.metadata,
                            score: mem.score || mem.relevance_score,
                            relevance_score: mem.relevance_score,
                            tags: mem.tags || mem.metadata?.tags
                        }
                    }))
                    setExcludedMemories(formattedExcludedMemories)
                } else {
                    setExcludedMemories([])
                }

                // Store filtering info
                setFilteringInfo(response.filtering_info || null)

                return true
            } else {
                setError('Failed to search memories')
                return false
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to search memories')
            return false
        } finally {
            setIsLoading(false)
        }
    }

    const deleteMemory = async (memoryId: string) => {
        try {
            const response = await memoryAPI.deleteMemory(memoryId)

            if (response.success) {
                // Remove the memory from search results
                setSearchResults((prev) => prev.filter(memory => memory.id !== memoryId))
                return true
            } else {
                setError('Failed to delete memory')
                return false
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to delete memory')
            return false
        }
    }

    const clearAllMemories = async () => {
        try {
            const response = await memoryAPI.clearAllMemories()

            if (response.success) {
                // Clear all local state
                setMemories([])
                setSearchResults([])
                setExcludedMemories([])
                setFilteringInfo(null)
                // Note: conversations are managed by usePersistentChat hook
                return { success: true, deletedCount: response.memories_deleted }
            } else {
                setError('Failed to clear all memories')
                return { success: false, deletedCount: 0 }
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to clear all memories')
            return { success: false, deletedCount: 0 }
        }
    }

    const getContext = useCallback(async () => {
        try {
            console.log('useMemoryAPI: Calling getContext...')
            const response = await memoryAPI.getContext()
            console.log('useMemoryAPI: getContext response:', response)
            if (response.success) {
                setCurrentContext(response.context)
                return response.context
            }
            console.log('useMemoryAPI: getContext failed - success was false')
            return null
        } catch (err) {
            console.error('useMemoryAPI: getContext error:', err)
            setError(err instanceof Error ? err.message : 'Failed to get context')
            return null
        }
    }, [memoryAPI])

    const updateContext = useCallback(async (context: {
        location?: string
        activity?: string
        people_present?: string[]
        weather?: string
        [key: string]: any
    }) => {
        try {
            const response = await memoryAPI.setContext(context)
            if (response.success) {
                setCurrentContext(response.context)
                return true
            }
            return false
        } catch (err) {
            console.error('useMemoryAPI: updateContext error:', err)
            setError(err instanceof Error ? err.message : 'Failed to update context')
            return false
        }
    }, [memoryAPI])

    const clearError = () => setError(null)

    const clearSearchResults = useCallback(() => {
        setSearchResults([])
        setExcludedMemories([])
        setFilteringInfo(null)
    }, [])

    return {
        // State
        memories,
        searchResults,
        excludedMemories,
        filteringInfo,
        isLoading,
        error,
        apiStatus,
        currentContext,
        groundingEnabled,

        // Actions
        saveMemory,
        askQuestion,
        searchMemories,
        deleteMemory,
        clearAllMemories,
        clearSearchResults,
        getContext,
        updateContext,
        setGroundingEnabled,
        clearError,
    }
}

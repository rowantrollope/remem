import { useState, useEffect, useCallback } from 'react'
import type { Conversation } from '@/types'
import type { RememberResponse, RecallResponse as ApiRecallResponse, ApiMemory } from '@/lib/api'

export interface MemorySaveResponse {
    success: boolean
    response: RememberResponse | null
    originalText: string
    timestamp: string
}

export interface RecallResponse {
    success: boolean
    response: ApiRecallResponse | null
    originalQuery: string
    timestamp: string
}

export interface SearchResponse {
    success: boolean
    query: string
    results: ApiMemory[]
    excludedMemories?: ApiMemory[]
    filteringInfo?: {
        min_similarity_threshold?: number
        total_candidates?: number
        excluded_count?: number
        included_count?: number
    }
    timestamp: string
}

interface ChatState {
    conversations: Conversation[]
    memorySaveResponses: MemorySaveResponse[]
    recallResponses: RecallResponse[]
    searchResponses: SearchResponse[]
    lastUpdated: string
}

const CHAT_STORAGE_KEY_PREFIX = 'memory-chat-history'
const STORAGE_VERSION = '1.0'

export function usePersistentChat(vectorSetName: string = 'memories') {
    const [conversations, setConversations] = useState<Conversation[]>([])
    const [memorySaveResponses, setMemorySaveResponses] = useState<MemorySaveResponse[]>([])
    const [recallResponses, setRecallResponses] = useState<RecallResponse[]>([])
    const [searchResponses, setSearchResponses] = useState<SearchResponse[]>([])
    const [isLoaded, setIsLoaded] = useState(false)

    // Create vectorset-specific storage key
    const chatStorageKey = `${CHAT_STORAGE_KEY_PREFIX}-${vectorSetName}`

    // Load chat history from localStorage on initialization
    useEffect(() => {
        try {
            const savedData = localStorage.getItem(chatStorageKey)
            if (savedData) {
                const parsed = JSON.parse(savedData)
                
                // Validate the data structure and version
                if (parsed && parsed.version === STORAGE_VERSION && parsed.data) {
                    const {
                        conversations: savedConversations,
                        memorySaveResponses: savedMemorySaves,
                        recallResponses: savedRecallResponses,
                        searchResponses: savedSearchResponses
                    } = parsed.data

                    // Validate and restore conversations
                    if (Array.isArray(savedConversations)) {
                        setConversations(savedConversations.filter(conv =>
                            conv &&
                            typeof conv.id === 'string' &&
                            typeof conv.question === 'string' &&
                            typeof conv.answer === 'string' &&
                            typeof conv.created_at === 'string'
                        ))
                    }

                    // Validate and restore memory save responses
                    if (Array.isArray(savedMemorySaves)) {
                        setMemorySaveResponses(savedMemorySaves.filter(save =>
                            save &&
                            typeof save.success === 'boolean' &&
                            typeof save.originalText === 'string' &&
                            typeof save.timestamp === 'string'
                        ))
                    }

                    // Validate and restore recall responses
                    if (Array.isArray(savedRecallResponses)) {
                        setRecallResponses(savedRecallResponses.filter(recall =>
                            recall &&
                            typeof recall.success === 'boolean' &&
                            typeof recall.originalQuery === 'string' &&
                            typeof recall.timestamp === 'string'
                        ))
                    }

                    // Validate and restore search responses
                    if (Array.isArray(savedSearchResponses)) {
                        setSearchResponses(savedSearchResponses.filter(search =>
                            search &&
                            typeof search.success === 'boolean' &&
                            typeof search.query === 'string' &&
                            typeof search.timestamp === 'string' &&
                            Array.isArray(search.results)
                        ))
                    }

                    console.log('Chat history loaded from localStorage', {
                        conversations: savedConversations?.length || 0,
                        memorySaves: savedMemorySaves?.length || 0,
                        recallResponses: savedRecallResponses?.length || 0,
                        searchResponses: savedSearchResponses?.length || 0
                    })
                }
            }
        } catch (error) {
            console.error('Failed to load chat history from localStorage:', error)
            // Clear corrupted data
            localStorage.removeItem(chatStorageKey)
        } finally {
            setIsLoaded(true)
        }
    }, [chatStorageKey])

    // Save chat history to localStorage whenever it changes
    const saveToStorage = useCallback((newConversations: Conversation[], newMemorySaves: MemorySaveResponse[], newRecallResponses: RecallResponse[], newSearchResponses: SearchResponse[]) => {
        try {
            const chatState: ChatState = {
                conversations: newConversations,
                memorySaveResponses: newMemorySaves,
                recallResponses: newRecallResponses,
                searchResponses: newSearchResponses,
                lastUpdated: new Date().toISOString()
            }

            const dataToSave = {
                version: STORAGE_VERSION,
                data: chatState,
                savedAt: new Date().toISOString()
            }

            localStorage.setItem(chatStorageKey, JSON.stringify(dataToSave))
            console.log(`Chat history saved to localStorage for vectorset: ${vectorSetName}`, {
                conversations: newConversations.length,
                memorySaves: newMemorySaves.length,
                recallResponses: newRecallResponses.length,
                searchResponses: newSearchResponses.length
            })
        } catch (error) {
            console.error('Failed to save chat history to localStorage:', error)
        }
    }, [chatStorageKey, vectorSetName])

    // Add a new conversation and persist
    const addConversation = useCallback((conversation: Conversation) => {
        setConversations(prev => {
            const updated = [...prev, conversation]
            saveToStorage(updated, memorySaveResponses, recallResponses, searchResponses)
            return updated
        })
    }, [memorySaveResponses, recallResponses, searchResponses, saveToStorage])

    // Add a new memory save response and persist
    const addMemorySaveResponse = useCallback((saveResponse: MemorySaveResponse) => {
        setMemorySaveResponses(prev => {
            const updated = [...prev, saveResponse]
            saveToStorage(conversations, updated, recallResponses, searchResponses)
            return updated
        })
    }, [conversations, recallResponses, searchResponses, saveToStorage])

    // Add a new recall response and persist
    const addRecallResponse = useCallback((recallResponse: RecallResponse) => {
        setRecallResponses(prev => {
            const updated = [...prev, recallResponse]
            saveToStorage(conversations, memorySaveResponses, updated, searchResponses)
            return updated
        })
    }, [conversations, memorySaveResponses, searchResponses, saveToStorage])

    // Add a new search response and persist
    const addSearchResponse = useCallback((searchResponse: SearchResponse) => {
        setSearchResponses(prev => {
            const updated = [...prev, searchResponse]
            saveToStorage(conversations, memorySaveResponses, recallResponses, updated)
            return updated
        })
    }, [conversations, memorySaveResponses, recallResponses, saveToStorage])

    // Update conversations array and persist
    const updateConversations = useCallback((newConversations: Conversation[]) => {
        setConversations(newConversations)
        saveToStorage(newConversations, memorySaveResponses, recallResponses, searchResponses)
    }, [memorySaveResponses, recallResponses, searchResponses, saveToStorage])

    // Update memory save responses array and persist
    const updateMemorySaveResponses = useCallback((newMemorySaves: MemorySaveResponse[] | ((prev: MemorySaveResponse[]) => MemorySaveResponse[])) => {
        if (typeof newMemorySaves === 'function') {
            setMemorySaveResponses(prev => {
                const updated = newMemorySaves(prev)
                saveToStorage(conversations, updated, recallResponses, searchResponses)
                return updated
            })
        } else {
            setMemorySaveResponses(newMemorySaves)
            saveToStorage(conversations, newMemorySaves, recallResponses, searchResponses)
        }
    }, [conversations, recallResponses, searchResponses, saveToStorage])

    // Update recall responses array and persist
    const updateRecallResponses = useCallback((newRecallResponses: RecallResponse[] | ((prev: RecallResponse[]) => RecallResponse[])) => {
        if (typeof newRecallResponses === 'function') {
            setRecallResponses(prev => {
                const updated = newRecallResponses(prev)
                saveToStorage(conversations, memorySaveResponses, updated, searchResponses)
                return updated
            })
        } else {
            setRecallResponses(newRecallResponses)
            saveToStorage(conversations, memorySaveResponses, newRecallResponses, searchResponses)
        }
    }, [conversations, memorySaveResponses, searchResponses, saveToStorage])

    // Update search responses array and persist
    const updateSearchResponses = useCallback((newSearchResponses: SearchResponse[] | ((prev: SearchResponse[]) => SearchResponse[])) => {
        if (typeof newSearchResponses === 'function') {
            setSearchResponses(prev => {
                const updated = newSearchResponses(prev)
                saveToStorage(conversations, memorySaveResponses, recallResponses, updated)
                return updated
            })
        } else {
            setSearchResponses(newSearchResponses)
            saveToStorage(conversations, memorySaveResponses, recallResponses, newSearchResponses)
        }
    }, [conversations, memorySaveResponses, recallResponses, saveToStorage])

    // Clear all chat history
    const clearChatHistory = useCallback(() => {
        setConversations([])
        setMemorySaveResponses([])
        setRecallResponses([])
        setSearchResponses([])
        localStorage.removeItem(chatStorageKey)
        console.log(`Chat history cleared for vectorset: ${vectorSetName}`)
    }, [chatStorageKey, vectorSetName])

    // Get total message count for debugging
    const getTotalMessageCount = useCallback(() => {
        return conversations.length * 2 + memorySaveResponses.length * 2 + recallResponses.length * 2 + searchResponses.length * 2 // Each conversation = 2 messages, each memory save = 2 messages, each recall = 2 messages, each search = 2 messages
    }, [conversations.length, memorySaveResponses.length, recallResponses.length, searchResponses.length])

    // Export chat history as JSON
    const exportChatHistory = useCallback(() => {
        const exportData = {
            conversations,
            memorySaveResponses,
            recallResponses,
            searchResponses,
            exportedAt: new Date().toISOString(),
            version: STORAGE_VERSION
        }
        return JSON.stringify(exportData, null, 2)
    }, [conversations, memorySaveResponses, recallResponses, searchResponses])

    // Import chat history from JSON
    const importChatHistory = useCallback((jsonData: string) => {
        try {
            const imported = JSON.parse(jsonData)
            if (imported.conversations && imported.memorySaveResponses) {
                const importedRecallResponses = imported.recallResponses || []
                const importedSearchResponses = imported.searchResponses || []
                setConversations(imported.conversations)
                setMemorySaveResponses(imported.memorySaveResponses)
                setRecallResponses(importedRecallResponses)
                setSearchResponses(importedSearchResponses)
                saveToStorage(imported.conversations, imported.memorySaveResponses, importedRecallResponses, importedSearchResponses)
                return true
            }
            return false
        } catch (error) {
            console.error('Failed to import chat history:', error)
            return false
        }
    }, [saveToStorage])

    return {
        // State
        conversations,
        memorySaveResponses,
        recallResponses,
        searchResponses,
        isLoaded,

        // Actions
        addConversation,
        addMemorySaveResponse,
        addRecallResponse,
        addSearchResponse,
        updateConversations,
        updateMemorySaveResponses,
        updateRecallResponses,
        updateSearchResponses,
        clearChatHistory,

        // Utilities
        getTotalMessageCount,
        exportChatHistory,
        importChatHistory
    }
}

"use client"

import { useState, useEffect, useRef } from "react"

// Components
import { PageLayout } from "@/components/PageLayout"
import { PageInputForm } from "@/components/PageInputForm"

import { ApiPageHeader } from "@/components/ApiPageHeader"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from "@/components/ui/table"
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog"
import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger,
} from "@/components/ui/tooltip"
import type { UnifiedChatMessage } from "@/components/ChatBox"
// Hooks
import { useMemoryAPI } from "@/hooks"
import { useSettings } from "@/hooks/useSettings"
import { usePersistentChat } from "@/hooks/usePersistentChat"
import type { SearchResponse } from "@/hooks/usePersistentChat"
import type { ApiMemory } from "@/lib/api"
import type { Memory } from "@/types"
// Icons
import { Anchor, Trash2, AlertTriangle, Info } from "lucide-react"

// Helper function to format memory ID (short version - 8 chars with ...)
function formatShortId(memoryId: string) {
    if (!memoryId) return 'N/A'
    if (memoryId.length <= 8) return memoryId
    return memoryId.substring(0, 8) + '...'
}

// Delete confirmation dialog component
interface DeleteConfirmationDialogProps {
    isOpen: boolean
    onClose: () => void
    onConfirm: () => void
    memoryId: string
    isDeleting: boolean
}

function DeleteConfirmationDialog({
    isOpen,
    onClose,
    onConfirm,
    memoryId,
    isDeleting
}: DeleteConfirmationDialogProps) {
    return (
        <Dialog open={isOpen} onOpenChange={onClose}>
            <DialogContent className="sm:max-w-md">
                <DialogHeader>
                    <DialogTitle className="flex items-center gap-2 text-red-600">
                        <AlertTriangle className="w-5 h-5" />
                        Delete Memory
                    </DialogTitle>
                    <DialogDescription className="text-left">
                        Are you sure you want to delete this memory?
                        <br />
                        <br />
                        <strong>Memory ID:</strong> <span className="font-mono text-sm">{formatShortId(memoryId)}</span>
                        <br />
                        <br />
                        <strong className="text-red-600">This action cannot be undone.</strong>
                    </DialogDescription>
                </DialogHeader>
                <DialogFooter className="flex-col sm:flex-row gap-2">
                    <Button
                        variant="outline"
                        onClick={onClose}
                        disabled={isDeleting}
                        className="w-full sm:w-auto"
                    >
                        Cancel
                    </Button>
                    <Button
                        variant="destructive"
                        onClick={onConfirm}
                        disabled={isDeleting}
                        className="w-full sm:w-auto"
                    >
                        {isDeleting ? (
                            <>
                                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                                Deleting...
                            </>
                        ) : (
                            <>
                                <Trash2 className="w-4 h-4 mr-2" />
                                Yes, Delete
                            </>
                        )}
                    </Button>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    )
}

// Component to display search results in a professional format
interface SearchResultsDisplayProps {
    results: ApiMemory[]
    excludedMemories?: ApiMemory[]
    filteringInfo?: {
        min_similarity_threshold?: number
        total_candidates?: number
        excluded_count?: number
        included_count?: number
    }
    onMemoryDeleted?: (memoryId: string) => void
}

function SearchResultsDisplay({ results, excludedMemories, filteringInfo, onMemoryDeleted }: SearchResultsDisplayProps) {
    const [selectedMemory, setSelectedMemory] = useState<ApiMemory | null>(null)
    const [isDialogOpen, setIsDialogOpen] = useState(false)
    const [deleteMemoryId, setDeleteMemoryId] = useState<string | null>(null)
    const [isDeleting, setIsDeleting] = useState(false)

    const { deleteMemory } = useMemoryAPI()

    const handleRowClick = (memory: ApiMemory) => {
        setSelectedMemory(memory)
        setIsDialogOpen(true)
    }

    const handleDeleteClick = (e: React.MouseEvent, memoryId: string) => {
        e.stopPropagation() // Prevent row click
        setDeleteMemoryId(memoryId)
    }

    const handleDeleteConfirm = async () => {
        if (!deleteMemoryId) return

        setIsDeleting(true)
        try {
            const success = await deleteMemory(deleteMemoryId)
            if (success && onMemoryDeleted) {
                onMemoryDeleted(deleteMemoryId)
            }
        } finally {
            setIsDeleting(false)
            setDeleteMemoryId(null)
        }
    }

    const handleDeleteCancel = () => {
        setDeleteMemoryId(null)
    }

    const getSimilarityColor = (score: number | undefined) => {
        if (score === undefined || score === null) return 'text-gray-600'
        if (score >= 0.9) return 'text-green-600'
        if (score >= 0.7) return 'text-yellow-600'
        return 'text-red-600'
    }

    const getRelevanceColor = (score: number | undefined) => {
        if (score === undefined || score === null) return 'text-gray-600'
        if (score >= 0.8) return 'text-green-600'
        if (score >= 0.6) return 'text-yellow-600'
        return 'text-red-600'
    }



    return (
        <div className="space-y-4">

            {/* Memory Results Table */}
            <div className="border rounded-lg overflow-hidden">
                <Table>
                    <TableHeader>
                        <TableRow>
                            <TableHead className="">Similarity</TableHead>
                            <TableHead className="">
                                <div className="flex items-center gap-1">
                                    Relevance
                                    <TooltipProvider>
                                        <Tooltip>
                                            <TooltipTrigger asChild>
                                                <Info className="w-3 h-3 text-gray-400 cursor-help" />
                                            </TooltipTrigger>
                                            <TooltipContent>
                                                <p className="max-w-xs text-sm">
                                                    Relevance score is a blend of similarity, temporal recency, access recency, and access frequency
                                                </p>
                                            </TooltipContent>
                                        </Tooltip>
                                    </TooltipProvider>
                                </div>
                            </TableHead>
                            <TableHead className="">Date</TableHead>
                            <TableHead className="">ID</TableHead>
                            <TableHead>Memory Text</TableHead>
                            <TableHead className="">Grounding</TableHead>
                        </TableRow>
                    </TableHeader>
                    <TableBody>
                        {results.map((memory, index) => {
                            const memoryId = memory.id || `memory-${index}`
                            const score = memory.metadata?.score || memory.score
                            const relevanceScore = memory.metadata?.relevance_score || memory.relevance_score
                            const content = memory.content || memory.text || memory.memory || 'No content available'
                            const shortContent = content.length > 100 ? content.substring(0, 100) + '...' : content

                            return (
                                <TableRow
                                    key={memoryId}
                                    className="group cursor-pointer hover:bg-gray-50"
                                    onClick={() => handleRowClick(memory)}
                                >
                                    <TableCell>
                                        <span className={`font-medium text-xs ${getSimilarityColor(score)}`}>
                                            {score !== undefined ? `${(score * 100).toFixed(1)}%` : 'N/A'}
                                        </span>
                                    </TableCell>
                                    <TableCell>
                                        <span className={`font-medium text-xs ${getRelevanceColor(relevanceScore)}`}>
                                            {relevanceScore !== undefined ? `${(relevanceScore * 100).toFixed(1)}%` : 'N/A'}
                                        </span>
                                    </TableCell>
                                    <TableCell className="text-xs text-gray-600">
                                        {memory.created_at ? new Date(memory.created_at).toLocaleDateString() : 'N/A'}
                                    </TableCell>
                                    <TableCell className="font-mono text-xs text-gray-500">
                                        {formatShortId(memoryId)}
                                    </TableCell>
                                    <TableCell className="text-sm">
                                        {shortContent}
                                    </TableCell>
                                    <TableCell className="flex w-full">
                                        {memory.grounding_applied ? (
                                            <Badge className="text-xs bg-orange-100 text-orange-800">
                                                <Anchor className="w-3 h-3 mr-1" />
                                                Yes
                                            </Badge>
                                        ) : (
                                            <span className="text-xs text-gray-400">No</span>
                                        )}
                                        <div className="grow"></div>
                                        <Button
                                            variant="ghost"
                                            size="icon"
                                            className="text-red-500 hover:text-red-700 hover:bg-red-50"
                                            onClick={(e) => handleDeleteClick(e, memoryId)}
                                            title="Delete memory"
                                        >
                                            <Trash2 className="group-hover:flex hidden" />
                                        </Button>
                                    </TableCell>
                                </TableRow>
                            )
                        })}
                    </TableBody>
                </Table>
            </div>

            {/* JSON Dialog */}
            <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
                <DialogContent className="max-w-4xl max-h-[80vh]">
                    <DialogHeader>
                        <DialogTitle className="flex items-center gap-2">
                            Memory Details
                            {selectedMemory?.id && (
                                <span className="font-mono text-sm text-gray-500">
                                    {formatShortId(selectedMemory.id)}
                                </span>
                            )}
                        </DialogTitle>
                    </DialogHeader>
                    <div className="overflow-auto">
                        {selectedMemory && (
                            <pre className="text-xs bg-gray-50 p-4 rounded border overflow-x-auto">
                                {JSON.stringify(selectedMemory, null, 2)}
                            </pre>
                        )}
                    </div>
                </DialogContent>
            </Dialog>

            {/* Delete Confirmation Dialog */}
            <DeleteConfirmationDialog
                isOpen={deleteMemoryId !== null}
                onClose={handleDeleteCancel}
                onConfirm={handleDeleteConfirm}
                memoryId={deleteMemoryId || ''}
                isDeleting={isDeleting}
            />

            {/* Excluded Memories */}
            {excludedMemories && excludedMemories.length > 0 && (
                <div className="mt-6">
                    <h4 className="text-sm font-medium text-gray-700 mb-3">
                        Excluded Memories ({excludedMemories.length})
                    </h4>
                    <div className="space-y-2">
                        {excludedMemories.map((memory, index) => {
                            const memoryId = memory.id || `excluded-${index}`
                            const score = memory.metadata?.score || memory.score
                            return (
                                <div key={memoryId} className="flex items-center justify-between p-2 bg-orange-50 border border-orange-200 rounded text-sm">
                                    <span className="text-orange-800">
                                        {index + 1}. {formatShortId(memoryId)}
                                    </span>
                                    {score !== undefined && (
                                        <Badge variant="outline" className="text-xs text-orange-600">
                                            {(score * 100).toFixed(1)}%
                                        </Badge>
                                    )}
                                </div>
                            )
                        })}
                    </div>
                </div>
            )}

            {/* Filtering Info */}
            {filteringInfo && (
                <div className="mt-6 p-3 bg-gray-50 rounded border">
                    <h4 className="text-sm font-medium text-gray-700 mb-2">Filtering Information</h4>
                    <div className="grid grid-cols-2 gap-2 text-xs text-gray-600">
                        <div>Min Similarity: {filteringInfo.min_similarity_threshold || 'N/A'}</div>
                        <div>Total Candidates: {filteringInfo.total_candidates || 'N/A'}</div>
                        <div>Included: {filteringInfo.included_count || 'N/A'}</div>
                        <div>Excluded: {filteringInfo.excluded_count || 'N/A'}</div>
                    </div>
                </div>
            )}
        </div>
    )
}

// Custom chat interface for search results
interface SearchChatInterfaceProps {
    messages: UnifiedChatMessage[]
    isLoading: boolean
    searchResults: Memory[]
    excludedMemories: Memory[]
    filteringInfo: {
        min_similarity_threshold?: number
        total_candidates?: number
        excluded_count?: number
        included_count?: number
    } | null
    onMemoryDeleted?: (memoryId: string) => void
}

function SearchChatInterface({
    messages,
    isLoading,
    onMemoryDeleted
}: SearchChatInterfaceProps) {
    const messagesEndRef = useRef<HTMLDivElement>(null)
    const scrollAreaRef = useRef<HTMLDivElement>(null)

    // Auto-scroll to bottom when new messages are added
    useEffect(() => {
        if (messagesEndRef.current && scrollAreaRef.current) {
            const viewport = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]') as HTMLElement
            if (viewport) {
                viewport.scrollTo({
                    top: viewport.scrollHeight,
                    behavior: 'smooth'
                })
            }
        }
    }, [messages])

    return (
        <div className="h-full">
            <div ref={scrollAreaRef} className="h-full overflow-y-auto">
                <div className="space-y-4">
                    {messages.map((message) => (
                        <div key={message.id} className="space-y-2">
                            {message.type === 'user' ? (
                                <div className="flex justify-end">
                                    <div className="max-w-[80%] bg-blue-500 text-white rounded-lg px-4 py-2">
                                        <div className="whitespace-pre-wrap">{message.content}</div>
                                        <div className="text-xs text-blue-100 mt-1">
                                            {new Date(message.created_at).toLocaleTimeString()}
                                        </div>
                                    </div>
                                </div>
                            ) : (
                                <div className="flex justify-start">
                                    <div className="max-w-[95%] space-y-2">
                                        <div className="bg-gray-100 text-gray-800 rounded-lg px-4 py-2">
                                            <div className="whitespace-pre-wrap">{message.content}</div>
                                            <div className="text-xs text-gray-500 mt-1">
                                                {new Date(message.created_at).toLocaleTimeString()}
                                            </div>
                                        </div>

                                        {/* Show SearchResultsDisplay for ALL assistant messages with search results */}
                                        {message.type === 'assistant' &&
                                         message.supporting_memories &&
                                         message.supporting_memories.length > 0 && (
                                            <div className="mt-4">
                                                <SearchResultsDisplay
                                                    results={message.supporting_memories as ApiMemory[]}
                                                    excludedMemories={message.excluded_memories as ApiMemory[]}
                                                    filteringInfo={message.filtering_info}
                                                    onMemoryDeleted={onMemoryDeleted}
                                                />
                                            </div>
                                        )}
                                    </div>
                                </div>
                            )}
                        </div>
                    ))}
                    {isLoading && (
                        <div className="flex justify-start">
                            <div className="max-w-[90%] bg-gray-100 text-gray-800 rounded-lg px-4 py-2">
                                <div className="flex items-center gap-2">
                                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-600"></div>
                                    <span>Searching memories...</span>
                                </div>
                            </div>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>
            </div>
        </div>
    )
}

export default function SearchPage() {
    const [input, setInput] = useState("")
    const [chatMessages, setChatMessages] = useState<UnifiedChatMessage[]>([])

    const {
        searchResults,
        excludedMemories,
        filteringInfo,
        isLoading,
        error,
        apiStatus,
        searchMemories,
        clearSearchResults,
        clearError,
    } = useMemoryAPI()

    // Handle memory deletion from search results
    const handleMemoryDeleted = (memoryId: string) => {
        // Update chat messages to remove the deleted memory from supporting_memories
        setChatMessages(prev => prev.map(message => {
            if (message.type === 'assistant' && message.supporting_memories) {
                return {
                    ...message,
                    supporting_memories: message.supporting_memories.filter(
                        (memory: ApiMemory) => memory.id !== memoryId
                    )
                }
            }
            return message
        }))
    }

    // Get settings for search top_k
    const { settings, updateSetting } = useSettings()

    // Handle vectorset change
    const handleVectorStoreChange = (newVectorStoreName: string) => {
        updateSetting('vectorSetName', newVectorStoreName)
    }

    // Use persistent chat hook for search responses (for persistence)
    const {
        searchResponses: persistentSearchResponses,
        addSearchResponse,
        updateSearchResponses,
    } = usePersistentChat(settings.vectorSetName)

    // Restore chat messages from persistent storage when vectorset changes
    useEffect(() => {
        if (persistentSearchResponses.length > 0) {
            const messages: UnifiedChatMessage[] = []

            persistentSearchResponses.forEach((searchResponse, index) => {
                // Add user message
                const userMessage: UnifiedChatMessage = {
                    id: `user-${index}-${Date.now()}`,
                    type: 'user',
                    content: searchResponse.query,
                    created_at: searchResponse.timestamp
                }
                messages.push(userMessage)

                // Add assistant response with search results or "No Memories Found"
                const assistantMessage: UnifiedChatMessage = {
                    id: `assistant-${index}-${Date.now()}`,
                    type: 'assistant',
                    content: searchResponse.results.length > 0
                        ? `Found ${searchResponse.results.length} matching memories`
                        : 'No Memories Found',
                    created_at: searchResponse.timestamp,
                    supporting_memories: searchResponse.results,
                    excluded_memories: searchResponse.excludedMemories,
                    filtering_info: searchResponse.filteringInfo
                }
                messages.push(assistantMessage)
            })

            setChatMessages(messages)
        } else {
            // Clear chat messages and search results when switching to vectorset with no history
            setChatMessages([])
            clearSearchResults()
        }
    }, [persistentSearchResponses, settings.vectorSetName, clearSearchResults])

    // Save search results when they are updated (after a successful search)
    useEffect(() => {
        if (chatMessages.length > 0) {
            // Check if the last message is already an assistant response with search results
            const lastMessage = chatMessages[chatMessages.length - 1]
            if (lastMessage.type === 'user') {
                if (searchResults.length > 0) {
                    // Add assistant response with search results
                    const assistantMessage: UnifiedChatMessage = {
                        id: `assistant-${Date.now()}`,
                        type: 'assistant',
                        content: `Found ${searchResults.length} matching memories`,
                        created_at: new Date().toISOString(),
                        supporting_memories: searchResults,
                        excluded_memories: excludedMemories,
                        filtering_info: filteringInfo || undefined
                    }
                    setChatMessages(prev => [...prev, assistantMessage])

                    // Save to persistent storage
                    const searchResponse: SearchResponse = {
                        success: true,
                        query: lastMessage.content,
                        results: searchResults,
                        excludedMemories: excludedMemories,
                        filteringInfo: filteringInfo || undefined,
                        timestamp: new Date().toISOString()
                    }
                    addSearchResponse(searchResponse)
                } else if (searchResults.length === 0 && !isLoading) {
                    // Add "No Memories Found" message when search returns no results
                    const noResultsMessage: UnifiedChatMessage = {
                        id: `assistant-${Date.now()}`,
                        type: 'assistant',
                        content: 'No Memories Found',
                        created_at: new Date().toISOString()
                    }
                    setChatMessages(prev => [...prev, noResultsMessage])

                    // Save empty search response to persistent storage
                    const searchResponse: SearchResponse = {
                        success: true,
                        query: lastMessage.content,
                        results: [],
                        excludedMemories: [],
                        filteringInfo: filteringInfo || undefined,
                        timestamp: new Date().toISOString()
                    }
                    addSearchResponse(searchResponse)
                }
            }
        }
    }, [searchResults, excludedMemories, filteringInfo, chatMessages, addSearchResponse, isLoading])

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        if (!input.trim()) return

        const currentInput = input
        setInput("")

        // Add user message immediately
        const userMessage: UnifiedChatMessage = {
            id: `user-${Date.now()}`,
            type: 'user',
            content: currentInput,
            created_at: new Date().toISOString()
        }
        setChatMessages(prev => [...prev, userMessage])

        try {
            const success = await searchMemories(currentInput, settings.questionTopK, settings.minSimilarity)
            if (!success) {
                // Add error message
                const errorMessage: UnifiedChatMessage = {
                    id: `error-${Date.now()}`,
                    type: 'assistant',
                    content: 'Failed to search memories. Please try again.',
                    created_at: new Date().toISOString()
                }
                setChatMessages(prev => [...prev, errorMessage])
            }
            // Success case is handled by the useEffect hook
        } catch (err) {
            // Add error message
            const errorMessage: UnifiedChatMessage = {
                id: `error-${Date.now()}`,
                type: 'assistant',
                content: `Error searching memories: ${err instanceof Error ? err.message : 'Unknown error'}`,
                created_at: new Date().toISOString()
            }
            setChatMessages(prev => [...prev, errorMessage])
        }
    }

    const clearSearch = () => {
        setChatMessages([])
        updateSearchResponses([])
        clearSearchResults()
    }

    // Check if there are any chat messages
    const hasMessages = chatMessages.length > 0

    return (
        <PageLayout
            error={error}
            apiStatus={apiStatus}
            onClearError={clearError}
        >
            {/* Search Content */}
            <div className="h-full flex flex-col">
                <ApiPageHeader
                    endpoint={`(POST) /api/memory/${settings.vectorSetName}/search`}
                    hasMessages={hasMessages}
                    onClearChat={clearSearch}
                    isLoading={isLoading}
                    title="Search Memory"
                    showSettingsButton={true}
                    showVectorStoreSelector={true}
                    vectorSetName={settings.vectorSetName}
                    onVectorStoreChange={handleVectorStoreChange}
                />
                {hasMessages ? (
                    // Layout when there are chat messages - input at bottom
                    <>
                        <div className="flex-1 min-h-0 p-4 bg-white">
                            <SearchChatInterface
                                messages={chatMessages}
                                isLoading={isLoading}
                                searchResults={searchResults}
                                excludedMemories={excludedMemories}
                                filteringInfo={filteringInfo}
                                onMemoryDeleted={handleMemoryDeleted}
                            />
                        </div>

                        {/* Input Form at bottom */}
                        <div className="flex-shrink-0 mt-4 p-4 rounded">
                            <PageInputForm
                                input={input}
                                setInput={setInput}
                                pageType="search"
                                isLoading={isLoading}
                                onSubmit={handleSubmit}
                            />
                        </div>
                    </>
                ) : (
                    // Layout when no messages - input centered vertically with prompt
                    <div className="flex-1 flex items-center justify-center">
                        <div className="w-full">
                            <div className="text-center mb-8">
                                <h1 className="text-3xl font-bold text-gray-900 mb-2">Search Memories</h1>
                                <p className="text-gray-600">
                                    Vector search for relevant nemes and return detailed JSON
                                </p>
                            </div>

                            <PageInputForm
                                input={input}
                                setInput={setInput}
                                pageType="search"
                                isLoading={isLoading}
                                onSubmit={handleSubmit}
                            />
                        </div>
                    </div>
                )}
            </div>
        </PageLayout>
    )
}

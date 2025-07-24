"use client"

import React, { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
    DialogTrigger,
} from '@/components/ui/dialog'
import { Brain, Clock, Target, EyeOff, Filter, Info, Copy, Check } from 'lucide-react'
import type { SessionMemory } from './ChatBox'

interface SessionMemoriesDialogProps {
    memories: SessionMemory[]
    excludedMemories?: SessionMemory[]
    filteringInfo?: {
        min_similarity_threshold?: number
        total_candidates?: number
        excluded_count?: number
        included_count?: number
    }
    className?: string
}

export function SessionMemoriesDialog({
    memories,
    excludedMemories,
    filteringInfo,
    className = ""
}: SessionMemoriesDialogProps) {
    const [isOpen, setIsOpen] = useState(false)
    const [copiedIds, setCopiedIds] = useState<Set<string>>(new Set())

    if (!memories || memories.length === 0) {
        return null
    }

    const formatTimestamp = (timestamp: string) => {
        try {
            return new Date(timestamp).toLocaleString()
        } catch {
            return timestamp
        }
    }

    const formatSimilarityScore = (score: number | undefined) => {
        if (score === undefined || score === null) return 'N/A'
        return (score * 100).toFixed(1) + '%'
    }

    const getSimilarityColor = (score: number | undefined) => {
        if (score === undefined || score === null) return 'text-gray-600 bg-gray-50'
        if (score >= 0.9) return 'text-green-600 bg-green-50'
        if (score >= 0.7) return 'text-yellow-600 bg-yellow-50'
        return 'text-red-600 bg-red-50'
    }

    const formatNemeId = (memoryId: string) => {
        // Extract the last component after the final dash
        return memoryId
    }

    const copyToClipboard = async (memoryId: string) => {
        try {
            await navigator.clipboard.writeText(memoryId)
            setCopiedIds(prev => new Set(prev).add(memoryId))
            // Clear the copied state after 2 seconds
            setTimeout(() => {
                setCopiedIds(prev => {
                    const newSet = new Set(prev)
                    newSet.delete(memoryId)
                    return newSet
                })
            }, 2000)
        } catch (err) {
            console.error('Failed to copy to clipboard:', err)
        }
    }

    return (
        <Dialog open={isOpen} onOpenChange={setIsOpen}>
            <DialogTrigger asChild>
                <Button variant="secondary" size="sm" className={className}>
                    <Brain className="w-3 h-3 mr-1" />
                    Memory Enhanced ({memories.length})
                    {excludedMemories && excludedMemories.length > 0 && (
                        <span className="ml-1 text-orange-600">
                            +{excludedMemories.length} excluded
                        </span>
                    )}
                </Button>
            </DialogTrigger>
            <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
                <DialogHeader>
                    <DialogTitle className="flex items-center gap-2">
                        <Brain className="w-5 h-5" />
                        Retrieved Memories ({memories.length})
                        {excludedMemories && excludedMemories.length > 0 && (
                            <span className="text-sm text-orange-600 font-normal">
                                +{excludedMemories.length} excluded
                            </span>
                        )}
                    </DialogTitle>
                    <DialogDescription>
                        These memories were used to enhance the response with personalized context.
                    </DialogDescription>
                    {filteringInfo && (
                        <div className="text-sm text-gray-600 space-y-1">
                            <div className="flex items-center gap-2">
                                <Filter className="w-4 h-4" />
                                <span>
                                    Threshold: {filteringInfo.min_similarity_threshold ?
                                        `${(filteringInfo.min_similarity_threshold * 100).toFixed(1)}%` :
                                        'N/A'
                                    }
                                </span>
                            </div>
                            {filteringInfo.total_candidates && (
                                <div className="text-xs text-gray-500">
                                    Total candidates: {filteringInfo.total_candidates} |
                                    Included: {filteringInfo.included_count || 0} |
                                    Excluded: {filteringInfo.excluded_count || 0}
                                </div>
                            )}
                        </div>
                    )}
                </DialogHeader>
                
                <div className="space-y-4">
                    {memories.map((memory, index) => (
                        <div key={memory.memory_id || `memory-${index}`} className="border rounded-lg p-4 space-y-3">
                            <div className="flex items-start justify-between gap-2">
                                <div className="flex-1">
                                    <h4 className="font-medium text-sm text-gray-900 mb-2">
                                        Memory #{index + 1}
                                    </h4>
                                    <p className="text-gray-700 leading-relaxed">
                                        {memory.grounded_text || memory.text}
                                    </p>
                                </div>
                                {memory.similarity_score !== undefined && (
                                    <Badge
                                        variant="outline"
                                        className={`text-xs ${getSimilarityColor(memory.similarity_score)}`}
                                    >
                                        <Target className="w-3 h-3 mr-1" />
                                        {formatSimilarityScore(memory.similarity_score)}
                                    </Badge>
                                )}
                            </div>

                            <div className="flex items-center gap-4 text-xs text-gray-500">
                                {memory.created_at && (
                                    <div className="flex items-center gap-1">
                                        <Clock className="w-3 h-3" />
                                        {formatTimestamp(memory.created_at)}
                                    </div>
                                )}
                                {memory.memory_id && (
                                    <button
                                        onClick={() => copyToClipboard(memory.memory_id!)}
                                        className="flex items-center gap-1 text-gray-400 hover:text-gray-600 transition-colors cursor-pointer"
                                        title="Click to copy full ID"
                                    >
                                        <span>Memory ID: {formatNemeId(memory.memory_id)}</span>
                                        {copiedIds.has(memory.memory_id) ? (
                                            <Check className="w-3 h-3 text-green-600" />
                                        ) : (
                                            <Copy className="w-3 h-3" />
                                        )}
                                    </button>
                                )}
                            </div>

                            {memory.grounded_text && memory.grounded_text !== memory.text && (
                                <div className="bg-gray-50 p-2 rounded text-xs">
                                    <span className="font-medium text-gray-600">Original: </span>
                                    <span className="text-gray-700">{memory.text}</span>
                                </div>
                            )}
                        </div>
                    ))}

                    {/* Excluded Memories Section */}
                    {excludedMemories && excludedMemories.length > 0 && (
                        <div className="space-y-4 pt-6 border-t border-orange-200">
                            <div className="flex items-center gap-2">
                                <EyeOff className="w-5 h-5 text-orange-600" />
                                <h3 className="text-lg font-medium text-orange-800">
                                    Excluded Memories ({excludedMemories.length})
                                </h3>
                            </div>
                            <div className="text-sm text-orange-700 bg-orange-50 p-3 rounded-lg">
                                <div className="flex items-center gap-2">
                                    <Info className="w-4 h-4" />
                                    <span>
                                        These memories were retrieved but excluded due to similarity threshold filtering.
                                    </span>
                                </div>
                            </div>

                            {excludedMemories.map((memory, index) => (
                                <div key={memory.memory_id || `excluded-memory-${index}`} className="border border-orange-200 bg-orange-50/30 rounded-lg p-4 space-y-3">
                                    <div className="flex items-start justify-between gap-2">
                                        <div className="flex-1">
                                            <h4 className="font-medium text-sm text-orange-800 mb-2">
                                                Excluded Memory #{index + 1}
                                            </h4>
                                            <p className="text-gray-700 leading-relaxed">
                                                {memory.grounded_text || memory.text}
                                            </p>
                                        </div>
                                        {memory.similarity_score !== undefined && (
                                            <Badge
                                                variant="outline"
                                                className={`text-xs text-orange-700 border-orange-300`}
                                            >
                                                <Target className="w-3 h-3 mr-1" />
                                                {formatSimilarityScore(memory.similarity_score)}
                                            </Badge>
                                        )}
                                    </div>

                                    <div className="flex items-center gap-4 text-xs text-gray-500">
                                        {memory.created_at && (
                                            <div className="flex items-center gap-1">
                                                <Clock className="w-3 h-3" />
                                                {formatTimestamp(memory.created_at)}
                                            </div>
                                        )}
                                        {memory.memory_id && (
                                            <button
                                                onClick={() => copyToClipboard(memory.memory_id!)}
                                                className="flex items-center gap-1 text-gray-400 hover:text-gray-600 transition-colors cursor-pointer"
                                                title="Click to copy full ID"
                                            >
                                                <span>Memory ID: {formatNemeId(memory.memory_id)}</span>
                                                {copiedIds.has(memory.memory_id) ? (
                                                    <Check className="w-3 h-3 text-green-600" />
                                                ) : (
                                                    <Copy className="w-3 h-3" />
                                                )}
                                            </button>
                                        )}
                                    </div>

                                    {memory.grounded_text && memory.grounded_text !== memory.text && (
                                        <div className="bg-orange-100/50 p-2 rounded text-xs border border-orange-200">
                                            <span className="font-medium text-gray-600">Original: </span>
                                            <span className="text-gray-700">{memory.text}</span>
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                <div className="mt-4 p-3 bg-blue-50 rounded-lg">
                    <p className="text-sm text-blue-800">
                        <strong>How it works:</strong> The AI retrieved these {memories.length} most relevant memories
                        based on similarity to your question, then used them to provide a more personalized and
                        contextual response.
                        {excludedMemories && excludedMemories.length > 0 && (
                            <span> Additionally, {excludedMemories.length} memories were excluded due to similarity threshold filtering.</span>
                        )}
                    </p>
                </div>
            </DialogContent>
        </Dialog>
    )
}

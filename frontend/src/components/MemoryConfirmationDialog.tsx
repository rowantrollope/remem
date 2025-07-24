"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogFooter,
} from "@/components/ui/dialog"
import {
    CheckCircle,
    Anchor,
    MapPin,
    Users,
    Cloud,
    Calendar,
    Clock,
    Thermometer,
    Heart,
    Eye,
    EyeOff,
    Copy,
    Check
} from "lucide-react"
import type { RememberResponse } from "@/lib/api"

interface MemoryConfirmationDialogProps {
    isOpen: boolean
    onClose: () => void
    response: RememberResponse | null
    originalText: string
}

export function MemoryConfirmationDialog({ 
    isOpen, 
    onClose, 
    response, 
    originalText 
}: MemoryConfirmationDialogProps) {
    const [showDetails, setShowDetails] = useState(false)
    const [copiedId, setCopiedId] = useState(false)

    if (!response) return null

    const hasGrounding = response.grounding_applied && response.grounding_info
    const hasContext = response.context_snapshot

    const formatNemeId = (memoryId: string) => {
        // Extract the last component after the final dash
        const parts = memoryId.split('-')
        return parts[parts.length - 1]
    }

    const copyToClipboard = async (memoryId: string) => {
        try {
            await navigator.clipboard.writeText(memoryId)
            setCopiedId(true)
            // Clear the copied state after 2 seconds
            setTimeout(() => {
                setCopiedId(false)
            }, 2000)
        } catch (err) {
            console.error('Failed to copy to clipboard:', err)
        }
    }

    return (
        <Dialog open={isOpen} onOpenChange={onClose}>
            <DialogContent className="max-w-2xl max-h-[85vh] flex flex-col">
                <DialogHeader>
                    <DialogTitle className="flex items-center gap-2">
                        <CheckCircle className="w-5 h-5 text-green-600" />
                        Memory Saved Successfully!
                    </DialogTitle>
                </DialogHeader>

                <ScrollArea className="flex-1 pr-4">
                    <div className="space-y-4">
                        {/* Basic Info */}
                        <Card>
                            <CardHeader className="pb-3">
                                <CardTitle className="text-sm flex items-center gap-2">
                                    <CheckCircle className="w-4 h-4 text-green-600" />
                                    Memory Details
                                </CardTitle>
                            </CardHeader>
                            <CardContent className="space-y-3">
                                <div>
                                    <h4 className="text-xs font-medium text-gray-700 mb-1">Memory ID</h4>
                                    <div className="flex items-center gap-2 text-sm font-mono bg-gray-100 p-2 rounded">
                                        <span className="flex-1">{formatNemeId(response.memory_id)}</span>
                                        <button
                                            onClick={() => copyToClipboard(response.memory_id)}
                                            className="flex items-center gap-1 text-gray-500 hover:text-gray-700 transition-colors cursor-pointer"
                                            title="Click to copy full ID"
                                        >
                                            {copiedId ? (
                                                <Check className="w-4 h-4 text-green-600" />
                                            ) : (
                                                <Copy className="w-4 h-4" />
                                            )}
                                        </button>
                                    </div>
                                    <p className="text-xs text-gray-500 mt-1">
                                        Full ID: {response.memory_id}
                                    </p>
                                </div>
                                
                                <div>
                                    <h4 className="text-xs font-medium text-gray-700 mb-1">Original Text</h4>
                                    <p className="text-sm bg-gray-50 p-2 rounded border-l-4 border-gray-300">
                                        {originalText}
                                    </p>
                                </div>

                                {response.grounded_text && response.grounded_text !== originalText && (
                                    <div>
                                        <h4 className="text-xs font-medium text-gray-700 mb-1 flex items-center gap-1">
                                            <Anchor className="w-3 h-3 text-blue-600" />
                                            Grounded Text
                                        </h4>
                                        <p className="text-sm bg-blue-50 p-2 rounded border-l-4 border-blue-300">
                                            {response.grounded_text}
                                        </p>
                                    </div>
                                )}

                                {hasGrounding && (
                                    <div className="flex items-center gap-2">
                                        <Badge className="text-xs bg-blue-100 text-blue-800">
                                            <Anchor className="w-3 h-3 mr-1" />
                                            Contextual Grounding Applied
                                        </Badge>
                                    </div>
                                )}
                            </CardContent>
                        </Card>

                        {/* Context Snapshot */}
                        {hasContext && (
                            <Card>
                                <CardHeader className="pb-3">
                                    <CardTitle className="text-sm flex items-center gap-2">
                                        <MapPin className="w-4 h-4 text-purple-600" />
                                        Context Snapshot
                                    </CardTitle>
                                </CardHeader>
                                <CardContent className="space-y-3">
                                    {response.context_snapshot?.temporal && (
                                        <div className="grid grid-cols-2 gap-2 text-xs">
                                            {response.context_snapshot.temporal.date && (
                                                <div className="flex items-center gap-1">
                                                    <Calendar className="w-3 h-3 text-gray-500" />
                                                    <span>{response.context_snapshot.temporal.date}</span>
                                                </div>
                                            )}
                                            {response.context_snapshot.temporal.time && (
                                                <div className="flex items-center gap-1">
                                                    <Clock className="w-3 h-3 text-gray-500" />
                                                    <span>{response.context_snapshot.temporal.time}</span>
                                                </div>
                                            )}
                                        </div>
                                    )}

                                    {response.context_snapshot?.spatial && (
                                        <div className="space-y-1">
                                            {response.context_snapshot.spatial.location && (
                                                <div className="flex items-center gap-1 text-xs">
                                                    <MapPin className="w-3 h-3 text-gray-500" />
                                                    <span>Location: {response.context_snapshot.spatial.location}</span>
                                                </div>
                                            )}
                                            {response.context_snapshot.spatial.activity && (
                                                <div className="flex items-center gap-1 text-xs">
                                                    <span>Activity: {response.context_snapshot.spatial.activity}</span>
                                                </div>
                                            )}
                                        </div>
                                    )}

                                    {response.context_snapshot?.social?.people_present && response.context_snapshot.social.people_present.length > 0 && (
                                        <div className="flex items-center gap-1 text-xs">
                                            <Users className="w-3 h-3 text-gray-500" />
                                            <span>People: {response.context_snapshot.social.people_present.join(', ')}</span>
                                        </div>
                                    )}

                                    {response.context_snapshot?.environmental && (
                                        <div className="grid grid-cols-2 gap-2 text-xs">
                                            {response.context_snapshot.environmental.weather && (
                                                <div className="flex items-center gap-1">
                                                    <Cloud className="w-3 h-3 text-gray-500" />
                                                    <span>{response.context_snapshot.environmental.weather}</span>
                                                </div>
                                            )}
                                            {response.context_snapshot.environmental.temperature && (
                                                <div className="flex items-center gap-1">
                                                    <Thermometer className="w-3 h-3 text-gray-500" />
                                                    <span>{response.context_snapshot.environmental.temperature}</span>
                                                </div>
                                            )}
                                            {response.context_snapshot.environmental.mood && (
                                                <div className="flex items-center gap-1">
                                                    <Heart className="w-3 h-3 text-gray-500" />
                                                    <span>Mood: {response.context_snapshot.environmental.mood}</span>
                                                </div>
                                            )}
                                        </div>
                                    )}
                                </CardContent>
                            </Card>
                        )}

                        {/* Grounding Details */}
                        {hasGrounding && (
                            <Card>
                                <CardHeader className="pb-3">
                                    <div className="flex items-center justify-between">
                                        <CardTitle className="text-sm flex items-center gap-2">
                                            <Anchor className="w-4 h-4 text-blue-600" />
                                            Grounding Information
                                        </CardTitle>
                                        <Button
                                            variant="outline"
                                            size="sm"
                                            onClick={() => setShowDetails(!showDetails)}
                                            className="text-xs h-6"
                                        >
                                            {showDetails ? (
                                                <>
                                                    <EyeOff className="w-3 h-3 mr-1" />
                                                    Hide Details
                                                </>
                                            ) : (
                                                <>
                                                    <Eye className="w-3 h-3 mr-1" />
                                                    Show Details
                                                </>
                                            )}
                                        </Button>
                                    </div>
                                </CardHeader>
                                {showDetails && (
                                    <CardContent className="space-y-3">
                                        {response.grounding_info?.dependencies_found && (
                                            <div className="space-y-2">
                                                <h4 className="text-xs font-medium text-gray-700">Dependencies Found</h4>
                                                {Object.entries(response.grounding_info.dependencies_found).map(([type, deps]) => (
                                                    deps && deps.length > 0 && (
                                                        <div key={type} className="text-xs">
                                                            <span className="font-medium capitalize">{type}:</span>
                                                            <div className="flex flex-wrap gap-1 mt-1">
                                                                {deps.map((dep, index) => (
                                                                    <Badge key={index} variant="secondary" className="text-xs">
                                                                        {dep}
                                                                    </Badge>
                                                                ))}
                                                            </div>
                                                        </div>
                                                    )
                                                ))}
                                            </div>
                                        )}

                                        {response.grounding_info?.changes_made && response.grounding_info.changes_made.length > 0 && (
                                            <div className="space-y-2">
                                                <h4 className="text-xs font-medium text-gray-700">Changes Made</h4>
                                                {response.grounding_info.changes_made.map((change, index) => (
                                                    <div key={index} className="text-xs bg-yellow-50 p-2 rounded border-l-4 border-yellow-300">
                                                        <div><strong>Original:</strong> {change.original}</div>
                                                        <div><strong>Replacement:</strong> {change.replacement}</div>
                                                        <div><strong>Type:</strong> {change.type}</div>
                                                    </div>
                                                ))}
                                            </div>
                                        )}
                                    </CardContent>
                                )}
                            </Card>
                        )}
                    </div>
                </ScrollArea>

                <DialogFooter>
                    <Button onClick={onClose} className="w-full">
                        Got it!
                    </Button>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    )
}

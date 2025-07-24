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
    DialogTrigger,
} from "@/components/ui/dialog"
import { 
    Anchor, 
    MapPin, 
    Users, 
    Cloud, 
    Calendar, 
    ArrowRight,
    Eye,
    EyeOff
} from "lucide-react"
import type { Memory } from "@/types"

interface GroundingInfoProps {
    memory: Memory
    className?: string
}

export function GroundingInfo({ memory, className }: GroundingInfoProps) {
    const [open, setOpen] = useState(false)
    const [showOriginal, setShowOriginal] = useState(false)

    if (!memory.grounding_applied || !memory.grounding_info) {
        return null
    }

    const hasChanges = memory.grounding_info.changes_made && memory.grounding_info.changes_made.length > 0
    const hasDependencies = memory.grounding_info.dependencies_found && 
        Object.values(memory.grounding_info.dependencies_found).some(deps => deps && deps.length > 0)

    return (
        <Dialog open={open} onOpenChange={setOpen}>
            <DialogTrigger asChild>
                <Button
                    variant="outline"
                    size="sm"
                    className={`${className} text-xs`}
                >
                    <Anchor className="w-3 h-3 mr-1" />
                    Grounding Info
                </Button>
            </DialogTrigger>
            <DialogContent className="max-w-3xl max-h-[85vh] flex flex-col">
                <DialogHeader>
                    <DialogTitle className="flex items-center gap-2">
                        <Anchor className="w-5 h-5 text-blue-600" />
                        Contextual Grounding Information
                    </DialogTitle>
                </DialogHeader>
                
                <div className="flex-1 overflow-hidden">
                    <ScrollArea className="h-[70vh] pr-4">
                        <div className="space-y-6">
                            {/* Original vs Grounded Text */}
                            {memory.original_text && memory.grounded_text && (
                                <Card>
                                    <CardHeader className="pb-3">
                                        <div className="flex items-center justify-between">
                                            <CardTitle className="text-lg">Text Comparison</CardTitle>
                                            <Button
                                                variant="outline"
                                                size="sm"
                                                onClick={() => setShowOriginal(!showOriginal)}
                                            >
                                                {showOriginal ? (
                                                    <>
                                                        <EyeOff className="w-4 h-4 mr-1" />
                                                        Show Grounded
                                                    </>
                                                ) : (
                                                    <>
                                                        <Eye className="w-4 h-4 mr-1" />
                                                        Show Original
                                                    </>
                                                )}
                                            </Button>
                                        </div>
                                    </CardHeader>
                                    <CardContent className="space-y-3">
                                        <div className="p-3 bg-gray-50 rounded-lg">
                                            <h4 className="text-sm font-medium text-gray-700 mb-2">
                                                {showOriginal ? "Original Text:" : "Grounded Text:"}
                                            </h4>
                                            <p className="text-sm">
                                                {showOriginal ? memory.original_text : memory.grounded_text}
                                            </p>
                                        </div>
                                    </CardContent>
                                </Card>
                            )}

                            {/* Changes Made */}
                            {hasChanges && (
                                <Card>
                                    <CardHeader className="pb-3">
                                        <CardTitle className="text-lg">Grounding Changes</CardTitle>
                                    </CardHeader>
                                    <CardContent>
                                        <div className="space-y-3">
                                            {memory.grounding_info.changes_made?.map((change, index) => (
                                                <div key={index} className="flex items-center gap-3 p-3 bg-blue-50 rounded-lg">
                                                    <Badge variant="outline" className="text-xs">
                                                        {change.type}
                                                    </Badge>
                                                    <div className="flex items-center gap-2 text-sm">
                                                        <span className="text-red-600 font-mono bg-red-100 px-2 py-1 rounded">
                                                            "{change.original}"
                                                        </span>
                                                        <ArrowRight className="w-4 h-4 text-gray-400" />
                                                        <span className="text-green-600 font-mono bg-green-100 px-2 py-1 rounded">
                                                            "{change.replacement}"
                                                        </span>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </CardContent>
                                </Card>
                            )}

                            {/* Dependencies Found */}
                            {hasDependencies && (
                                <Card>
                                    <CardHeader className="pb-3">
                                        <CardTitle className="text-lg">Dependencies Detected</CardTitle>
                                    </CardHeader>
                                    <CardContent>
                                        <div className="space-y-4">
                                            {memory.grounding_info.dependencies_found?.spatial && 
                                             memory.grounding_info.dependencies_found.spatial.length > 0 && (
                                                <div>
                                                    <h4 className="text-sm font-medium text-gray-700 mb-2 flex items-center gap-1">
                                                        <MapPin className="w-4 h-4" />
                                                        Spatial
                                                    </h4>
                                                    <div className="flex flex-wrap gap-2">
                                                        {memory.grounding_info.dependencies_found.spatial.map((dep, index) => (
                                                            <Badge key={index} className="text-xs bg-green-100 text-green-800">
                                                                {dep}
                                                            </Badge>
                                                        ))}
                                                    </div>
                                                </div>
                                            )}

                                            {memory.grounding_info.dependencies_found?.environmental && 
                                             memory.grounding_info.dependencies_found.environmental.length > 0 && (
                                                <div>
                                                    <h4 className="text-sm font-medium text-gray-700 mb-2 flex items-center gap-1">
                                                        <Cloud className="w-4 h-4" />
                                                        Environmental
                                                    </h4>
                                                    <div className="flex flex-wrap gap-2">
                                                        {memory.grounding_info.dependencies_found.environmental.map((dep, index) => (
                                                            <Badge key={index} className="text-xs bg-yellow-100 text-yellow-800">
                                                                {dep}
                                                            </Badge>
                                                        ))}
                                                    </div>
                                                </div>
                                            )}

                                            {memory.grounding_info.dependencies_found?.social && 
                                             memory.grounding_info.dependencies_found.social.length > 0 && (
                                                <div>
                                                    <h4 className="text-sm font-medium text-gray-700 mb-2 flex items-center gap-1">
                                                        <Users className="w-4 h-4" />
                                                        Social
                                                    </h4>
                                                    <div className="flex flex-wrap gap-2">
                                                        {memory.grounding_info.dependencies_found.social.map((dep, index) => (
                                                            <Badge key={index} className="text-xs bg-blue-100 text-blue-800">
                                                                {dep}
                                                            </Badge>
                                                        ))}
                                                    </div>
                                                </div>
                                            )}

                                            {memory.grounding_info.dependencies_found?.temporal && 
                                             memory.grounding_info.dependencies_found.temporal.length > 0 && (
                                                <div>
                                                    <h4 className="text-sm font-medium text-gray-700 mb-2 flex items-center gap-1">
                                                        <Calendar className="w-4 h-4" />
                                                        Temporal
                                                    </h4>
                                                    <div className="flex flex-wrap gap-2">
                                                        {memory.grounding_info.dependencies_found.temporal.map((dep, index) => (
                                                            <Badge key={index} className="text-xs bg-purple-100 text-purple-800">
                                                                {dep}
                                                            </Badge>
                                                        ))}
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    </CardContent>
                                </Card>
                            )}

                            {/* Context Snapshot */}
                            {memory.context_snapshot && (
                                <Card>
                                    <CardHeader className="pb-3">
                                        <CardTitle className="text-lg">Context Snapshot</CardTitle>
                                    </CardHeader>
                                    <CardContent>
                                        <div className="space-y-4">
                                            {memory.context_snapshot.temporal && (
                                                <div>
                                                    <h4 className="text-sm font-medium text-gray-700 mb-2 flex items-center gap-1">
                                                        <Calendar className="w-4 h-4" />
                                                        Time
                                                    </h4>
                                                    <div className="flex flex-wrap gap-2">
                                                        {memory.context_snapshot.temporal.date && (
                                                            <Badge variant="outline" className="text-xs">
                                                                {memory.context_snapshot.temporal.date}
                                                            </Badge>
                                                        )}
                                                        {memory.context_snapshot.temporal.time && (
                                                            <Badge variant="outline" className="text-xs">
                                                                {memory.context_snapshot.temporal.time}
                                                            </Badge>
                                                        )}
                                                        {memory.context_snapshot.temporal.day_of_week && (
                                                            <Badge variant="outline" className="text-xs">
                                                                {memory.context_snapshot.temporal.day_of_week}
                                                            </Badge>
                                                        )}
                                                    </div>
                                                </div>
                                            )}

                                            {memory.context_snapshot.spatial && (
                                                <div>
                                                    <h4 className="text-sm font-medium text-gray-700 mb-2 flex items-center gap-1">
                                                        <MapPin className="w-4 h-4" />
                                                        Location & Activity
                                                    </h4>
                                                    <div className="flex flex-wrap gap-2">
                                                        {memory.context_snapshot.spatial.location && (
                                                            <Badge className="text-xs bg-green-100 text-green-800">
                                                                {memory.context_snapshot.spatial.location}
                                                            </Badge>
                                                        )}
                                                        {memory.context_snapshot.spatial.activity && (
                                                            <Badge className="text-xs bg-purple-100 text-purple-800">
                                                                {memory.context_snapshot.spatial.activity}
                                                            </Badge>
                                                        )}
                                                    </div>
                                                </div>
                                            )}

                                            {memory.context_snapshot.social?.people_present && 
                                             memory.context_snapshot.social.people_present.length > 0 && (
                                                <div>
                                                    <h4 className="text-sm font-medium text-gray-700 mb-2 flex items-center gap-1">
                                                        <Users className="w-4 h-4" />
                                                        People Present
                                                    </h4>
                                                    <div className="flex flex-wrap gap-2">
                                                        {memory.context_snapshot.social.people_present.map((person, index) => (
                                                            <Badge key={index} className="text-xs bg-blue-100 text-blue-800">
                                                                {person}
                                                            </Badge>
                                                        ))}
                                                    </div>
                                                </div>
                                            )}

                                            {memory.context_snapshot.environmental && (
                                                <div>
                                                    <h4 className="text-sm font-medium text-gray-700 mb-2 flex items-center gap-1">
                                                        <Cloud className="w-4 h-4" />
                                                        Environment
                                                    </h4>
                                                    <div className="flex flex-wrap gap-2">
                                                        {Object.entries(memory.context_snapshot.environmental).map(([key, value], index) => (
                                                            value && (
                                                                <Badge key={index} className="text-xs bg-yellow-100 text-yellow-800">
                                                                    {key}: {value}
                                                                </Badge>
                                                            )
                                                        ))}
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    </CardContent>
                                </Card>
                            )}

                            {/* Unresolved References */}
                            {memory.grounding_info.unresolved_references && 
                             memory.grounding_info.unresolved_references.length > 0 && (
                                <Card className="border-orange-200 bg-orange-50">
                                    <CardHeader className="pb-3">
                                        <CardTitle className="text-lg text-orange-800">Unresolved References</CardTitle>
                                    </CardHeader>
                                    <CardContent>
                                        <div className="flex flex-wrap gap-2">
                                            {memory.grounding_info.unresolved_references.map((ref, index) => (
                                                <Badge key={index} variant="outline" className="text-xs border-orange-300 text-orange-700">
                                                    {ref}
                                                </Badge>
                                            ))}
                                        </div>
                                        <p className="text-xs text-orange-600 mt-2">
                                            These references could not be resolved with the current context.
                                        </p>
                                    </CardContent>
                                </Card>
                            )}
                        </div>
                    </ScrollArea>
                </div>
            </DialogContent>
        </Dialog>
    )
}

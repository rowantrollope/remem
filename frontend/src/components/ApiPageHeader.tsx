"use client"

import { Button } from "@/components/ui/button"
import { Trash2 } from "lucide-react"
import { RecallSettingsDialog } from "./RecallSettingsDialog"
import { VectorStoreSelector } from "./VectorStoreSelector"

export interface ApiPageHeaderProps {
    endpoint: string
    hasMessages: boolean
    onClearChat: () => void
    isLoading?: boolean
    title?: string
    showSettingsButton?: boolean
    showVectorStoreSelector?: boolean
    vectorSetName?: string
    onVectorStoreChange?: (value: string) => void
}

export function ApiPageHeader({
    endpoint,
    hasMessages,
    onClearChat,
    isLoading = false,
    title,
    showSettingsButton = false,
    showVectorStoreSelector = false,
    vectorSetName,
    onVectorStoreChange
}: ApiPageHeaderProps) {
    return (
        <div className="flex-shrink-0 py-2 space-y-2 text-black ">
            <div className="flex justify-between items-center">
                <h1 className="text-lg font-semibold text-gray-900">
                    {showVectorStoreSelector && vectorSetName && onVectorStoreChange && (
                        <div className="flex items-center gap-2">
                            <span className="text-sm font-medium text-gray-700">Vector Set:</span>
                            <VectorStoreSelector
                                value={vectorSetName}
                                onValueChange={onVectorStoreChange}
                                disabled={isLoading}
                                className="w-[300px]"
                            />
                        </div>
                    )}
                </h1>
                <div className="flex items-center gap-2">
                    {hasMessages && (
                        <Button
                            variant="outline"
                            size="sm"
                            onClick={onClearChat}
                            className="flex items-center gap-2 text-red-600 hover:text-red-700 hover:bg-red-50"
                            disabled={isLoading}
                        >
                            <Trash2 className="w-4 h-4" />

                        </Button>
                    )}
                    {showSettingsButton && (
                        <RecallSettingsDialog />
                    )}
                </div>


                <div className="font-mono text-sm text-muted-foreground">
                    {endpoint}
                </div>
            </div>


        </div>
    )
}

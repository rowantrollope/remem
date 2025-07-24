"use client"

import React, { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
    DialogTrigger,
} from '@/components/ui/dialog'
import { Settings, RotateCcw } from 'lucide-react'
import { useSettings } from '@/hooks/useSettings'
import { VectorStoreSelector } from './VectorStoreSelector'

export function RecallSettingsDialog() {
    const { settings, updateSetting, resetSettings } = useSettings()
    const [tempTopK, setTempTopK] = useState<string>('')
    const [tempMinSimilarity, setTempMinSimilarity] = useState<string>('')
    const [isOpen, setIsOpen] = useState(false)

    // Sync temp values with settings when settings change
    useEffect(() => {
        setTempTopK(settings.questionTopK.toString())
        setTempMinSimilarity(settings.minSimilarity.toString())
    }, [settings.questionTopK, settings.minSimilarity])

    // Handle dialog open/close and sync temp values when opening
    const handleOpenChange = (open: boolean) => {
        setIsOpen(open)
        if (open) {
            // Sync temp values when dialog opens
            setTempTopK(settings.questionTopK.toString())
            setTempMinSimilarity(settings.minSimilarity.toString())
        }
    }

    const handleSaveTopK = () => {
        const value = parseInt(tempTopK)
        console.log('handleSaveTopK called with value:', value)
        console.log('Current settings before update:', settings)
        console.log('tempTopK:', tempTopK, 'parsed value:', value)
        console.log('isTopKValid():', isTopKValid(), 'isTopKChanged():', isTopKChanged())

        if (value >= 1 && value <= 50) {
            console.log('Calling updateSetting with questionTopK =', value)
            updateSetting('questionTopK', value)
        } else {
            console.log('Value validation failed:', value)
        }
    }

    const handleSaveMinSimilarity = () => {
        const value = parseFloat(tempMinSimilarity)
        if (value >= 0.0 && value <= 1.0) {
            updateSetting('minSimilarity', value)
        }
    }

    const handleReset = () => {
        resetSettings()
        // Reset temp values to defaults
        setTempTopK('10')
        setTempMinSimilarity('0.7')
    }

    const isTopKValid = () => {
        const value = parseInt(tempTopK)
        return !isNaN(value) && value >= 1 && value <= 50
    }

    const isMinSimilarityValid = () => {
        const value = parseFloat(tempMinSimilarity)
        return !isNaN(value) && value >= 0.0 && value <= 1.0
    }

    const isTopKChanged = () => {
        return tempTopK !== settings.questionTopK.toString()
    }

    const isMinSimilarityChanged = () => {
        return tempMinSimilarity !== settings.minSimilarity.toString()
    }

    return (
        <Dialog open={isOpen} onOpenChange={handleOpenChange}>
            <DialogTrigger asChild>
                <Button
                    variant="outline"
                    size="sm"
                    className="flex items-center gap-2"
                >
                    <Settings className="w-4 h-4" />
                </Button>
            </DialogTrigger>
            <DialogContent className="sm:max-w-md">
                <DialogHeader>
                    <DialogTitle className="flex items-center gap-2">
                        <Settings className="w-5 h-5" />
                        Options
                    </DialogTitle>
                    <DialogDescription>
                        Configure how many memories to retrieve, the similarity threshold for API calls, and which vectorset to use.
                    </DialogDescription>
                </DialogHeader>
                
                <div className="space-y-6">
                    {/* Top-K Setting */}
                    <div className="space-y-2">
                        <Label htmlFor="top-k" className="text-sm font-medium">
                            Top-K (Number of memories to retrieve)
                        </Label>
                        <p className="text-xs text-gray-500">
                            Controls how many relevant memories are retrieved. Higher values provide more context but may include less relevant information.
                        </p>
                        <div className="flex items-center gap-3">
                            <Input
                                id="top-k"
                                type="number"
                                min="1"
                                max="50"
                                value={tempTopK}
                                onChange={(e) => setTempTopK(e.target.value)}
                                className="w-24"
                            />
                            <Button
                                onClick={handleSaveTopK}
                                size="sm"
                                disabled={!isTopKValid() || !isTopKChanged()}
                            >
                                Save
                            </Button>
                        </div>
                        <p className="text-xs text-gray-600">
                            Current: <span className="font-mono font-medium">{settings.questionTopK}</span> memories
                        </p>
                    </div>

                    {/* Min Similarity Setting */}
                    <div className="space-y-2">
                        <Label htmlFor="min-similarity" className="text-sm font-medium">
                            Minimum Similarity (0.0 - 1.0)
                        </Label>
                        <p className="text-xs text-gray-500">
                            Minimum similarity threshold for memory retrieval. Higher values return only more relevant memories.
                        </p>
                        <div className="flex items-center gap-3">
                            <Input
                                id="min-similarity"
                                type="number"
                                min="0.0"
                                max="1.0"
                                step="0.1"
                                value={tempMinSimilarity}
                                onChange={(e) => setTempMinSimilarity(e.target.value)}
                                className="w-24"
                            />
                            <Button
                                onClick={handleSaveMinSimilarity}
                                size="sm"
                                disabled={!isMinSimilarityValid() || !isMinSimilarityChanged()}
                            >
                                Save
                            </Button>
                        </div>
                        <p className="text-xs text-gray-600">
                            Current: <span className="font-mono font-medium">{settings.minSimilarity}</span>
                        </p>
                    </div>

                    {/* Vectorstore Selection */}
                    <div className="space-y-2">
                        <Label className="text-sm font-medium">
                            Vectorstore Name
                        </Label>
                        <p className="text-xs text-gray-500">
                            Select which vectorset to use for memory operations. Demo vectorsets are pre-configured for specific use cases.
                        </p>
                        <VectorStoreSelector
                            value={settings.vectorSetName}
                            onValueChange={(value) => updateSetting('vectorSetName', value)}
                            className="w-full"
                        />
                        <p className="text-xs text-gray-600">
                            Current: <span className="font-mono font-medium">{settings.vectorSetName}</span>
                        </p>
                    </div>

                    {/* Reset Button */}
                    <div className="pt-4 border-t">
                        <Button
                            onClick={handleReset}
                            variant="outline"
                            size="sm"
                            className="flex items-center gap-2"
                        >
                            <RotateCcw className="w-4 h-4" />
                            Reset to Defaults
                        </Button>
                    </div>
                </div>
            </DialogContent>
        </Dialog>
    )
}

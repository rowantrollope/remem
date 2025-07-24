import React, { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { 
    Sparkles, 
    DollarSign, 
    Shield, 
    Brain, 
    Server, 
    Zap, 
    Clock, 
    HelpCircle,
    ChevronRight
} from 'lucide-react'
import { LLM_PRESETS, type LLMPresetConfig } from '@/types'

interface ConfigurationPresetsProps {
    onApplyPreset: (presetId: string) => void
    className?: string
}

export function ConfigurationPresets({ onApplyPreset, className }: ConfigurationPresetsProps) {
    const [selectedPreset, setSelectedPreset] = useState<LLMPresetConfig | null>(null)
    const [showDetails, setShowDetails] = useState(false)

    const getPresetIcon = (presetId: string) => {
        switch (presetId) {
            case 'cost-optimized':
                return <DollarSign className="w-5 h-5 text-green-600" />
            case 'performance-optimized':
                return <Zap className="w-5 h-5 text-blue-600" />
            case 'privacy-first':
                return <Shield className="w-5 h-5 text-purple-600" />
            default:
                return <Sparkles className="w-5 h-5 text-gray-600" />
        }
    }

    const getPresetColor = (presetId: string) => {
        switch (presetId) {
            case 'cost-optimized':
                return 'border-green-200 bg-green-50 hover:bg-green-100'
            case 'performance-optimized':
                return 'border-blue-200 bg-blue-50 hover:bg-blue-100'
            case 'privacy-first':
                return 'border-purple-200 bg-purple-50 hover:bg-purple-100'
            default:
                return 'border-gray-200 bg-gray-50 hover:bg-gray-100'
        }
    }

    const handleApplyPreset = (preset: LLMPresetConfig) => {
        onApplyPreset(preset.id)
        setShowDetails(false)
    }

    const PresetCard = ({ preset }: { preset: LLMPresetConfig }) => (
        <Card className={`cursor-pointer transition-all ${getPresetColor(preset.id)}`}>
            <CardContent className="p-4">
                <div className="flex items-start gap-3">
                    <div className="mt-1">
                        {getPresetIcon(preset.id)}
                    </div>
                    <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between mb-2">
                            <h3 className="font-semibold text-gray-900">{preset.name}</h3>
                            <Button
                                variant="ghost"
                                size="sm"
                                onClick={(e) => {
                                    e.stopPropagation()
                                    setSelectedPreset(preset)
                                    setShowDetails(true)
                                }}
                            >
                                <HelpCircle className="w-4 h-4" />
                            </Button>
                        </div>
                        <p className="text-sm text-gray-600 mb-3">{preset.description}</p>
                        
                        <div className="space-y-2">
                            <div className="flex flex-wrap gap-1">
                                {preset.tags.map((tag, index) => (
                                    <Badge key={index} variant="secondary" className="text-xs">
                                        {tag}
                                    </Badge>
                                ))}
                            </div>
                            
                            <div className="text-xs text-gray-500 space-y-1">
                                <div className="flex items-center gap-2">
                                    <Brain className="w-3 h-3" />
                                    <span>Tier 1: {preset.config.tier1.provider} • {preset.config.tier1.model}</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <Server className="w-3 h-3" />
                                    <span>Tier 2: {preset.config.tier2.provider} • {preset.config.tier2.model}</span>
                                </div>
                            </div>
                        </div>
                        
                        <Button
                            className="w-full mt-3"
                            size="sm"
                            onClick={() => handleApplyPreset(preset)}
                        >
                            Apply Preset
                            <ChevronRight className="w-4 h-4 ml-1" />
                        </Button>
                    </div>
                </div>
            </CardContent>
        </Card>
    )

    return (
        <div className={className}>
            <div className="mb-4">
                <h3 className="font-medium text-gray-900 mb-2">Configuration Presets</h3>
                <p className="text-sm text-gray-600">
                    Quick start with pre-configured LLM setups optimized for different use cases.
                </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {LLM_PRESETS.map(preset => (
                    <PresetCard key={preset.id} preset={preset} />
                ))}
            </div>

            {/* Preset Details Dialog */}
            <Dialog open={showDetails} onOpenChange={setShowDetails}>
                <DialogContent className="max-w-2xl">
                    <DialogHeader>
                        <DialogTitle className="flex items-center gap-2">
                            {selectedPreset && getPresetIcon(selectedPreset.id)}
                            {selectedPreset?.name} Configuration
                        </DialogTitle>
                    </DialogHeader>
                    
                    {selectedPreset && (
                        <div className="space-y-6">
                            <div>
                                <p className="text-gray-600 mb-4">{selectedPreset.description}</p>
                                <div className="flex flex-wrap gap-2">
                                    {selectedPreset.tags.map((tag, index) => (
                                        <Badge key={index} variant="secondary">{tag}</Badge>
                                    ))}
                                </div>
                            </div>

                            {/* Tier Configurations */}
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <Card>
                                    <CardHeader className="pb-3">
                                        <CardTitle className="text-sm flex items-center gap-2">
                                            <Zap className="w-4 h-4" />
                                            Tier 1 - Primary
                                        </CardTitle>
                                    </CardHeader>
                                    <CardContent className="space-y-2 text-sm">
                                        <div className="flex justify-between">
                                            <span className="text-gray-600">Provider:</span>
                                            <span className="font-medium">{selectedPreset.config.tier1.provider}</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-gray-600">Model:</span>
                                            <span className="font-medium">{selectedPreset.config.tier1.model}</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-gray-600">Temperature:</span>
                                            <span className="font-medium">{selectedPreset.config.tier1.temperature}</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-gray-600">Max Tokens:</span>
                                            <span className="font-medium">{selectedPreset.config.tier1.max_tokens}</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-gray-600">Timeout:</span>
                                            <span className="font-medium">{selectedPreset.config.tier1.timeout}s</span>
                                        </div>
                                    </CardContent>
                                </Card>

                                <Card>
                                    <CardHeader className="pb-3">
                                        <CardTitle className="text-sm flex items-center gap-2">
                                            <Shield className="w-4 h-4" />
                                            Tier 2 - Internal
                                        </CardTitle>
                                    </CardHeader>
                                    <CardContent className="space-y-2 text-sm">
                                        <div className="flex justify-between">
                                            <span className="text-gray-600">Provider:</span>
                                            <span className="font-medium">{selectedPreset.config.tier2.provider}</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-gray-600">Model:</span>
                                            <span className="font-medium">{selectedPreset.config.tier2.model}</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-gray-600">Temperature:</span>
                                            <span className="font-medium">{selectedPreset.config.tier2.temperature}</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-gray-600">Max Tokens:</span>
                                            <span className="font-medium">{selectedPreset.config.tier2.max_tokens}</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-gray-600">Timeout:</span>
                                            <span className="font-medium">{selectedPreset.config.tier2.timeout}s</span>
                                        </div>
                                    </CardContent>
                                </Card>
                            </div>

                            {/* Use Case Information */}
                            <div className="p-4 bg-gray-50 rounded-lg">
                                <h4 className="font-medium text-gray-900 mb-2">Best For:</h4>
                                <div className="text-sm text-gray-600 space-y-1">
                                    {selectedPreset.id === 'cost-optimized' && (
                                        <ul className="list-disc list-inside space-y-1">
                                            <li>Budget-conscious deployments</li>
                                            <li>Mixed cloud and local infrastructure</li>
                                            <li>High-volume internal operations</li>
                                            <li>Development and testing environments</li>
                                        </ul>
                                    )}
                                    {selectedPreset.id === 'performance-optimized' && (
                                        <ul className="list-disc list-inside space-y-1">
                                            <li>Production environments requiring high quality</li>
                                            <li>Customer-facing applications</li>
                                            <li>Complex reasoning and analysis tasks</li>
                                            <li>Real-time response requirements</li>
                                        </ul>
                                    )}
                                    {selectedPreset.id === 'privacy-first' && (
                                        <ul className="list-disc list-inside space-y-1">
                                            <li>Sensitive data processing</li>
                                            <li>Offline or air-gapped environments</li>
                                            <li>Organizations with strict data policies</li>
                                            <li>Edge computing deployments</li>
                                        </ul>
                                    )}
                                </div>
                            </div>

                            <div className="flex gap-3">
                                <Button
                                    onClick={() => handleApplyPreset(selectedPreset)}
                                    className="flex-1"
                                >
                                    Apply This Configuration
                                </Button>
                                <Button
                                    variant="outline"
                                    onClick={() => setShowDetails(false)}
                                >
                                    Cancel
                                </Button>
                            </div>
                        </div>
                    )}
                </DialogContent>
            </Dialog>
        </div>
    )
} 
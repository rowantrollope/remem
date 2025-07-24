import React, { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Button } from '@/components/ui/button'
import { Zap, Shield, Save, RotateCcw, AlertCircle, CheckCircle } from 'lucide-react'
import { useLLMConfig } from '@/hooks/useLLMConfig'
import { TierConfigForm } from './TierConfigForm'
import { ConfigurationPresets } from './ConfigurationPresets'
import type { LLMTierConfig } from '@/types'

interface LLMConfigurationPanelProps {
    className?: string
}

export function LLMConfigurationPanel({ className }: LLMConfigurationPanelProps) {
    const {
        tempConfig,
        isLoading,
        error,
        hasUnsavedChanges,
        setTempConfig,
        saveTempConfig,
        resetTempConfig,
        applyPreset
    } = useLLMConfig()

    const [activeTab, setActiveTab] = useState('tier1')
    const [isSaving, setIsSaving] = useState(false)
    const [saveSuccess, setSaveSuccess] = useState(false)

    const handleTierConfigChange = (tier: 'tier1' | 'tier2', newConfig: LLMTierConfig) => {
        if (tempConfig) {
            setTempConfig({
                ...tempConfig,
                [tier]: newConfig
            })
        }
        // Clear save success when making changes
        setSaveSuccess(false)
    }

    const handleSave = async () => {
        setIsSaving(true)
        setSaveSuccess(false)

        try {
            const success = await saveTempConfig()
            if (success) {
                setSaveSuccess(true)
                // Clear success message after 3 seconds
                setTimeout(() => setSaveSuccess(false), 3000)
            }
        } catch (err) {
            // Error is handled by the hook
        } finally {
            setIsSaving(false)
        }
    }

    const handleReset = () => {
        resetTempConfig()
        setSaveSuccess(false)
    }

    const handleApplyPreset = (presetId: string) => {
        applyPreset(presetId)
        setSaveSuccess(false)
    }

    if (isLoading) {
        return (
            <Card className={className}>
                <CardContent className="flex items-center justify-center py-12">
                    <div className="text-center">
                        <div className="w-8 h-8 border-4 border-purple-600 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
                        <p className="text-gray-600">Loading LLM configuration...</p>
                    </div>
                </CardContent>
            </Card>
        )
    }

    return (
        <div className={className}>

            {/* Status Messages */}
            {error && (
                <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg flex items-center gap-2">
                    <AlertCircle className="w-5 h-5 text-red-500" />
                    <p className="text-red-700">{error}</p>
                </div>
            )}

            {saveSuccess && (
                <div className="mb-4 p-4 bg-green-50 border border-green-200 rounded-lg flex items-center gap-2">
                    <CheckCircle className="w-5 h-5 text-green-500" />
                    <p className="text-green-700">LLM configuration saved successfully!</p>
                </div>
            )}

            {/* Configuration Tabs */}
            <Card>
                <CardHeader>
                    <CardTitle className="flex items-center justify-between">
                        <span className="flex items-center gap-2">
                            <Zap className="w-5 h-5" />
                            LLM Configuration
                        </span>

                        {/* Save/Reset Actions */}
                        <div className="flex items-center gap-2">
                            <Button
                                variant="outline"
                                size="sm"
                                onClick={handleReset}
                                disabled={!hasUnsavedChanges || isSaving}
                            >
                                <RotateCcw className="w-4 h-4 mr-1" />
                                Reset
                            </Button>
                            <Button
                                onClick={handleSave}
                                disabled={!hasUnsavedChanges || isSaving}
                                size="sm"
                            >
                                {isSaving ? (
                                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-1" />
                                ) : (
                                    <Save className="w-4 h-4 mr-1" />
                                )}
                                {isSaving ? 'Saving...' : 'Save Configuration'}
                            </Button>
                        </div>
                    </CardTitle>

                    <Tabs value={activeTab} onValueChange={setActiveTab}>
                        <TabsList className="grid w-full grid-cols-2">
                            <TabsTrigger value="tier1" className="flex items-center gap-2">
                                <Zap className="w-4 h-4" />
                                Tier 1 - Primary
                            </TabsTrigger>
                            <TabsTrigger value="tier2" className="flex items-center gap-2">
                                <Shield className="w-4 h-4" />
                                Tier 2 - Internal
                            </TabsTrigger>
                        </TabsList>
                    </Tabs>
                </CardHeader>

                <CardContent>
                    <Tabs value={activeTab} onValueChange={setActiveTab}>
                        <TabsContent value="tier1" className="mt-0">
                            <div className="mb-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                                <h4 className="font-medium text-blue-900 mb-2">Tier 1 - Primary/Conversational</h4>
                                <p className="text-sm text-blue-700">
                                    Handles main user interactions, chat responses, and primary reasoning. 
                                    This tier directly impacts user experience and response quality.
                                </p>
                            </div>
                            
                            {tempConfig && (
                                <TierConfigForm
                                    config={tempConfig.tier1}
                                    onChange={(newConfig) => handleTierConfigChange('tier1', newConfig)}
                                />
                            )}
                        </TabsContent>

                        <TabsContent value="tier2" className="mt-0">
                            <div className="mb-4 p-4 bg-green-50 border border-green-200 rounded-lg">
                                <h4 className="font-medium text-green-900 mb-2">Tier 2 - Internal/Utility</h4>
                                <p className="text-sm text-green-700">
                                    Handles internal operations like query optimization, memory extraction, and relevance filtering. 
                                    Can use a more cost-effective model for background processing.
                                </p>
                            </div>
                            
                            {tempConfig && (
                                <TierConfigForm
                                    config={tempConfig.tier2}
                                    onChange={(newConfig) => handleTierConfigChange('tier2', newConfig)}
                                />
                            )}
                        </TabsContent>
                    </Tabs>
                </CardContent>
            </Card>
        </div>
    )
} 
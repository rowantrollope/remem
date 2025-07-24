"use client"

import React, { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Switch } from "@/components/ui/switch"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Loader2, Settings, CheckCircle, AlertCircle } from "lucide-react"
import { useConfiguredAPI } from "@/hooks/useConfiguredAPI"
import type { SystemConfig, CacheConfigUpdateRequest } from "@/lib/api"

interface CacheSettingsCardProps {
    className?: string
}

export function CacheSettingsCard({ className }: CacheSettingsCardProps) {
    const [config, setConfig] = useState<SystemConfig | null>(null)
    const [isLoading, setIsLoading] = useState(true)
    const [isSaving, setIsSaving] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [success, setSuccess] = useState<string | null>(null)
    const [tempMinSimilarity, setTempMinSimilarity] = useState<string>("0.95")
    const [tempTtlMinutes, setTempTtlMinutes] = useState<string>("2")
    const { api, isLoaded: apiConfigLoaded } = useConfiguredAPI()

    // Load current configuration
    useEffect(() => {
        if (!apiConfigLoaded) return

        const loadConfig = async () => {
            try {
                setIsLoading(true)
                setError(null)
                const response = await api.getSystemConfig()
                console.log('System config response:', response)

                if (response.success && response.config) {
                    setConfig(response.config)
                    // Set temp values from config
                    if (response.config.langcache?.minimum_similarity !== undefined) {
                        setTempMinSimilarity(response.config.langcache.minimum_similarity.toString())
                    }
                    if (response.config.langcache?.ttl_minutes !== undefined) {
                        setTempTtlMinutes(response.config.langcache.ttl_minutes.toString())
                    }
                } else {
                    console.error('System config response failed or missing config:', response)
                    setError('Failed to load system configuration')
                }
            } catch (err) {
                console.error('Error loading cache config:', err)

                // If the API endpoint doesn't exist, provide default values
                if (err instanceof Error && (err.message.includes('404') || err.message.includes('Not Found'))) {
                    console.log('Cache config API not available, using default values')
                    setConfig({
                        langcache: {
                            enabled: true,
                            minimum_similarity: 0.95,
                            ttl_minutes: 2,
                            cache_types: {
                                memory_extraction: false,
                                query_optimization: true,
                                embedding_optimization: true,
                                context_analysis: false,
                                memory_grounding: true
                            }
                        }
                    } as SystemConfig)
                    setError('Cache configuration API not available. Showing default values.')
                } else {
                    setError(`Failed to load cache configuration: ${err instanceof Error ? err.message : 'Unknown error'}`)
                }
            } finally {
                setIsLoading(false)
            }
        }

        loadConfig()
    }, [api, apiConfigLoaded])

    // Clear success message after 3 seconds
    useEffect(() => {
        if (success) {
            const timer = setTimeout(() => setSuccess(null), 3000)
            return () => clearTimeout(timer)
        }
    }, [success])

    const handleMasterToggle = async (enabled: boolean) => {
        if (!config) return

        const updateRequest: CacheConfigUpdateRequest = {
            langcache: {
                enabled
            }
        }

        await updateConfig(updateRequest)
    }

    const handleCacheTypeToggle = async (cacheType: keyof NonNullable<SystemConfig['langcache']>['cache_types'], enabled: boolean) => {
        if (!config) return

        const updateRequest: CacheConfigUpdateRequest = {
            langcache: {
                cache_types: {
                    [cacheType]: enabled
                }
            }
        }

        await updateConfig(updateRequest)
    }

    const handleMinSimilarityUpdate = async () => {
        if (!config) return

        const value = parseFloat(tempMinSimilarity)
        if (isNaN(value) || value < 0.0 || value > 1.0) return

        const updateRequest: CacheConfigUpdateRequest = {
            langcache: {
                minimum_similarity: value
            }
        }

        const success = await updateConfig(updateRequest)
        if (success && config.langcache) {
            // Update temp value to match the saved value
            setTempMinSimilarity(value.toString())
        }
    }

    const handleTtlUpdate = async () => {
        if (!config) return

        const value = parseFloat(tempTtlMinutes)
        if (isNaN(value) || value < 0.1 || value > 1440) return

        const updateRequest: CacheConfigUpdateRequest = {
            langcache: {
                ttl_minutes: value
            }
        }

        const success = await updateConfig(updateRequest)
        if (success && config.langcache) {
            // Update temp value to match the saved value
            setTempTtlMinutes(value.toString())
        }
    }

    const updateConfig = async (updateRequest: CacheConfigUpdateRequest): Promise<boolean> => {
        try {
            setIsSaving(true)
            setError(null)
            setSuccess(null)

            const response = await api.updateCacheConfig(updateRequest)
            if (response.success) {
                setConfig(response.config)
                setSuccess(response.message || 'Cache configuration updated successfully')
                return true
            } else {
                setError('Failed to update cache configuration')
                return false
            }
        } catch (err) {
            console.error('Error updating cache config:', err)
            setError('Failed to update cache configuration')
            return false
        } finally {
            setIsSaving(false)
        }
    }

    if (isLoading) {
        return (
            <Card className={className}>
                <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                        <Settings className="w-5 h-5" />
                        Cache Settings
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="flex items-center justify-center py-8">
                        <Loader2 className="w-6 h-6 animate-spin" />
                        <span className="ml-2">Loading cache configuration...</span>
                    </div>
                </CardContent>
            </Card>
        )
    }

    if (!config) {
        return (
            <Card className={className}>
                <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                        <Settings className="w-5 h-5" />
                        Cache Settings
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    <Alert>
                        <AlertCircle className="w-4 h-4" />
                        <AlertDescription>
                            {error || "Unable to load system configuration. Please try refreshing the page."}
                        </AlertDescription>
                    </Alert>
                </CardContent>
            </Card>
        )
    }

    // If langcache is not available in the config, show a message
    if (!config.langcache) {
        return (
            <Card className={className}>
                <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                        <Settings className="w-5 h-5" />
                        Cache Settings
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    <Alert>
                        <AlertCircle className="w-4 h-4" />
                        <AlertDescription>
                            Cache configuration is not available in the current server version.
                            The unified cache configuration API has not been implemented yet.
                        </AlertDescription>
                    </Alert>
                    <div className="mt-4 p-4 bg-gray-50 rounded-lg">
                        <h4 className="font-medium text-gray-900 mb-2">Expected Cache Configuration:</h4>
                        <div className="text-sm text-gray-600 space-y-1">
                            <div>• Memory Extraction: Disabled</div>
                            <div>• Query Optimization: Enabled</div>
                            <div>• Embedding Optimization: Enabled</div>
                            <div>• Context Analysis: Disabled</div>
                            <div>• Memory Grounding: Enabled</div>
                        </div>
                        <p className="text-xs text-gray-500 mt-2">
                            These settings will be configurable once the server implements the unified cache configuration API.
                        </p>
                    </div>
                </CardContent>
            </Card>
        )
    }

    return (
        <Card className={className}>
            <CardHeader>
                <CardTitle className="flex items-center gap-2">
                    <Settings className="w-5 h-5" />
                    Cache Settings
                </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
                {/* Success/Error Messages */}
                {success && (
                    <Alert>
                        <CheckCircle className="w-4 h-4" />
                        <AlertDescription className="text-green-700">
                            {success}
                        </AlertDescription>
                    </Alert>
                )}

                {error && (
                    <Alert variant="destructive">
                        <AlertCircle className="w-4 h-4" />
                        <AlertDescription>
                            {error}
                        </AlertDescription>
                    </Alert>
                )}

                {/* Master Toggle */}
                <div className="flex items-center justify-between p-4 border rounded-lg bg-gray-50">
                    <div className="space-y-1">
                        <Label className="text-base font-medium">Enable LangCache</Label>
                        <p className="text-sm text-gray-600">
                            Master toggle for all caching functionality
                        </p>
                    </div>
                    <Switch
                        checked={config?.langcache?.enabled || false}
                        onCheckedChange={handleMasterToggle}
                        disabled={isSaving}
                    />
                </div>

                {/* Minimum Similarity Setting */}
                <div className="p-4 border rounded-lg">
                    <div className="space-y-3">
                        <div className="space-y-1">
                            <Label className="text-base font-medium">Minimum Similarity Threshold</Label>
                            <p className="text-sm text-gray-600">
                                Minimum similarity score (0.0-1.0) required for cache hits. Higher values require more precise matches.
                            </p>
                        </div>
                        <div className="flex items-center gap-3">
                            <Input
                                type="number"
                                min="0.0"
                                max="1.0"
                                step="0.01"
                                value={tempMinSimilarity}
                                onChange={(e) => setTempMinSimilarity(e.target.value)}
                                className="w-24"
                                disabled={isSaving || !config?.langcache?.enabled}
                            />
                            <Button
                                onClick={handleMinSimilarityUpdate}
                                size="sm"
                                disabled={
                                    isSaving ||
                                    !config?.langcache?.enabled ||
                                    tempMinSimilarity === (config?.langcache?.minimum_similarity?.toString() || "0.95") ||
                                    parseFloat(tempMinSimilarity) < 0.0 ||
                                    parseFloat(tempMinSimilarity) > 1.0 ||
                                    isNaN(parseFloat(tempMinSimilarity))
                                }
                            >
                                Save
                            </Button>
                        </div>
                        <p className="text-xs text-gray-600">
                            Current setting:{" "}
                            <span className="font-mono font-medium">
                                {config?.langcache?.minimum_similarity || 0.95}
                            </span>
                        </p>
                    </div>
                </div>

                {/* TTL Setting */}
                <div className="p-4 border rounded-lg">
                    <div className="space-y-3">
                        <div className="space-y-1">
                            <Label className="text-base font-medium">Cache Entry TTL (Time To Live)</Label>
                            <p className="text-sm text-gray-600">
                                How long cache entries remain valid (in minutes). Range: 0.1 to 1440 minutes (24 hours).
                            </p>
                        </div>
                        <div className="flex items-center gap-3">
                            <Input
                                type="number"
                                min="0.1"
                                max="1440"
                                step="0.1"
                                value={tempTtlMinutes}
                                onChange={(e) => setTempTtlMinutes(e.target.value)}
                                className="w-24"
                                disabled={isSaving || !config?.langcache?.enabled}
                            />
                            <span className="text-sm text-gray-600">minutes</span>
                            <Button
                                onClick={handleTtlUpdate}
                                size="sm"
                                disabled={
                                    isSaving ||
                                    !config?.langcache?.enabled ||
                                    tempTtlMinutes === (config?.langcache?.ttl_minutes?.toString() || "2") ||
                                    parseFloat(tempTtlMinutes) < 0.1 ||
                                    parseFloat(tempTtlMinutes) > 1440 ||
                                    isNaN(parseFloat(tempTtlMinutes))
                                }
                            >
                                Save
                            </Button>
                        </div>
                        <p className="text-xs text-gray-600">
                            Current setting:{" "}
                            <span className="font-mono font-medium">
                                {config?.langcache?.ttl_minutes || 2} minutes
                            </span>
                        </p>
                    </div>
                </div>

                {/* Individual Cache Type Toggles */}
                <div className="space-y-4">
                    <h3 className="text-lg font-medium">Cache Types</h3>
                    
                    <div className="space-y-3">
                        <div className="flex items-center justify-between">
                            <div className="space-y-1">
                                <Label className="text-sm font-medium">Memory Extraction</Label>
                                <p className="text-xs text-gray-500">
                                    Cache memory extraction operations
                                </p>
                            </div>
                            <Switch
                                checked={config?.langcache?.cache_types?.memory_extraction || false}
                                onCheckedChange={(checked) => handleCacheTypeToggle('memory_extraction', checked)}
                                disabled={isSaving || !config?.langcache?.enabled}
                            />
                        </div>

                        <div className="flex items-center justify-between">
                            <div className="space-y-1">
                                <Label className="text-sm font-medium">Query Optimization</Label>
                                <p className="text-xs text-gray-500">
                                    Cache query optimization results
                                </p>
                            </div>
                            <Switch
                                checked={config?.langcache?.cache_types?.query_optimization || false}
                                onCheckedChange={(checked) => handleCacheTypeToggle('query_optimization', checked)}
                                disabled={isSaving || !config?.langcache?.enabled}
                            />
                        </div>

                        <div className="flex items-center justify-between">
                            <div className="space-y-1">
                                <Label className="text-sm font-medium">Embedding Optimization</Label>
                                <p className="text-xs text-gray-500">
                                    Cache embedding generation and optimization
                                </p>
                            </div>
                            <Switch
                                checked={config?.langcache?.cache_types?.embedding_optimization || false}
                                onCheckedChange={(checked) => handleCacheTypeToggle('embedding_optimization', checked)}
                                disabled={isSaving || !config?.langcache?.enabled}
                            />
                        </div>

                        <div className="flex items-center justify-between">
                            <div className="space-y-1">
                                <Label className="text-sm font-medium">Context Analysis</Label>
                                <p className="text-xs text-gray-500">
                                    Cache context analysis operations
                                </p>
                            </div>
                            <Switch
                                checked={config?.langcache?.cache_types?.context_analysis || false}
                                onCheckedChange={(checked) => handleCacheTypeToggle('context_analysis', checked)}
                                disabled={isSaving || !config?.langcache?.enabled}
                            />
                        </div>

                        <div className="flex items-center justify-between">
                            <div className="space-y-1">
                                <Label className="text-sm font-medium">Memory Grounding</Label>
                                <p className="text-xs text-gray-500">
                                    Cache memory grounding operations
                                </p>
                            </div>
                            <Switch
                                checked={config?.langcache?.cache_types?.memory_grounding || false}
                                onCheckedChange={(checked) => handleCacheTypeToggle('memory_grounding', checked)}
                                disabled={isSaving || !config?.langcache?.enabled}
                            />
                        </div>
                    </div>
                </div>

                {/* Loading indicator when saving */}
                {isSaving && (
                    <div className="flex items-center justify-center py-2">
                        <Loader2 className="w-4 h-4 animate-spin mr-2" />
                        <span className="text-sm text-gray-600">Updating configuration...</span>
                    </div>
                )}
            </CardContent>
        </Card>
    )
}

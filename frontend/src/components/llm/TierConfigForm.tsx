import React, { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { Badge } from '@/components/ui/badge'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { AlertCircle, CheckCircle, Eye, EyeOff, HelpCircle, TestTube, Zap } from 'lucide-react'
import { ProviderSelector } from './ProviderSelector'
import { ModelSelector } from './ModelSelector'
import type { LLMTierConfig } from '@/types'

interface TierConfigFormProps {
    config: LLMTierConfig
    onChange: (config: LLMTierConfig) => void
    className?: string
}

export function TierConfigForm({ config, onChange, className }: TierConfigFormProps) {
    const [showApiKey, setShowApiKey] = useState(false)
    const [validationErrors, setValidationErrors] = useState<Record<string, string>>({})

    const validateField = (field: string, value: any): string | null => {
        switch (field) {
            case 'temperature':
                if (value < 0 || value > 2) return 'Temperature must be between 0.0 and 2.0'
                break
            case 'max_tokens':
                if (value < 1 || value > 8000) return 'Max tokens must be between 1 and 8000'
                break
            case 'timeout':
                if (value < 1 || value > 300) return 'Timeout must be between 1 and 300 seconds'
                break
            case 'api_key':
                if (config.provider === 'openai' && (!value || !value.startsWith('sk-'))) {
                    return 'OpenAI API key must start with "sk-"'
                }
                break
            case 'base_url':
                if (config.provider === 'ollama' && !value) {
                    return 'Base URL is required for Ollama'
                }
                if (value && !value.match(/^https?:\/\/.+/)) {
                    return 'Base URL must be a valid HTTP/HTTPS URL'
                }
                break
            case 'model':
                if (!value || value.trim() === '') return 'Model is required'
                break
        }
        return null
    }

    const handleFieldChange = (field: keyof LLMTierConfig, value: any) => {
        const error = validateField(field, value)
        setValidationErrors(prev => ({
            ...prev,
            [field]: error || ''
        }))

        onChange({
            ...config,
            [field]: value
        })
    }

    const getFieldError = (field: string) => validationErrors[field]

    return (
        <div className={className}>
            <div className="space-y-6">
                {/* Provider Selection */}
                <div className="space-y-2">
                    <Label className="text-base font-medium">Provider</Label>
                    <ProviderSelector
                        value={config.provider}
                        onChange={(provider) => {
                            // Reset provider-specific fields when switching
                            const baseConfig = {
                                ...config,
                                provider,
                                model: '', // Reset model when switching provider
                            }
                            
                            // Set defaults based on provider
                            if (provider === 'ollama') {
                                baseConfig.base_url = baseConfig.base_url || 'http://localhost:11434'
                                baseConfig.api_key = undefined
                            } else {
                                baseConfig.base_url = undefined
                            }
                            
                            onChange(baseConfig)
                        }}
                    />
                </div>

                {/* Model Selection */}
                <div className="space-y-2">
                    <Label className="text-base font-medium">Model</Label>
                    <ModelSelector
                        provider={config.provider}
                        value={config.model}
                        onChange={(model) => handleFieldChange('model', model)}
                        error={getFieldError('model')}
                        baseUrl={config.provider === 'ollama' ? config.base_url : undefined}
                    />
                </div>

                {/* Provider-specific Configuration */}
                {config.provider === 'openai' && (
                    <div className="space-y-2">
                        <Label htmlFor="api-key" className="text-base font-medium">
                            OpenAI API Key
                        </Label>
                        <div className="relative">
                            <Input
                                id="api-key"
                                type={showApiKey ? 'text' : 'password'}
                                value={config.api_key || ''}
                                onChange={(e) => handleFieldChange('api_key', e.target.value)}
                                placeholder="sk-..."
                                className={getFieldError('api_key') ? 'border-red-500' : ''}
                            />
                            <Button
                                type="button"
                                variant="ghost"
                                size="sm"
                                className="absolute right-0 top-0 h-full px-3"
                                onClick={() => setShowApiKey(!showApiKey)}
                            >
                                {showApiKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                            </Button>
                        </div>
                        {getFieldError('api_key') && (
                            <p className="text-sm text-red-600 flex items-center gap-1">
                                <AlertCircle className="w-4 h-4" />
                                {getFieldError('api_key')}
                            </p>
                        )}
                    </div>
                )}

                {config.provider === 'ollama' && (
                    <div className="space-y-2">
                        <Label htmlFor="base-url" className="text-base font-medium">
                            Ollama Base URL
                        </Label>
                        <Input
                            id="base-url"
                            type="url"
                            value={config.base_url || ''}
                            onChange={(e) => handleFieldChange('base_url', e.target.value)}
                            placeholder="http://localhost:11434"
                            className={getFieldError('base_url') ? 'border-red-500' : ''}
                        />
                        {getFieldError('base_url') && (
                            <p className="text-sm text-red-600 flex items-center gap-1">
                                <AlertCircle className="w-4 h-4" />
                                {getFieldError('base_url')}
                            </p>
                        )}
                    </div>
                )}

                {/* Temperature */}
                <div className="space-y-3">
                    <div className="flex items-center gap-2">
                        <Label className="text-base font-medium">Temperature</Label>
                        <TooltipProvider>
                            <Tooltip>
                                <TooltipTrigger>
                                    <HelpCircle className="w-4 h-4 text-gray-400" />
                                </TooltipTrigger>
                                <TooltipContent>
                                    <p className="max-w-xs">
                                        Controls randomness. Lower values (0.1-0.3) for focused, consistent responses. 
                                        Higher values (0.7-1.0) for creative, varied responses.
                                    </p>
                                </TooltipContent>
                            </Tooltip>
                        </TooltipProvider>
                        <Badge variant="outline" className="ml-auto">
                            {config.temperature.toFixed(1)}
                        </Badge>
                    </div>
                    <Slider
                        value={[config.temperature]}
                        onValueChange={([value]) => handleFieldChange('temperature', value)}
                        min={0}
                        max={2}
                        step={0.1}
                        className="w-full"
                    />
                    <div className="flex justify-between text-xs text-gray-500">
                        <span>Conservative (0.0)</span>
                        <span>Balanced (1.0)</span>
                        <span>Creative (2.0)</span>
                    </div>
                    {getFieldError('temperature') && (
                        <p className="text-sm text-red-600 flex items-center gap-1">
                            <AlertCircle className="w-4 h-4" />
                            {getFieldError('temperature')}
                        </p>
                    )}
                </div>

                {/* Max Tokens */}
                <div className="space-y-2">
                    <div className="flex items-center gap-2">
                        <Label htmlFor="max-tokens" className="text-base font-medium">
                            Max Tokens
                        </Label>
                        <TooltipProvider>
                            <Tooltip>
                                <TooltipTrigger>
                                    <HelpCircle className="w-4 h-4 text-gray-400" />
                                </TooltipTrigger>
                                <TooltipContent>
                                    <p className="max-w-xs">
                                        Maximum number of tokens in the response. Higher values allow longer responses 
                                        but increase cost and latency.
                                    </p>
                                </TooltipContent>
                            </Tooltip>
                        </TooltipProvider>
                    </div>
                    <Input
                        id="max-tokens"
                        type="number"
                        min="1"
                        max="8000"
                        value={config.max_tokens}
                        onChange={(e) => handleFieldChange('max_tokens', parseInt(e.target.value) || 1)}
                        className={getFieldError('max_tokens') ? 'border-red-500' : ''}
                    />
                    {getFieldError('max_tokens') && (
                        <p className="text-sm text-red-600 flex items-center gap-1">
                            <AlertCircle className="w-4 h-4" />
                            {getFieldError('max_tokens')}
                        </p>
                    )}
                </div>

                {/* Timeout */}
                <div className="space-y-2">
                    <div className="flex items-center gap-2">
                        <Label htmlFor="timeout" className="text-base font-medium">
                            Timeout (seconds)
                        </Label>
                        <TooltipProvider>
                            <Tooltip>
                                <TooltipTrigger>
                                    <HelpCircle className="w-4 h-4 text-gray-400" />
                                </TooltipTrigger>
                                <TooltipContent>
                                    <p className="max-w-xs">
                                        How long to wait for a response before timing out. 
                                        Larger models may need longer timeouts.
                                    </p>
                                </TooltipContent>
                            </Tooltip>
                        </TooltipProvider>
                    </div>
                    <Input
                        id="timeout"
                        type="number"
                        min="1"
                        max="300"
                        value={config.timeout}
                        onChange={(e) => handleFieldChange('timeout', parseInt(e.target.value) || 1)}
                        className={getFieldError('timeout') ? 'border-red-500' : ''}
                    />
                    {getFieldError('timeout') && (
                        <p className="text-sm text-red-600 flex items-center gap-1">
                            <AlertCircle className="w-4 h-4" />
                            {getFieldError('timeout')}
                        </p>
                    )}
                </div>
            </div>
        </div>
    )
} 
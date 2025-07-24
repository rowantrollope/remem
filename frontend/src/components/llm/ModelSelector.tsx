import React, { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import { AlertCircle, Brain, Server, Star, Zap, DollarSign, Clock, RefreshCw, Loader2 } from 'lucide-react'
import { POPULAR_MODELS, type OllamaModel } from '@/types'
import { useLLMConfig } from '@/hooks/useLLMConfig'

interface ModelSelectorProps {
    provider: 'openai' | 'ollama'
    value: string
    onChange: (model: string) => void
    error?: string
    className?: string
    baseUrl?: string // For Ollama instances
}

export function ModelSelector({ provider, value, onChange, error, className, baseUrl }: ModelSelectorProps) {
    const [inputMode, setInputMode] = useState<'dropdown' | 'custom'>('dropdown')
    const [customModel, setCustomModel] = useState('')
    const [ollamaModels, setOllamaModels] = useState<OllamaModel[]>([])
    const [isLoadingModels, setIsLoadingModels] = useState(false)
    const [modelLoadError, setModelLoadError] = useState<string | null>(null)

    const { getOllamaModels } = useLLMConfig()

    const models = provider === 'openai' ? POPULAR_MODELS[provider] : ollamaModels.map(m => m.name)

    // Load Ollama models when provider changes to Ollama or baseUrl changes
    useEffect(() => {
        if (provider === 'ollama') {
            loadOllamaModels()
        }
    }, [provider, baseUrl])

    const loadOllamaModels = async () => {
        setIsLoadingModels(true)
        setModelLoadError(null)
        
        try {
            const response = await getOllamaModels(baseUrl)
            if (response.success) {
                setOllamaModels(response.models)
                // If current value is not in the list, clear it
                if (value && !response.models.some(m => m.name === value)) {
                    onChange('')
                }
            } else {
                setModelLoadError(response.message || 'Failed to load Ollama models')
                setOllamaModels([])
            }
        } catch (error) {
            console.error('Error loading Ollama models:', error)
            setModelLoadError(error instanceof Error ? error.message : 'Failed to load Ollama models')
            setOllamaModels([])
        } finally {
            setIsLoadingModels(false)
        }
    }

    // Model information and recommendations for OpenAI
    const openaiModelInfo = {
        'gpt-4': { 
            description: 'Most capable model, best for complex tasks',
            tags: ['High Quality', 'Expensive', 'Slower'],
            recommended: ['Primary Tier', 'Complex Reasoning']
        },
        'gpt-4-turbo': { 
            description: 'Faster GPT-4 with lower costs',
            tags: ['Balanced', 'Fast', 'Cost-Effective'],
            recommended: ['Primary Tier', 'Most Use Cases']
        },
        'gpt-3.5-turbo': { 
            description: 'Fast and affordable, good for most tasks',
            tags: ['Fast', 'Affordable', 'Good Quality'],
            recommended: ['Both Tiers', 'Budget Friendly']
        },
        'gpt-3.5-turbo-16k': { 
            description: 'Extended context version of GPT-3.5',
            tags: ['Long Context', 'Affordable'],
            recommended: ['Long Documents']
        }
    }

    const handleModeChange = (mode: 'dropdown' | 'custom') => {
        setInputMode(mode)
        if (mode === 'dropdown') {
            // If switching back to dropdown and current value is not in available models,
            // select the first available model
            if (!models.includes(value)) {
                onChange(models[0] || '')
            }
        } else {
            setCustomModel(value)
        }
    }

    const handleCustomModelChange = (model: string) => {
        setCustomModel(model)
        onChange(model)
    }

    const getOpenAIModelInfo = (modelName: string) => {
        return openaiModelInfo[modelName as keyof typeof openaiModelInfo]
    }

    const getOllamaModelInfo = (modelName: string) => {
        return ollamaModels.find(m => m.name === modelName)
    }

    const formatBytes = (bytes: number) => {
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB']
        if (bytes === 0) return '0 Bytes'
        const i = Math.floor(Math.log(bytes) / Math.log(1024))
        return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i]
    }

    return (
        <div className={`space-y-4 ${className}`}>
            {/* Mode Toggle */}
            <div className="flex items-center gap-2 bg-gray-50 p-1 rounded-lg">
                <Button
                    variant={inputMode === 'dropdown' ? 'default' : 'ghost'}
                    size="sm"
                    onClick={() => handleModeChange('dropdown')}
                    className="flex-1"
                    disabled={provider === 'ollama' && isLoadingModels}
                >
                    <Star className="w-4 h-4 mr-1" />
                    {provider === 'openai' ? 'Popular Models' : 'Available Models'}
                </Button>
                <Button
                    variant={inputMode === 'custom' ? 'default' : 'ghost'}
                    size="sm"
                    onClick={() => handleModeChange('custom')}
                    className="flex-1"
                >
                    Custom Model
                </Button>
                {provider === 'ollama' && (
                    <Button
                        variant="ghost"
                        size="sm"
                        onClick={loadOllamaModels}
                        disabled={isLoadingModels}
                        className="px-2"
                    >
                        {isLoadingModels ? (
                            <Loader2 className="w-4 h-4 animate-spin" />
                        ) : (
                            <RefreshCw className="w-4 h-4" />
                        )}
                    </Button>
                )}
            </div>

            {/* Model Selection */}
            {inputMode === 'dropdown' ? (
                <div className="space-y-3">
                    {/* Loading State for Ollama */}
                    {provider === 'ollama' && isLoadingModels && (
                        <div className="flex items-center justify-center py-8">
                            <div className="text-center">
                                <Loader2 className="w-6 h-6 animate-spin mx-auto mb-2 text-gray-400" />
                                <p className="text-sm text-gray-600">Loading available models...</p>
                            </div>
                        </div>
                    )}

                    {/* Error State for Ollama */}
                    {provider === 'ollama' && modelLoadError && (
                        <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                            <div className="flex items-center gap-2 text-red-700 mb-2">
                                <AlertCircle className="w-4 h-4" />
                                <span className="font-medium">Failed to load models</span>
                            </div>
                            <p className="text-sm text-red-600 mb-3">{modelLoadError}</p>
                            <div className="space-y-1 text-xs text-red-600">
                                <p><strong>Troubleshooting:</strong></p>
                                <ul className="list-disc list-inside space-y-1 ml-2">
                                    <li>Ensure Ollama is running at the specified URL</li>
                                    <li>Check that the base URL is correct</li>
                                    <li>Verify network connectivity to the Ollama instance</li>
                                    <li>Try refreshing the model list</li>
                                </ul>
                            </div>
                        </div>
                    )}

                    {/* No Models Available */}
                    {provider === 'ollama' && !isLoadingModels && !modelLoadError && models.length === 0 && (
                        <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                            <div className="flex items-center gap-2 text-yellow-700 mb-2">
                                <AlertCircle className="w-4 h-4" />
                                <span className="font-medium">No models available</span>
                            </div>
                            <p className="text-sm text-yellow-600 mb-2">No models found on this Ollama instance.</p>
                            <div className="space-y-1 text-xs text-yellow-600">
                                <p><strong>To install models:</strong></p>
                                <ul className="list-disc list-inside space-y-1 ml-2">
                                    <li>Run: <code className="bg-yellow-100 px-1 rounded">ollama pull llama2</code></li>
                                    <li>Or: <code className="bg-yellow-100 px-1 rounded">ollama pull mistral</code></li>
                                    <li>Then refresh this list</li>
                                </ul>
                            </div>
                        </div>
                    )}

                    {/* Model Dropdown */}
                    {(!isLoadingModels || provider === 'openai') && models.length > 0 && (
                        <>
                            <Select value={value} onValueChange={onChange}>
                                <SelectTrigger className={error ? 'border-red-500' : ''}>
                                    <SelectValue placeholder={`Select a ${provider} model`} />
                                </SelectTrigger>
                                <SelectContent>
                                    {models.map((model) => (
                                        <SelectItem key={model} value={model}>
                                            <div className="flex items-center gap-2">
                                                {provider === 'openai' ? (
                                                    <Brain className="w-4 h-4" />
                                                ) : (
                                                    <Server className="w-4 h-4" />
                                                )}
                                                {model}
                                                {provider === 'ollama' && (() => {
                                                    const modelInfo = getOllamaModelInfo(model)
                                                    return modelInfo ? (
                                                        <span className="text-xs text-gray-500 ml-2">
                                                            ({formatBytes(modelInfo.size)})
                                                        </span>
                                                    ) : null
                                                })()}
                                            </div>
                                        </SelectItem>
                                    ))}
                                </SelectContent>
                            </Select>

                            {/* Model Information */}
                            {value && provider === 'openai' && getOpenAIModelInfo(value) && (
                                <Card className="bg-blue-50 border-blue-200">
                                    <CardContent className="p-4">
                                        <div className="space-y-3">
                                            <div>
                                                <h4 className="font-medium text-blue-900 mb-1">{value}</h4>
                                                <p className="text-sm text-blue-700">
                                                    {getOpenAIModelInfo(value)?.description}
                                                </p>
                                            </div>

                                            <div className="flex flex-wrap gap-2">
                                                {getOpenAIModelInfo(value)?.tags.map((tag, index) => (
                                                    <Badge key={index} variant="secondary" className="text-xs bg-blue-100 text-blue-700">
                                                        {tag}
                                                    </Badge>
                                                ))}
                                            </div>

                                            <div>
                                                <p className="text-xs text-blue-600 font-medium mb-1">Recommended for:</p>
                                                <div className="flex flex-wrap gap-1">
                                                    {getOpenAIModelInfo(value)?.recommended.map((rec, index) => (
                                                        <span key={index} className="text-xs text-blue-700 bg-blue-100/50 px-2 py-1 rounded">
                                                            {rec}
                                                        </span>
                                                    ))}
                                                </div>
                                            </div>
                                        </div>
                                    </CardContent>
                                </Card>
                            )}

                            {/* Ollama Model Information */}
                            {value && provider === 'ollama' && (() => {
                                const modelInfo = getOllamaModelInfo(value)
                                return modelInfo ? (
                                    <Card className="bg-green-50 border-green-200">
                                        <CardContent className="p-4">
                                            <div className="space-y-3">
                                                <div>
                                                    <h4 className="font-medium text-green-900 mb-1">{value}</h4>
                                                    <p className="text-sm text-green-700">
                                                        Local Ollama model • {formatBytes(modelInfo.size)}
                                                    </p>
                                                </div>

                                                <div className="flex flex-wrap gap-2">
                                                    <Badge variant="secondary" className="text-xs bg-green-100 text-green-700">
                                                        Local
                                                    </Badge>
                                                    <Badge variant="secondary" className="text-xs bg-green-100 text-green-700">
                                                        {formatBytes(modelInfo.size)}
                                                    </Badge>
                                                    {modelInfo.details?.parameter_size && (
                                                        <Badge variant="secondary" className="text-xs bg-green-100 text-green-700">
                                                            {modelInfo.details.parameter_size}
                                                        </Badge>
                                                    )}
                                                </div>

                                                {modelInfo.details && (
                                                    <div className="text-xs text-green-600 space-y-1">
                                                        {modelInfo.details.family && (
                                                            <div>Family: {modelInfo.details.family}</div>
                                                        )}
                                                        {modelInfo.details.format && (
                                                            <div>Format: {modelInfo.details.format}</div>
                                                        )}
                                                        <div>Modified: {new Date(modelInfo.modified_at).toLocaleDateString()}</div>
                                                    </div>
                                                )}
                                            </div>
                                        </CardContent>
                                    </Card>
                                ) : null
                            })()}
                        </>
                    )}
                </div>
            ) : (
                <div className="space-y-2">
                    <Input
                        type="text"
                        value={customModel}
                        onChange={(e) => handleCustomModelChange(e.target.value)}
                        placeholder={provider === 'openai' ? 'e.g., gpt-4, gpt-3.5-turbo' : 'e.g., llama2:7b, mistral:latest'}
                        className={error ? 'border-red-500' : ''}
                    />
                    <p className="text-xs text-gray-600">
                        Enter the exact model name. For {provider === 'openai' ? 'OpenAI' : 'Ollama'}, make sure the model is {provider === 'openai' ? 'available in your account' : 'installed locally'}.
                    </p>
                </div>
            )}

            {/* Error Display */}
            {error && (
                <div className="flex items-center gap-2 text-sm text-red-600">
                    <AlertCircle className="w-4 h-4" />
                    {error}
                </div>
            )}

            {/* Provider-specific Help */}
            <div className="p-3 bg-gray-50 rounded-lg">
                <p className="text-xs text-gray-600">
                    <strong>{provider === 'openai' ? 'OpenAI' : 'Ollama'} Models:</strong> {' '}
                    {provider === 'openai' 
                        ? 'Use your OpenAI account models. Pricing varies by model. GPT-4 offers best quality but higher costs.'
                        : 'Models must be installed locally. Use "ollama pull <model-name>" to install new models, then refresh the list.'
                    }
                </p>
            </div>
        </div>
    )
} 
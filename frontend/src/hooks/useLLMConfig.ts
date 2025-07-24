import { useState, useEffect, useCallback } from 'react'
import { useConfiguredAPI } from './useConfiguredAPI'
import type { 
    LLMConfig, 
    LLMTierConfig, 
    LLM_PRESETS,
    OllamaModelsResponse
} from '@/types'

interface UseLLMConfigReturn {
    // State
    config: LLMConfig | null
    isLoading: boolean
    error: string | null
    hasUnsavedChanges: boolean
    
    // Actions
    loadConfig: () => Promise<void>
    updateConfig: (newConfig: LLMConfig) => Promise<boolean>
    resetToDefaults: () => void
    applyPreset: (presetId: string) => void
    getOllamaModels: (baseUrl?: string) => Promise<OllamaModelsResponse>
    
    // Temporary state for form editing
    tempConfig: LLMConfig | null
    setTempConfig: (config: LLMConfig) => void
    resetTempConfig: () => void
    saveTempConfig: () => Promise<boolean>
}

export function useLLMConfig(): UseLLMConfigReturn {
    const { api, isLoaded: apiConfigLoaded } = useConfiguredAPI()
    const [config, setConfig] = useState<LLMConfig | null>(null)
    const [tempConfig, setTempConfig] = useState<LLMConfig | null>(null)
    const [isLoading, setIsLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)

    // Default configuration
    const defaultConfig: LLMConfig = {
        tier1: {
            provider: 'openai',
            model: 'gpt-3.5-turbo',
            temperature: 0.7,
            max_tokens: 2000,
            base_url: null,  // null for OpenAI
            api_key: '',     // Empty string for default
            timeout: 30
        },
        tier2: {
            provider: 'openai',
            model: 'gpt-3.5-turbo',
            temperature: 0.3,
            max_tokens: 1000,
            base_url: null,  // null for OpenAI
            api_key: '',     // Empty string for default
            timeout: 30
        }
    }

    // Check if there are unsaved changes
    const hasUnsavedChanges = tempConfig !== null && 
        JSON.stringify(tempConfig) !== JSON.stringify(config)

    // Load configuration from API
    const loadConfig = useCallback(async () => {
        try {
            setIsLoading(true)
            setError(null)
            
            const response = await api.getLLMConfig()
            if (response.success) {
                setConfig(response.llm_config)
                setTempConfig(response.llm_config)
            } else {
                // If no config exists, use default
                setConfig(defaultConfig)
                setTempConfig(defaultConfig)
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load LLM configuration')
            // Use default config on error
            setConfig(defaultConfig)
            setTempConfig(defaultConfig)
        } finally {
            setIsLoading(false)
        }
    }, [api])

    // Update configuration
    const updateConfig = useCallback(async (newConfig: LLMConfig): Promise<boolean> => {
        try {
            setIsLoading(true)
            setError(null)
            
            // Backend expects direct tier objects, not nested in "config"
            const response = await api.updateLLMConfig({
                tier1: newConfig.tier1,
                tier2: newConfig.tier2
            })
            if (response.success) {
                setConfig(newConfig)
                setTempConfig(newConfig)
                
                // Show restart message if needed
                if (response.requires_restart) {
                    setError(`Configuration updated successfully. ${response.message}`)
                }
                
                return true
            } else {
                setError(response.message || 'Failed to update configuration')
                return false
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to update configuration')
            return false
        } finally {
            setIsLoading(false)
        }
    }, [api])

    // Reset to defaults
    const resetToDefaults = useCallback(() => {
        setTempConfig(defaultConfig)
    }, [])

    // Apply a preset
    const applyPreset = useCallback((presetId: string) => {
        const preset = LLM_PRESETS.find(p => p.id === presetId)
        if (preset) {
            setTempConfig(preset.config)
        }
    }, [])

    // Reset temp config to current saved config
    const resetTempConfig = useCallback(() => {
        setTempConfig(config)
    }, [config])

    // Save temp config
    const saveTempConfig = useCallback(async (): Promise<boolean> => {
        if (tempConfig) {
            return await updateConfig(tempConfig)
        }
        return false
    }, [tempConfig, updateConfig])

    // Load config on mount
    useEffect(() => {
        // Don't try to load until API configuration is loaded
        if (!apiConfigLoaded) {
            return
        }
        loadConfig()
    }, [loadConfig, apiConfigLoaded])

    // Get Ollama models
    const getOllamaModels = useCallback(async (baseUrl?: string): Promise<OllamaModelsResponse> => {
        try {
            console.log('Fetching Ollama models from:', baseUrl || 'default URL')
            return await api.getOllamaModels(baseUrl)
        } catch (err) {
            console.error('Error in getOllamaModels:', err)
            
            // Check if this is a network error or API not found
            const errorMessage = err instanceof Error ? err.message : 'Failed to get Ollama models'
            
            return {
                success: false,
                models: [],
                message: errorMessage
            }
        }
    }, [api])

    return {
        config,
        isLoading,
        error,
        hasUnsavedChanges,
        loadConfig,
        updateConfig,
        resetToDefaults,
        applyPreset,
        getOllamaModels,
        tempConfig,
        setTempConfig,
        resetTempConfig,
        saveTempConfig
    }
} 
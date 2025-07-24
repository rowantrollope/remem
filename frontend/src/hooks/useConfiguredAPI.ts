import { useMemo } from 'react'
import { MemoryAgentAPI } from '@/lib/api'
import { useSettings } from './useSettings'

export function useConfiguredAPI() {
    const { isLoaded, getApiBaseUrl, settings } = useSettings()

    // Create a new API instance when settings change
    // This ensures that vectorset changes are properly reflected
    const api = useMemo(() => {
        const baseUrl = isLoaded ? getApiBaseUrl() : 'http://localhost:5001'
        const vectorSetName = isLoaded ? settings.vectorSetName : 'memories'

        console.log('useConfiguredAPI: Creating API instance with:', { baseUrl, vectorSetName })
        return new MemoryAgentAPI(baseUrl, vectorSetName)
    }, [isLoaded, getApiBaseUrl, settings.vectorSetName])

    return {
        api,
        isLoaded,
        baseUrl: isLoaded ? getApiBaseUrl() : 'http://localhost:5001',
        vectorSetName: isLoaded ? settings.vectorSetName : 'memories'
    }
}

// Export a function to get a default API instance
// This can be used in places where hooks can't be used
// Note: This will use default settings and won't reflect user preferences
export function getConfiguredAPI(): MemoryAgentAPI {
    return new MemoryAgentAPI('http://localhost:5001', 'memories')
}

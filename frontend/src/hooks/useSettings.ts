import { useState, useEffect, useCallback } from 'react'

// Remove custom event system to avoid circular updates

export interface UserSettings {
    questionTopK: number // Number of memories to use when answering questions
    minSimilarity: number // Minimum similarity threshold for memory retrieval (0.0 to 1.0)
    serverUrl: string // Base URL for the REMEM server
    serverPort: number // Port for the REMEM server
    vectorSetName: string // Name of the vectorset to use for memory operations
}

const DEFAULT_SETTINGS: UserSettings = {
    questionTopK: 10, // Default value matching the API default
    minSimilarity: 0.7, // Default minimum similarity threshold
    serverUrl: 'http://localhost', // Default server URL
    serverPort: 5001, // Default server port
    vectorSetName: 'memories' // Default vectorset name
}

const SETTINGS_STORAGE_KEY = 'memory-app-settings'
const SETTINGS_VERSION = '1.0'

export function useSettings() {
    const [settings, setSettings] = useState<UserSettings>(DEFAULT_SETTINGS)
    const [isLoaded, setIsLoaded] = useState(false)

    // Load settings from localStorage on initialization
    useEffect(() => {
        console.log('useSettings: Loading settings from localStorage')
        try {
            const savedData = localStorage.getItem(SETTINGS_STORAGE_KEY)
            console.log('useSettings: Saved data:', savedData)
            if (savedData) {
                const parsed = JSON.parse(savedData)

                // Validate the data structure and version
                if (parsed && parsed.version === SETTINGS_VERSION && parsed.data) {
                    const savedSettings = parsed.data

                    // Validate and merge with defaults
                    const validatedSettings: UserSettings = {
                        questionTopK: typeof savedSettings.questionTopK === 'number' &&
                                     savedSettings.questionTopK >= 1 &&
                                     savedSettings.questionTopK <= 50
                                     ? savedSettings.questionTopK
                                     : DEFAULT_SETTINGS.questionTopK,
                        minSimilarity: typeof savedSettings.minSimilarity === 'number' &&
                                      savedSettings.minSimilarity >= 0.0 &&
                                      savedSettings.minSimilarity <= 1.0
                                      ? savedSettings.minSimilarity
                                      : DEFAULT_SETTINGS.minSimilarity,
                        serverUrl: typeof savedSettings.serverUrl === 'string' &&
                                  savedSettings.serverUrl.trim().length > 0 &&
                                  (savedSettings.serverUrl.startsWith('http://') || savedSettings.serverUrl.startsWith('https://'))
                                  ? savedSettings.serverUrl.trim()
                                  : DEFAULT_SETTINGS.serverUrl,
                        serverPort: typeof savedSettings.serverPort === 'number' &&
                                   savedSettings.serverPort >= 1 &&
                                   savedSettings.serverPort <= 65535
                                   ? savedSettings.serverPort
                                   : DEFAULT_SETTINGS.serverPort,
                        vectorSetName: typeof savedSettings.vectorSetName === 'string' &&
                                        savedSettings.vectorSetName.trim().length > 0
                                        ? savedSettings.vectorSetName.trim()
                                        : DEFAULT_SETTINGS.vectorSetName
                    }

                    setSettings(validatedSettings)
                    console.log('useSettings: Settings loaded from localStorage:', validatedSettings)
                } else {
                    console.log('useSettings: Using default settings (invalid stored data)')
                }
            } else {
                console.log('useSettings: Using default settings (no stored data)')
            }
        } catch (error) {
            console.error('useSettings: Failed to load settings from localStorage:', error)
            // Clear corrupted data
            localStorage.removeItem(SETTINGS_STORAGE_KEY)
        } finally {
            setIsLoaded(true)
            console.log('useSettings: Settings loading complete, isLoaded set to true')
        }
    }, [])

    // Removed event listener system to avoid circular updates

    // Save settings to localStorage
    const saveToStorage = useCallback((newSettings: UserSettings) => {
        try {
            const dataToSave = {
                version: SETTINGS_VERSION,
                data: newSettings,
                savedAt: new Date().toISOString()
            }

            localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(dataToSave))
            console.log('Settings saved to localStorage:', newSettings)
        } catch (error) {
            console.error('Failed to save settings to localStorage:', error)
        }
    }, [])

    // Update a specific setting
    const updateSetting = useCallback(<K extends keyof UserSettings>(
        key: K,
        value: UserSettings[K]
    ) => {
        console.log('updateSetting called:', { key, value })
        setSettings(prev => {
            const updated = { ...prev, [key]: value }
            console.log('Settings updated from', prev[key], 'to', value)
            saveToStorage(updated)
            return updated
        })
    }, [saveToStorage])

    // Update multiple settings at once
    const updateSettings = useCallback((newSettings: Partial<UserSettings>) => {
        setSettings(prev => {
            const updated = { ...prev, ...newSettings }
            saveToStorage(updated)
            return updated
        })
    }, [saveToStorage])

    // Reset settings to defaults
    const resetSettings = useCallback(() => {
        setSettings(DEFAULT_SETTINGS)
        saveToStorage(DEFAULT_SETTINGS)
    }, [saveToStorage])

    // Helper function to get the full API base URL
    const getApiBaseUrl = useCallback(() => {
        return `${settings.serverUrl}:${settings.serverPort}`
    }, [settings.serverUrl, settings.serverPort])

    return {
        // State
        settings,
        isLoaded,

        // Actions
        updateSetting,
        updateSettings,
        resetSettings,

        // Helpers
        getApiBaseUrl
    }
}

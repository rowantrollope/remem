"use client"

import { useSettings } from "@/hooks/useSettings"
import { RecallSettingsDialog } from "@/components/RecallSettingsDialog"
import { useConfiguredAPI } from "@/hooks/useConfiguredAPI"
import { useState } from "react"

export default function TestSettingsPage() {
    const { settings, isLoaded } = useSettings()
    const { api } = useConfiguredAPI()
    const [testResult, setTestResult] = useState<string>("")

    return (
        <div className="p-8">
            <h1 className="text-2xl font-bold mb-4">Settings Test Page</h1>
            
            <div className="mb-4">
                <RecallSettingsDialog />
            </div>
            
            <div className="bg-gray-100 p-4 rounded">
                <h2 className="text-lg font-semibold mb-2">Current Settings:</h2>
                <p><strong>Is Loaded:</strong> {isLoaded ? 'Yes' : 'No'}</p>
                <p><strong>Question Top-K:</strong> {settings.questionTopK}</p>
                <p><strong>Min Similarity:</strong> {settings.minSimilarity}</p>
                <p><strong>Server URL:</strong> {settings.serverUrl}</p>
                <p><strong>Server Port:</strong> {settings.serverPort}</p>
                <p><strong>Vector Store Name:</strong> {settings.vectorSetName}</p>
                <p><strong>API Vector Store:</strong> {api.getVectorStoreName()}</p>
            </div>
            
            <div className="mt-4 space-x-2">
                <button
                    onClick={() => console.log('Current settings:', settings)}
                    className="bg-blue-500 text-white px-4 py-2 rounded"
                >
                    Log Settings to Console
                </button>
                <button
                    onClick={() => {
                        const stored = localStorage.getItem('memory-app-settings')
                        console.log('Raw localStorage data:', stored)
                        if (stored) {
                            try {
                                const parsed = JSON.parse(stored)
                                console.log('Parsed localStorage data:', parsed)
                            } catch (e) {
                                console.error('Failed to parse localStorage data:', e)
                            }
                        }
                    }}
                    className="bg-green-500 text-white px-4 py-2 rounded"
                >
                    Check localStorage
                </button>
                <button
                    onClick={() => {
                        localStorage.removeItem('memory-app-settings')
                        window.location.reload()
                    }}
                    className="bg-red-500 text-white px-4 py-2 rounded"
                >
                    Clear localStorage & Reload
                </button>
                <button
                    onClick={async () => {
                        console.log('Testing API call with current settings:', settings)
                        console.log('API vectorset name:', api.getVectorStoreName())
                        try {
                            const response = await api.recall("test query", settings.questionTopK, settings.minSimilarity)
                            console.log('API Response:', response)
                            setTestResult(`API called with vectorset: ${api.getVectorStoreName()}, topK: ${settings.questionTopK}, minSimilarity: ${settings.minSimilarity}`)
                        } catch (error) {
                            console.error('API Error:', error)
                            setTestResult(`API Error: ${error}`)
                        }
                    }}
                    className="bg-purple-500 text-white px-4 py-2 rounded"
                >
                    Test API Call
                </button>
            </div>

            {testResult && (
                <div className="mt-4 p-4 bg-yellow-100 rounded">
                    <h3 className="font-semibold">Test Result:</h3>
                    <p>{testResult}</p>
                </div>
            )}
        </div>
    )
}

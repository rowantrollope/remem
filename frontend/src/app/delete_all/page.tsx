"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Trash2, AlertTriangle, CheckCircle, Info } from "lucide-react"
import { ApiPageHeader } from "@/components/ApiPageHeader"
import { PageLayout } from "@/components/PageLayout"
import { useMemoryAPI } from "@/hooks/useMemoryAPI"
import { useSettings } from "@/hooks/useSettings"
import { useConfiguredAPI } from "@/hooks/useConfiguredAPI"
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog"

export default function DeleteAllPage() {
    const [isConfirmDialogOpen, setIsConfirmDialogOpen] = useState(false)
    const [isDeleting, setIsDeleting] = useState(false)
    const [deleteSuccess, setDeleteSuccess] = useState(false)
    const [deletedCount, setDeletedCount] = useState(0)
    const [memoryCount, setMemoryCount] = useState<number | null>(null)

    const {
        isLoading,
        error,
        apiStatus,
        clearError,
        clearAllMemories,
    } = useMemoryAPI()

    const { settings, updateSetting } = useSettings()
    const { api: configuredAPI } = useConfiguredAPI()

    // Get memory info to show current count
    useEffect(() => {
        if (apiStatus !== 'ready') return

        const fetchMemoryInfo = async () => {
            try {
                // Import and create a fresh API instance to avoid stale references
                const { MemoryAgentAPI } = await import('@/lib/api')
                const freshAPI = new MemoryAgentAPI(
                    `${settings.serverUrl}:${settings.serverPort}`,
                    settings.vectorSetName
                )

                console.log('🚀 EXACT FETCH COMMAND:')
                console.log(`fetch('${freshAPI.getBaseUrl()}/api/memory/${freshAPI.getVectorStoreName()}', {`)
                console.log(`  method: 'GET',`)
                console.log(`  headers: { 'Content-Type': 'application/json' }`)
                console.log(`})`)

                const info = await freshAPI.getMemoryInfo()

                console.log('📦 FULL RESPONSE:')
                console.log(JSON.stringify(info, null, 2))

                setMemoryCount(info.memory_count || 0)
            } catch (error) {
                console.error('Failed to fetch memory info:', error)
                setMemoryCount(0)
            }
        }

        fetchMemoryInfo()
    }, [apiStatus, settings.vectorSetName, settings.serverUrl, settings.serverPort])

    // Handle vectorset change
    const handleVectorStoreChange = (newVectorStoreName: string) => {
        updateSetting('vectorSetName', newVectorStoreName)
        setDeleteSuccess(false)
        setDeletedCount(0)
        setMemoryCount(null) // Reset count to trigger refetch
    }

    const handleDeleteAll = async () => {
        setIsDeleting(true)
        setDeleteSuccess(false)
        clearError()

        try {
            console.log('🗑️ Starting delete all operation for vectorset:', settings.vectorSetName)
            const response = await clearAllMemories()
            console.log('🗑️ Delete all response:', response)

            if (response && response.success) {
                setDeletedCount(response.deletedCount || 0)
                setDeleteSuccess(true)
                setMemoryCount(0)
                console.log('✅ Delete all successful, deleted count:', response.deletedCount)
            } else {
                console.error('❌ Delete all failed - response indicates failure:', response)
                throw new Error('Failed to delete all memories')
            }
        } catch (err) {
            console.error('❌ Delete all failed with error:', err)
        } finally {
            setIsDeleting(false)
            setIsConfirmDialogOpen(false)
        }
    }

    const handleConfirm = () => {
        setIsConfirmDialogOpen(true)
    }

    return (
        <PageLayout
            error={error}
            apiStatus={apiStatus}
            onClearError={clearError}
        >
            <div className="container mx-auto p-4 max-w-4xl">
                <ApiPageHeader
                    endpoint="/api/memory/{vectorset_name}/all"
                    hasMessages={false}
                    onClearChat={() => {}}
                    isLoading={isLoading || isDeleting}
                    title="Delete All Memories"
                    showVectorStoreSelector={true}
                    vectorSetName={settings.vectorSetName}
                    onVectorStoreChange={handleVectorStoreChange}
                />

            <div className="space-y-6">
                {/* API Status */}
                {apiStatus !== 'ready' && (
                    <Alert>
                        <Info className="h-4 w-4" />
                        <AlertDescription>
                            {apiStatus === 'not_initialized' 
                                ? 'Memory Agent API is not available. Please check your server connection.'
                                : 'Checking API status...'
                            }
                        </AlertDescription>
                    </Alert>
                )}

                {/* Error Display */}
                {error && (
                    <Alert variant="destructive">
                        <AlertTriangle className="h-4 w-4" />
                        <AlertDescription>{error}</AlertDescription>
                    </Alert>
                )}

                {/* Success Message */}
                {deleteSuccess && (
                    <Alert className="border-green-200 bg-green-50">
                        <CheckCircle className="h-4 w-4 text-green-600" />
                        <AlertDescription className="text-green-800">
                            Successfully deleted {deletedCount} memories from vectorset "{settings.vectorSetName}".
                        </AlertDescription>
                    </Alert>
                )}

                {/* Main Content */}
                <Card>
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2 text-red-600">
                            <Trash2 className="w-5 h-5" />
                            Delete All Memories
                        </CardTitle>
                        <CardDescription>
                            Permanently delete all memories from the selected vectorset. This action cannot be undone.
                        </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        {/* Memory Count Info */}
                        {memoryCount !== null && (
                            <div className="p-4 bg-gray-50 rounded-lg">
                                <div className="flex items-center justify-between">
                                    <span className="text-sm font-medium text-gray-700">
                                        Current memory count in "{settings.vectorSetName}":
                                    </span>
                                    <span className="text-lg font-bold text-gray-900">
                                        {memoryCount.toLocaleString()}
                                    </span>
                                </div>
                            </div>
                        )}

                        {/* Warning */}
                        <Alert variant="destructive">
                            <AlertTriangle className="h-4 w-4" />
                            <AlertDescription>
                                <strong>Warning:</strong> This action will permanently delete ALL memories from the vectorset "{settings.vectorSetName}". 
                                This operation cannot be undone and will affect all applications using this vectorset.
                            </AlertDescription>
                        </Alert>

                        {/* Delete Button */}
                        <div className="flex justify-center pt-4">
                            <Button
                                variant="destructive"
                                size="lg"
                                onClick={handleConfirm}
                                disabled={isLoading || isDeleting || apiStatus !== 'ready' || memoryCount === 0}
                                className="flex items-center gap-2"
                            >
                                <Trash2 className="w-4 h-4" />
                                {isDeleting ? 'Deleting All Memories...' : 'Delete All Memories'}
                            </Button>
                        </div>

                        {memoryCount === 0 && (
                            <p className="text-center text-sm text-gray-500">
                                No memories to delete in this vectorset.
                            </p>
                        )}
                    </CardContent>
                </Card>
            </div>

            {/* Confirmation Dialog */}
            <Dialog open={isConfirmDialogOpen} onOpenChange={setIsConfirmDialogOpen}>
                <DialogContent className="sm:max-w-md">
                    <DialogHeader>
                        <DialogTitle className="flex items-center gap-2 text-red-600">
                            <AlertTriangle className="w-5 h-5" />
                            Confirm Delete All Memories
                        </DialogTitle>
                        <DialogDescription className="text-left">
                            You are about to permanently delete <strong>ALL {memoryCount?.toLocaleString() || 0} memories</strong> from
                            the vectorset "<strong>{settings.vectorSetName}</strong>".
                            <br /><br />
                            <span className="text-red-600 font-semibold">
                                This action cannot be undone.
                            </span>
                            <br /><br />
                            Are you absolutely sure you want to continue?
                        </DialogDescription>
                    </DialogHeader>
                    <DialogFooter className="flex-col sm:flex-row gap-2">
                        <Button 
                            variant="outline" 
                            onClick={() => setIsConfirmDialogOpen(false)}
                            disabled={isDeleting}
                            className="w-full sm:w-auto"
                        >
                            Cancel
                        </Button>
                        <Button 
                            variant="destructive" 
                            onClick={handleDeleteAll}
                            disabled={isDeleting}
                            className="w-full sm:w-auto"
                        >
                            {isDeleting ? (
                                <>
                                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                                    Deleting...
                                </>
                            ) : (
                                <>
                                    <Trash2 className="w-4 h-4 mr-2" />
                                    Yes, Delete All {memoryCount?.toLocaleString() || 0} Memories
                                </>
                            )}
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>
            </div>
        </PageLayout>
    )
}

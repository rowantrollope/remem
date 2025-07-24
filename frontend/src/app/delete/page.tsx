"use client"

import { useState } from "react"

// Components
import { PageLayout } from "@/components/PageLayout"
import { ApiPageHeader } from "@/components/ApiPageHeader"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog"

// Hooks
import { useMemoryAPI } from "@/hooks"
import { useSettings } from "@/hooks/useSettings"

// Icons
import { Trash2, AlertTriangle, CheckCircle } from "lucide-react"

export default function DeletePage() {
    const [memoryId, setMemoryId] = useState("")
    const [isConfirmDialogOpen, setIsConfirmDialogOpen] = useState(false)
    const [isDeleting, setIsDeleting] = useState(false)
    const [deleteSuccess, setDeleteSuccess] = useState(false)
    const [lastDeletedId, setLastDeletedId] = useState("")

    const {
        isLoading,
        error,
        apiStatus,
        deleteMemory,
        clearError,
    } = useMemoryAPI()

    const { settings, updateSetting } = useSettings()

    // Handle vectorset change
    const handleVectorStoreChange = (newVectorStoreName: string) => {
        updateSetting('vectorSetName', newVectorStoreName)
    }

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault()
        if (!memoryId.trim()) return
        setIsConfirmDialogOpen(true)
    }

    const handleDeleteConfirm = async () => {
        if (!memoryId.trim()) return
        
        setIsDeleting(true)
        setDeleteSuccess(false)
        
        try {
            const success = await deleteMemory(memoryId.trim())
            if (success) {
                setLastDeletedId(memoryId.trim())
                setDeleteSuccess(true)
                setMemoryId("")
            }
        } finally {
            setIsDeleting(false)
            setIsConfirmDialogOpen(false)
        }
    }

    const handleDeleteCancel = () => {
        setIsConfirmDialogOpen(false)
    }

    const formatShortId = (id: string) => {
        if (!id) return 'N/A'
        if (id.length <= 8) return id
        return id.substring(0, 8) + '...'
    }

    return (
        <PageLayout
            error={error}
            apiStatus={apiStatus}
            onClearError={clearError}
        >
            <div className="h-full flex flex-col">
                <ApiPageHeader
                    endpoint={`(DELETE) /api/memory/${settings.vectorSetName}/{memory_id}`}
                    hasMessages={false}
                    onClearChat={() => {}}
                    isLoading={isLoading || isDeleting}
                    title="Delete Memory"
                    showSettingsButton={false}
                    showVectorStoreSelector={true}
                    vectorSetName={settings.vectorSetName}
                    onVectorStoreChange={handleVectorStoreChange}
                />

                <div className="flex-1 flex items-center justify-center bg-white">
                    <div className="w-full max-w-md">
                        <div className="text-center mb-8">
                            <h1 className="text-3xl font-bold text-gray-900 mb-2">Delete Memory</h1>
                            <p className="text-gray-600">
                                Permanently remove a memory by its ID
                            </p>
                        </div>

                        {/* Success Message */}
                        {deleteSuccess && (
                            <Card className="mb-6 border-green-200 bg-green-50">
                                <CardContent className="pt-6">
                                    <div className="flex items-center gap-3">
                                        <CheckCircle className="w-5 h-5 text-green-600" />
                                        <div>
                                            <p className="text-green-800 font-medium">Memory deleted successfully!</p>
                                            <p className="text-green-700 text-sm">
                                                Memory ID: <span className="font-mono">{formatShortId(lastDeletedId)}</span>
                                            </p>
                                        </div>
                                    </div>
                                </CardContent>
                            </Card>
                        )}

                        {/* Delete Form */}
                        <Card>
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <Trash2 className="w-5 h-5 text-red-600" />
                                    Delete Memory
                                </CardTitle>
                                <CardDescription>
                                    Enter the memory ID to delete. This action cannot be undone.
                                </CardDescription>
                            </CardHeader>
                            <CardContent>
                                <form onSubmit={handleSubmit} className="space-y-4">
                                    <div className="space-y-2">
                                        <Label htmlFor="memory-id">Memory ID</Label>
                                        <Input
                                            id="memory-id"
                                            type="text"
                                            placeholder="Enter memory ID (e.g., neme-123abc...)"
                                            value={memoryId}
                                            onChange={(e) => setMemoryId(e.target.value)}
                                            className="font-mono"
                                            disabled={isDeleting}
                                        />
                                        <p className="text-xs text-gray-500">
                                            You can find memory IDs in search results or memory details
                                        </p>
                                    </div>

                                    <Button
                                        type="submit"
                                        variant="destructive"
                                        className="w-full"
                                        disabled={!memoryId.trim() || isDeleting}
                                    >
                                        <Trash2 className="w-4 h-4 mr-2" />
                                        Delete Memory
                                    </Button>
                                </form>
                            </CardContent>
                        </Card>
                    </div>
                </div>
            </div>

            {/* Delete Confirmation Dialog */}
            <Dialog open={isConfirmDialogOpen} onOpenChange={setIsConfirmDialogOpen}>
                <DialogContent className="sm:max-w-md">
                    <DialogHeader>
                        <DialogTitle className="flex items-center gap-2 text-red-600">
                            <AlertTriangle className="w-5 h-5" />
                            Confirm Delete
                        </DialogTitle>
                        <DialogDescription className="text-left">
                            Are you sure you want to delete this memory?
                            <br />
                            <br />
                            <strong>Memory ID:</strong> <span className="font-mono text-sm">{formatShortId(memoryId)}</span>
                            <br />
                            <br />
                            <strong className="text-red-600">This action cannot be undone.</strong>
                        </DialogDescription>
                    </DialogHeader>
                    <DialogFooter className="flex-col sm:flex-row gap-2">
                        <Button 
                            variant="outline" 
                            onClick={handleDeleteCancel}
                            disabled={isDeleting}
                            className="w-full sm:w-auto"
                        >
                            Cancel
                        </Button>
                        <Button 
                            variant="destructive" 
                            onClick={handleDeleteConfirm}
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
                                    Yes, Delete
                                </>
                            )}
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>
        </PageLayout>
    )
}

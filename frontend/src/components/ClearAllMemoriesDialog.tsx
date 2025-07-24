"use client"

import type React from "react"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
    DialogTrigger,
} from "@/components/ui/dialog"
import { Trash2, AlertTriangle } from "lucide-react"

interface ClearAllMemoriesDialogProps {
    onConfirm: () => Promise<void>
    isLoading?: boolean
    memoryCount?: number
}

export function ClearAllMemoriesDialog({ 
    onConfirm, 
    isLoading = false, 
    memoryCount = 0 
}: ClearAllMemoriesDialogProps) {
    const [isOpen, setIsOpen] = useState(false)

    const handleConfirm = async () => {
        await onConfirm()
        setIsOpen(false)
    }

    return (
        <Dialog open={isOpen} onOpenChange={setIsOpen}>
            <DialogTrigger asChild>
                <Button 
                    variant="destructive" 
                    className="flex items-center gap-2"
                    disabled={isLoading || memoryCount === 0}
                >
                    <Trash2 className="w-4 h-4" />
                    Clear All Memories
                </Button>
            </DialogTrigger>
            <DialogContent className="sm:max-w-md">
                <DialogHeader>
                    <DialogTitle className="flex items-center gap-2 text-red-600">
                        <AlertTriangle className="w-5 h-5" />
                        Clear All Memories
                    </DialogTitle>
                    <DialogDescription className="text-left">
                        This action will permanently delete all {memoryCount} memories from your memory bank and clear your local chat history.
                        <br />
                        <br />
                        <strong className="text-red-600">This action cannot be undone.</strong>
                        <br />
                        <br />
                        Are you sure you want to continue?
                    </DialogDescription>
                </DialogHeader>
                <DialogFooter className="flex-col sm:flex-row gap-2">
                    <Button 
                        variant="outline" 
                        onClick={() => setIsOpen(false)}
                        disabled={isLoading}
                        className="w-full sm:w-auto"
                    >
                        Cancel
                    </Button>
                    <Button 
                        variant="destructive" 
                        onClick={handleConfirm}
                        disabled={isLoading}
                        className="w-full sm:w-auto"
                    >
                        {isLoading ? (
                            <>
                                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                                Clearing...
                            </>
                        ) : (
                            <>
                                <Trash2 className="w-4 h-4 mr-2" />
                                Yes, Clear All
                            </>
                        )}
                    </Button>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    )
}

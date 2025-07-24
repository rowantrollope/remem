"use client"

import { ReactNode } from "react"
import { Card } from "@/components/ui/card"
import { Navbar, ErrorAlert, StatusAlert } from "@/components"
import type { ApiStatus } from "@/types"

interface PageLayoutProps {
    children: ReactNode
    error?: string | null
    apiStatus?: ApiStatus
    onClearError?: () => void
    className?: string
}

export function PageLayout({ 
    children, 
    error, 
    apiStatus, 
    onClearError,
    className = ""
}: PageLayoutProps) {
    return (
        <div className="h-screen flex flex-col">
            <Navbar />

            {/* Main content area */}
            <div className="flex-1 px-6 min-h-0">
                <Card className={`h-full flex flex-col max-w-4xl mx-auto p-2 shadow-none border-none ${className}`}>
                    
                    {/* Error Display */}
                    {error && onClearError && (
                        <ErrorAlert error={error} onDismiss={onClearError} />
                    )}

                    {/* API Status Warning */}
                    {apiStatus && (
                        <StatusAlert apiStatus={apiStatus} />
                    )}

                    {/* Page Content */}
                    <div className="flex-1 min-h-0">
                        {children}
                    </div>
                </Card>
            </div>
        </div>
    )
}

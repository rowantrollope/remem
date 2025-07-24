"use client"

// Components
import { PageLayout } from "@/components/PageLayout"
import { ContextTab } from "@/components/tabs/ContextTab"

// Hooks
import { useMemoryAPI } from "@/hooks"

export default function ContextPage() {
    const {
        isLoading,
        error,
        apiStatus,
        currentContext,
        getContext,
        updateContext,
        clearError,
    } = useMemoryAPI()

    return (
        <PageLayout 
            error={error} 
            apiStatus={apiStatus} 
            onClearError={clearError}
        >
            {/* Context Content */}
            <div className=" h-full overflow-y-auto px-4 bg-white">
                <div className=" w-full bg-white/50 flex justify-between items-center">
                    <div className="grow"></div>
                    <div className="font-mono text-muted-foreground">
                        (GET & POST) /api/memory/memories/context
                    </div>
                </div>
                <ContextTab
                    currentContext={currentContext}
                    onUpdateContext={updateContext}
                    onGetContext={getContext}
                    isLoading={isLoading}
                />
            </div>
        </PageLayout>
    )
}

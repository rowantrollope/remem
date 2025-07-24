import { AlertCircle } from "lucide-react"
import type { ApiStatus } from "@/types"

interface StatusAlertProps {
    apiStatus: ApiStatus
}

export function StatusAlert({ apiStatus }: StatusAlertProps) {
    if (apiStatus === 'ready') return null

    return (
        <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-yellow-500" />
            <p className="text-sm text-yellow-700">
                {apiStatus === 'not_initialized'
                    ? 'Memory Agent API is not initialized. Please check the server.'
                    : 'Checking API status...'}
            </p>
        </div>
    )
}

import { AlertCircle } from "lucide-react"

interface ErrorAlertProps {
    error: string
    onDismiss: () => void
}

export function ErrorAlert({ error, onDismiss }: ErrorAlertProps) {
    return (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-red-500" />
            <p className="text-sm text-red-700">{error}</p>
            <button
                onClick={onDismiss}
                className="ml-auto text-red-500 hover:text-red-700"
            >
                ×
            </button>
        </div>
    )
}

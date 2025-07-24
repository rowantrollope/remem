import { cn } from "@/lib/utils"

interface ConfidencePillProps {
    confidence: 'high' | 'medium' | 'low' | number
    className?: string
}

export function ConfidencePill({ confidence, className }: ConfidencePillProps) {
    const getConfidenceStyles = () => {
        if (typeof confidence === 'number') {
            if (confidence >= 0.8) return 'bg-green-100 text-green-800 border-green-200'
            if (confidence >= 0.6) return 'bg-yellow-100 text-yellow-800 border-yellow-200'
            return 'bg-red-100 text-red-800 border-red-200'
        }

        switch (confidence) {
            case 'high':
                return 'bg-green-100 text-green-800 border-green-200'
            case 'medium':
                return 'bg-yellow-100 text-yellow-800 border-yellow-200'
            case 'low':
                return 'bg-red-100 text-red-800 border-red-200'
            default:
                return 'bg-gray-100 text-gray-800 border-gray-200'
        }
    }

    const getConfidenceText = () => {
        if (typeof confidence === 'number') {
            return `${(confidence * 100).toFixed(1)}% confidence`
        }
        return `${confidence} confidence`
    }

    return (
        <span
            className={cn(
                'inline-flex items-center px-2 py-1 rounded-full text-xs font-medium border',
                getConfidenceStyles(),
                className
            )}
        >
            {getConfidenceText()}
        </span>
    )
}

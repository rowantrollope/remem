import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Card, CardContent } from "@/components/ui/card"
import { Info, Anchor } from "lucide-react"
import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger,
} from "@/components/ui/tooltip"

interface GroundingToggleProps {
    enabled: boolean
    onChange: (enabled: boolean) => void
    className?: string
}

export function GroundingToggle({ enabled, onChange, className }: GroundingToggleProps) {
    return (
        <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
                <Anchor className="w-5 h-5 text-blue-600" />
                <div className="space-y-1">
                    <Label htmlFor="grounding-toggle" className="text-sm font-medium">
                        Contextual Grounding
                    </Label>
                    <p className="text-xs text-gray-500">
                        Enhance memories with current context
                    </p>
                </div>
            </div>
            <div className="flex items-center space-x-2">
                <TooltipProvider>
                    <Tooltip>
                        <TooltipTrigger asChild>
                            <Info className="w-4 h-4 text-gray-400 cursor-help" />
                        </TooltipTrigger>
                        <TooltipContent className="max-w-xs">
                            <p className="text-sm">
                                When enabled, memories will be enhanced with contextual information
                                like location, time, and environment to make them more specific and searchable.
                            </p>
                        </TooltipContent>
                    </Tooltip>
                </TooltipProvider>
                <Switch
                    id="grounding-toggle"
                    checked={enabled}
                    onCheckedChange={onChange}
                />
            </div>
        </div>
    )
}

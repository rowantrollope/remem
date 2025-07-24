import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { MapPin, Users, Activity, Cloud, Save, Loader2, Plus, X } from "lucide-react"
import type { ContextInfo } from "@/types"

interface ContextTabProps {
    currentContext: ContextInfo | null
    onUpdateContext: (context: {
        location?: string
        activity?: string
        people_present?: string[]
        weather?: string
        [key: string]: any
    }) => Promise<boolean>
    onGetContext: () => Promise<ContextInfo | null>
    isLoading: boolean
}

export function ContextTab({ currentContext, onUpdateContext, onGetContext, isLoading }: ContextTabProps) {
    const [location, setLocation] = useState("")
    const [activity, setActivity] = useState("")
    const [peoplePresent, setPeoplePresent] = useState("")
    const [weather, setWeather] = useState("")
    const [customFields, setCustomFields] = useState<Array<{key: string, value: string}>>([])
    const [isUpdating, setIsUpdating] = useState(false)
    const [hasInitialized, setHasInitialized] = useState(false)

    // Load current context on mount
    useEffect(() => {
        onGetContext()
    }, [onGetContext])

    // Update form when context changes
    useEffect(() => {
        if (currentContext) {
            // Handle both API response formats:
            // GET: { spatial: {location, activity}, social: {people_present}, environmental: {weather, ...custom} }
            // POST: { location, activity, people_present, environment: {weather, ...custom} }
            const spatial = currentContext.spatial || {}
            const social = currentContext.social || {}
            const environmental = currentContext.environmental || currentContext.environment || {}

            // Handle both flat and nested formats
            setLocation(spatial.location || currentContext.location || "")
            setActivity(spatial.activity || currentContext.activity || "")
            setPeoplePresent(social.people_present?.join(", ") || currentContext.people_present?.join(", ") || "")
            setWeather(environmental.weather || "")

            // Extract custom fields from environmental/environment section
            const predefinedFields = new Set(['weather'])
            const customFieldsFromContext = Object.entries(environmental)
                .filter(([key, value]) => !predefinedFields.has(key) && value !== undefined && value !== null && value !== "")
                .map(([key, value]) => ({ key, value: String(value) }))

            setCustomFields(customFieldsFromContext)
            setHasInitialized(true)
        } else if (hasInitialized) {
            // Clear form if context becomes null after being initialized
            setLocation("")
            setActivity("")
            setPeoplePresent("")
            setWeather("")
            setCustomFields([])
        }
    }, [currentContext, hasInitialized])

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        setIsUpdating(true)

        // Try both flat and structured approaches to see which one works
        // First, let's try the flat approach that was working before
        const flatContext: any = {
            location: location.trim() || undefined,
            activity: activity.trim() || undefined,
            people_present: peoplePresent.trim() ? peoplePresent.split(",").map(p => p.trim()).filter(Boolean) : undefined,
            weather: weather.trim() || undefined,
        }

        // Add custom fields to flat context
        customFields.forEach(({ key, value }) => {
            if (key.trim() && value.trim()) {
                flatContext[key.trim()] = value.trim()
            }
        })

        const success = await onUpdateContext(flatContext)
        if (success) {
            // Context will be updated via the hook and form will be updated automatically
        }
        setIsUpdating(false)
    }

    const addCustomField = () => {
        setCustomFields([...customFields, { key: "", value: "" }])
    }

    const removeCustomField = (index: number) => {
        setCustomFields(customFields.filter((_, i) => i !== index))
    }

    const updateCustomField = (index: number, field: 'key' | 'value', newValue: string) => {
        const updated = [...customFields]
        updated[index][field] = newValue
        setCustomFields(updated)
    }

    const formatDateTime = (temporal: any) => {
        if (!temporal) return null
        return `${temporal.date || ''} ${temporal.time || ''}`.trim()
    }

    return (
        <div className="flex flex-col h-full">
            <ScrollArea className="flex-1">
                <div className="space-y-6 p-2">
                    {/* Current Context Display */}
                    {isLoading ? (
                        <Card className="border border-gray-200 bg-gray-50/50">
                            <CardHeader className="pb-3">
                                <CardTitle className="text-lg flex items-center gap-2">
                                    <Loader2 className="w-5 h-5 text-gray-600 animate-spin" />
                                    Loading Context...
                                </CardTitle>
                            </CardHeader>
                        </Card>
                    ) : currentContext ? (
                        <Card className="border border-blue-200 bg-blue-50/50">
                            <CardHeader className="pb-3">
                                <CardTitle className="text-lg flex items-center gap-2">
                                    <MapPin className="w-5 h-5 text-blue-600" />
                                    Current Context
                                </CardTitle>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                {/* Temporal Info */}
                                {currentContext.temporal && (
                                    <div className="space-y-2">
                                        <h4 className="font-medium text-sm text-gray-700">Time</h4>
                                        <div className="flex flex-wrap gap-2">
                                            {formatDateTime(currentContext.temporal) && (
                                                <Badge variant="outline" className="text-xs">
                                                    {formatDateTime(currentContext.temporal)}
                                                </Badge>
                                            )}
                                            {currentContext.temporal.day_of_week && (
                                                <Badge variant="outline" className="text-xs">
                                                    {currentContext.temporal.day_of_week}
                                                </Badge>
                                            )}
                                        </div>
                                    </div>
                                )}

                                {/* Spatial Info - handle both nested and flat formats */}
                                {(() => {
                                    const location = currentContext.spatial?.location || currentContext.location
                                    const activity = currentContext.spatial?.activity || currentContext.activity
                                    return (location || activity) && (
                                        <div className="space-y-2">
                                            <h4 className="font-medium text-sm text-gray-700 flex items-center gap-1">
                                                <MapPin className="w-4 h-4" />
                                                Location & Activity
                                            </h4>
                                            <div className="flex flex-wrap gap-2">
                                                {location && (
                                                    <Badge className="text-xs bg-green-100 text-green-800">
                                                        {location}
                                                    </Badge>
                                                )}
                                                {activity && (
                                                    <Badge className="text-xs bg-purple-100 text-purple-800">
                                                        {activity}
                                                    </Badge>
                                                )}
                                            </div>
                                        </div>
                                    )
                                })()}

                                {/* Social Info - handle both nested and flat formats */}
                                {(() => {
                                    const peoplePresent = currentContext.social?.people_present || currentContext.people_present
                                    return peoplePresent && peoplePresent.length > 0 && (
                                        <div className="space-y-2">
                                            <h4 className="font-medium text-sm text-gray-700 flex items-center gap-1">
                                                <Users className="w-4 h-4" />
                                                People Present
                                            </h4>
                                            <div className="flex flex-wrap gap-2">
                                                {peoplePresent.map((person, index) => (
                                                    <Badge key={index} className="text-xs bg-blue-100 text-blue-800">
                                                        {person}
                                                    </Badge>
                                                ))}
                                            </div>
                                        </div>
                                    )
                                })()}

                                {/* Environmental Info - handle both environmental and environment */}
                                {(currentContext.environmental || currentContext.environment) && (
                                    <div className="space-y-2">
                                        <h4 className="font-medium text-sm text-gray-700 flex items-center gap-1">
                                            <Cloud className="w-4 h-4" />
                                            Environment
                                        </h4>
                                        <div className="flex flex-wrap gap-2">
                                            {(() => {
                                                const envData = currentContext.environmental || currentContext.environment || {}
                                                return (
                                                    <>
                                                        {envData.weather && (
                                                            <Badge className="text-xs bg-yellow-100 text-yellow-800">
                                                                Weather: {envData.weather}
                                                            </Badge>
                                                        )}
                                                        {/* Display all other fields as custom fields */}
                                                        {Object.entries(envData)
                                                            .filter(([key, value]) =>
                                                                key !== 'weather' &&
                                                                value !== undefined &&
                                                                value !== null &&
                                                                value !== ""
                                                            )
                                                            .map(([key, value]) => (
                                                                <Badge key={key} className="text-xs bg-gray-100 text-gray-800">
                                                                    {key}: {String(value)}
                                                                </Badge>
                                                            ))
                                                        }
                                                    </>
                                                )
                                            })()}
                                        </div>
                                    </div>
                                )}
                            </CardContent>
                        </Card>
                    ) : (
                        <Card className="border border-yellow-200 bg-yellow-50/50">
                            <CardHeader className="pb-3">
                                <CardTitle className="text-lg flex items-center gap-2">
                                    <MapPin className="w-5 h-5 text-yellow-600" />
                                    No Context Set
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                <p className="text-sm text-gray-600">
                                    No context information is currently available. Use the form below to set your current context.
                                </p>
                            </CardContent>
                        </Card>
                    )}

                    {/* Context Update Form */}
                    <Card>
                        <CardHeader className="pb-3">
                            <CardTitle className="text-lg">Update Context</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <form onSubmit={handleSubmit} className="space-y-4">
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    <div className="space-y-2">
                                        <Label htmlFor="location" className="flex items-center gap-1">
                                            <MapPin className="w-4 h-4" />
                                            Location
                                        </Label>
                                        <Input
                                            id="location"
                                            value={location}
                                            onChange={(e) => setLocation(e.target.value)}
                                            placeholder="e.g., Tokyo, Japan"
                                            disabled={isLoading || isUpdating}
                                        />
                                    </div>

                                    <div className="space-y-2">
                                        <Label htmlFor="activity" className="flex items-center gap-1">
                                            <Activity className="w-4 h-4" />
                                            Activity
                                        </Label>
                                        <Input
                                            id="activity"
                                            value={activity}
                                            onChange={(e) => setActivity(e.target.value)}
                                            placeholder="e.g., vacation, work meeting"
                                            disabled={isLoading || isUpdating}
                                        />
                                    </div>

                                    <div className="space-y-2">
                                        <Label htmlFor="people" className="flex items-center gap-1">
                                            <Users className="w-4 h-4" />
                                            People Present
                                        </Label>
                                        <Input
                                            id="people"
                                            value={peoplePresent}
                                            onChange={(e) => setPeoplePresent(e.target.value)}
                                            placeholder="e.g., Sarah, Mike (comma separated)"
                                            disabled={isLoading || isUpdating}
                                        />
                                    </div>

                                    <div className="space-y-2">
                                        <Label htmlFor="weather" className="flex items-center gap-1">
                                            <Cloud className="w-4 h-4" />
                                            Weather
                                        </Label>
                                        <Input
                                            id="weather"
                                            value={weather}
                                            onChange={(e) => setWeather(e.target.value)}
                                            placeholder="e.g., sunny, rainy"
                                            disabled={isLoading || isUpdating}
                                        />
                                    </div>
                                </div>

                                {/* Custom Fields Section */}
                                <div className="space-y-4">
                                    <div className="flex items-center justify-between">
                                        <Label className="text-sm font-medium text-gray-700">
                                            Custom Fields
                                        </Label>
                                        <Button
                                            type="button"
                                            variant="outline"
                                            size="sm"
                                            onClick={addCustomField}
                                            disabled={isLoading || isUpdating}
                                            className="flex items-center gap-1"
                                        >
                                            <Plus className="w-4 h-4" />
                                            Add Field
                                        </Button>
                                    </div>

                                    {customFields.length > 0 && (
                                        <div className="space-y-3">
                                            {customFields.map((field, index) => (
                                                <div key={index} className="flex gap-2 items-end">
                                                    <div className="flex-1">
                                                        <Label htmlFor={`custom-key-${index}`} className="text-xs text-gray-600">
                                                            Key
                                                        </Label>
                                                        <Input
                                                            id={`custom-key-${index}`}
                                                            value={field.key}
                                                            onChange={(e) => updateCustomField(index, 'key', e.target.value)}
                                                            placeholder="e.g., project, priority"
                                                            disabled={isLoading || isUpdating}
                                                            className="text-sm"
                                                        />
                                                    </div>
                                                    <div className="flex-1">
                                                        <Label htmlFor={`custom-value-${index}`} className="text-xs text-gray-600">
                                                            Value
                                                        </Label>
                                                        <Input
                                                            id={`custom-value-${index}`}
                                                            value={field.value}
                                                            onChange={(e) => updateCustomField(index, 'value', e.target.value)}
                                                            placeholder="e.g., website redesign, high"
                                                            disabled={isLoading || isUpdating}
                                                            className="text-sm"
                                                        />
                                                    </div>
                                                    <Button
                                                        type="button"
                                                        variant="outline"
                                                        size="sm"
                                                        onClick={() => removeCustomField(index)}
                                                        disabled={isLoading || isUpdating}
                                                        className="flex items-center gap-1 text-red-600 hover:text-red-700 hover:bg-red-50"
                                                    >
                                                        <X className="w-4 h-4" />
                                                    </Button>
                                                </div>
                                            ))}
                                        </div>
                                    )}

                                    {customFields.length === 0 && (
                                        <p className="text-sm text-gray-500 italic">
                                            No custom fields added. Click "Add Field" to create custom key-value pairs.
                                        </p>
                                    )}
                                </div>

                                <Button
                                    type="submit"
                                    disabled={isLoading || isUpdating}
                                    className="w-full bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700"
                                >
                                    {isUpdating ? (
                                        <>
                                            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                                            Updating Context...
                                        </>
                                    ) : (
                                        <>
                                            <Save className="w-4 h-4 mr-2" />
                                            Update Context
                                        </>
                                    )}
                                </Button>
                            </form>
                        </CardContent>
                    </Card>
                </div>
            </ScrollArea>
        </div>
    )
}

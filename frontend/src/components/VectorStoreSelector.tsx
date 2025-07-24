"use client"

import React, { useState, useEffect, useCallback } from "react"
import { Check, ChevronsUpDown, Database } from "lucide-react"

import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import {
    Command,
    CommandEmpty,
    CommandGroup,
    CommandInput,
    CommandItem,
    CommandList,
} from "@/components/ui/command"
import {
    Popover,
    PopoverContent,
    PopoverTrigger,
} from "@/components/ui/popover"
import { Badge } from "@/components/ui/badge"

// Predefined vectorset options for demos
const PREDEFINED_VECTORSTORES = [
    { value: "travel_agent_memory", label: "travel_agent_memory", description: "Travel demo Vector Set" },
    { value: "retail_agent_memory", label: "retail_agent_memory", description: "Retail demo Vector Set" },
    { value: "investment_agent_memory", label: "investment_agent_memory", description: "Investment demo Vector Set" },
]

// localStorage key for custom vectorset names
const CUSTOM_VECTORSTORES_STORAGE_KEY = 'custom-vectorset-names'

interface VectorStoreSelectorProps {
    value: string
    onValueChange: (value: string) => void
    className?: string
    disabled?: boolean
    placeholder?: string
}

export function VectorStoreSelector({
    value,
    onValueChange,
    className,
    disabled = false,
    placeholder = "Select vectorset..."
}: VectorStoreSelectorProps) {
    const [open, setOpen] = useState(false)
    const [customValue, setCustomValue] = useState("")
    const [customVectorStores, setCustomVectorStores] = useState<string[]>([])

    // Load custom vectorset names from localStorage on mount
    useEffect(() => {
        try {
            const saved = localStorage.getItem(CUSTOM_VECTORSTORES_STORAGE_KEY)
            if (saved) {
                const parsed = JSON.parse(saved)
                if (Array.isArray(parsed)) {
                    setCustomVectorStores(parsed)
                }
            }
        } catch (error) {
            console.error('Failed to load custom vectorset names:', error)
        }
    }, [])

    // Save custom vectorset names to localStorage
    const saveCustomVectorStores = useCallback((names: string[]) => {
        try {
            localStorage.setItem(CUSTOM_VECTORSTORES_STORAGE_KEY, JSON.stringify(names))
        } catch (error) {
            console.error('Failed to save custom vectorset names:', error)
        }
    }, [])

    // Add a new custom vectorset name
    const addCustomVectorStore = useCallback((name: string) => {
        const trimmedName = name.trim()
        if (!trimmedName) return

        // Check if it's already in predefined or custom lists
        const isPredefined = PREDEFINED_VECTORSTORES.some(store => store.value === trimmedName)
        if (isPredefined) return

        setCustomVectorStores(prev => {
            if (prev.includes(trimmedName)) return prev
            const updated = [...prev, trimmedName]
            saveCustomVectorStores(updated)
            return updated
        })
    }, [saveCustomVectorStores])

    // Check if current value is in predefined list
    const selectedPredefined = PREDEFINED_VECTORSTORES.find(store => store.value === value)
    const isCustomValue = !selectedPredefined && value

    // Automatically add new custom vectorset names to the persistent list
    useEffect(() => {
        if (value && !selectedPredefined && !customVectorStores.includes(value)) {
            addCustomVectorStore(value)
        }
    }, [value, selectedPredefined, customVectorStores, addCustomVectorStore])

    const handleSelect = (selectedValue: string) => {
        if (selectedValue === value) {
            setOpen(false)
            return
        }
        
        onValueChange(selectedValue)
        setOpen(false)
        setCustomValue("")
    }

    const handleCustomSubmit = () => {
        if (customValue.trim()) {
            const trimmedValue = customValue.trim()
            addCustomVectorStore(trimmedValue)
            onValueChange(trimmedValue)
            setOpen(false)
            setCustomValue("")
        }
    }

    const displayValue = selectedPredefined?.label || value || placeholder

    return (
        <Popover open={open} onOpenChange={setOpen}>
            <PopoverTrigger asChild>
                <Button
                    variant="outline"
                    role="combobox"
                    aria-expanded={open}
                    className={cn("w-[280px] justify-between", className)}
                    disabled={disabled}
                >
                    <div className="flex items-center gap-2">
                        <Database className="h-4 w-4" />
                        <span className="truncate">{displayValue}</span>
                        {isCustomValue && (
                            <Badge variant="secondary" className="text-xs">
                                Custom
                            </Badge>
                        )}
                    </div>
                    <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
                </Button>
            </PopoverTrigger>
            <PopoverContent className="w-[280px] p-0">
                <Command>
                    <CommandInput 
                        placeholder="Search or type custom name..." 
                        value={customValue}
                        onValueChange={setCustomValue}
                    />
                    <CommandList>
                        <CommandEmpty>
                            {customValue.trim() ? (
                                <div className="p-2">
                                    <Button 
                                        variant="ghost" 
                                        className="w-full justify-start"
                                        onClick={handleCustomSubmit}
                                    >
                                        <Database className="mr-2 h-4 w-4" />
                                        Use "{customValue.trim()}"
                                    </Button>
                                </div>
                            ) : (
                                "No vectorset found."
                            )}
                        </CommandEmpty>
                        <CommandGroup heading="Demo Vector Sets">
                            {PREDEFINED_VECTORSTORES.map((store) => (
                                <CommandItem
                                    key={store.value}
                                    value={store.value}
                                    onSelect={handleSelect}
                                >
                                    <Check
                                        className={cn(
                                            "mr-2 h-4 w-4",
                                            value === store.value ? "opacity-100" : "opacity-0"
                                        )}
                                    />
                                    <div className="flex flex-col">
                                        <span className="font-medium">{store.label}</span>
                                        <span className="text-xs text-muted-foreground">
                                            {store.description}
                                        </span>
                                    </div>
                                </CommandItem>
                            ))}
                        </CommandGroup>
                        {customVectorStores.length > 0 && (
                            <CommandGroup heading="Custom Vectorstores">
                                {customVectorStores.map((storeName) => (
                                    <CommandItem
                                        key={storeName}
                                        value={storeName}
                                        onSelect={handleSelect}
                                    >
                                        <Check
                                            className={cn(
                                                "mr-2 h-4 w-4",
                                                value === storeName ? "opacity-100" : "opacity-0"
                                            )}
                                        />
                                        <div className="flex flex-col">
                                            <span className="font-medium">{storeName}</span>
                                            <span className="text-xs text-muted-foreground">
                                                Custom vectorset
                                            </span>
                                        </div>
                                    </CommandItem>
                                ))}
                            </CommandGroup>
                        )}
                        {customValue.trim() && (
                            <CommandGroup heading="New Custom">
                                <CommandItem
                                    value={customValue.trim()}
                                    onSelect={handleSelect}
                                >
                                    <Database className="mr-2 h-4 w-4" />
                                    <div className="flex flex-col">
                                        <span className="font-medium">{customValue.trim()}</span>
                                        <span className="text-xs text-muted-foreground">
                                            Create new custom vectorset
                                        </span>
                                    </div>
                                </CommandItem>
                            </CommandGroup>
                        )}
                    </CommandList>
                </Command>
            </PopoverContent>
        </Popover>
    )
}

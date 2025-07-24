"use client"

import React from 'react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Send, Trash2 } from 'lucide-react'
import { ChatBox, type UnifiedChatMessage, type ChatMessage } from '@/components/ChatBox'

export interface ChatCardProps {
    title: string
    subtitle: string
    messages: UnifiedChatMessage[] | ChatMessage[]
    input: string
    onInputChange: (value: string) => void
    onSubmit: (e: React.FormEvent) => void
    onClearChat?: () => void
    isLoading: boolean
    placeholder?: string
    headerIcon?: React.ReactNode
    borderColor?: string
    headerBgColor?: string
    messageBgColor?: string
    buttonColor?: string
    loadingText?: string
    showMemoryIndicators?: boolean
    badge?: React.ReactNode
    className?: string
    
    // ChatBox specific props
    showTimestamps?: boolean
    enableMemoryExpansion?: boolean
    copiedId?: string | null
    onCopyId?: (id: string) => void
}

export function ChatCard({
    title,
    subtitle,
    messages,
    input,
    onInputChange,
    onSubmit,
    onClearChat,
    isLoading,
    placeholder = "Ask a question...",
    headerIcon,
    borderColor = "border-gray-200",
    headerBgColor = "bg-gray-50",
    messageBgColor = "bg-gray-100",
    buttonColor = "bg-blue-600 hover:bg-blue-700",
    loadingText = "Thinking...",
    showMemoryIndicators = false,
    badge,
    className = "",
    showTimestamps = true,
    enableMemoryExpansion = true,
    copiedId,
    onCopyId
}: ChatCardProps) {
    return (
        <div className={`h-[600px] flex flex-col rounded-lg border bg-card text-card-foreground shadow-sm ${borderColor} ${className}`}>
            {/* Header */}
            <div className={`flex flex-col space-y-1.5 p-6 ${headerBgColor} border-b ${borderColor} rounded-t-lg`}>
                <div className="flex items-center justify-between">
                    <h3 className="text-xl font-semibold leading-none tracking-tight flex items-center gap-2">
                        {headerIcon}
                        {title}
                        {badge}
                    </h3>
                    {onClearChat && messages.length > 0 && (
                        <Button
                            variant="ghost"
                            size="sm"
                            onClick={onClearChat}
                            className="text-red-600 hover:text-red-700 hover:bg-red-50"
                            disabled={isLoading}
                        >
                            <Trash2 className="w-4 h-4" />
                        </Button>
                    )}
                </div>
                <p className="text-sm text-muted-foreground">{subtitle}</p>
            </div>

            {/* Content */}
            <div className="flex-1 flex flex-col p-0 overflow-hidden">
                {/* Messages */}
                <div className="flex-1 min-h-0 p-4">
                    <ChatBox
                        messages={messages}
                        isLoading={isLoading}
                        loadingText={loadingText}
                        showMemoryIndicators={showMemoryIndicators}
                        showTimestamps={showTimestamps}
                        enableMemoryExpansion={enableMemoryExpansion}
                        copiedId={copiedId}
                        onCopyId={onCopyId}
                        className="h-full"
                    />
                </div>
                
                {/* Input Form */}
                <div className={`border-t ${borderColor} p-4`}>
                    <form onSubmit={onSubmit} className="flex gap-2">
                        <Input
                            value={input}
                            onChange={(e) => onInputChange(e.target.value)}
                            placeholder={placeholder}
                            disabled={isLoading}
                            className="flex-1"
                        />
                        <Button
                            type="submit"
                            disabled={isLoading || !input.trim()}
                            className={buttonColor}
                        >
                            <Send className="w-4 h-4" />
                        </Button>
                    </form>
                </div>
            </div>
        </div>
    )
}

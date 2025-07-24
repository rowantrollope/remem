"use client"

import { useState, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
    DialogTrigger,
} from "@/components/ui/dialog"
import { ShoppingCart, Brain, HelpCircle, Package, Plus, Check } from "lucide-react"
import { PageLayout } from "@/components/PageLayout"
import { ChatCard } from "@/components/ChatCard"
import { ChatMessage } from "@/components/ChatBox"
import { useMemoryAPI } from "@/hooks"
import { useConfiguredAPI } from "@/hooks/useConfiguredAPI"
import { useSettings } from "@/hooks/useSettings"
import { MemoryAgentAPI } from "@/lib/api"

interface RetailChatMessage extends ChatMessage {
    hasMemory?: boolean
    memory_context?: any
}

const SAMPLE_BASKET = `Current Shopping Basket:
- Organic bananas (2 lbs)
- Whole wheat bread (1 loaf)
- Greek yogurt (32 oz container)
- Chicken breast (2 lbs)
- Broccoli (1 head)
- Olive oil (500ml bottle)
- Brown rice (2 lb bag)
- Cheddar cheese (8 oz block)`

const RETAIL_AGENT_PROMPT = `You are a helpful grocery shopping assistant. You help customers analyze their shopping baskets and suggest missing items based on their shopping patterns and preferences. When analyzing a basket, consider nutritional balance, meal planning, and common grocery shopping patterns.`

const MEMORY_ENHANCED_PROMPT = `You are a personalized grocery shopping assistant with access to the customer's shopping history, dietary preferences, and household information stored in memory. When analyzing their current basket, use their past purchases and stated preferences to provide highly personalized recommendations. Always reference specific preferences or patterns from their shopping history when making suggestions.`

const CUSTOMER_PROFILE_MEMORIES = [
    "I am a health-conscious shopper who prefers organic and whole food options",
    "I follow a Mediterranean diet with lots of vegetables, fish, and olive oil",
    "I have a family of 4 including two teenagers who eat a lot",
    "I typically shop for the week and meal prep on Sundays",
    "I'm lactose intolerant so I avoid regular dairy products",
    "I love cooking Italian and Mexican cuisine at home",
    "I always buy fresh herbs like basil, cilantro, and parsley for cooking",
    "I prefer buying seasonal fruits and vegetables when possible",
    "I stock up on pantry staples like quinoa, lentils, and nuts",
    "I usually buy almond milk instead of regular milk"
]

export default function RetailDemo() {
    const { error, clearError, apiStatus } = useMemoryAPI()
    const { baseUrl } = useConfiguredAPI()
    const { settings } = useSettings()

    // Create dedicated API instance for retail demo
    const retailAPI = new MemoryAgentAPI(baseUrl, 'retail_agent_memory')

    // Session management
    const [standardSessionId, setStandardSessionId] = useState<string | null>(null)
    const [memorySessionId, setMemorySessionId] = useState<string | null>(null)
    const [hasLoadedProfile, setHasLoadedProfile] = useState(false)

    // Chat states
    const [standardMessages, setStandardMessages] = useState<RetailChatMessage[]>([])
    const [memoryMessages, setMemoryMessages] = useState<RetailChatMessage[]>([])
    const [standardQuestion, setStandardQuestion] = useState("")
    const [memoryQuestion, setMemoryQuestion] = useState("")
    const [isStandardLoading, setIsStandardLoading] = useState(false)
    const [isMemoryLoading, setIsMemoryLoading] = useState(false)

    // UI states
    const [showHelpDialog, setShowHelpDialog] = useState(false)
    const [showMemoriesDialog, setShowMemoriesDialog] = useState(false)
    const [isLoadingMemories, setIsLoadingMemories] = useState(false)
    const [copiedId, setCopiedId] = useState<string | null>(null)

    // Manual function to load memories with UI feedback
    const handleAddMemories = async () => {
        if (hasLoadedProfile) {
            console.log('Customer profile already loaded, skipping')
            setShowMemoriesDialog(false)
            return
        }

        setIsLoadingMemories(true)
        try {
            for (const memory of CUSTOMER_PROFILE_MEMORIES) {
                // Use API directly to disable grounding for sample memories
                const result = await retailAPI.remember(memory, false)
                if (!result.success) {
                    console.error('Failed to save memory:', memory)
                }
            }
            setHasLoadedProfile(true)
            setShowMemoriesDialog(false)
        } catch (error) {
            console.error('Failed to add memories:', error)
        } finally {
            setIsLoadingMemories(false)
        }
    }

    // Create standard session (no memory) - Retail Demo specific
    const createStandardSession = useCallback(async () => {
        if (standardSessionId) return standardSessionId

        try {
            const response = await retailAPI.createChatSession({
                system_prompt: RETAIL_AGENT_PROMPT,
                session_id: `retail-standard-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`,
                config: {
                    use_memory: false,
                    model: "gpt-3.5-turbo",
                    temperature: 0.7,
                    max_tokens: 1000
                }
            })
            if (response.success) {
                setStandardSessionId(response.session_id)
                console.log('Retail Demo: Created standard session:', response.session_id)
                return response.session_id
            }
        } catch (error) {
            console.error('Failed to create standard session:', error)
        }
        return null
    }, [retailAPI, standardSessionId])

    // Create memory session (with memory retrieval) - Retail Demo specific
    const createMemorySession = useCallback(async () => {
        if (memorySessionId) return memorySessionId

        try {
            const response = await retailAPI.createChatSession({
                system_prompt: MEMORY_ENHANCED_PROMPT,
                session_id: `retail-memory-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`,
                config: {
                    use_memory: true,
                    model: "gpt-3.5-turbo",
                    temperature: 0.7,
                    max_tokens: 1000,
                    top_k: settings.questionTopK
                }
            })
            if (response.success) {
                setMemorySessionId(response.session_id)
                console.log('Retail Demo: Created memory session:', response.session_id)
                return response.session_id
            }
        } catch (error) {
            console.error('Failed to create memory session:', error)
        }
        return null
    }, [retailAPI, memorySessionId, settings.questionTopK])

    const handleStandardSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        if (!standardQuestion.trim()) return

        setIsStandardLoading(true)

        try {
            // Ensure session is created and get the session ID
            let sessionId = standardSessionId
            if (!sessionId) {
                sessionId = await createStandardSession()
            }

            // Use session-based chat API (no memory)
            if (sessionId) {
                const messageWithBasket = `${SAMPLE_BASKET}\n\nQuestion: ${standardQuestion}`
                const response = await retailAPI.chatWithSession({
                    session_id: sessionId,
                    message: messageWithBasket,
                    top_k: settings.questionTopK,
                    min_similarity: settings.minSimilarity
                })

                if (response.success) {
                    const newMessage: RetailChatMessage = {
                        id: Date.now().toString(),
                        question: standardQuestion,
                        answer: response.message,
                        created_at: new Date(),
                        hasMemory: false
                    }

                    setStandardMessages(prev => [...prev, newMessage])
                    setStandardQuestion("")
                }
            }
        } catch (error) {
            console.error('Failed to get standard response:', error)
        } finally {
            setIsStandardLoading(false)
        }
    }

    const handleMemorySubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        if (!memoryQuestion.trim()) return

        setIsMemoryLoading(true)

        try {
            // Ensure session is created and get the session ID
            let sessionId = memorySessionId
            if (!sessionId) {
                sessionId = await createMemorySession()
            }

            // Use session-based chat API (with memory)
            if (sessionId) {
                const messageWithBasket = `${SAMPLE_BASKET}\n\nQuestion: ${memoryQuestion}`
                const response = await retailAPI.chatWithSession({
                    session_id: sessionId,
                    message: messageWithBasket,
                    top_k: settings.questionTopK,
                    min_similarity: settings.minSimilarity
                })

                if (response.success) {
                    const newMessage: RetailChatMessage = {
                        id: Date.now().toString(),
                        question: memoryQuestion,
                        answer: response.message,
                        created_at: new Date(),
                        hasMemory: true,
                        memory_context: response.memory_context
                    }

                    setMemoryMessages(prev => [...prev, newMessage])
                    setMemoryQuestion("")
                }
            }
        } catch (error) {
            console.error('Failed to get memory response:', error)
        } finally {
            setIsMemoryLoading(false)
        }
    }

    const clearStandardChat = () => {
        setStandardMessages([])
        setStandardSessionId(null)
    }

    const clearMemoryChat = () => {
        setMemoryMessages([])
        setMemorySessionId(null)
    }

    const handleCopyId = (id: string) => {
        navigator.clipboard.writeText(id)
        setCopiedId(id)
        setTimeout(() => setCopiedId(null), 2000)
    }

    const sampleQuestions = [
        "What's missing from this basket?",
        "What should I add for a complete week of meals?",
        "Any suggestions for healthy snacks?",
        "What ingredients do I need for meal prep?",
        "What pantry staples am I missing?"
    ]

    return (
        <PageLayout
            error={error}
            apiStatus={apiStatus}
            onClearError={clearError}
        >
            <div className="max-w-7xl mx-auto p-6 space-y-6">
                {/* Header */}
                <div className="text-center space-y-4">
                    <div className="flex items-center justify-center gap-3">
                        <ShoppingCart className="w-8 h-8 text-green-600" />
                        <h1 className="text-3xl font-bold text-gray-900">Retail Shopping Agent</h1>
                        <div className="flex gap-2">
                            <Dialog open={showMemoriesDialog} onOpenChange={setShowMemoriesDialog}>
                                <DialogTrigger asChild>
                                    <Button variant="outline" size="sm">
                                        {hasLoadedProfile ? (
                                            <>
                                                <Check className="w-4 h-4 mr-2 text-green-600" />
                                                Memories Loaded
                                            </>
                                        ) : (
                                            <>
                                                <Plus className="w-4 h-4 mr-2" />
                                                Add Shopping Memories
                                            </>
                                        )}
                                    </Button>
                                </DialogTrigger>
                                <DialogContent className="max-w-2xl">
                                    <DialogHeader>
                                        <DialogTitle>Customer Shopping Profile</DialogTitle>
                                        <DialogDescription>
                                            These memories will be added to help the memory-enhanced agent provide personalized recommendations.
                                        </DialogDescription>
                                    </DialogHeader>
                                    <div className="space-y-4">
                                        <div className="max-h-60 overflow-y-auto">
                                            <ul className="space-y-2 text-sm">
                                                {CUSTOMER_PROFILE_MEMORIES.map((memory, index) => (
                                                    <li key={index} className="flex items-start gap-2 p-2 bg-gray-50 rounded">
                                                        <span className="w-2 h-2 bg-green-500 rounded-full mt-2 flex-shrink-0"></span>
                                                        <span>{memory}</span>
                                                    </li>
                                                ))}
                                            </ul>
                                        </div>
                                        <div className="flex justify-end gap-2">
                                            <Button
                                                variant="outline"
                                                onClick={() => setShowMemoriesDialog(false)}
                                                disabled={isLoadingMemories}
                                            >
                                                Cancel
                                            </Button>
                                            <Button
                                                onClick={handleAddMemories}
                                                disabled={isLoadingMemories || hasLoadedProfile}
                                            >
                                                {isLoadingMemories ? (
                                                    "Adding Memories..."
                                                ) : hasLoadedProfile ? (
                                                    "Already Added"
                                                ) : (
                                                    "Add to Memory"
                                                )}
                                            </Button>
                                        </div>
                                    </div>
                                </DialogContent>
                            </Dialog>
                            <Dialog open={showHelpDialog} onOpenChange={setShowHelpDialog}>
                                <DialogTrigger asChild>
                                    <Button variant="outline" size="sm">
                                        <HelpCircle className="w-4 h-4 mr-2" />
                                        Sample Questions
                                    </Button>
                                </DialogTrigger>
                                <DialogContent className="max-w-2xl">
                                    <DialogHeader>
                                        <DialogTitle>Sample Questions to Try</DialogTitle>
                                        <DialogDescription>
                                            Try these questions with both the standard and memory-enhanced shopping assistants to see the difference.
                                        </DialogDescription>
                                    </DialogHeader>
                                    <div className="space-y-4">
                                        <div>
                                            <h4 className="font-semibold mb-2">Basket Analysis Questions:</h4>
                                            <ul className="space-y-1 text-sm text-gray-600">
                                                {sampleQuestions.map((question, index) => (
                                                    <li key={index} className="flex items-center gap-2">
                                                        <span className="w-2 h-2 bg-green-500 rounded-full"></span>
                                                        {question}
                                                    </li>
                                                ))}
                                            </ul>
                                        </div>
                                    </div>
                                </DialogContent>
                            </Dialog>
                        </div>
                    </div>
                    <p className="text-lg text-gray-600 max-w-3xl mx-auto">
                        Compare how a standard shopping agent vs. a memory-enhanced agent analyzes your grocery basket and provides recommendations.
                    </p>
                </div>

                {/* Current Basket Display */}
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
                    <h3 className="text-lg font-semibold text-blue-900 mb-3 flex items-center gap-2">
                        <Package className="w-5 h-5" />
                        Current Shopping Basket
                    </h3>
                    <div className="bg-white rounded-lg p-4 border">
                        <pre className="text-sm text-gray-700 whitespace-pre-wrap">{SAMPLE_BASKET}</pre>
                    </div>
                </div>

                {/* Chat Interface */}
                <Tabs defaultValue="standard" className="w-full">
                    <TabsList className="grid w-full grid-cols-2">
                        <TabsTrigger value="standard" className="flex items-center gap-2">
                            <ShoppingCart className="w-4 h-4" />
                            Standard Agent
                        </TabsTrigger>
                        <TabsTrigger value="memory" className="flex items-center gap-2">
                            <Brain className="w-4 h-4" />
                            Memory-Enhanced Agent
                        </TabsTrigger>
                    </TabsList>

                    <TabsContent value="standard" className="space-y-6">
                        <ChatCard
                            title="Standard Shopping Agent"
                            subtitle="Analyzes your basket but has no memory of your preferences, dietary restrictions, or shopping history"
                            messages={standardMessages}
                            input={standardQuestion}
                            onInputChange={setStandardQuestion}
                            onSubmit={handleStandardSubmit}
                            onClearChat={clearStandardChat}
                            isLoading={isStandardLoading}
                            placeholder="Ask about your shopping basket..."
                            headerIcon={<ShoppingCart className="w-5 h-5" />}
                            borderColor="border-orange-200"
                            headerBgColor="bg-orange-50 text-orange-800"
                            messageBgColor="bg-gray-100 text-gray-800"
                            buttonColor="bg-orange-600 hover:bg-orange-700"
                            loadingText="Analyzing basket..."
                            showMemoryIndicators={false}
                        />
                    </TabsContent>

                    <TabsContent value="memory" className="space-y-6">
                        <ChatCard
                            title="Memory-Enhanced Shopping Agent"
                            subtitle="Analyzes your basket with full knowledge of your dietary preferences, shopping patterns, and household needs"
                            messages={memoryMessages}
                            input={memoryQuestion}
                            onInputChange={setMemoryQuestion}
                            onSubmit={handleMemorySubmit}
                            onClearChat={clearMemoryChat}
                            isLoading={isMemoryLoading}
                            placeholder="Ask about your shopping basket..."
                            headerIcon={<Brain className="w-5 h-5" />}
                            borderColor="border-green-200"
                            headerBgColor="bg-green-50 text-green-800"
                            messageBgColor="bg-green-100 text-green-800"
                            buttonColor="bg-green-600 hover:bg-green-700"
                            loadingText="Analyzing with your shopping profile + history..."
                            showMemoryIndicators={true}
                            copiedId={copiedId}
                            onCopyId={handleCopyId}
                        />
                    </TabsContent>
                </Tabs>
            </div>
        </PageLayout>
    )
}
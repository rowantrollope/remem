"use client"

import { useState, useEffect, useCallback } from "react"
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
import { Plane, Brain, HelpCircle, Compass, Map, GitBranch, Upload, CheckCircle } from "lucide-react"
import { PageLayout } from "@/components/PageLayout"
import { ChatCard } from "@/components/ChatCard"
import { ChatMessage } from "@/components/ChatBox"
import { useMemoryAPI } from "@/hooks"
import { useConfiguredAPI } from "@/hooks/useConfiguredAPI"
import { useSettings } from "@/hooks/useSettings"
import { MemoryAgentAPI } from "@/lib/api"

// Travel agent system prompt
const TRAVEL_AGENT_PROMPT = `You are an expert travel agent helping users plan amazing trips. You provide personalized recommendations for destinations, accommodations, activities, restaurants, and travel logistics. You consider factors like budget, travel style, interests, dietary restrictions, and past travel experiences when making suggestions. Be helpful, enthusiastic, and detailed in your responses.`

// Sample travel memories to demonstrate the memory-enhanced agent
const sampleMemories = [
    "I prefer boutique hotels over large chain hotels because I enjoy unique, personalized experiences",
    "My budget for international trips is typically around $3000-4000 per person for a week-long vacation",
    "I'm vegetarian and always need to research restaurant options before traveling to new destinations",
    "I love adventure activities like hiking, rock climbing, and water sports when I travel",
    "I prefer traveling during shoulder seasons (spring/fall) to avoid crowds and get better prices",
    "I have a fear of flying so I prefer destinations I can reach by train or car when possible",
    "I'm interested in cultural experiences, museums, and historical sites rather than beach vacations",
    "I always book accommodations with kitchen facilities because I like to cook some of my own meals",
    "I prefer small group tours or self-guided travel rather than large tour groups",
    "I have food allergies to nuts and shellfish, so I need to be very careful about restaurant choices",
    "I love photography and always seek out scenic viewpoints and unique architecture",
    "I prefer destinations with good public transportation since I don't like renting cars abroad",
    "I enjoy wine tasting and often plan trips around visiting vineyards and wine regions",
    "I'm an early riser and like to start sightseeing early to beat the crowds",
    "I prefer staying in city centers or walkable neighborhoods rather than suburban areas"
]

// Sample questions that demonstrate memory value for travel planning
const sampleQuestions = [
    "I want to plan a week-long vacation in Europe. What do you recommend?",
    "What are the best restaurants in Tokyo for someone who loves sushi?",
    "I'm planning a family trip with kids. Where should we go?",
    "What's the best time to visit Thailand?",
    "I have a $3000 budget for a romantic getaway. Any suggestions?",
    "What should I pack for a trip to Iceland in winter?",
    "Can you recommend some off-the-beaten-path destinations?",
    "I'm a solo female traveler. Where would be safe and fun to visit?"
]



export default function TravelDemo() {
    const [activeTab, setActiveTab] = useState("standard")
    const [standardQuestion, setStandardQuestion] = useState("")
    const [memoryQuestion, setMemoryQuestion] = useState("")
    const [langgraphQuestion, setLanggraphQuestion] = useState("")
    const [standardMessages, setStandardMessages] = useState<ChatMessage[]>([])
    const [memoryMessages, setMemoryMessages] = useState<ChatMessage[]>([])
    const [langgraphMessages, setLanggraphMessages] = useState<ChatMessage[]>([])
    const [isStandardLoading, setIsStandardLoading] = useState(false)
    const [isMemoryLoading, setIsMemoryLoading] = useState(false)
    const [isLanggraphLoading, setIsLanggraphLoading] = useState(false)
    const [showHelpDialog, setShowHelpDialog] = useState(false)
    const [showMemoriesDialog, setShowMemoriesDialog] = useState(false)
    const [isLoadingMemories, setIsLoadingMemories] = useState(false)
    const [memoriesLoaded, setMemoriesLoaded] = useState(false)

    const [standardSessionId, setStandardSessionId] = useState<string | null>(null)
    const [memorySessionId, setMemorySessionId] = useState<string | null>(null)
    const { apiStatus, error, clearError, askQuestion } = useMemoryAPI()
    const { baseUrl } = useConfiguredAPI()
    const { settings } = useSettings()

    // Create dedicated API instance for travel demo
    const travelAPI = new MemoryAgentAPI(baseUrl, 'travel_agent_memory')



    // Create standard session (no memory) - Travel Demo specific
    const createStandardSession = useCallback(async () => {
        if (standardSessionId) return standardSessionId

        try {
            const standardSessionResponse = await travelAPI.createChatSession({
                system_prompt: TRAVEL_AGENT_PROMPT,
                session_id: `travel-standard-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`,
                config: {
                    use_memory: false,
                    model: "gpt-3.5-turbo",
                    temperature: 0.7,
                    max_tokens: 1000
                }
            })

            if (standardSessionResponse.success) {
                setStandardSessionId(standardSessionResponse.session_id)
                console.log('Travel Demo: Created standard session:', standardSessionResponse.session_id)
                return standardSessionResponse.session_id
            }
        } catch (error) {
            console.error('Failed to create standard session:', error)
        }
        return null
    }, [travelAPI, standardSessionId])

    // Create memory session (with memory retrieval) - Travel Demo specific
    const createMemorySession = useCallback(async () => {
        if (memorySessionId) return memorySessionId

        try {
            const memorySessionResponse = await travelAPI.createChatSession({
                system_prompt: TRAVEL_AGENT_PROMPT,
                session_id: `travel-memory-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`,
                config: {
                    use_memory: true,
                    model: "gpt-3.5-turbo",
                    temperature: 0.7,
                    max_tokens: 1000,
                    top_k: settings.questionTopK
                }
            })
            if (memorySessionResponse.success) {
                setMemorySessionId(memorySessionResponse.session_id)
                console.log('Travel Demo: Created memory session:', memorySessionResponse.session_id)
                return memorySessionResponse.session_id
            }
        } catch (error) {
            console.error('Failed to create memory session:', error)
        }
        return null
    }, [travelAPI, memorySessionId, settings.questionTopK])

    // Create sessions on component mount
    useEffect(() => {
        createStandardSession()
        createMemorySession()
    }, [createStandardSession, createMemorySession])

    // Generate standard travel agent response (no memory) using session-based API
    const generateStandardResponse = async (question: string): Promise<string> => {
        try {
            // Ensure we have a standard session for fallback responses
            if (!standardSessionId) {
                await createStandardSession()
            }

            if (standardSessionId) {
                const response = await travelAPI.chatWithSession({
                    session_id: standardSessionId,
                    message: question,
                    min_similarity: settings.minSimilarity
                })

                if (response.success) {
                    return response.message
                } else {
                    return "I'm sorry, I'm having trouble processing your request right now. Please try again."
                }
            } else {
                return "I'm sorry, I'm having trouble processing your request right now. Please try again."
            }
        } catch (error) {
            console.error('Failed to get standard response:', error)
            return "I apologize, but I'm experiencing technical difficulties. Please try again later."
        }
    }

    const handleStandardSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        if (!standardQuestion.trim()) return

        setIsStandardLoading(true)

        try {
            // Ensure session is created
            if (!standardSessionId) {
                await createStandardSession()
            }

            // Use session-based chat API (no memory)
            if (standardSessionId) {
                const response = await travelAPI.chatWithSession({
                    session_id: standardSessionId,
                    message: standardQuestion,
                    min_similarity: settings.minSimilarity
                })

                if (response.success) {
                    const newMessage: ChatMessage = {
                        id: Date.now().toString(),
                        question: standardQuestion,
                        answer: response.message,
                        created_at: new Date(),
                        hasMemory: false,
                        session_memories: response.memory_context?.memories || [],
                        excluded_memories: response.memory_context?.excluded_memories || [],
                        filtering_info: response.memory_context?.filtering_info
                    }
                    setStandardMessages(prev => [...prev, newMessage])
                } else {
                    // Fallback to standard response
                    const answer = await generateStandardResponse(standardQuestion)
                    const newMessage: ChatMessage = {
                        id: Date.now().toString(),
                        question: standardQuestion,
                        answer: answer,
                        created_at: new Date(),
                        hasMemory: false
                    }
                    setStandardMessages(prev => [...prev, newMessage])
                }
            } else {
                // Fallback to standard response if session creation failed
                const answer = await generateStandardResponse(standardQuestion)
                const newMessage: ChatMessage = {
                    id: Date.now().toString(),
                    question: standardQuestion,
                    answer: answer,
                    created_at: new Date(),
                    hasMemory: false
                }
                setStandardMessages(prev => [...prev, newMessage])
            }

            setStandardQuestion("")
        } catch (error) {
            console.error('Failed to get session response:', error)
            // Fallback to standard response
            const answer = await generateStandardResponse(standardQuestion)
            const newMessage: ChatMessage = {
                id: Date.now().toString(),
                question: standardQuestion,
                answer: answer,
                created_at: new Date(),
                hasMemory: false
            }
            setStandardMessages(prev => [...prev, newMessage])
            setStandardQuestion("")
        } finally {
            setIsStandardLoading(false)
        }
    }

    const handleMemorySubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        if (!memoryQuestion.trim()) return

        setIsMemoryLoading(true)

        try {
            // Ensure session is created
            if (!memorySessionId) {
                await createMemorySession()
            }

            // Use session-based chat API (with memory)
            if (memorySessionId) {
                const response = await travelAPI.chatWithSession({
                    session_id: memorySessionId,
                    message: memoryQuestion,
                    top_k: settings.questionTopK,
                    min_similarity: settings.minSimilarity
                })

                if (response.success) {
                    console.log('Memory session response:', response)
                    console.log('Memory context:', response.memory_context)
                    console.log('Memories:', response.memory_context?.memories)

                    const newMessage: ChatMessage = {
                        id: Date.now().toString(),
                        question: memoryQuestion,
                        answer: response.message,
                        created_at: new Date(),
                        hasMemory: true,
                        session_memories: response.memory_context?.memories || [],
                        excluded_memories: response.memory_context?.excluded_memories || [],
                        filtering_info: response.memory_context?.filtering_info
                    }
                    console.log('New message with memories:', newMessage)
                    setMemoryMessages(prev => [...prev, newMessage])
                } else {
                    // Fallback to memory API approach
                    const memoryResponse = await askQuestion(memoryQuestion, settings.questionTopK)
                    let contextualAnswer = ""
                    let confidence: 'high' | 'medium' | 'low' | undefined
                    let reasoning: string | undefined
                    let supporting_memories: unknown[] | undefined

                    if (memoryResponse && typeof memoryResponse === 'object' && memoryResponse.success) {
                        contextualAnswer = memoryResponse.conversation.answer
                        confidence = memoryResponse.conversation.confidence
                        reasoning = memoryResponse.conversation.reasoning
                        supporting_memories = memoryResponse.conversation.supporting_memories
                    } else {
                        contextualAnswer = await generateStandardResponse(memoryQuestion)
                    }

                    const newMessage: ChatMessage = {
                        id: Date.now().toString(),
                        question: memoryQuestion,
                        answer: contextualAnswer,
                        created_at: new Date(),
                        hasMemory: true,
                        confidence,
                        reasoning,
                        supporting_memories
                    }
                    setMemoryMessages(prev => [...prev, newMessage])
                }
            } else {
                // Fallback to memory API approach if session creation failed
                const memoryResponse = await askQuestion(memoryQuestion, settings.questionTopK)
                let contextualAnswer = ""
                let confidence: 'high' | 'medium' | 'low' | undefined
                let reasoning: string | undefined
                let supporting_memories: unknown[] | undefined

                if (memoryResponse && typeof memoryResponse === 'object' && memoryResponse.success) {
                    contextualAnswer = memoryResponse.conversation.answer
                    confidence = memoryResponse.conversation.confidence
                    reasoning = memoryResponse.conversation.reasoning
                    supporting_memories = memoryResponse.conversation.supporting_memories
                } else {
                    contextualAnswer = await generateStandardResponse(memoryQuestion)
                }

                const newMessage: ChatMessage = {
                    id: Date.now().toString(),
                    question: memoryQuestion,
                    answer: contextualAnswer,
                    created_at: new Date(),
                    hasMemory: true,
                    confidence,
                    reasoning,
                    supporting_memories
                }
                setMemoryMessages(prev => [...prev, newMessage])
            }

            setMemoryQuestion("")
        } catch (error) {
            console.error('Failed to get memory-enhanced response:', error)
            // Fallback to standard response
            const fallbackAnswer = await generateStandardResponse(memoryQuestion)
            const newMessage: ChatMessage = {
                id: Date.now().toString(),
                question: memoryQuestion,
                answer: fallbackAnswer,
                created_at: new Date(),
                hasMemory: false
            }
            setMemoryMessages(prev => [...prev, newMessage])
            setMemoryQuestion("")
        } finally {
            setIsMemoryLoading(false)
        }
    }

    const handleLanggraphSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        if (!langgraphQuestion.trim()) return

        setIsLanggraphLoading(true)

        try {
            const response = await fetch('http://localhost:5001/api/agent/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: langgraphQuestion,
                    system_prompt: "You are a helpful travel assistant"
                })
            })

            if (response.ok) {
                const data = await response.json()
                const newMessage: ChatMessage = {
                    id: Date.now().toString(),
                    question: langgraphQuestion,
                    answer: data.response || data.message || "I received your message but couldn't generate a response.",
                    created_at: new Date(),
                    hasMemory: false
                }
                setLanggraphMessages(prev => [...prev, newMessage])
            } else {
                const newMessage: ChatMessage = {
                    id: Date.now().toString(),
                    question: langgraphQuestion,
                    answer: "I'm sorry, I'm having trouble processing your request right now. Please try again.",
                    created_at: new Date(),
                    hasMemory: false
                }
                setLanggraphMessages(prev => [...prev, newMessage])
            }

            setLanggraphQuestion("")
        } catch (error) {
            console.error('Failed to get Langgraph response:', error)
            const newMessage: ChatMessage = {
                id: Date.now().toString(),
                question: langgraphQuestion,
                answer: "I apologize, but I'm experiencing technical difficulties. Please try again later.",
                created_at: new Date(),
                hasMemory: false
            }
            setLanggraphMessages(prev => [...prev, newMessage])
            setLanggraphQuestion("")
        } finally {
            setIsLanggraphLoading(false)
        }
    }

    const handleSampleQuestion = (question: string) => {
        setStandardQuestion(question)
        setMemoryQuestion(question)
        setLanggraphQuestion(question)
    }

    const clearStandardChat = async () => {
        setStandardMessages([])
        setStandardSessionId(null)
        // Create a new session for fresh conversation
        await createStandardSession()
    }

    const clearMemoryChat = async () => {
        setMemoryMessages([])
        setMemorySessionId(null)
        // Create a new session for fresh conversation
        await createMemorySession()
    }

    const clearLanggraphChat = () => {
        setLanggraphMessages([])
    }

    const loadSampleMemories = async () => {
        setIsLoadingMemories(true)
        try {
            let successCount = 0
            let failureCount = 0

            for (const memory of sampleMemories) {
                try {
                    // Use API directly to disable grounding for sample memories
                    const result = await travelAPI.remember(memory, false)
                    if (result.success) {
                        successCount++
                    } else {
                        failureCount++
                    }
                } catch (error) {
                    console.error('Failed to save memory:', memory, error)
                    failureCount++
                }
                // Small delay between requests to avoid overwhelming the API
                await new Promise(resolve => setTimeout(resolve, 100))
            }

            setMemoriesLoaded(true)
            console.log(`Loaded ${successCount} sample memories successfully, ${failureCount} failed`)

            // Show a success message in the memory chat
            const successMessage: ChatMessage = {
                id: Date.now().toString(),
                question: "Load Sample Memories",
                answer: `Successfully loaded ${successCount} sample travel memories! These memories include your travel preferences, budget, dietary restrictions, and travel style. Now ask me about travel recommendations and I'll use these memories to provide personalized suggestions.`,
                created_at: new Date(),
                hasMemory: true
            }
            setMemoryMessages(prev => [...prev, successMessage])

        } catch (error) {
            console.error('Failed to load sample memories:', error)
        } finally {
            setIsLoadingMemories(false)
        }
    }

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
                        <Plane className="w-8 h-8 text-blue-600" />
                        <h1 className="text-3xl font-bold text-gray-900">Travel Agent Demo</h1>
                        <div className="flex flex-col gap-2">
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
                                            Click any question to populate both travel agent tabs with the same query
                                        </DialogDescription>
                                    </DialogHeader>
                                    <div className="grid gap-2 max-h-96 overflow-y-auto">
                                        {sampleQuestions.map((question, index) => (
                                            <Button
                                                key={index}
                                                variant="ghost"
                                                className="justify-start text-left h-auto p-3 whitespace-normal"
                                                onClick={() => {
                                                    handleSampleQuestion(question)
                                                    setShowHelpDialog(false)
                                                }}
                                            >
                                                {question}
                                            </Button>
                                        ))}
                                    </div>
                                </DialogContent>
                            </Dialog>
                            <Dialog open={showMemoriesDialog} onOpenChange={setShowMemoriesDialog}>
                                <DialogTrigger asChild>
                                    <Button
                                        variant="outline"
                                        size="sm"
                                        disabled={memoriesLoaded}
                                    >
                                        {memoriesLoaded ? (
                                            <>
                                                <CheckCircle className="w-4 h-4 mr-2 text-green-600" />
                                                Memories Loaded
                                            </>
                                        ) : (
                                            <>
                                                <Upload className="w-4 h-4 mr-2" />
                                                Load Sample Memories
                                            </>
                                        )}
                                    </Button>
                                </DialogTrigger>
                                <DialogContent className="max-w-3xl max-h-[80vh] overflow-hidden flex flex-col">
                                    <DialogHeader>
                                        <DialogTitle>Sample Travel Memories</DialogTitle>
                                        <DialogDescription>
                                            These sample memories will help the memory-enhanced travel agent provide personalized recommendations based on your preferences.
                                        </DialogDescription>
                                    </DialogHeader>
                                    <div className="flex-1 overflow-y-auto space-y-2 pr-2">
                                        {sampleMemories.map((memory, index) => (
                                            <div key={index} className="p-3 bg-gray-50 rounded-lg border text-sm">
                                                {memory}
                                            </div>
                                        ))}
                                    </div>
                                    <div className="flex justify-end gap-2 pt-4 border-t">
                                        <Button
                                            variant="outline"
                                            onClick={() => setShowMemoriesDialog(false)}
                                        >
                                            Cancel
                                        </Button>
                                        <Button
                                            onClick={async () => {
                                                await loadSampleMemories()
                                                setShowMemoriesDialog(false)
                                            }}
                                            disabled={isLoadingMemories || memoriesLoaded}
                                        >
                                            {isLoadingMemories ? (
                                                <>
                                                    <div className="w-4 h-4 mr-2 animate-spin rounded-full border-2 border-gray-300 border-t-blue-600" />
                                                    Loading...
                                                </>
                                            ) : (
                                                <>
                                                    <Upload className="w-4 h-4 mr-2" />
                                                    Load Memories
                                                </>
                                            )}
                                        </Button>
                                    </div>
                                </DialogContent>
                            </Dialog>
                        </div>
                    </div>
                    <p className="text-lg text-gray-600 max-w-4xl mx-auto">
                        Compare different travel agent implementations. Switch between tabs to see how the standard agent treats each question independently,
                        the memory-enhanced agent learns your preferences, and the Langgraph agent provides advanced conversational capabilities.
                    </p>
                    <p className="text-sm text-gray-500 max-w-3xl mx-auto">
                        💡 <strong>Tip:</strong> Click &ldquo;Load Sample Memories&rdquo; to populate the memory-enhanced agent with sample travel preferences,
                        then ask questions to see how it provides personalized recommendations based on your stored preferences.
                    </p>
                </div>



                {/* Tabbed Chat Interface */}
                <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
                    <TabsList className="grid w-full grid-cols-3 mb-6 overflow-hidden">
                        <TabsTrigger value="standard" className="flex justify-start items-center gap-2 overflow-hidden">
                            <Compass className="w-4 h-4" />
                            Travel Agent (No Memory)
                        </TabsTrigger>
                        <TabsTrigger value="memory" className="flex justify-start items-center gap-2 overflow-hidden">
                            <Brain className="w-4 h-4" />
                            Travel Agent + Remem
                        </TabsTrigger>
                        <TabsTrigger value="langgraph" className="flex justify-start items-center gap-2 overflow-hidden overflow-ellipsis">
                            <GitBranch className="w-4 h-4" />
                            <div>
                            Travel Agent, Langgraph + Remem
                            </div>
                        </TabsTrigger>
                    </TabsList>

                    <TabsContent value="standard" className="space-y-6">
                        <ChatCard
                            title="Standard Travel Agent"
                            subtitle="Provides general travel advice but doesn't remember your preferences"
                            messages={standardMessages}
                            input={standardQuestion}
                            onInputChange={setStandardQuestion}
                            onSubmit={handleStandardSubmit}
                            onClearChat={clearStandardChat}
                            isLoading={isStandardLoading}
                            placeholder="Ask about travel plans..."
                            headerIcon={<Compass className="w-5 h-5" />}
                            borderColor="border-orange-200"
                            headerBgColor="bg-orange-50 text-orange-800"
                            messageBgColor="bg-orange-100 text-orange-800"
                            buttonColor="bg-orange-600 hover:bg-orange-700"
                            loadingText="Thinking..."
                            showMemoryIndicators={false}
                        />
                    </TabsContent>

                    <TabsContent value="memory" className="space-y-6">
                        <ChatCard
                            title="Travel Agent, with Remem for Memory"
                            subtitle="Learns your preferences and provides personalized recommendations"
                            messages={memoryMessages}
                            input={memoryQuestion}
                            onInputChange={setMemoryQuestion}
                            onSubmit={handleMemorySubmit}
                            onClearChat={clearMemoryChat}
                            isLoading={isMemoryLoading}
                            placeholder="Ask about travel plans..."
                            headerIcon={
                                <>
                                    <Brain className="w-5 h-5" />
                                    <Map className="w-5 h-5" />
                                </>
                            }
                            borderColor="border-green-200"
                            headerBgColor="bg-green-50 text-green-800"
                            messageBgColor="bg-green-100 text-green-800"
                            buttonColor="bg-green-600 hover:bg-green-700"
                            loadingText="Thinking..."
                            showMemoryIndicators={true}
                        />
                    </TabsContent>

                    <TabsContent value="langgraph" className="space-y-6">
                        <ChatCard
                            title="Travel Agent, Langgraph + Remem"
                            subtitle="Powered by Langgraph API + Remem for advanced travel assistance"
                            messages={langgraphMessages}
                            input={langgraphQuestion}
                            onInputChange={setLanggraphQuestion}
                            onSubmit={handleLanggraphSubmit}
                            onClearChat={clearLanggraphChat}
                            isLoading={isLanggraphLoading}
                            placeholder="Ask about travel plans..."
                            headerIcon={<GitBranch className="w-5 h-5" />}
                            borderColor="border-purple-200"
                            headerBgColor="bg-purple-50 text-purple-800"
                            messageBgColor="bg-purple-100 text-purple-800"
                            buttonColor="bg-purple-600 hover:bg-purple-700"
                            loadingText="Processing with Langgraph..."
                            showMemoryIndicators={false}
                        />
                    </TabsContent>
                </Tabs>

                {/* Instructions */}
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
                    <h3 className="text-lg font-semibold text-blue-900 mb-3">How to Use This Demo</h3>
                    <div className="grid md:grid-cols-2 gap-4 text-sm text-blue-800">
                        <div>
                            <h4 className="font-medium mb-2">1. Load Sample Memories</h4>
                            <p>Click &ldquo;Load Sample Memories&rdquo; to populate the memory-enhanced agent with sample travel preferences and constraints.</p>
                        </div>
                        <div>
                            <h4 className="font-medium mb-2">2. Switch Between Tabs</h4>
                            <p>Use the tabs above to switch between the Standard, Memory-Enhanced, and Langgraph travel agents.</p>
                        </div>
                        <div>
                            <h4 className="font-medium mb-2">3. Ask Similar Questions</h4>
                            <p>Try asking the same questions in all tabs to compare the different response styles and capabilities.</p>
                        </div>
                        <div>
                            <h4 className="font-medium mb-2">4. Notice the Difference</h4>
                            <p>See how the memory-enhanced agent provides more personalized and contextual recommendations based on stored preferences.</p>
                        </div>
                    </div>
                </div>
            </div>
        </PageLayout>
    )
}

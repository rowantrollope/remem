"use client"

import { useState, useEffect, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
    DialogTrigger,
} from "@/components/ui/dialog"
import { Brain, Code, GitPullRequest, Lightbulb, HelpCircle } from "lucide-react"
import { PageLayout } from "@/components/PageLayout"
import { ChatCard } from "@/components/ChatCard"
import { ChatMessage } from "@/components/ChatBox"
import { useMemoryAPI } from "@/hooks"
import { useConfiguredAPI } from "@/hooks/useConfiguredAPI"
import { useSettings } from "@/hooks/useSettings"

// Code review agent system prompt
const CODE_REVIEW_PROMPT = `You are an expert code review assistant helping developers improve their code quality. You provide detailed, constructive feedback on code snippets, focusing on best practices, potential issues, performance optimizations, and maintainability. You consider the developer's coding style preferences, past feedback patterns, and project context when making suggestions. Be thorough, helpful, and educational in your responses.`

// Sample code snippets for the demo
const codeSnippets = [
    {
        id: 1,
        title: "React Component",
        language: "typescript",
        code: `function UserProfile({ userId }: { userId: string }) {
    const [user, setUser] = useState(null)
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        fetch(\`/api/users/\${userId}\`)
            .then(res => res.json())
            .then(data => {
                setUser(data)
                setLoading(false)
            })
    }, [userId])

    if (loading) return <div>Loading...</div>

    return (
        <div>
            <h1>{user.name}</h1>
            <p>{user.email}</p>
        </div>
    )
}`
    },
    {
        id: 2,
        title: "API Endpoint",
        language: "python",
        code: `def get_user_orders(user_id):
    orders = []
    for order in db.orders.find({"user_id": user_id}):
        orders.append({
            "id": order["_id"],
            "total": order["total"],
            "items": order["items"]
        })
    return orders`
    },
    {
        id: 3,
        title: "Database Query",
        language: "sql",
        code: `SELECT u.name, u.email, COUNT(o.id) as order_count
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.created_at > '2024-01-01'
GROUP BY u.id
ORDER BY order_count DESC`
    }
]

// Sample questions that demonstrate memory value for code reviews
const sampleQuestions = [
    "Please review this React component for best practices and potential improvements.",
    "What are the potential issues with this API endpoint implementation?",
    "How can I optimize this database query for better performance?",
    "Are there any security concerns with this code?",
    "What TypeScript improvements would you suggest for this component?",
    "How can I make this code more maintainable and readable?",
    "What error handling should I add to this function?",
    "Are there any accessibility issues with this React component?"
]

export default function AgentDemo() {
    const [selectedSnippet, setSelectedSnippet] = useState(codeSnippets[0])
    const [activeTab, setActiveTab] = useState("standard")
    const [standardQuestion, setStandardQuestion] = useState("")
    const [memoryQuestion, setMemoryQuestion] = useState("")
    const [standardMessages, setStandardMessages] = useState<ChatMessage[]>([])
    const [memoryMessages, setMemoryMessages] = useState<ChatMessage[]>([])
    const [isStandardLoading, setIsStandardLoading] = useState(false)
    const [isMemoryLoading, setIsMemoryLoading] = useState(false)
    const [showHelpDialog, setShowHelpDialog] = useState(false)
    const [standardSessionId, setStandardSessionId] = useState<string | null>(null)
    const [memorySessionId, setMemorySessionId] = useState<string | null>(null)
    const { apiStatus, error, clearError, askQuestion } = useMemoryAPI()
    const { api } = useConfiguredAPI()
    const { settings } = useSettings()

    // Create standard session (no memory) - Agent Demo specific
    const createStandardSession = useCallback(async () => {
        if (standardSessionId) return standardSessionId

        try {
            const standardSessionResponse = await api.createChatSession({
                system_prompt: CODE_REVIEW_PROMPT,
                session_id: `agent-standard-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`,
                config: {
                    use_memory: false,
                    model: "gpt-3.5-turbo",
                    temperature: 0.7,
                    max_tokens: 1000
                }
            })

            if (standardSessionResponse.success) {
                setStandardSessionId(standardSessionResponse.session_id)
                console.log('Agent Demo: Created standard session:', standardSessionResponse.session_id)
                return standardSessionResponse.session_id
            }
        } catch (error) {
            console.error('Failed to create standard session:', error)
        }
        return null
    }, [api, standardSessionId])

    // Create memory session (with memory retrieval) - Agent Demo specific
    const createMemorySession = useCallback(async () => {
        if (memorySessionId) return memorySessionId

        try {
            const memorySessionResponse = await api.createChatSession({
                system_prompt: CODE_REVIEW_PROMPT,
                session_id: `agent-memory-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`,
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
                console.log('Agent Demo: Created memory session:', memorySessionResponse.session_id)
                return memorySessionResponse.session_id
            }
        } catch (error) {
            console.error('Failed to create memory session:', error)
        }
        return null
    }, [api, memorySessionId, settings.questionTopK])

    // Create sessions on component mount
    useEffect(() => {
        createStandardSession()
        createMemorySession()
    }, [createStandardSession, createMemorySession])

    // Generate standard code review response (no memory) using session-based API
    const generateStandardResponse = async (question: string): Promise<string> => {
        try {
            // Ensure we have a standard session for fallback responses
            if (!standardSessionId) {
                await createStandardSession()
            }

            if (standardSessionId) {
                const response = await api.chatWithSession({
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
                const response = await api.chatWithSession({
                    session_id: standardSessionId,
                    message: standardQuestion,
                    min_similarity: settings.minSimilarity
                })

                if (response.success) {
                    const newMessage: ChatMessage = {
                        id: Date.now().toString(),
                        question: standardQuestion,
                        answer: response.message,
                        timestamp: new Date(),
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
                        timestamp: new Date(),
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
                    timestamp: new Date(),
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
                timestamp: new Date(),
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
                const response = await api.chatWithSession({
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
                    const memoryResponse = await askQuestion(memoryQuestion, settings.questionTopK, settings.minSimilarity)
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
                        timestamp: new Date(),
                        hasMemory: true,
                        confidence,
                        reasoning,
                        supporting_memories
                    }
                    setMemoryMessages(prev => [...prev, newMessage])
                }
            } else {
                // Fallback to memory API approach if session creation failed
                const memoryResponse = await askQuestion(memoryQuestion, settings.questionTopK, settings.minSimilarity)
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

    const handleSampleQuestion = (question: string) => {
        setStandardQuestion(question)
        setMemoryQuestion(question)
    }

    const handleCodeSnippetSelect = (snippet: typeof codeSnippets[0]) => {
        setSelectedSnippet(snippet)
        const codeContext = `Please review this ${snippet.language} code and provide feedback:\n\n${snippet.title}:\n${snippet.code}`
        setStandardQuestion(codeContext)
        setMemoryQuestion(codeContext)
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
                        <GitPullRequest className="w-8 h-8 text-blue-600" />
                        <h1 className="text-3xl font-bold text-gray-900">Code Review Assistant Demo</h1>
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
                                        Click any question to populate both code review agent tabs with the same query
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
                    </div>
                    <p className="text-lg text-gray-600 max-w-4xl mx-auto">
                        Compare how memory transforms code reviews. Switch between tabs to see how the standard agent provides generic feedback,
                        while the memory-enhanced agent learns your coding preferences and provides increasingly personalized reviews.
                    </p>
                </div>



                {/* Code Selection */}
                <Card>
                    <CardHeader>
                        <CardTitle>Select Code to Review</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            {codeSnippets.map((snippet) => (
                                <Button
                                    key={snippet.id}
                                    variant={selectedSnippet.id === snippet.id ? "default" : "outline"}
                                    onClick={() => handleCodeSnippetSelect(snippet)}
                                    className="h-auto p-4 text-left"
                                >
                                    <div>
                                        <div className="font-medium">{snippet.title}</div>
                                        <div className="text-sm text-muted-foreground">{snippet.language}</div>
                                    </div>
                                </Button>
                            ))}
                        </div>
                    </CardContent>
                </Card>

                {/* Code Display */}
                <Card>
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <Code className="w-5 h-5" />
                            {selectedSnippet.title}
                            <Badge variant="secondary">{selectedSnippet.language}</Badge>
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        <pre className="bg-gray-100 p-4 rounded-lg overflow-x-auto">
                            <code>{selectedSnippet.code}</code>
                        </pre>
                    </CardContent>
                </Card>

                {/* Tabbed Chat Interface */}
                <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
                    <TabsList className="grid w-full grid-cols-2 mb-6">
                        <TabsTrigger value="standard" className="flex items-center gap-2">
                            <Code className="w-4 h-4" />
                            Standard Code Review Agent
                        </TabsTrigger>
                        <TabsTrigger value="memory" className="flex items-center gap-2">
                            <Brain className="w-4 h-4" />
                            Memory-Enhanced Code Review Agent
                        </TabsTrigger>
                    </TabsList>

                    <TabsContent value="standard" className="space-y-6">
                        <ChatCard
                            title="Standard Code Review Agent"
                            subtitle="Provides general code feedback but doesn't remember your preferences"
                            messages={standardMessages}
                            input={standardQuestion}
                            onInputChange={setStandardQuestion}
                            onSubmit={handleStandardSubmit}
                            onClearChat={clearStandardChat}
                            isLoading={isStandardLoading}
                            placeholder="Ask for code review..."
                            headerIcon={<Code className="w-5 h-5" />}
                            borderColor="border-orange-200"
                            headerBgColor="bg-orange-50 text-orange-800"
                            messageBgColor="bg-orange-100 text-orange-800"
                            buttonColor="bg-orange-600 hover:bg-orange-700"
                            loadingText="Reviewing code..."
                            showMemoryIndicators={false}
                        />
                    </TabsContent>

                    <TabsContent value="memory" className="space-y-6">
                        <ChatCard
                            title="Memory-Enhanced Code Review Agent"
                            subtitle="Learns your coding preferences and provides personalized feedback"
                            messages={memoryMessages}
                            input={memoryQuestion}
                            onInputChange={setMemoryQuestion}
                            onSubmit={handleMemorySubmit}
                            onClearChat={clearMemoryChat}
                            isLoading={isMemoryLoading}
                            placeholder="Ask for code review..."
                            headerIcon={
                                <>
                                    <Brain className="w-5 h-5" />
                                    <Code className="w-5 h-5" />
                                </>
                            }
                            borderColor="border-green-200"
                            headerBgColor="bg-green-50 text-green-800"
                            messageBgColor="bg-green-100 text-green-800"
                            buttonColor="bg-green-600 hover:bg-green-700"
                            loadingText="Checking my memory and reviewing code..."
                            showMemoryIndicators={true}
                        />
                    </TabsContent>
                </Tabs>

                {/* Instructions */}
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
                    <h3 className="text-lg font-semibold text-blue-900 mb-3">How to Use This Demo</h3>
                    <div className="grid md:grid-cols-2 gap-4 text-sm text-blue-800">
                        <div>
                            <h4 className="font-medium mb-2">1. Select Code to Review</h4>
                            <p>Choose from the sample code snippets above to automatically populate both agent tabs with a review request.</p>
                        </div>
                        <div>
                            <h4 className="font-medium mb-2">2. Switch Between Tabs</h4>
                            <p>Use the tabs to switch between the Standard and Memory-Enhanced code review agents.</p>
                        </div>
                        <div>
                            <h4 className="font-medium mb-2">3. Share Your Preferences</h4>
                            <p>Tell the memory-enhanced agent about your coding style, team conventions, or specific concerns.</p>
                        </div>
                        <div>
                            <h4 className="font-medium mb-2">4. Compare Responses</h4>
                            <p>Notice how the memory-enhanced agent provides more personalized and contextual feedback over time.</p>
                        </div>
                    </div>
                </div>

                {/* Explanation Section */}
                <Card className="mt-8">
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <Lightbulb className="w-5 h-5" />
                            Why Memory Matters for Code Review Agents
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div className="grid md:grid-cols-2 gap-6">
                            <div>
                                <h4 className="font-semibold text-orange-800 mb-2">Without Memory</h4>
                                <ul className="space-y-1 text-sm text-gray-600">
                                    <li>• Generic, one-size-fits-all feedback</li>
                                    <li>• No learning from past code reviews</li>
                                    <li>• Repetitive suggestions</li>
                                    <li>• No awareness of coding style preferences</li>
                                    <li>• Limited understanding of team conventions</li>
                                </ul>
                            </div>
                            <div>
                                <h4 className="font-semibold text-green-800 mb-2">With Memory</h4>
                                <ul className="space-y-1 text-sm text-gray-600">
                                    <li>• Personalized code review feedback</li>
                                    <li>• Learns from your coding patterns</li>
                                    <li>• Remembers your style preferences</li>
                                    <li>• Aware of project context and standards</li>
                                    <li>• Improves recommendations over time</li>
                                </ul>
                            </div>
                        </div>
                        <div className="bg-blue-50 p-4 rounded-lg">
                            <p className="text-blue-800 text-sm">
                                <strong>Real-world Impact:</strong> Memory-enhanced code review agents can reduce review time by 40-60%
                                by providing more relevant, actionable feedback that aligns with team standards and individual coding preferences.
                            </p>
                        </div>
                    </CardContent>
                </Card>
            </div>
        </PageLayout>
    )
}

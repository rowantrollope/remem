"use client"

import { useState, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Textarea } from "@/components/ui/textarea"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
    DialogTrigger,
} from "@/components/ui/dialog"
import { TrendingUp, Brain, Database, HelpCircle, FileText, BarChart3, Eye, Trash2, CheckCircle } from "lucide-react"
import { PageLayout } from "@/components/PageLayout"
import { ChatCard } from "@/components/ChatCard"
import { ChatMessage } from "@/components/ChatBox"
import { useMemoryAPI } from "@/hooks"
import { useSettings } from "@/hooks/useSettings"
import { useConfiguredAPI } from "@/hooks/useConfiguredAPI"
import { MemoryAgentAPI } from "@/lib/api"

// Complete Microsoft 2024 Annual Report content for comprehensive analysis
const microsoft2024Annual = `MICROSOFT CORPORATION
2024 ANNUAL REPORT
For the Year Ended June 30, 2024

CONSOLIDATED INCOME STATEMENTS
(In millions, except per share amounts)

Year Ended June 30,                    2024        2023        2022
Revenue:
Product                              $69,274     $72,732     $72,732
Service and other                    176,075     139,267     125,538
Total revenue                        245,122     211,915     198,270

Cost of revenue:
Product                               19,715      18,219      19,715
Service and other                     46,078      43,586      46,078
Total cost of revenue                 65,525      62,650      65,525

Gross margin                         179,597     149,265     132,745

Research and development              29,515      27,195      24,512
Sales and marketing                   22,759      22,759      21,825
General and administrative            7,395       6,768       5,900

Operating income                     119,928      93,543      80,508
Other income (expense), net            3,593       1,768       1,186
Income before income taxes           123,521      95,311      81,694
Provision for income taxes            19,621      16,950      10,978
Net income                          $103,900     $78,361     $70,716

Earnings per share:
Basic                                 $13.95      $10.51      $9.50
Diluted                               $13.89      $10.44      $9.42

BALANCE SHEET HIGHLIGHTS (June 30, 2024):
Total assets: $512.1 billion
Cash and cash equivalents: $75.0 billion
Total stockholders' equity: $238.4 billion
Total debt: $97.1 billion

BUSINESS SEGMENT PERFORMANCE:
Productivity and Business Processes: $69.3 billion (+12% YoY)
- Microsoft 365 Commercial revenue: $28.0 billion (+15% YoY)
- Microsoft Teams: 320 million monthly active users
- LinkedIn revenue: $15.7 billion (+10% YoY)

Intelligent Cloud: $87.9 billion (+19% YoY)
- Azure and other cloud services: +31% YoY growth
- SQL Server: +16% YoY growth
- Windows Server: +18% YoY growth

More Personal Computing: $54.7 billion (+17% YoY)
- Windows revenue: $13.5 billion (+11% YoY)
- Xbox content and services: $16.2 billion (+61% YoY)
- Search and news advertising: $12.0 billion (+8% YoY)

CASH FLOW STATEMENT HIGHLIGHTS:
Operating cash flow: $118.3 billion
Free cash flow: $101.9 billion
Capital expenditures: $16.4 billion
Share repurchases: $16.1 billion
Dividends paid: $22.0 billion

KEY FINANCIAL METRICS:
Return on equity: 43.6%
Operating margin: 48.9%
Debt-to-equity ratio: 0.41
Current ratio: 1.7

STRATEGIC INITIATIVES & INVESTMENTS:
AI and Machine Learning:
- $13.0 billion investment in OpenAI partnership
- Azure AI services revenue growth of 60%
- GitHub Copilot: 1.8 million paid subscribers
- Microsoft 365 Copilot adoption across Fortune 500

Cloud Infrastructure:
- 60+ Azure regions globally
- $50+ billion in cloud infrastructure investments
- 99.9% uptime SLA across all services

Acquisitions:
- Activision Blizzard: $68.7 billion (completed October 2023)
- Nuance Communications integration completed
- Multiple AI and gaming studio acquisitions

RISK FACTORS:
Operational Risks:
- Intense competition in cloud computing and AI services
- Cybersecurity threats and data protection requirements
- Regulatory scrutiny and antitrust investigations
- Dependence on third-party suppliers and partners
- Talent acquisition and retention in competitive markets

Financial Risks:
- Foreign exchange rate fluctuations
- Interest rate changes affecting debt and investments
- Economic downturns affecting enterprise IT spending
- Credit risks from customer defaults

Technology Risks:
- Rapid technological changes requiring continuous innovation
- AI safety and ethical considerations
- Data privacy and protection regulations
- Intellectual property disputes and litigation

Market Risks:
- Competition from Google, Amazon, Apple, and emerging players
- Shift in customer preferences and buying patterns
- Geopolitical tensions affecting international operations
- Supply chain disruptions and component shortages

FORWARD-LOOKING STATEMENTS:
Microsoft expects continued strong growth in cloud services, driven by AI and machine learning capabilities. The company is investing heavily in data center infrastructure, AI research, and strategic partnerships to maintain competitive advantages. Key growth drivers include:

1. AI Integration: Expanding Copilot across all product lines
2. Cloud Expansion: Targeting $100B+ annual cloud revenue
3. Gaming Growth: Leveraging Activision Blizzard content
4. Enterprise Solutions: Deepening Microsoft 365 penetration
5. Emerging Markets: Expanding presence in developing economies

The company projects 15-20% annual revenue growth over the next 3-5 years, with operating margins remaining above 45%. Capital expenditures are expected to increase to $20-25 billion annually to support AI and cloud infrastructure expansion.

MANAGEMENT DISCUSSION:
CEO Satya Nadella emphasized Microsoft's transformation into an AI-first company, stating: "We are moving from talking about AI to helping our customers translate it into real outcomes. Our comprehensive AI platform, from infrastructure to applications, positions us uniquely for the next decade of growth."

CFO Amy Hood highlighted the company's financial strength: "Our diversified revenue streams, strong cash generation, and disciplined capital allocation enable us to invest aggressively in growth opportunities while returning significant value to shareholders."

SUSTAINABILITY AND ESG:
- Carbon negative by 2030 commitment
- $1 billion Climate Innovation Fund
- 100% renewable energy for operations by 2025
- Diversity and inclusion initiatives across all business units
- $250 million employee giving and volunteering programs`

// Curated investor profile memories for realistic financial analysis
const investorProfileMemories = [
    "I am a value investor focused on undervalued companies with strong fundamentals and sustainable competitive advantages",
    "I prioritize Free Cash Flow (FCF) as my primary valuation metric, looking for companies generating consistent cash flows",
    "I prefer dividend-paying stocks for steady income and companies with a history of dividend growth",
    "I avoid speculative tech stocks and focus on established companies with proven business models",
    "I invest with a 5-10 year holding period and am not concerned with short-term market volatility",
    "I look for companies with strong balance sheets, low debt-to-equity ratios, and conservative financial management",
    "I prefer companies with economic moats - sustainable competitive advantages that protect market share",
    "I focus on Return on Equity (ROE) above 15% as an indicator of management effectiveness",
    "I avoid companies with high capital expenditure requirements that eat into free cash flow",
    "I prefer companies trading below their intrinsic value based on discounted cash flow analysis",
    "I look for management teams with strong track records and shareholder-friendly capital allocation",
    "I prefer companies with diversified revenue streams to reduce business risk and cyclicality"
]

// Enhanced sample questions that demonstrate memory-enhanced financial analysis
const sampleQuestions = [
    "Based on my investment criteria, is Microsoft a good value investment?",
    "How does Microsoft's free cash flow generation align with my investment philosophy?",
    "Does Microsoft's dividend policy match my income-focused investment strategy?",
    "What is Microsoft's economic moat and how sustainable is it?",
    "How does Microsoft's ROE compare to my 15% minimum threshold?",
    "Should I be concerned about Microsoft's capital expenditure increases?",
    "How does Microsoft's debt level align with my preference for conservative balance sheets?",
    "Does Microsoft's diversified revenue model reduce investment risk for my portfolio?",
    "What are the key risks to Microsoft's competitive advantages?",
    "How does Microsoft's valuation compare to my intrinsic value calculations?"
]

// Memory-enhanced analysis context for financial document processing
const documentAnalysisMemories = [
    "When analyzing financial statements, always cross-reference with the investor's stated preferences and criteria",
    "Focus on metrics that align with the investor's value-oriented approach: FCF, ROE, debt levels, and dividend sustainability",
    "Consider the investor's 5-10 year time horizon when evaluating short-term fluctuations and quarterly results",
    "Emphasize sustainable competitive advantages and economic moats when discussing business performance",
    "Relate financial performance to the investor's preference for established companies over speculative investments"
]



export default function InvestmentDemo() {
    const [activeTab, setActiveTab] = useState("standard")
    const [standardQuestion, setStandardQuestion] = useState("")
    const [memoryQuestion, setMemoryQuestion] = useState("")
    const [standardMessages, setStandardMessages] = useState<ChatMessage[]>([])
    const [memoryMessages, setMemoryMessages] = useState<ChatMessage[]>([])
    const [isStandardLoading, setIsStandardLoading] = useState(false)
    const [isMemoryLoading, setIsMemoryLoading] = useState(false)
    const [hasLoadedProfile, setHasLoadedProfile] = useState(false)
    const [showHelpDialog, setShowHelpDialog] = useState(false)
    const [showMemoriesDialog, setShowMemoriesDialog] = useState(false)
    const [showLoadMemoriesDialog, setShowLoadMemoriesDialog] = useState(false)
    const [isLoadingProfile, setIsLoadingProfile] = useState(false)
    const [currentMemories, setCurrentMemories] = useState<any[]>([])
    const [documentContent, setDocumentContent] = useState(microsoft2024Annual)
    const [standardSessionId, setStandardSessionId] = useState<string | null>(null)
    const [memorySessionId, setMemorySessionId] = useState<string | null>(null)
    const { apiStatus, error, clearError } = useMemoryAPI()
    const { baseUrl } = useConfiguredAPI()
    const { settings } = useSettings()

    // Create dedicated API instance for investment demo
    const investmentAPI = new MemoryAgentAPI(baseUrl, 'investment_agent_memory')



    // Create standard session (no memory) - Investment Demo specific
    const createStandardSession = useCallback(async () => {
        if (standardSessionId) return standardSessionId

        try {
            const response = await investmentAPI.createChatSession({
                system_prompt: `You are a financial analyst helping to analyze Microsoft's 2024 Annual Report. You have access to the complete financial document but no memory of previous conversations. Provide detailed analysis based on the document content. Here is the document: ${documentContent}`,
                session_id: `investment-standard-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`,
                config: {
                    use_memory: false,
                    model: "gpt-3.5-turbo",
                    temperature: 0.7,
                    max_tokens: 1000
                }
            })
            if (response.success) {
                setStandardSessionId(response.session_id)
                console.log('Investment Demo: Created standard session:', response.session_id)
                return response.session_id
            }
        } catch (error) {
            console.error('Failed to create standard session:', error)
        }
        return null
    }, [standardSessionId, documentContent])

    // Create memory session (with memory retrieval) - Investment Demo specific
    const createMemorySession = useCallback(async () => {
        if (memorySessionId) return memorySessionId

        try {
            const response = await investmentAPI.createChatSession({
                system_prompt: `You are a personalized financial advisor with access to Microsoft's 2024 Annual Report and the investor's profile and preferences stored in memory. Use both the document and the investor's stated criteria to provide tailored investment analysis. Always reference the investor's specific preferences when making recommendations. Here is the document: ${documentContent}`,
                session_id: `investment-memory-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`,
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
                console.log('Investment Demo: Created memory session:', response.session_id)
                return response.session_id
            }
        } catch (error) {
            console.error('Failed to create memory session:', error)
        }
        return null
    }, [memorySessionId, documentContent, settings.questionTopK])

    // Load sample investor profile into memory
    const loadInvestorProfile = async () => {
        if (hasLoadedProfile) return

        setIsLoadingProfile(true)
        try {
            // Load investor profile memories with grounding disabled
            for (const memory of investorProfileMemories) {
                await investmentAPI.remember(memory, false)
            }

            // Load document analysis context with grounding disabled
            for (const memory of documentAnalysisMemories) {
                await investmentAPI.remember(memory, false)
            }

            setHasLoadedProfile(true)
            await refreshCurrentMemories()
        } catch (error) {
            console.error('Failed to load investor profile:', error)
        } finally {
            setIsLoadingProfile(false)
        }
    }

    // View current memories
    const refreshCurrentMemories = async () => {
        try {
            const response = await investmentAPI.recall("investor profile financial analysis", 20)
            if (response.success) {
                setCurrentMemories(response.memories || [])
            }
        } catch (error) {
            console.error('Failed to fetch current memories:', error)
        }
    }

    // Clear all memories
    const clearAllMemories = async () => {
        try {
            await investmentAPI.clearAllMemories()
            setHasLoadedProfile(false)
            setCurrentMemories([])
        } catch (error) {
            console.error('Failed to clear memories:', error)
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
                const response = await investmentAPI.chatWithSession({
                    session_id: standardSessionId,
                    message: standardQuestion,
                    top_k: settings.questionTopK,
                    min_similarity: settings.minSimilarity
                })

                if (response.success) {
                    const newMessage: ChatMessage = {
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
            // Ensure investor profile is loaded
            if (!hasLoadedProfile) {
                await loadInvestorProfile()
            }

            // Ensure session is created
            if (!memorySessionId) {
                await createMemorySession()
            }

            // Use session-based chat API (with memory)
            if (memorySessionId) {
                const response = await investmentAPI.chatWithSession({
                    session_id: memorySessionId,
                    message: memoryQuestion,
                    top_k: settings.questionTopK,
                    min_similarity: settings.minSimilarity
                })

                if (response.success) {
                    const newMessage: ChatMessage = {
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
            console.error('Failed to get memory-enhanced response:', error)
        } finally {
            setIsMemoryLoading(false)
        }
    }

    const handleSampleQuestion = (question: string) => {
        setStandardQuestion(question)
        setMemoryQuestion(question)
    }

    const clearStandardChat = () => {
        setStandardMessages([])
    }

    const clearMemoryChat = () => {
        setMemoryMessages([])
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
                        <TrendingUp className="w-8 h-8 text-green-600" />
                        <h1 className="text-3xl font-bold text-gray-900">Investment Agent Demo</h1>
                        <div className="flex gap-2">
                            <Dialog open={showHelpDialog} onOpenChange={setShowHelpDialog}>
                                <DialogTrigger asChild>
                                    <Button variant="outline" size="sm">
                                        <HelpCircle className="w-4 h-4 mr-2" />
                                        Sample Questions
                                    </Button>
                                </DialogTrigger>
                                <DialogContent className="max-w-3xl">
                                    <DialogHeader>
                                        <DialogTitle>Sample Investment Questions</DialogTitle>
                                        <DialogDescription>
                                            These questions demonstrate how memory-enhanced analysis provides personalized investment insights
                                        </DialogDescription>
                                    </DialogHeader>
                                    <div className="space-y-3 py-4">
                                        {sampleQuestions.map((question, index) => (
                                            <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                                                <span className="text-sm">{question}</span>
                                                <Button
                                                    size="sm"
                                                    variant="outline"
                                                    onClick={() => {
                                                        handleSampleQuestion(question)
                                                        setShowHelpDialog(false)
                                                    }}
                                                >
                                                    Use This
                                                </Button>
                                            </div>
                                        ))}
                                    </div>
                                </DialogContent>
                            </Dialog>
                            <Dialog open={showMemoriesDialog} onOpenChange={setShowMemoriesDialog}>
                                <DialogTrigger asChild>
                                    <Button variant="outline" size="sm">
                                        <Eye className="w-4 h-4 mr-2" />
                                        View Memories ({currentMemories.length})
                                    </Button>
                                </DialogTrigger>
                                <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto">
                                    <DialogHeader>
                                        <DialogTitle>Current Investor Profile & Analysis Context</DialogTitle>
                                        <DialogDescription>
                                            These memories shape how the memory-enhanced analyst provides personalized investment advice
                                        </DialogDescription>
                                    </DialogHeader>
                                    <div className="space-y-3 py-4">
                                        {currentMemories.length === 0 ? (
                                            <p className="text-gray-500 text-center py-8">No memories loaded. Load the sample investor profile to see how memory enhances analysis.</p>
                                        ) : (
                                            currentMemories.map((memory, index) => (
                                                <div key={index} className="p-3 bg-blue-50 rounded-lg border border-blue-200">
                                                    <p className="text-sm text-blue-900">{memory.text || memory.content}</p>
                                                    <div className="flex justify-between items-center mt-2 text-xs text-blue-600">
                                                        <span>Relevance: {(memory.relevance_score || memory.score || 0).toFixed(3)}</span>
                                                        <span>{new Date(memory.created_at).toLocaleDateString()}</span>
                                                    </div>
                                                </div>
                                            ))
                                        )}
                                    </div>
                                </DialogContent>
                            </Dialog>
                        </div>
                    </div>
                    <p className="text-lg text-gray-600 max-w-5xl mx-auto">
                        Experience how conversational memory transforms generic financial document analysis into personalized investment agent.
                        The memory-enhanced analyst remembers your investment philosophy, preferences, and criteria to provide tailored insights
                        that align with your specific investment strategy.
                    </p>
                </div>

                {/* Memory Management Section */}
                <Card className="bg-gradient-to-r from-purple-50 to-blue-50 border-purple-200">
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <Brain className="w-5 h-5 text-purple-600" />
                            Investor Profile & Memory Management
                        </CardTitle>
                        <p className="text-sm text-gray-600">
                            Load a sample investor profile to see how memory personalizes financial analysis
                        </p>
                    </CardHeader>
                    <CardContent>
                        <div className="flex flex-wrap gap-3">
                            <Dialog open={showLoadMemoriesDialog} onOpenChange={setShowLoadMemoriesDialog}>
                                <DialogTrigger asChild>
                                    <Button
                                        disabled={hasLoadedProfile}
                                        className="bg-purple-600 hover:bg-purple-700"
                                    >
                                        <Database className="w-4 h-4 mr-2" />
                                        {hasLoadedProfile ? "Profile Loaded" : "Load Sample Investor Profile"}
                                    </Button>
                                </DialogTrigger>
                                <DialogContent className="max-w-4xl max-h-[80vh] overflow-hidden flex flex-col">
                                    <DialogHeader>
                                        <DialogTitle>Sample Investor Profile & Analysis Context</DialogTitle>
                                        <DialogDescription>
                                            These memories will help the memory-enhanced investment agent provide personalized financial analysis based on your investment philosophy and preferences.
                                        </DialogDescription>
                                    </DialogHeader>
                                    <div className="flex-1 overflow-y-auto space-y-4 pr-2">
                                        <div>
                                            <h4 className="font-semibold text-gray-800 mb-2">Investor Profile Memories:</h4>
                                            <div className="space-y-2">
                                                {investorProfileMemories.map((memory, index) => (
                                                    <div key={index} className="p-3 bg-purple-50 rounded-lg border text-sm">
                                                        {memory}
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                        <div>
                                            <h4 className="font-semibold text-gray-800 mb-2">Document Analysis Context:</h4>
                                            <div className="space-y-2">
                                                {documentAnalysisMemories.map((memory, index) => (
                                                    <div key={index} className="p-3 bg-blue-50 rounded-lg border text-sm">
                                                        {memory}
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    </div>
                                    <div className="flex justify-end gap-2 pt-4 border-t">
                                        <Button
                                            variant="outline"
                                            onClick={() => setShowLoadMemoriesDialog(false)}
                                        >
                                            Cancel
                                        </Button>
                                        <Button
                                            onClick={async () => {
                                                await loadInvestorProfile()
                                                setShowLoadMemoriesDialog(false)
                                            }}
                                            disabled={isLoadingProfile || hasLoadedProfile}
                                            className="bg-purple-600 hover:bg-purple-700"
                                        >
                                            {isLoadingProfile ? (
                                                <>
                                                    <div className="w-4 h-4 mr-2 animate-spin rounded-full border-2 border-gray-300 border-t-purple-600" />
                                                    Loading...
                                                </>
                                            ) : (
                                                <>
                                                    <Database className="w-4 h-4 mr-2" />
                                                    Load Profile
                                                </>
                                            )}
                                        </Button>
                                    </div>
                                </DialogContent>
                            </Dialog>
                            <Button
                                onClick={() => {
                                    refreshCurrentMemories()
                                    setShowMemoriesDialog(true)
                                }}
                                variant="outline"
                            >
                                <Eye className="w-4 h-4 mr-2" />
                                View Current Memories ({currentMemories.length})
                            </Button>
                            <Button
                                onClick={clearAllMemories}
                                variant="outline"
                                className="text-red-600 hover:text-red-700"
                            >
                                <Trash2 className="w-4 h-4 mr-2" />
                                Clear All Memories
                            </Button>
                        </div>
                        {hasLoadedProfile && (
                            <div className="mt-4 p-3 bg-green-50 border border-green-200 rounded-lg">
                                <div className="flex items-center gap-2">
                                    <CheckCircle className="w-4 h-4 text-green-600" />
                                    <span className="text-sm font-medium text-green-800">
                                        Investor Profile Loaded: Value-focused, FCF-prioritizing, dividend-seeking investor with 5-10 year horizon
                                    </span>
                                </div>
                            </div>
                        )}
                    </CardContent>
                </Card>

                {/* Document Analysis Section */}
                <Card className="bg-gradient-to-r from-blue-50 to-green-50 border-blue-200">
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <FileText className="w-5 h-5 text-green-600" />
                            Microsoft 2024 Annual Report
                        </CardTitle>
                        <p className="text-sm text-gray-600">
                            Complete annual report with financial statements, business segments, and strategic initiatives
                        </p>
                    </CardHeader>
                    <CardContent>
                        <Textarea
                            value={documentContent}
                            onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setDocumentContent(e.target.value)}
                            className="min-h-[200px] font-mono text-xs"
                            placeholder="Complete Microsoft 2024 Annual Report content..."
                        />
                        <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                            <div>
                                <p className="text-lg font-bold text-green-600">$245.1B</p>
                                <p className="text-sm text-gray-600">Annual Revenue</p>
                            </div>
                            <div>
                                <p className="text-lg font-bold text-green-600">+16%</p>
                                <p className="text-sm text-gray-600">YoY Growth</p>
                            </div>
                            <div>
                                <p className="text-lg font-bold text-blue-600">$103.9B</p>
                                <p className="text-sm text-gray-600">Net Income</p>
                            </div>
                            <div>
                                <p className="text-lg font-bold text-purple-600">43.6%</p>
                                <p className="text-sm text-gray-600">ROE</p>
                            </div>
                        </div>
                    </CardContent>
                </Card>

                {/* Guided Experience Section */}
                <Card className="bg-gradient-to-r from-yellow-50 to-orange-50 border-yellow-200">
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <HelpCircle className="w-5 h-5 text-orange-600" />
                            How to Experience the Demo
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="grid md:grid-cols-3 gap-4">
                            <div className="text-center">
                                <div className="w-8 h-8 bg-purple-600 text-white rounded-full flex items-center justify-center mx-auto mb-2 font-bold">1</div>
                                <h4 className="font-semibold text-purple-800 mb-1">Load Investor Profile</h4>
                                <p className="text-sm text-gray-600">Click "Load Sample Investor Profile" to set up a value-focused investor's preferences and criteria</p>
                            </div>
                            <div className="text-center">
                                <div className="w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center mx-auto mb-2 font-bold">2</div>
                                <h4 className="font-semibold text-blue-800 mb-1">Compare Responses</h4>
                                <p className="text-sm text-gray-600">Ask the same question to both analysts and see how memory provides personalized insights</p>
                            </div>
                            <div className="text-center">
                                <div className="w-8 h-8 bg-green-600 text-white rounded-full flex items-center justify-center mx-auto mb-2 font-bold">3</div>
                                <h4 className="font-semibold text-green-800 mb-1">Build Context</h4>
                                <p className="text-sm text-gray-600">Continue the conversation to see how memory builds sophisticated investment analysis over time</p>
                            </div>
                        </div>
                    </CardContent>
                </Card>

                {/* Tabbed Chat Interface */}
                <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
                    <TabsList className="grid w-full grid-cols-2 mb-6">
                        <TabsTrigger value="standard" className="flex items-center gap-2">
                            <BarChart3 className="w-4 h-4" />
                            Standard Financial Agent
                        </TabsTrigger>
                        <TabsTrigger value="memory" className="flex items-center gap-2">
                            <Brain className="w-4 h-4" />
                            Memory-Enhanced Investment Agent
                        </TabsTrigger>
                    </TabsList>

                    <TabsContent value="standard" className="space-y-6">
                        <ChatCard
                            title="Standard Financial Agent"
                            subtitle="Analyzes Microsoft's annual report but has no memory of your investment preferences or previous conversations"
                            messages={standardMessages}
                            input={standardQuestion}
                            onInputChange={setStandardQuestion}
                            onSubmit={handleStandardSubmit}
                            onClearChat={clearStandardChat}
                            isLoading={isStandardLoading}
                            placeholder="Ask about Microsoft's financial performance..."
                            headerIcon={<BarChart3 className="w-5 h-5" />}
                            borderColor="border-orange-200"
                            headerBgColor="bg-orange-50 text-orange-800"
                            messageBgColor="bg-gray-100 text-gray-800"
                            buttonColor="bg-orange-600 hover:bg-orange-700"
                            loadingText="Analyzing document..."
                            showMemoryIndicators={false}
                        />
                    </TabsContent>

                    <TabsContent value="memory" className="space-y-6">
                        <ChatCard
                            title="Memory-Enhanced Investment Agent"
                            subtitle="Analyzes Microsoft's annual report with full knowledge of your investment philosophy, preferences, and conversation history"
                            messages={memoryMessages}
                            input={memoryQuestion}
                            onInputChange={setMemoryQuestion}
                            onSubmit={handleMemorySubmit}
                            onClearChat={clearMemoryChat}
                            isLoading={isMemoryLoading}
                            placeholder="Ask about Microsoft as an investment opportunity..."
                            headerIcon={<Brain className="w-5 h-5" />}
                            borderColor="border-green-200"
                            headerBgColor="bg-green-50 text-green-800"
                            messageBgColor="bg-green-100 text-green-800"
                            buttonColor="bg-green-600 hover:bg-green-700"
                            loadingText="Analyzing with your investment profile + conversation history..."
                            showMemoryIndicators={true}
                            badge={hasLoadedProfile && (
                                <Badge variant="secondary" className="bg-green-100 text-green-700">
                                    <Brain className="w-3 h-3 mr-1" />
                                    Investor Profile Active
                                </Badge>
                            )}
                        />
                    </TabsContent>
                </Tabs>

                {/* Explanation Section */}
                <Card className="mt-8">
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <Brain className="w-5 h-5" />
                            Why Memory Transforms Investment Analysis
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div className="grid md:grid-cols-2 gap-6">
                            <div>
                                <h4 className="font-semibold text-orange-800 mb-2">Standard Financial Agent</h4>
                                <ul className="space-y-1 text-sm text-gray-600">
                                    <li>• Analyzes Microsoft's annual report objectively</li>
                                    <li>• Provides generic financial insights</li>
                                    <li>• Treats each question independently</li>
                                    <li>• No awareness of your investment preferences</li>
                                    <li>• Cannot tailor advice to your specific criteria</li>
                                    <li>• Repeats similar analysis for related questions</li>
                                </ul>
                            </div>
                            <div>
                                <h4 className="font-semibold text-green-800 mb-2">Memory-Enhanced Investment Agent</h4>
                                <ul className="space-y-1 text-sm text-gray-600">
                                    <li>• Knows your value-focused investment philosophy</li>
                                    <li>• Prioritizes FCF and ROE metrics you care about</li>
                                    <li>• Considers your 5-10 year investment horizon</li>
                                    <li>• Factors in your dividend income preferences</li>
                                    <li>• Builds sophisticated, personalized analysis over time</li>
                                    <li>• References your criteria in every recommendation</li>
                                </ul>
                            </div>
                        </div>
                        <div className="bg-purple-50 p-4 rounded-lg">
                            <p className="text-purple-800 text-sm">
                                <strong>Key Insight:</strong> Memory transforms generic financial document analysis into personalized investment advice.
                                The memory-enhanced advisor doesn't just analyze Microsoft's financials—it evaluates Microsoft specifically against
                                YOUR investment criteria, philosophy, and preferences.
                            </p>
                        </div>
                        <div className="bg-blue-50 p-4 rounded-lg">
                            <p className="text-blue-800 text-sm">
                                <strong>For Investors:</strong> Experience how AI can remember your investment strategy and provide
                                consistently personalized advice that aligns with your specific goals, risk tolerance, and preferences.
                            </p>
                        </div>
                        <div className="bg-green-50 p-4 rounded-lg">
                            <p className="text-green-800 text-sm">
                                <strong>For Developers:</strong> This demonstrates how conversational memory enables AI systems to provide
                                truly personalized experiences by remembering user context, preferences, and building sophisticated analysis over time.
                            </p>
                        </div>
                    </CardContent>
                </Card>

                {/* Developer Implementation Section */}
                <Card className="mt-8 bg-gradient-to-r from-gray-50 to-slate-50 border-gray-200">
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <FileText className="w-5 h-5 text-gray-700" />
                            For Developers: Memory-Enhanced Chat Implementation
                        </CardTitle>
                        <p className="text-sm text-gray-600">
                            Here's the actual code used to implement memory-enhanced financial analysis in this demo
                        </p>
                    </CardHeader>
                    <CardContent className="space-y-6">
                        <div className="grid md:grid-cols-2 gap-6">
                            <div>
                                <h4 className="font-semibold text-gray-800 mb-3">1. Load Investor Profile Memories</h4>
                                <div className="bg-gray-900 text-gray-100 p-4 rounded-lg text-xs font-mono overflow-x-auto">
                                    <pre>{`// Load curated investor profile
const loadInvestorProfile = async () => {
  const investorMemories = [
    "I am a value investor focused on undervalued companies",
    "I prioritize Free Cash Flow (FCF) as my primary metric",
    "I prefer dividend-paying stocks for steady income",
    "I invest with a 5-10 year holding period"
  ];

  for (const memory of investorMemories) {
    await memoryAPI.remember(memory, true);
  }
};`}</pre>
                                </div>
                            </div>
                            <div>
                                <h4 className="font-semibold text-gray-800 mb-3">2. Create Memory-Enhanced Session</h4>
                                <div className="bg-gray-900 text-gray-100 p-4 rounded-lg text-xs font-mono overflow-x-auto">
                                    <pre>{`// Create session with memory enabled
const createMemorySession = async () => {
  const response = await memoryAPI.createChatSession({
    system_prompt: \`You are a personalized financial advisor
    with access to the investor's profile stored in memory.
    Use both the document and investor's criteria to provide
    tailored investment analysis.\`,
    config: {
      use_memory: true,
      model: "gpt-3.5-turbo",
      temperature: 0.7,
      max_tokens: 1000,
      top_k: 5
    }
  });
  return response.session_id;
};`}</pre>
                                </div>
                            </div>
                        </div>

                        <div className="grid md:grid-cols-2 gap-6">
                            <div>
                                <h4 className="font-semibold text-gray-800 mb-3">3. Chat with Memory Context</h4>
                                <div className="bg-gray-900 text-gray-100 p-4 rounded-lg text-xs font-mono overflow-x-auto">
                                    <pre>{`// Send message with memory retrieval
const chatWithMemory = async (message) => {
  const response = await memoryAPI.chatWithSession({
    session_id: memorySessionId,
    message: message,
    top_k: 5,
    min_similarity: 0.7
  });

  // Response includes memory context
  console.log(response.memory_context.memories);
  return response.message;
};`}</pre>
                                </div>
                            </div>
                            <div>
                                <h4 className="font-semibold text-gray-800 mb-3">4. View Retrieved Memories</h4>
                                <div className="bg-gray-900 text-gray-100 p-4 rounded-lg text-xs font-mono overflow-x-auto">
                                    <pre>{`// Display memory transparency
const showMemoryContext = (memoryContext) => {
  memoryContext.memories.forEach(memory => {
    console.log(\`Memory: \${memory.text}\`);
    console.log(\`Relevance: \${memory.similarity_score}\`);
    console.log(\`Created: \${memory.created_at}\`);
  });

  // Show filtering info
  console.log(\`Used \${memoryContext.memories_used} memories\`);
  console.log(\`Excluded \${memoryContext.excluded_memories?.length || 0}\`);
};`}</pre>
                                </div>
                            </div>
                        </div>

                        <div>
                            <h4 className="font-semibold text-gray-800 mb-3">5. Complete API Integration Example</h4>
                            <div className="bg-gray-900 text-gray-100 p-4 rounded-lg text-xs font-mono overflow-x-auto">
                                <pre>{`// Full implementation pattern
import { memoryAPI } from '@/lib/api';

const MemoryEnhancedFinancialChat = () => {
  const [sessionId, setSessionId] = useState(null);
  const [hasLoadedProfile, setHasLoadedProfile] = useState(false);

  // 1. Load investor profile into memory
  const setupInvestorProfile = async () => {
    const profileMemories = [
      "I am a value investor focused on undervalued companies with strong fundamentals",
      "I prioritize Free Cash Flow (FCF) as my primary valuation metric",
      "I prefer dividend-paying stocks for steady income and dividend growth",
      "I avoid speculative tech stocks and focus on established companies",
      "I invest with a 5-10 year holding period and ignore short-term volatility"
    ];

    for (const memory of profileMemories) {
      await memoryAPI.remember(memory, true);
    }
    setHasLoadedProfile(true);
  };

  // 2. Create memory-enabled chat session
  const createSession = async () => {
    const response = await memoryAPI.createChatSession({
      system_prompt: \`You are a personalized financial advisor with access to Microsoft's
      annual report and the investor's profile stored in memory. Use both the document
      and the investor's stated criteria to provide tailored investment analysis.\`,
      config: {
        use_memory: true,
        model: "gpt-3.5-turbo",
        temperature: 0.7,
        max_tokens: 1000,
        top_k: 5
      }
    });
    setSessionId(response.session_id);
  };

  // 3. Send message and get personalized response
  const askQuestion = async (question) => {
    if (!hasLoadedProfile) await setupInvestorProfile();
    if (!sessionId) await createSession();

    const response = await memoryAPI.chatWithSession({
      session_id: sessionId,
      message: question,
      top_k: 5,
      min_similarity: 0.7
    });

    // The response is now personalized based on investor profile
    return {
      answer: response.message,
      memories_used: response.memory_context?.memories || [],
      conversation_length: response.conversation_length
    };
  };

  return (
    <div>
      <button onClick={setupInvestorProfile}>Load Investor Profile</button>
      <ChatInterface onSubmit={askQuestion} />
    </div>
  );
};`}</pre>
                            </div>
                        </div>

                        <div className="bg-blue-50 p-4 rounded-lg">
                            <h4 className="font-semibold text-blue-800 mb-2">Key Implementation Points:</h4>
                            <ul className="text-sm text-blue-700 space-y-1">
                                <li>• <strong>Memory Loading:</strong> Use <code>memoryAPI.remember()</code> to store investor preferences</li>
                                <li>• <strong>Session Creation:</strong> Set <code>use_memory: true</code> in session config</li>
                                <li>• <strong>Personalized Prompts:</strong> System prompt should reference memory usage</li>
                                <li>• <strong>Memory Retrieval:</strong> <code>top_k</code> and <code>min_similarity</code> control memory selection</li>
                                <li>• <strong>Transparency:</strong> Access <code>memory_context</code> to show which memories influenced the response</li>
                            </ul>
                        </div>

                        <div className="bg-green-50 p-4 rounded-lg">
                            <h4 className="font-semibold text-green-800 mb-2">API Endpoints Used:</h4>
                            <ul className="text-sm text-green-700 space-y-1">
                                <li>• <code>POST /api/memory/memories</code> - Store investor profile memories</li>
                                <li>• <code>POST /api/agent/session</code> - Create memory-enabled chat session</li>
                                <li>• <code>POST /api/agent/session/{'{session_id}'}</code> - Send messages with memory context</li>
                                <li>• <code>POST /api/memory/memories/search</code> - Search and view current memories</li>
                                <li>• <code>DELETE /api/memory/memories/all</code> - Clear all memories</li>
                            </ul>
                        </div>
                    </CardContent>
                </Card>
            </div>
        </PageLayout>
    )
}

'use client'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Database, MessageSquare, Zap } from "lucide-react"
import { Navbar } from "@/components/Navbar"

export default function ApiDocsPage() {
    return (
        <>
            <Navbar />
            <div className="container mx-auto px-4 py-8 max-w-6xl">
                <div className="mb-8">
                    <h1 className="text-4xl font-bold mb-4">REMEM API Documentation</h1>
                    <p className="text-xl text-muted-foreground mb-6">
                        Complete API reference for building memory-enhanced AI agents
                    </p>
                    <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
                        <p className="text-sm">
                            <strong>Base URL:</strong> <code className="bg-blue-100 px-2 py-1 rounded">http://localhost:5001</code>
                        </p>
                    </div>
                    
                </div>

                <Tabs defaultValue="remem" className="w-full">
                    <TabsList className="grid w-full grid-cols-2">
                        <TabsTrigger value="remem">REMEM API</TabsTrigger>
                        <TabsTrigger value="demo">DEMO API</TabsTrigger>
                    </TabsList>

                    <TabsContent value="remem" className="space-y-6">
                        <div className="grid gap-6">

                            {/* NEME API */}
                            <Card>
                                <CardHeader>
                                    <CardTitle className="flex items-center gap-2">
                                        <Database className="h-5 w-5" />
                                        Atomic Memory Operations
                                    </CardTitle>
                                    <CardDescription>
                                        Store, search, and manage individual memories
                                    </CardDescription>
                                </CardHeader>
                                <CardContent className="space-y-4">

                                    {/* Create Memory */}
                                    <div className="border rounded-lg p-4">
                                        <div className="flex items-center gap-2 mb-2">
                                            <Badge variant="default">POST</Badge>
                                            <code className="text-sm">/api/memory/{`{vectorset_name}`}</code>
                                            <span className="text-sm text-muted-foreground">Create a memory</span>
                                        </div>
                                        <div className="bg-gray-50 rounded p-3 text-sm">
                                            <strong>Parameters:</strong>
                                            <div className="mt-1 text-xs mb-2">
                                                • <code>vectorset_name</code> (string, required): Name of the vectorset (e.g., "memories")
                                            </div>
                                            <strong>Request Body:</strong>
                                            <pre className="mt-1 text-xs">{`{
  "text": "I went to Mario's Italian Restaurant",
  "apply_grounding": true
}`}</pre>
                                            <strong className="block mt-2">Example URL:</strong>
                                            <div className="mt-1 text-xs text-blue-600">
                                                <code>POST /api/memory/memories</code>
                                            </div>
                                            <strong className="block mt-2">Response:</strong>
                                            <pre className="mt-1 text-xs">{`{
  "success": true,
  "memory_id": "memory:550e8400-e29b-41d4...",
  "message": "Memory stored successfully"
}`}</pre>
                                        </div>
                                    </div>

                                    {/* Search Memories */}
                                    <div className="border rounded-lg p-4">
                                        <div className="flex items-center gap-2 mb-2">
                                            <Badge variant="default">POST</Badge>
                                            <code className="text-sm">/api/memory/{`{vectorset_name}`}/search</code>
                                            <span className="text-sm text-muted-foreground">Search for memories</span>
                                        </div>
                                        <div className="bg-gray-50 rounded p-3 text-sm">
                                            <strong>Parameters:</strong>
                                            <div className="mt-1 text-xs mb-2">
                                                • <code>vectorset_name</code> (string, required): Name of the vectorset (e.g., "memories")
                                            </div>
                                            <strong>Request Body:</strong>
                                            <pre className="mt-1 text-xs">{`{
  "query": "Italian restaurant food",
  "top_k": 5,
  "min_similarity": 0.7
}`}</pre>
                                            <strong className="block mt-2">Example URL:</strong>
                                            <div className="mt-1 text-xs text-blue-600">
                                                <code>POST /api/memory/memories/search</code>
                                            </div>
                                            <strong className="block mt-2">Response:</strong>
                                            <pre className="mt-1 text-xs">{`{
  "success": true,
  "memories": [...],
  "count": 3,
  "excluded_memories": [...]
}`}</pre>
                                        </div>
                                    </div>

                                    {/* Ask Question */}
                                    <div className="border rounded-lg p-4">
                                        <div className="flex items-center gap-2 mb-2">
                                            <Badge variant="default">POST</Badge>
                                            <code className="text-sm">/api/memory/{`{vectorset_name}`}/ask</code>
                                            <span className="text-sm text-muted-foreground">Ask a question with cognitive reasoning</span>
                                        </div>
                                        <div className="bg-gray-50 rounded p-3 text-sm">
                                            <strong>Parameters:</strong>
                                            <div className="mt-1 text-xs mb-2">
                                                • <code>vectorset_name</code> (string, required): Name of the vectorset (e.g., "memories")
                                            </div>
                                            <strong>Request Body:</strong>
                                            <pre className="mt-1 text-xs">{`{
  "question": "What restaurants do I like?",
  "top_k": 5
}`}</pre>
                                            <strong className="block mt-2">Example URL:</strong>
                                            <div className="mt-1 text-xs text-blue-600">
                                                <code>POST /api/memory/memories/ask</code>
                                            </div>
                                            <strong className="block mt-2">Response:</strong>
                                            <pre className="mt-1 text-xs">{`{
  "success": true,
  "question": "What restaurants do I like?",
  "answer": "Based on your memories...",
  "confidence": "high",
  "reasoning": "...",
  "supporting_memories": [...],
  "total_memories_searched": 10,
  "relevant_memories_used": 3,
  "type": "answer"
}`}</pre>
                                        </div>
                                    </div>

                                    {/* Get Memory Info */}
                                    <div className="border rounded-lg p-4">
                                        <div className="flex items-center gap-2 mb-2">
                                            <Badge variant="secondary">GET</Badge>
                                            <code className="text-sm">/api/memory/{`{vectorset_name}`}</code>
                                            <span className="text-sm text-muted-foreground">Get memory system information</span>
                                        </div>
                                        <div className="bg-gray-50 rounded p-3 text-sm">
                                            <strong>Parameters:</strong>
                                            <div className="mt-1 text-xs mb-2">
                                                • <code>vectorset_name</code> (string, required): Name of the vectorset (e.g., "memories")
                                            </div>
                                            <strong className="block mt-2">Example URL:</strong>
                                            <div className="mt-1 text-xs text-blue-600">
                                                <code>GET /api/memory/memories</code>
                                            </div>
                                            <strong className="block mt-2">Response:</strong>
                                            <pre className="mt-1 text-xs">{`{
  "success": true,
  "memory_count": 42,
  "vector_dimension": 1536,
  "vectorset_name": "memories",
  "embedding_model": "text-embedding-ada-002"
}`}</pre>
                                        </div>
                                    </div>

                                    {/* Context Management */}
                                    <div className="border rounded-lg p-4">
                                        <div className="space-y-2 mb-2">
                                            <div className="flex items-center gap-2">
                                                <Badge variant="default">POST</Badge>
                                                <code className="text-sm">/api/memory/{`{vectorset_name}`}/context</code>
                                                <span className="text-sm text-muted-foreground">Set context</span>
                                            </div>
                                            <div className="flex items-center gap-2">
                                                <Badge variant="secondary">GET</Badge>
                                                <code className="text-sm">/api/memory/{`{vectorset_name}`}/context</code>
                                                <span className="text-sm text-muted-foreground">Get context</span>
                                            </div>
                                        </div>
                                        <div className="bg-gray-50 rounded p-3 text-sm">
                                            <strong>Parameters:</strong>
                                            <div className="mt-1 text-xs mb-2">
                                                • <code>vectorset_name</code> (string, required): Name of the vectorset (e.g., "memories")
                                            </div>
                                            <strong>Example URLs:</strong>
                                            <div className="mt-1 text-xs text-blue-600 space-y-1">
                                                <div><code>POST /api/memory/memories/context</code></div>
                                                <div><code>GET /api/memory/memories/context</code></div>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Memory Management */}
                                    <div className="border rounded-lg p-4">
                                        <div className="space-y-2">
                                            <div className="flex items-center gap-2">
                                                <Badge variant="destructive">DELETE</Badge>
                                                <code className="text-sm">/api/memory/{`{vectorset_name}`}/{`{memory_id}`}</code>
                                                <span className="text-sm text-muted-foreground">Delete specific memory</span>
                                            </div>
                                            <div className="flex items-center gap-2">
                                                <Badge variant="destructive">DELETE</Badge>
                                                <code className="text-sm">/api/memory/{`{vectorset_name}`}/all</code>
                                                <span className="text-sm text-muted-foreground">Clear all memories</span>
                                            </div>
                                        </div>
                                        <div className="bg-gray-50 rounded p-3 text-sm mt-2">
                                            <strong>Parameters:</strong>
                                            <div className="mt-1 text-xs mb-2">
                                                • <code>vectorset_name</code> (string, required): Name of the vectorset (e.g., "memories")<br/>
                                                • <code>memory_id</code> (string, required for specific delete): ID of the memory to delete
                                            </div>
                                            <strong>Example URLs:</strong>
                                            <div className="mt-1 text-xs text-blue-600 space-y-1">
                                                <div><code>DELETE /api/memory/memories/memory:123-456</code></div>
                                                <div><code>DELETE /api/memory/memories/all</code></div>
                                            </div>
                                        </div>
                                    </div>

                                </CardContent>
                            </Card>

                        </div>
                    </TabsContent>

                    <TabsContent value="demo" className="space-y-6">
                        <Card>
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <MessageSquare className="h-5 w-5" />
                                    Chat Session API
                                </CardTitle>
                                <CardDescription>
                                    Conversational interface with session management
                                </CardDescription>
                            </CardHeader>
                            <CardContent className="space-y-4">

                                {/* Create Session */}
                                <div className="border rounded-lg p-4">
                                    <div className="flex items-center gap-2 mb-2">
                                        <Badge variant="default">POST</Badge>
                                        <code className="text-sm">/api/agent/session</code>
                                        <span className="text-sm text-muted-foreground">Create a chat session</span>
                                    </div>
                                    <div className="bg-gray-50 rounded p-3 text-sm">
                                        <strong>Request:</strong>
                                        <pre className="mt-1 text-xs">{`{
  "system_prompt": "You are a helpful assistant",
  "session_id": "optional-custom-id",
  "config": {
    "use_memory": true,
    "model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 1000
  }
}`}</pre>
                                        <strong className="block mt-2">Response:</strong>
                                        <pre className="mt-1 text-xs">{`{
  "success": true,
  "session_id": "session-uuid",
  "created_at": "2024-01-15T18:30:00Z",
  "use_memory": true
}`}</pre>
                                    </div>
                                </div>

                                {/* Chat with Session */}
                                <div className="border rounded-lg p-4">
                                    <div className="flex items-center gap-2 mb-2">
                                        <Badge variant="default">POST</Badge>
                                        <code className="text-sm">/api/agent/session/{`{session_id}`}</code>
                                        <span className="text-sm text-muted-foreground">Chat with session</span>
                                    </div>
                                    <div className="bg-gray-50 rounded p-3 text-sm">
                                        <strong>Request:</strong>
                                        <pre className="mt-1 text-xs">{`{
  "message": "What restaurants have I been to?",
  "top_k": 5,
  "min_similarity": 0.7
}`}</pre>
                                        <strong className="block mt-2">Response:</strong>
                                        <pre className="mt-1 text-xs">{`{
  "success": true,
  "message": "Based on your memories...",
  "session_id": "session-uuid",
  "memory_context": {
    "memories_used": 3,
    "memories": [...],
    "excluded_memories": [...]
  }
}`}</pre>
                                    </div>
                                </div>

                                {/* Session Management */}
                                <div className="border rounded-lg p-4">
                                    <div className="space-y-2">
                                        <div className="flex items-center gap-2">
                                            <Badge variant="secondary">GET</Badge>
                                            <code className="text-sm">/api/agent/session/{`{session_id}`}</code>
                                            <span className="text-sm text-muted-foreground">Get session info</span>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            <Badge variant="secondary">GET</Badge>
                                            <code className="text-sm">/api/agent/sessions</code>
                                            <span className="text-sm text-muted-foreground">List all sessions</span>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            <Badge variant="destructive">DELETE</Badge>
                                            <code className="text-sm">/api/agent/session/{`{session_id}`}</code>
                                            <span className="text-sm text-muted-foreground">Delete session</span>
                                        </div>
                                    </div>
                                </div>

                            </CardContent>
                        </Card>
                    </TabsContent>
                </Tabs>

                
                {/* Quick Start */}
                <Card className="mt-8">
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <Zap className="h-5 w-5" />
                            Quick Start
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="space-y-6">
                            <div className="bg-gray-50 rounded p-4">
                                <h4 className="font-semibold mb-2">Basic Memory Flow:</h4>
                                <ol className="list-decimal list-inside space-y-1 text-sm">
                                    <li>Store memories: <code>POST /api/memory/memories</code></li>
                                    <li>Search memories: <code>POST /api/memory/memories/search</code></li>
                                    <li>Ask questions: <code>POST /api/memory/memories/ask</code></li>
                                    <li>Create chat session: <code>POST /api/agent/session</code></li>
                                    <li>Chat with memory: <code>POST /api/agent/session/{`{id}`}</code></li>
                                </ol>
                            </div>

                            <div className="bg-blue-50 rounded p-4">
                                <h4 className="font-semibold mb-3">Example cURL Commands:</h4>
                                <div className="space-y-3 text-xs">
                                    <div>
                                        <div className="font-medium mb-1">1. Store a memory:</div>
                                        <pre className="bg-white p-2 rounded border overflow-x-auto">{`curl -X POST http://localhost:5001/api/memory/memories \\
  -H "Content-Type: application/json" \\
  -d '{"text": "I love Italian food", "apply_grounding": true}'`}</pre>
                                    </div>

                                    <div>
                                        <div className="font-medium mb-1">2. Search memories:</div>
                                        <pre className="bg-white p-2 rounded border overflow-x-auto">{`curl -X POST http://localhost:5001/api/memory/memories/search \\
  -H "Content-Type: application/json" \\
  -d '{"query": "Italian food", "top_k": 5}'`}</pre>
                                    </div>

                                    <div>
                                        <div className="font-medium mb-1">3. Ask a question:</div>
                                        <pre className="bg-white p-2 rounded border overflow-x-auto">{`curl -X POST http://localhost:5001/api/memory/memories/ask \\
  -H "Content-Type: application/json" \\
  -d '{"question": "What food do I like?", "top_k": 5}'`}</pre>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </CardContent>
                </Card>
            </div>
        </>
    )
}

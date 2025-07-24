"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Copy, Check } from "lucide-react"

export function CodeExamplesSection() {
    const [copiedIndex, setCopiedIndex] = useState<number | null>(null)

    const copyToClipboard = async (text: string, index: number) => {
        try {
            await navigator.clipboard.writeText(text)
            setCopiedIndex(index)
            setTimeout(() => setCopiedIndex(null), 2000)
        } catch (err) {
            console.error('Failed to copy text: ', err)
        }
    }

    const examples = [
        {
            title: "Store a Memory",
            code: `curl -X POST http://localhost:5001/api/memory/memories \\
  -H "Content-Type: application/json" \\
  -d '{
    "text": "Had lunch with Sarah at the new Italian place",
    "apply_grounding": true
  }'`
        },
        {
            title: "Ask a Question",
            code: `curl -X POST http://localhost:5001/api/memory/memories/ask \\
  -H "Content-Type: application/json" \\
  -d '{
    "question": "Where did I eat Italian food recently?",
    "top_k": 5
  }'`
        },
        {
            title: "Search Memories",
            code: `curl -X POST http://localhost:5001/api/memory/memories/search \\
  -H "Content-Type: application/json" \\
  -d '{
    "query": "lunch with friends",
    "top_k": 10
  }'`
        },
        {
            title: "Set Context",
            code: `curl -X POST http://localhost:5001/api/memory/memories/context \\
  -H "Content-Type: application/json" \\
  -d '{
    "location": "San Francisco, CA",
    "activity": "business meeting",
    "people_present": ["Alice", "Bob"]
  }'`
        }
    ]

    return (
        <div className="space-y-6">
            <div className="text-center lg:text-left">
                <h2 className="text-2xl font-bold mb-4">
                    Quick Start
                </h2>
                <p className="text-muted-foreground">
                    Get started with these simple API calls.
                </p>
            </div>
            
            <div className="space-y-4">
                {examples.map((example, index) => (
                    <Card key={index}>
                        <CardHeader className="pb-3">
                            <div className="flex items-center justify-between">
                                <CardTitle className="text-base">{example.title}</CardTitle>
                                <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => copyToClipboard(example.code, index)}
                                    className="h-8 w-8 p-0"
                                >
                                    {copiedIndex === index ? (
                                        <Check className="h-4 w-4 text-green-500" />
                                    ) : (
                                        <Copy className="h-4 w-4" />
                                    )}
                                </Button>
                            </div>
                        </CardHeader>
                        <CardContent>
                            <pre className="text-xs bg-muted p-3 rounded-md overflow-x-auto">
                                <code>{example.code}</code>
                            </pre>
                        </CardContent>
                    </Card>
                ))}
            </div>
        </div>
    )
}

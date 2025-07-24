"use client"

import { Card, CardHeader, CardContent, CardTitle } from "@/components/ui/card"
import { Brain, Zap, Shield, Globe, Network } from "lucide-react"

export function InfoCards() {
    return (
        <div className="space-y-6">
            <div className="text-center lg:text-left">
                <h2 className="text-2xl font-bold mb-4">
                    Why Choose Remem?
                </h2>
                <p className="text-muted-foreground">
                    Built for developers who need intelligent memory capabilities in their applications.
                </p>
            </div>
            
            <div className="grid gap-4 grid-cols-1 md:grid-cols-2 lg:grid-cols-3">
                <Card>
                    <CardHeader className="pb-3">
                        <div className="flex items-center gap-2">
                            <Brain className="h-5 w-5 text-blue-500" />
                            <CardTitle className="text-lg">Contextual Grounding</CardTitle>
                        </div>
                    </CardHeader>
                    <CardContent>
                        <p className="text-sm text-muted-foreground">
                            Intelligently resolves context-dependent references in memories with our advanced grounding system.
                            Transforms "it's hot here today" into precise, searchable memories with location, time, and social
                            context—making your AI's memory truly timeless and location-independent.
                        </p>
                    </CardContent>
                </Card>

                <Card>
                    <CardHeader className="pb-3">
                        <div className="flex items-center gap-2">
                            <Zap className="h-5 w-5 text-yellow-500" />
                            <CardTitle className="text-lg">Enhanced Relevance Scoring</CardTitle>
                        </div>
                    </CardHeader>
                    <CardContent>
                        <p className="text-sm text-muted-foreground">
                            Combines vector similarity with temporal recency and usage patterns for truly intelligent memory retrieval.
                            Memories that matter most rise to the top through our sophisticated three-component scoring system that
                            mimics human memory prioritization.
                        </p>
                    </CardContent>
                </Card>

                <Card>
                    <CardHeader className="pb-3">
                        <div className="flex items-center gap-2">
                            <Network className="h-5 w-5 text-indigo-500" />
                            <CardTitle className="text-lg">Minsky-Inspired Cognitive Architecture</CardTitle>
                        </div>
                    </CardHeader>
                    <CardContent>
                        <p className="text-sm text-muted-foreground">
                            Built on cognitive science principles with our three-layer memory system: atomic Nemes (fundamental memories),
                            K-lines (mental states), and full Agent orchestration—creating a complete cognitive architecture for your
                            AI applications.
                        </p>
                    </CardContent>
                </Card>

                <Card>
                    <CardHeader className="pb-3">
                        <div className="flex items-center gap-2">
                            <Shield className="h-5 w-5 text-green-500" />
                            <CardTitle className="text-lg">Confidence-Based Reasoning</CardTitle>
                        </div>
                    </CardHeader>
                    <CardContent>
                        <p className="text-sm text-muted-foreground">
                            Every answer includes confidence assessment, supporting evidence, and transparent reasoning chains.
                            Our system knows when it knows and when it doesn't, providing structured responses with clear
                            reasoning paths.
                        </p>
                    </CardContent>
                </Card>

                <Card>
                    <CardHeader className="pb-3">
                        <div className="flex items-center gap-2">
                            <Globe className="h-5 w-5 text-purple-500" />
                            <CardTitle className="text-lg">Simple REST based API</CardTitle>
                        </div>
                    </CardHeader>
                    <CardContent>
                        <p className="text-sm text-muted-foreground">
                            Comprehensive three-layer API architecture with dedicated endpoints for atomic memory operations,
                            mental state re-construction, and full agent orchestration—all accessible through clean RESTful interfaces.
                        </p>
                    </CardContent>
                </Card>
            </div>
        </div>
    )
}

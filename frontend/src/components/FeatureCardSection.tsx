"use client"

import { Card, CardHeader, CardContent, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { MessageCircle, Brain, Save, Search, MapPin, Settings, GitPullRequest, TrendingUp } from "lucide-react"
import Link from "next/link"

function FeatureCard({
    title,
    description,
    icon: Icon,
    href,
    color
}: {
    title: string
    description: string
    icon: any
    href: string
    color: string
}) {
    return (
        <Card className="h-full hover:shadow-lg transition-shadow">
            <CardHeader>
                <div className="flex items-center gap-3">
                    <div className={`p-2 rounded-lg ${color} text-white`}>
                        <Icon className="w-5 h-5" />
                    </div>
                    <CardTitle className="text-lg">{title}</CardTitle>
                </div>
            </CardHeader>
            <CardContent className="space-y-4">
                <p className="text-muted-foreground">{description}</p>
                <Link href={href}>
                    <Button variant="outline" size="sm" className="w-full">
                        Try {title}
                    </Button>
                </Link>
            </CardContent>
        </Card>
    )
}

export function FeatureCardSection() {
    const features = [
        {
            title: "Chat Demo",
            description: "Experience conversational AI that remembers context across interactions. Perfect for building chatbots and virtual assistants.",
            icon: MessageCircle,
            href: "/chat-demo",
            color: "bg-blue-500"
        },
        {
            title: "Code Review Demo",
            description: "See how memory transforms AI agents from generic to personalized. Compare code review assistance with and without memory.",
            icon: GitPullRequest,
            href: "/agent-demo",
            color: "bg-indigo-500"
        },
        {
            title: "Investment Agent",
            description: "Side-by-side comparison of investment advice with and without memory. Real-time chat interface for financial guidance.",
            icon: TrendingUp,
            href: "/investment-demo",
            color: "bg-emerald-500"
        },
        {
            title: "Ask API",
            description: "Get intelligent answers with confidence scoring and supporting evidence. Ideal for Q&A systems and knowledge retrieval.",
            icon: Brain,
            href: "/ask",
            color: "bg-green-500"
        },
        {
            title: "Save API",
            description: "Store memories with automatic contextual grounding. Transform simple text into rich, searchable knowledge.",
            icon: Save,
            href: "/save",
            color: "bg-purple-500"
        },
        {
            title: "Browse API",
            description: "Semantic search through stored memories. Find information by meaning, not just keywords.",
            icon: Search,
            href: "/search",
            color: "bg-orange-500"
        },
        {
            title: "Context API",
            description: "Set environmental context for better memory grounding. Location, time, people, and activity awareness.",
            icon: MapPin,
            href: "/context",
            color: "bg-teal-500"
        },
        {
            title: "System Info",
            description: "Monitor your memory system with detailed analytics and configuration options.",
            icon: Settings,
            href: "/memory-info",
            color: "bg-gray-500"
        }
    ]

    return (
        <section className="container py-12 md:py-24 mx-auto">
            <div className="mx-auto flex max-w-[980px] flex-col items-center gap-4 text-center">
                <h2 className="text-3xl font-bold leading-tight tracking-tighter md:text-4xl">
                    Explore the API
                </h2>
                <p className="max-w-[750px] text-lg text-muted-foreground">
                    Try out different endpoints and see how Remem can enhance your applications
                </p>
            </div>
            <div className="grid gap-6 grid-cols-1 md:grid-cols-2 mt-12">
                {features.map((feature, index) => (
                    <FeatureCard
                        key={index}
                        title={feature.title}
                        description={feature.description}
                        icon={feature.icon}
                        href={feature.href}
                        color={feature.color}
                    />
                ))}
            </div>
        </section>
    )
}

import React from 'react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import { Brain, Server, Zap, Shield, DollarSign, Clock } from 'lucide-react'

interface ProviderSelectorProps {
    value: 'openai' | 'ollama'
    onChange: (provider: 'openai' | 'ollama') => void
    className?: string
}

export function ProviderSelector({ value, onChange, className }: ProviderSelectorProps) {
    const providers = [
        {
            id: 'openai' as const,
            name: 'OpenAI',
            description: 'Cloud-based AI service with GPT models',
            icon: Brain,
            features: ['High Quality', 'Fast Response', 'Latest Models'],
            pros: ['State-of-the-art models', 'Reliable uptime', 'Extensive documentation'],
            cons: ['Usage-based pricing', 'Internet required', 'Data sent to OpenAI'],
            badges: [
                { text: 'Cloud', variant: 'default' as const },
                { text: 'Paid', variant: 'secondary' as const }
            ]
        },
        {
            id: 'ollama' as const,
            name: 'Ollama',
            description: 'Local AI models running on your hardware',
            icon: Server,
            features: ['Privacy First', 'Local Control', 'No Usage Limits'],
            pros: ['Complete privacy', 'No ongoing costs', 'Works offline'],
            cons: ['Requires local setup', 'Hardware dependent', 'Limited model selection'],
            badges: [
                { text: 'Local', variant: 'default' as const },
                { text: 'Free', variant: 'secondary' as const }
            ]
        }
    ]

    return (
        <div className={`grid grid-cols-1 md:grid-cols-2 gap-4 ${className}`}>
            {providers.map((provider) => {
                const Icon = provider.icon
                const isSelected = value === provider.id

                return (
                    <Card
                        key={provider.id}
                        className={`cursor-pointer transition-all hover:shadow-md ${
                            isSelected 
                                ? 'ring-2 ring-purple-500 border-purple-500 bg-purple-50' 
                                : 'hover:border-gray-300'
                        }`}
                        onClick={() => onChange(provider.id)}
                    >
                        <CardContent className="p-4">
                            <div className="flex items-start justify-between mb-3">
                                <div className="flex items-center gap-3">
                                    <div className={`p-2 rounded-lg ${
                                        isSelected ? 'bg-purple-100' : 'bg-gray-100'
                                    }`}>
                                        <Icon className={`w-5 h-5 ${
                                            isSelected ? 'text-purple-600' : 'text-gray-600'
                                        }`} />
                                    </div>
                                    <div>
                                        <h3 className="font-semibold text-gray-900">{provider.name}</h3>
                                        <p className="text-sm text-gray-600">{provider.description}</p>
                                    </div>
                                </div>
                                {isSelected && (
                                    <Badge className="bg-purple-100 text-purple-700 border-purple-200">
                                        Selected
                                    </Badge>
                                )}
                            </div>

                            <div className="flex flex-wrap gap-2 mb-3">
                                {provider.badges.map((badge, index) => (
                                    <Badge key={index} variant={badge.variant} className="text-xs">
                                        {badge.text}
                                    </Badge>
                                ))}
                            </div>

                            <div className="space-y-2 text-sm">
                                <div className="flex flex-wrap gap-1">
                                    {provider.features.map((feature, index) => (
                                        <span 
                                            key={index} 
                                            className="inline-flex items-center px-2 py-1 rounded bg-gray-100 text-gray-700 text-xs"
                                        >
                                            {feature}
                                        </span>
                                    ))}
                                </div>

                                <div className="pt-2">
                                    <div className="text-xs text-green-600 mb-1">
                                        <strong>Pros:</strong>
                                    </div>
                                    <ul className="text-xs text-gray-600 space-y-1">
                                        {provider.pros.map((pro, index) => (
                                            <li key={index} className="flex items-center gap-1">
                                                <div className="w-1 h-1 bg-green-500 rounded-full flex-shrink-0" />
                                                {pro}
                                            </li>
                                        ))}
                                    </ul>
                                </div>

                                <div className="pt-1">
                                    <div className="text-xs text-orange-600 mb-1">
                                        <strong>Considerations:</strong>
                                    </div>
                                    <ul className="text-xs text-gray-600 space-y-1">
                                        {provider.cons.map((con, index) => (
                                            <li key={index} className="flex items-center gap-1">
                                                <div className="w-1 h-1 bg-orange-500 rounded-full flex-shrink-0" />
                                                {con}
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            </div>
                        </CardContent>
                    </Card>
                )
            })}
        </div>
    )
} 
export interface Memory {
    id: string
    content: string
    text?: string
    original_text?: string
    grounded_text?: string
    created_at: string
    last_accessed_at?: string
    grounding_applied?: boolean
    grounding_info?: {
        dependencies_found?: {
            spatial?: string[]
            environmental?: string[]
            temporal?: string[]
            social?: string[]
        }
        changes_made?: Array<{
            original: string
            replacement: string
            type: string
        }>
        unresolved_references?: string[]
    }
    context_snapshot?: {
        temporal?: {
            date?: string
            time?: string
            iso_date?: string
            day_of_week?: string
            month?: string
            year?: number
        }
        spatial?: {
            location?: string
            activity?: string
        }
        social?: {
            people_present?: string[]
        }
        environmental?: {
            weather?: string
            temperature?: string
            mood?: string
            [key: string]: any
        }
    }
    metadata?: {
        location?: string
        tags?: string[]
        confidence?: number
        score?: number
        relevance_score?: number
    }
}

export interface Conversation {
    id: string
    question: string
    answer: string
    created_at: string
    confidence?: 'high' | 'medium' | 'low'
    reasoning?: string
    supporting_memories?: Memory[]
    excluded_memories?: Memory[]
    filtering_info?: {
        min_similarity_threshold?: number
        total_candidates?: number
        excluded_count?: number
        included_count?: number
    }
    // K-line specific fields
    kline?: {
        coherence_score: number
        mental_state: string
    }
    total_memories_searched?: number
    relevant_memories_used?: number
    type?: 'answer' | 'help'
}

export interface ContextInfo {
    // API returns flat structure for some fields
    location?: string
    activity?: string
    people_present?: string[]

    // Temporal info (if supported by API)
    temporal?: {
        date?: string
        time?: string
        iso_date?: string
        day_of_week?: string
        month?: string
        year?: number
    }

    // Environmental info can be in either 'environmental' or 'environment' key
    environmental?: {
        weather?: string
        [key: string]: any
    }
    environment?: {
        weather?: string
        [key: string]: any
    }

    // Legacy nested structure support
    spatial?: {
        location?: string
        activity?: string
    }
    social?: {
        people_present?: string[]
    }
}

export type ApiStatus = 'ready' | 'not_initialized' | 'unknown'

// LLM Configuration Types
export interface LLMTierConfig {
    provider: 'openai' | 'ollama'
    model: string
    temperature: number        // 0.0 - 2.0
    max_tokens: number        // 1 - 8000
    base_url: string | null   // null for OpenAI, required URL for Ollama  
    api_key: string           // Required for OpenAI, can be empty string for Ollama
    timeout: number           // 1 - 300 seconds
}

export interface LLMConfig {
    tier1: LLMTierConfig
    tier2: LLMTierConfig
}

export interface LLMConfigResponse {
    success: boolean
    llm_config: LLMConfig
    runtime?: {
        timestamp: string
        llm_manager_initialized: boolean
        tier1_provider: string
        tier2_provider: string
        tier1_model: string
        tier2_model: string
    }
    message?: string
}

export interface LLMConfigUpdateRequest {
    tier1: LLMTierConfig
    tier2: LLMTierConfig
}

export interface LLMConfigUpdateResponse {
    success: boolean
    changes_made: string[]
    requires_restart: boolean
    warnings: string[]
    message: string
}

export interface OllamaModel {
    name: string
    size: number
    digest: string
    modified_at: string
    details?: {
        format?: string
        family?: string
        families?: string[]
        parameter_size?: string
        quantization_level?: string
    }
}

export interface OllamaModelsResponse {
    success: boolean
    models: OllamaModel[]
    message?: string
}

export interface LLMPresetConfig {
    id: string
    name: string
    description: string
    config: LLMConfig
    tags: string[]
}

export const LLM_PRESETS: LLMPresetConfig[] = [
    {
        id: 'cost-optimized',
        name: 'Cost Optimized',
        description: 'OpenAI for main interactions, Ollama for internal operations',
        config: {
            tier1: {
                provider: 'openai',
                model: 'gpt-3.5-turbo',
                temperature: 0.7,
                max_tokens: 2000,
                base_url: null,
                api_key: '',
                timeout: 30
            },
            tier2: {
                provider: 'ollama',
                model: 'llama2:7b',
                temperature: 0.3,
                max_tokens: 1000,
                base_url: 'http://localhost:11434',
                api_key: '',
                timeout: 60
            }
        },
        tags: ['cost-effective', 'hybrid']
    },
    {
        id: 'performance-optimized',
        name: 'Performance Optimized', 
        description: 'GPT-4 for both tiers for maximum capability',
        config: {
            tier1: {
                provider: 'openai',
                model: 'gpt-4',
                temperature: 0.7,
                max_tokens: 3000,
                base_url: null,
                api_key: '',
                timeout: 45
            },
            tier2: {
                provider: 'openai',
                model: 'gpt-4',
                temperature: 0.3,
                max_tokens: 2000,
                base_url: null,
                api_key: '',
                timeout: 45
            }
        },
        tags: ['high-performance', 'premium']
    },
    {
        id: 'local-only',
        name: 'Local Only',
        description: 'Ollama for both tiers for complete privacy',
        config: {
            tier1: {
                provider: 'ollama',
                model: 'llama2:13b',
                temperature: 0.7,
                max_tokens: 2000,
                base_url: 'http://localhost:11434',
                api_key: '',
                timeout: 90
            },
            tier2: {
                provider: 'ollama',
                model: 'llama2:7b',
                temperature: 0.3,
                max_tokens: 1000,
                base_url: 'http://localhost:11434',
                api_key: '',
                timeout: 60
            }
        },
        tags: ['privacy', 'offline', 'local']
    }
]

export const POPULAR_MODELS = {
    openai: [
        'gpt-4',
        'gpt-4-turbo',
        'gpt-3.5-turbo',
        'gpt-3.5-turbo-16k'
    ],
    ollama: [] // Will be fetched dynamically from Ollama instance
}

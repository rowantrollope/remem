# LLM Configuration Architecture Design

## Overview
This document outlines the design for a two-tier LLM system supporting both OpenAI and Ollama providers.

## Configuration Structure

```json
{
  "llm": {
    "tier1": {
      "provider": "openai|ollama",
      "model": "gpt-4|llama2|mistral|etc",
      "temperature": 0.7,
      "max_tokens": 2000,
      "base_url": "http://localhost:11434",  // Only for Ollama
      "api_key": "sk-...",                   // Only for OpenAI
      "timeout": 30
    },
    "tier2": {
      "provider": "ollama|openai", 
      "model": "llama2|gpt-3.5-turbo|etc",
      "temperature": 0.1,
      "max_tokens": 1000,
      "base_url": "http://localhost:11434",  // Only for Ollama
      "api_key": "sk-...",                   // Only for OpenAI
      "timeout": 30
    }
  }
}
```

## Tier Usage Classification

### Tier 1 (Primary/Conversational)
- User-facing chat sessions (`/chat/session`)
- LangGraph memory agent responses
- Main question answering with full context
- Complex reasoning tasks

### Tier 2 (Internal/Utility)
- Memory extraction from conversations
- Context dependency analysis
- Memory grounding operations
- Input validation and preprocessing
- Memory filtering and relevance scoring
- Configuration testing

## LLM Client Abstraction

```python
class LLMClient:
    def chat_completion(self, messages, temperature=None, max_tokens=None, **kwargs):
        """Unified interface for chat completions"""
        pass

class OpenAIClient(LLMClient):
    def __init__(self, api_key, model, base_url=None):
        pass

class OllamaClient(LLMClient):
    def __init__(self, base_url, model):
        pass

class LLMManager:
    def __init__(self, tier1_config, tier2_config):
        self.tier1_client = self._create_client(tier1_config)
        self.tier2_client = self._create_client(tier2_config)
    
    def get_tier1_client(self):
        return self.tier1_client
    
    def get_tier2_client(self):
        return self.tier2_client
```

## Migration Strategy

1. Create LLM abstraction layer
2. Update configuration schema
3. Add Ollama dependency and client
4. Refactor existing LLM calls to use tier system
5. Update configuration APIs
6. Add validation and testing

## Configuration API Changes

The `/api/config` endpoint will be extended to handle:
- LLM tier configuration
- Provider validation
- Model availability checking
- Connection testing for both OpenAI and Ollama

# LLM Configuration System

The memory system now supports a flexible two-tier LLM configuration that allows you to use different LLM providers (OpenAI and Ollama) for different types of operations.

## Overview

### Two-Tier Architecture

**Tier 1 (Primary/Conversational)**
- User-facing chat sessions
- Main question answering with full context
- Complex reasoning tasks
- LangGraph memory agent responses

**Tier 2 (Internal/Utility)**
- Memory extraction from conversations
- Context dependency analysis
- Memory grounding operations
- Input validation and preprocessing
- Memory filtering and relevance scoring
- Configuration testing

### Supported Providers

- **OpenAI**: GPT-3.5-turbo, GPT-4, and other OpenAI models
- **Ollama**: Local models like Llama2, Mistral, CodeLlama, etc.

## Configuration Methods

### 1. API Configuration (Recommended)

#### Get Current Configuration
```bash
curl http://localhost:5001/api/config
```

#### Update Configuration
```bash
curl -X PUT http://localhost:5001/api/config \
  -H "Content-Type: application/json" \
  -d '{
    "llm": {
      "tier1": {
        "provider": "openai",
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 2000,
        "api_key": "your-openai-api-key",
        "timeout": 30
      },
      "tier2": {
        "provider": "ollama",
        "model": "llama2",
        "temperature": 0.1,
        "max_tokens": 1000,
        "base_url": "http://localhost:11434",
        "timeout": 30
      }
    }
  }'
```

#### Test Configuration
```bash
curl -X POST http://localhost:5001/api/config/test \
  -H "Content-Type: application/json" \
  -d '{
    "llm": {
      "tier1": {
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        "api_key": "your-api-key"
      }
    }
  }'
```

#### Reload Configuration
```bash
curl -X POST http://localhost:5001/api/config/reload
```

### 2. Environment Variables

Set these in your `.env` file:

```bash
# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Optional LLM Configuration
LLM_TIER1_PROVIDER=openai
LLM_TIER1_MODEL=gpt-4
LLM_TIER1_TEMPERATURE=0.7
LLM_TIER1_MAX_TOKENS=2000

LLM_TIER2_PROVIDER=ollama
LLM_TIER2_MODEL=llama2
LLM_TIER2_TEMPERATURE=0.1
LLM_TIER2_MAX_TOKENS=1000
LLM_TIER2_BASE_URL=http://localhost:11434
```

## Configuration Examples

### Example 1: OpenAI Only (Default)
```json
{
  "llm": {
    "tier1": {
      "provider": "openai",
      "model": "gpt-4",
      "temperature": 0.7,
      "max_tokens": 2000,
      "api_key": "your-openai-api-key"
    },
    "tier2": {
      "provider": "openai",
      "model": "gpt-3.5-turbo",
      "temperature": 0.1,
      "max_tokens": 1000,
      "api_key": "your-openai-api-key"
    }
  }
}
```

### Example 2: Ollama Only (Local)
```json
{
  "llm": {
    "tier1": {
      "provider": "ollama",
      "model": "llama2",
      "temperature": 0.7,
      "max_tokens": 2000,
      "base_url": "http://localhost:11434"
    },
    "tier2": {
      "provider": "ollama",
      "model": "mistral",
      "temperature": 0.1,
      "max_tokens": 1000,
      "base_url": "http://localhost:11434"
    }
  }
}
```

### Example 3: Hybrid (OpenAI + Ollama)
```json
{
  "llm": {
    "tier1": {
      "provider": "openai",
      "model": "gpt-4",
      "temperature": 0.7,
      "max_tokens": 2000,
      "api_key": "your-openai-api-key"
    },
    "tier2": {
      "provider": "ollama",
      "model": "llama2",
      "temperature": 0.1,
      "max_tokens": 1000,
      "base_url": "http://localhost:11434"
    }
  }
}
```

## Setup Instructions

### OpenAI Setup
1. Get an API key from [OpenAI](https://platform.openai.com/api-keys)
2. Set `OPENAI_API_KEY` in your `.env` file
3. Configure tier(s) to use `"provider": "openai"`

### Ollama Setup
1. Install Ollama: `curl -fsSL https://ollama.ai/install.sh | sh`
2. Start Ollama: `ollama serve`
3. Pull a model: `ollama pull llama2`
4. Configure tier(s) to use `"provider": "ollama"`
5. Set `base_url` to your Ollama server (default: `http://localhost:11434`)

## Testing

### Test Script
Run the included test script:
```bash
python tests/test_llm_config.py
```

### Manual Testing
1. Start the web server: `python web_app.py`
2. Test configuration: `curl -X POST http://localhost:5001/api/config/test -H "Content-Type: application/json" -d '{"llm": {...}}'`
3. Check logs for connection status

## Performance Tips

1. **Cost Optimization**: Use Ollama for Tier 2 operations to reduce API costs
2. **Speed**: Use faster models (gpt-3.5-turbo) for Tier 2 operations
3. **Quality**: Use better models (gpt-4) for Tier 1 user interactions
4. **Consistency**: Set lower temperature (0.1-0.3) for Tier 2 operations
5. **Capacity**: Set appropriate max_tokens based on use case

## Troubleshooting

### Common Issues

**"LLM manager not initialized"**
- Ensure the web server started successfully
- Check that at least one tier has valid configuration
- Verify API keys and model availability

**"Connection test failed"**
- For OpenAI: Check API key validity and model access
- For Ollama: Ensure server is running and model is pulled
- Check network connectivity and timeouts

**"Model not found"**
- For OpenAI: Verify model name and API access
- For Ollama: Run `ollama list` to see available models

### Debug Mode
Enable debug logging by setting `debug=True` in the web server configuration.

## Migration from Old System

The system maintains backward compatibility. Existing configurations will continue to work, but you can gradually migrate to the new tier-based system for better performance and cost optimization.

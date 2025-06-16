# Configuration Management API

The Memory Agent now includes a comprehensive configuration management API that allows developers to dynamically configure all aspects of the system at runtime.

## Overview

The Configuration API provides endpoints to:
- **Get** current system configuration
- **Update** configuration settings
- **Test** configuration changes before applying them
- **Reload** the system with new configuration

## API Endpoints

### GET /api/config
Get the current system configuration.

**Response:**
```json
{
  "success": true,
  "config": {
    "redis": {
      "host": "localhost",
      "port": 6381,
      "db": 0,
      "vectorset_key": "memories"
    },
    "openai": {
      "api_key": "sk-proj-...A",
      "organization": "",
      "embedding_model": "text-embedding-ada-002",
      "embedding_dimension": 1536,
      "chat_model": "gpt-3.5-turbo",
      "temperature": 0.1
    },
    "langgraph": {
      "model_name": "gpt-3.5-turbo",
      "temperature": 0.1,
      "system_prompt_enabled": true
    },
    "memory_agent": {
      "default_top_k": 5,
      "apply_grounding_default": true,
      "validation_enabled": true
    },
    "web_server": {
      "host": "0.0.0.0",
      "port": 5001,
      "debug": true,
      "cors_enabled": true
    }
  },
  "runtime": {
    "memory_agent_initialized": true,
    "memory_count": 42,
    "redis_connected": true,
    "actual_redis_host": "localhost",
    "actual_redis_port": 6381,
    "timestamp": "2024-01-15T10:30:00"
  }
}
```

### PUT /api/config
Update system configuration. You can update any subset of configuration categories.

**Request Body:**
```json
{
  "redis": {
    "host": "redis.example.com",
    "port": 6379,
    "db": 1
  },
  "openai": {
    "temperature": 0.2,
    "chat_model": "gpt-4"
  },
  "memory_agent": {
    "default_top_k": 10
  }
}
```

**Response:**
```json
{
  "success": true,
  "changes_made": [
    "redis.host: localhost → redis.example.com",
    "redis.port: 6381 → 6379",
    "openai.temperature: 0.1 → 0.2",
    "memory_agent.default_top_k: 5 → 10"
  ],
  "requires_restart": true,
  "warnings": [],
  "message": "Configuration updated. Memory agent restart required for changes to take effect."
}
```

### POST /api/config/test
Test configuration changes without applying them. Validates settings and tests connections.

**Request Body:** Same format as PUT /api/config

**Response:**
```json
{
  "success": true,
  "test_results": {
    "overall_valid": true,
    "tests": {
      "redis": {
        "valid": true,
        "connection_successful": true,
        "errors": [],
        "warnings": []
      },
      "openai": {
        "valid": true,
        "api_connection_successful": true,
        "errors": [],
        "warnings": ["Temperature should be between 0 and 2"]
      }
    }
  },
  "message": "Configuration test completed"
}
```

### POST /api/config/reload
Reload configuration and restart the memory agent with current settings.

**Response:**
```json
{
  "success": true,
  "message": "Configuration reloaded and memory agent restarted successfully",
  "reload_details": {
    "agent_was_initialized": true,
    "agent_now_initialized": true,
    "redis_connected": true,
    "memory_count_before": 42,
    "memory_count_after": 42,
    "config_applied": {
      "redis_host": "localhost",
      "redis_port": 6381,
      "redis_db": 0,
      "langgraph_model": "gpt-3.5-turbo",
      "langgraph_temperature": 0.1
    }
  }
}
```

## Configuration Categories

### Redis Configuration
- **host**: Redis server hostname (default: "localhost")
- **port**: Redis server port (default: 6381)
- **db**: Redis database number (default: 0)
- **vectorset_key**: Name of the vector set for memories (default: "memories")

### OpenAI Configuration
- **api_key**: OpenAI API key
- **organization**: OpenAI organization ID (optional)
- **embedding_model**: Model for embeddings (default: "text-embedding-ada-002")
- **embedding_dimension**: Embedding vector dimension (default: 1536)
- **chat_model**: Model for chat completions (default: "gpt-3.5-turbo")
- **temperature**: Temperature for OpenAI calls (default: 0.1)

### LangGraph Configuration
- **model_name**: Model for LangGraph agent (default: "gpt-3.5-turbo")
- **temperature**: Temperature for LangGraph model (default: 0.1)
- **system_prompt_enabled**: Whether to use system prompts (default: true)

### Memory Agent Configuration
- **default_top_k**: Default number of memories to retrieve (default: 5)
- **apply_grounding_default**: Default contextual grounding setting (default: true)
- **validation_enabled**: Whether to validate user input (default: true)

### Web Server Configuration
- **host**: Server bind address (default: "0.0.0.0")
- **port**: Server port (default: 5001)
- **debug**: Debug mode (default: true)
- **cors_enabled**: CORS support (default: true)

*Note: Web server changes require application restart to take effect.*

## Usage Examples

### Change Redis Server
```bash
curl -X PUT http://localhost:5001/api/config \
  -H "Content-Type: application/json" \
  -d '{
    "redis": {
      "host": "redis.example.com",
      "port": 6379,
      "db": 1
    }
  }'
```

### Update AI Model Settings
```bash
curl -X PUT http://localhost:5001/api/config \
  -H "Content-Type: application/json" \
  -d '{
    "openai": {
      "chat_model": "gpt-4",
      "temperature": 0.2
    },
    "langgraph": {
      "model_name": "gpt-4",
      "temperature": 0.15
    }
  }'
```

### Test Configuration Before Applying
```bash
curl -X POST http://localhost:5001/api/config/test \
  -H "Content-Type: application/json" \
  -d '{
    "redis": {
      "host": "new-redis-server",
      "port": 6379
    }
  }'
```

### Reload After Changes
```bash
curl -X POST http://localhost:5001/api/config/reload
```

## Testing

Run the test script to see the configuration API in action:

```bash
python test_config_api.py
```

This will demonstrate all configuration management features and show example requests/responses.

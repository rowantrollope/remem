"""
Global configuration management for the Memory Agent API.
"""

import os
from typing import Dict, Any
from .constants import CACHE_TYPES


# Global configuration store
app_config: Dict[str, Any] = {
    "redis": {
        "host": os.getenv("REDIS_HOST", "localhost"),
        "port": int(os.getenv("REDIS_PORT", "6379")),
        "db": int(os.getenv("REDIS_DB", "0")),
        "vectorset_key": "augment:remem"
    },
    "llm": {
        "tier1": {
            "provider": "openai",  # "openai" or "ollama"
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 2000,
            "base_url": None,  # For Ollama: "http://localhost:11434"
            "api_key": os.getenv("OPENAI_API_KEY", ""),
            "timeout": 30
        },
        "tier2": {
            "provider": "openai",  # "openai" or "ollama"
            "model": "gpt-3.5-turbo",
            "temperature": 0.1,
            "max_tokens": 1000,
            "base_url": None,  # For Ollama: "http://localhost:11434"
            "api_key": os.getenv("OPENAI_API_KEY", ""),
            "timeout": 30
        }
    },
    "embedding": {
        "provider": os.getenv("EMBEDDING_PROVIDER", "openai"),  # "openai" or "ollama"
        "model": os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002"),
        "dimension": int(os.getenv("EMBEDDING_DIMENSION", "1536")),
        "base_url": os.getenv("EMBEDDING_BASE_URL", None),  # For Ollama: "http://localhost:11434"
        "api_key": os.getenv("EMBEDDING_API_KEY", "") or os.getenv("OPENAI_API_KEY", ""),
        "timeout": int(os.getenv("EMBEDDING_TIMEOUT", "30"))
    },
    "openai": {
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "organization": os.getenv("OPENAI_ORG_ID", ""),
        "embedding_model": "text-embedding-ada-002",  # Deprecated: use embedding.model instead
        "embedding_dimension": 1536,  # Deprecated: use embedding.dimension instead
        "chat_model": "gpt-3.5-turbo",
        "temperature": 0.1
    },
    "langgraph": {
        "model_name": "gpt-3.5-turbo",
        "temperature": 0.1,
        "system_prompt_enabled": True
    },
    "memory_agent": {
        "default_top_k": 5,
        "apply_grounding_default": True,
        "validation_enabled": True
    },
    "web_server": {
        "host": "0.0.0.0",
        "port": 5001,
        "debug": True,
        "cors_enabled": True
    },
    "langcache": {
        "enabled": True,  # Master switch for all caching
        "cache_types": CACHE_TYPES.copy(),
        "minimum_similarity": 0.95,  # Minimum similarity threshold for cache hits
        "ttl_minutes": 2  # Time to live in minutes for cache entries
    }
}


def get_config() -> Dict[str, Any]:
    """Get the current application configuration."""
    return app_config


def update_config(section: str, updates: Dict[str, Any]) -> None:
    """Update a configuration section."""
    if section in app_config:
        app_config[section].update(updates)
    else:
        app_config[section] = updates

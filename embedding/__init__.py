"""
Embedding module for supporting multiple embedding providers.
"""

from .embedding_client import (
    EmbeddingClient,
    EmbeddingConfig,
    OpenAIEmbeddingClient,
    OllamaEmbeddingClient,
    create_embedding_client,
    get_embedding_config_from_app_config
)

__all__ = [
    "EmbeddingClient",
    "EmbeddingConfig", 
    "OpenAIEmbeddingClient",
    "OllamaEmbeddingClient",
    "create_embedding_client",
    "get_embedding_config_from_app_config"
]

"""
Embedding client abstraction for supporting multiple embedding providers.

This module provides a unified interface for generating embeddings using different
providers like OpenAI and Ollama, allowing for easy configuration and switching
between embedding models.
"""

import os
import requests
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import openai


@dataclass
class EmbeddingConfig:
    """Configuration for embedding providers."""
    provider: str  # "openai" or "ollama"
    model: str
    dimension: int
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout: int = 30


class EmbeddingClient(ABC):
    """Abstract base class for embedding clients."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
    
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for the given text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
            
        Raises:
            Exception: If embedding generation fails
        """
        pass
    
    @abstractmethod
    def test_connection(self) -> Dict[str, Any]:
        """Test connection to the embedding provider.
        
        Returns:
            Dictionary with success status and details
        """
        pass


class OpenAIEmbeddingClient(EmbeddingClient):
    """OpenAI embedding client implementation."""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self.client = openai.OpenAI(
            api_key=config.api_key or os.getenv("OPENAI_API_KEY"),
            base_url=config.base_url
        )
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate OpenAI embedding for text."""
        try:
            response = self.client.embeddings.create(
                model=self.config.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            raise Exception(f"OpenAI embedding error: {str(e)}")
    
    def test_connection(self) -> Dict[str, Any]:
        """Test OpenAI API connection."""
        try:
            # Test with a simple embedding request
            response = self.client.embeddings.create(
                model=self.config.model,
                input="test"
            )
            
            embedding = response.data[0].embedding
            actual_dimension = len(embedding)
            
            return {
                'success': True,
                'provider': 'openai',
                'model': self.config.model,
                'configured_dimension': self.config.dimension,
                'actual_dimension': actual_dimension,
                'dimension_match': actual_dimension == self.config.dimension
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'provider': 'openai'
            }


class OllamaEmbeddingClient(EmbeddingClient):
    """Ollama embedding client implementation."""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self.base_url = config.base_url or "http://localhost:11434"
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate Ollama embedding for text."""
        try:
            payload = {
                "model": self.config.model,
                "prompt": text
            }
            
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get('embedding', [])
        except Exception as e:
            raise Exception(f"Ollama embedding error: {str(e)}")
    
    def test_connection(self) -> Dict[str, Any]:
        """Test Ollama API connection."""
        try:
            # First check if Ollama is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            
            # Check if model is available
            models = response.json().get('models', [])
            model_names = [model.get('name', '') for model in models]
            
            if self.config.model not in model_names:
                return {
                    'success': False,
                    'error': f"Model '{self.config.model}' not found. Available models: {model_names}",
                    'provider': 'ollama',
                    'available_models': model_names
                }
            
            # Test embedding generation
            test_embedding = self.get_embedding("test")
            actual_dimension = len(test_embedding)
            
            return {
                'success': True,
                'provider': 'ollama',
                'model': self.config.model,
                'configured_dimension': self.config.dimension,
                'actual_dimension': actual_dimension,
                'dimension_match': actual_dimension == self.config.dimension,
                'available_models': model_names
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'provider': 'ollama'
            }


def create_embedding_client(config: EmbeddingConfig) -> EmbeddingClient:
    """Factory function to create embedding clients based on provider.
    
    Args:
        config: Embedding configuration
        
    Returns:
        Appropriate embedding client instance
        
    Raises:
        ValueError: If provider is not supported
    """
    if config.provider.lower() == "openai":
        return OpenAIEmbeddingClient(config)
    elif config.provider.lower() == "ollama":
        return OllamaEmbeddingClient(config)
    else:
        raise ValueError(f"Unsupported embedding provider: {config.provider}")


def get_embedding_config_from_app_config(app_config: Dict[str, Any]) -> EmbeddingConfig:
    """Extract embedding configuration from app config.
    
    Args:
        app_config: Application configuration dictionary
        
    Returns:
        EmbeddingConfig instance
    """
    embedding_config = app_config.get("embedding", {})
    
    return EmbeddingConfig(
        provider=embedding_config.get("provider", "openai"),
        model=embedding_config.get("model", "text-embedding-ada-002"),
        dimension=embedding_config.get("dimension", 1536),
        base_url=embedding_config.get("base_url"),
        api_key=embedding_config.get("api_key"),
        timeout=embedding_config.get("timeout", 30)
    )

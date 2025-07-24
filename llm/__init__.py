"""
LLM Package - Unified interface for OpenAI and Ollama LLM providers

This package provides a unified interface for interacting with different LLM providers
(OpenAI and Ollama) through a two-tier system for different use cases.
"""

from .llm_manager import (
    LLMConfig,
    LLMClient,
    OpenAIClient,
    OllamaClient,
    LLMManager,
    get_llm_manager,
    init_llm_manager
)

__all__ = [
    'LLMConfig',
    'LLMClient', 
    'OpenAIClient',
    'OllamaClient',
    'LLMManager',
    'get_llm_manager',
    'init_llm_manager'
]

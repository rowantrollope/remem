#!/usr/bin/env python3
"""
LLM Manager - Unified interface for OpenAI and Ollama LLM providers

This module provides a unified interface for interacting with different LLM providers
(OpenAI and Ollama) through a two-tier system for different use cases.
"""

import os
import json
import requests
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import openai


@dataclass
class LLMConfig:
    """Configuration for an LLM provider."""
    provider: str  # "openai" or "ollama"
    model: str
    temperature: float = 0.7
    max_tokens: int = 1000
    base_url: Optional[str] = None  # For Ollama
    api_key: Optional[str] = None   # For OpenAI
    timeout: int = 30


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def chat_completion(self, messages: List[Dict[str, str]], 
                       temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None,
                       **kwargs) -> Dict[str, Any]:
        """
        Generate a chat completion.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Dict with 'content' key containing the response text
        """
        pass
    
    @abstractmethod
    def test_connection(self) -> Dict[str, Any]:
        """
        Test the connection to the LLM provider.
        
        Returns:
            Dict with 'success' boolean and optional 'error' message
        """
        pass


class OpenAIClient(LLMClient):
    """OpenAI LLM client implementation."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = openai.OpenAI(
            api_key=config.api_key or os.getenv("OPENAI_API_KEY"),
            base_url=config.base_url
        )
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None,
                       **kwargs) -> Dict[str, Any]:
        """Generate chat completion using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=temperature or self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens,
                timeout=self.config.timeout,
                **kwargs
            )
            
            return {
                'content': response.choices[0].message.content,
                'usage': response.usage.model_dump() if response.usage else None,
                'model': response.model
            }
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def test_connection(self) -> Dict[str, Any]:
        """Test OpenAI API connection."""
        try:
            # Simple test with minimal tokens
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
                timeout=10
            )
            return {
                'success': True,
                'model': response.model,
                'provider': 'openai'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'provider': 'openai'
            }


class OllamaClient(LLMClient):
    """Ollama LLM client implementation."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = config.base_url or "http://localhost:11434"
        
    def chat_completion(self, messages: List[Dict[str, str]], 
                       temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None,
                       **kwargs) -> Dict[str, Any]:
        """Generate chat completion using Ollama API."""
        try:
            # Convert messages to Ollama format
            prompt = self._messages_to_prompt(messages)
            
            payload = {
                "model": self.config.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature or self.config.temperature,
                    "num_predict": max_tokens or self.config.max_tokens,
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return {
                'content': result.get('response', ''),
                'model': result.get('model', self.config.model),
                'usage': {
                    'prompt_tokens': result.get('prompt_eval_count', 0),
                    'completion_tokens': result.get('eval_count', 0),
                    'total_tokens': result.get('prompt_eval_count', 0) + result.get('eval_count', 0)
                }
            }
        except Exception as e:
            raise Exception(f"Ollama API error: {str(e)}")
    
    def test_connection(self) -> Dict[str, Any]:
        """Test Ollama API connection."""
        try:
            # Test basic connectivity
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
            
            # Test simple generation
            test_response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.config.model,
                    "prompt": "Hi",
                    "stream": False,
                    "options": {"num_predict": 5}
                },
                timeout=30
            )
            test_response.raise_for_status()
            
            return {
                'success': True,
                'model': self.config.model,
                'provider': 'ollama',
                'available_models': model_names
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'provider': 'ollama'
            }
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI-style messages to a single prompt for Ollama."""
        prompt_parts = []
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"Human: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)


class LLMManager:
    """Manager for two-tier LLM system."""
    
    def __init__(self, tier1_config: LLMConfig, tier2_config: LLMConfig):
        self.tier1_config = tier1_config
        self.tier2_config = tier2_config
        self.tier1_client = self._create_client(tier1_config)
        self.tier2_client = self._create_client(tier2_config)
    
    def _create_client(self, config: LLMConfig) -> LLMClient:
        """Create appropriate client based on provider."""
        if config.provider.lower() == 'openai':
            return OpenAIClient(config)
        elif config.provider.lower() == 'ollama':
            return OllamaClient(config)
        else:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")
    
    def get_tier1_client(self) -> LLMClient:
        """Get Tier 1 (primary/conversational) LLM client."""
        return self.tier1_client
    
    def get_tier2_client(self) -> LLMClient:
        """Get Tier 2 (internal/utility) LLM client."""
        return self.tier2_client
    
    def test_all_connections(self) -> Dict[str, Any]:
        """Test connections for both tiers."""
        return {
            'tier1': self.tier1_client.test_connection(),
            'tier2': self.tier2_client.test_connection()
        }
    
    def reload_clients(self, tier1_config: LLMConfig = None, tier2_config: LLMConfig = None):
        """Reload clients with new configurations."""
        if tier1_config:
            self.tier1_config = tier1_config
            self.tier1_client = self._create_client(tier1_config)
        
        if tier2_config:
            self.tier2_config = tier2_config
            self.tier2_client = self._create_client(tier2_config)


# Global LLM manager instance
llm_manager: Optional[LLMManager] = None


def get_llm_manager() -> LLMManager:
    """Get the global LLM manager instance."""
    global llm_manager
    if llm_manager is None:
        raise RuntimeError("LLM manager not initialized. Call init_llm_manager() first.")
    return llm_manager


def init_llm_manager(tier1_config: LLMConfig, tier2_config: LLMConfig) -> LLMManager:
    """Initialize the global LLM manager."""
    global llm_manager
    llm_manager = LLMManager(tier1_config, tier2_config)
    return llm_manager

#!/usr/bin/env python3
"""
Redis LangCache Client

This module provides a client for the Redis LangCache API for semantic prompt caching.
Uses the REST API endpoints for storing and retrieving cached LLM responses.
"""

import os
import requests
from typing import Dict, Any, Optional, List


def is_cache_enabled_for_operation(operation_type: str) -> bool:
    """
    Check if caching is enabled for a specific operation type.

    Args:
        operation_type: The operation type to check

    Returns:
        True if caching is enabled for this operation
    """
    # Disable caching for memory operations with long prompts
    # These prompts are typically unique and often exceed LangCache's 1024 char limit
    memory_extraction_operations = {
        'memory_extraction',
        'memory_grounding',
        'context_analysis',
        'query_optimization'  # Validation prompts can be quite long
    }

    if operation_type in memory_extraction_operations:
        return False

    try:
        # Import here to avoid circular imports
        import sys
        if 'web_app' in sys.modules:
            web_app = sys.modules['web_app']
            if hasattr(web_app, 'app_config'):
                config = web_app.app_config
                # Check master switch first
                if not config.get('langcache', {}).get('enabled', True):
                    return False
                # Check specific operation type
                return config.get('langcache', {}).get('cache_types', {}).get(operation_type, True)
    except Exception:
        pass

    # Default to True if config not available
    return True


class LangCacheClient:
    """Client for Redis LangCache API."""

    def __init__(self, host: str = None, api_key: str = None, cache_id: str = None):
        """
        Initialize LangCache client.

        Args:
            host: LangCache API base URL (from LANGCACHE_HOST env var if not provided)
            api_key: LangCache API key (from LANGCACHE_API_KEY env var if not provided)
            cache_id: Cache ID (from LANGCACHE_CACHE_ID env var if not provided)
        """
        self.host = host or os.getenv("LANGCACHE_HOST")
        self.api_key = api_key or os.getenv("LANGCACHE_API_KEY")
        self.cache_id = cache_id or os.getenv("LANGCACHE_CACHE_ID")
        
        if not self.host:
            raise ValueError("LangCache host must be provided via LANGCACHE_HOST environment variable or host parameter")
        if not self.api_key:
            raise ValueError("LangCache API key must be provided via LANGCACHE_API_KEY environment variable or api_key parameter")
        if not self.cache_id:
            raise ValueError("LangCache cache ID must be provided via LANGCACHE_CACHE_ID environment variable or cache_id parameter")
        
        # Ensure host doesn't end with slash
        self.host = self.host.rstrip('/')
        
        # Base URL for API calls
        self.base_url = f"{self.host}/v1/caches/{self.cache_id}"
        
        # Headers for API calls
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'stores': 0,
            'errors': 0
        }
    
    def _create_prompt_key(self, messages: List[Dict[str, str]]) -> str:
        """
        Create a prompt key from messages for LangCache semantic similarity.

        Args:
            messages: Chat messages

        Returns:
            Plain text content for semantic similarity matching
        """
        # Extract just the user's text content for semantic similarity
        # LangCache expects plain text, not JSON structures
        user_content = ""

        # Find the last user message content
        for message in reversed(messages):
            if message.get("role") == "user":
                user_content = message.get("content", "")
                break

        # If no user message found, use the first message content as fallback
        if not user_content and messages:
            user_content = messages[0].get("content", "")

        # Ensure we have some content and it's within reasonable length
        if not user_content:
            user_content = "empty_prompt"

        # Truncate if too long (LangCache has 1024 char limit)
        if len(user_content) > 1000:  # Leave some buffer
            user_content = user_content[:1000] + "..."

        return user_content
    
    def search_cache(self, messages: List[Dict[str, str]],
                    attributes: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Search for cached response using semantic similarity.

        Args:
            messages: Chat messages
            attributes: Additional attributes for scoping

        Returns:
            Cached response dict or None if not found
        """
        try:
            prompt = self._create_prompt_key(messages)
            
            # Prepare request data
            request_data = {
                "prompt": prompt
            }
            
            # Add attributes if provided
            if attributes:
                request_data["attributes"] = attributes
            
            # Make API call
            response = requests.post(
                f"{self.base_url}/entries/search",
                headers=self.headers,
                json=request_data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result and 'response' in result:
                    self.stats['hits'] += 1
                    return {
                        'content': result['response'],
                        '_cache_hit': True,
                        '_cached_at': result.get('created_at'),
                        '_similarity_score': result.get('similarity_score')
                    }
                else:
                    self.stats['misses'] += 1
                    return None
            elif response.status_code == 404:
                # No matching cache entry found
                self.stats['misses'] += 1
                return None
            else:
                self.stats['errors'] += 1
                print(f"⚠️ LangCache search error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            self.stats['errors'] += 1
            print(f"⚠️ LangCache search exception: {e}")
            return None
    
    def store_cache(self, messages: List[Dict[str, str]], response: str,
                   attributes: Dict[str, Any] = None) -> bool:
        """
        Store response in cache.

        Args:
            messages: Chat messages
            response: LLM response to cache
            attributes: Additional attributes

        Returns:
            True if stored successfully
        """
        try:
            prompt = self._create_prompt_key(messages)
            
            # Prepare request data
            request_data = {
                "prompt": prompt,
                "response": response
            }
            
            # Add attributes if provided (but LangCache API is very strict about allowed attributes)
            if attributes:
                request_data["attributes"] = attributes
            
            # Make API call
            api_response = requests.post(
                f"{self.base_url}/entries",
                headers=self.headers,
                json=request_data,
                timeout=10
            )
            
            if api_response.status_code in [200, 201]:
                self.stats['stores'] += 1
                return True
            else:
                self.stats['errors'] += 1
                print(f"⚠️ LangCache store error: {api_response.status_code} - {api_response.text}")
                return False
                
        except Exception as e:
            self.stats['errors'] += 1
            print(f"⚠️ LangCache store exception: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check cache health.
        
        Returns:
            Health status dict
        """
        try:
            response = requests.get(
                f"{self.base_url}/health",
                headers=self.headers,
                timeout=5
            )
            
            if response.status_code == 200:
                return {
                    'healthy': True,
                    'status': response.json()
                }
            else:
                return {
                    'healthy': False,
                    'error': f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e)
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self.stats,
            'hit_rate_percent': round(hit_rate, 2),
            'total_requests': total_requests
        }
    
    def clear_stats(self):
        """Clear cache statistics."""
        self.stats = {
            'hits': 0,
            'misses': 0,
            'stores': 0,
            'errors': 0
        }


class CachedLLMClient:
    """LLM client wrapper with LangCache integration."""
    
    def __init__(self, llm_client, langcache_client: LangCacheClient):
        """
        Initialize cached LLM client.
        
        Args:
            llm_client: Original LLM client
            langcache_client: LangCache client instance
        """
        self.llm_client = llm_client
        self.cache = langcache_client
    
    def chat_completion(self, messages: List[Dict[str, str]],
                       operation_type: str = 'conversation',
                       temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None,
                       **kwargs) -> Dict[str, Any]:
        """
        Generate chat completion with caching.

        Args:
            messages: Chat messages
            operation_type: Type of operation for cache categorization
            temperature: LLM temperature
            max_tokens: Max tokens
            **kwargs: Additional LLM parameters

        Returns:
            LLM response with cache metadata
        """
        # Check if caching is enabled for this operation type
        if not is_cache_enabled_for_operation(operation_type):
            # Caching disabled for this operation, call LLM directly
            response = self.llm_client.chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            response['_cache_hit'] = False
            response['_cache_disabled'] = True
            return response

        # Try to get from cache first
        cached_response = self.cache.search_cache(
            messages=messages
        )

        if cached_response:
            return cached_response

        # Call original LLM client
        response = self.llm_client.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        # Cache the response
        if 'content' in response:
            self.cache.store_cache(
                messages=messages,
                response=response['content']
            )

        response['_cache_hit'] = False
        return response
    
    def test_connection(self) -> Dict[str, Any]:
        """Test connection (delegates to original client)."""
        return self.llm_client.test_connection()

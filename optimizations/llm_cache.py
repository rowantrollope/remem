#!/usr/bin/env python3
"""
LLM Response Caching System

This module provides Redis-based caching for LLM responses to reduce API calls
and improve performance. Supports configurable TTL, cache keys based on content
hashing, and A/B testing capabilities.
"""

import hashlib
import json
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import redis
import os
from dataclasses import dataclass


@dataclass
class CacheConfig:
    """Configuration for LLM response caching."""
    enabled: bool = True
    default_ttl: int = 3600  # 1 hour default
    max_key_length: int = 250
    cache_prefix: str = "llm_cache"
    
    # TTL settings for different operation types
    ttl_settings: Dict[str, int] = None
    
    def __post_init__(self):
        if self.ttl_settings is None:
            self.ttl_settings = {
                'query_optimization': 7200,      # 2 hours - queries don't change often
                'memory_relevance': 1800,        # 30 minutes - context dependent
                'context_analysis': 3600,        # 1 hour - relatively stable
                'memory_grounding': 1800,        # 30 minutes - context dependent
                'extraction_evaluation': 900,   # 15 minutes - user behavior dependent
                'conversation': 300,             # 5 minutes - highly dynamic
                'answer_generation': 1800        # 30 minutes - depends on memory context
            }


class LLMCache:
    """Redis-based cache for LLM responses with intelligent key generation."""
    
    def __init__(self, redis_client: redis.Redis, config: CacheConfig = None):
        """
        Initialize LLM cache.
        
        Args:
            redis_client: Redis client instance
            config: Cache configuration (uses defaults if None)
        """
        self.redis_client = redis_client
        self.config = config or CacheConfig()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'stores': 0,
            'errors': 0
        }
    
    def _generate_cache_key(self, operation_type: str, content: str, 
                          context: Dict[str, Any] = None) -> str:
        """
        Generate a cache key based on operation type, content, and context.
        
        Args:
            operation_type: Type of LLM operation (e.g., 'query_optimization')
            content: Main content to hash
            context: Additional context that affects the response
            
        Returns:
            Cache key string
        """
        # Create a deterministic hash of the content and context
        hash_input = {
            'content': content.strip(),
            'context': context or {}
        }
        
        # Sort context keys for consistent hashing
        if context:
            hash_input['context'] = {k: v for k, v in sorted(context.items())}
        
        content_hash = hashlib.sha256(
            json.dumps(hash_input, sort_keys=True).encode('utf-8')
        ).hexdigest()[:16]  # Use first 16 chars of hash
        
        cache_key = f"{self.config.cache_prefix}:{operation_type}:{content_hash}"
        
        # Ensure key length doesn't exceed Redis limits
        if len(cache_key) > self.config.max_key_length:
            cache_key = cache_key[:self.config.max_key_length]
        
        return cache_key
    
    def get(self, operation_type: str, content: str, 
           context: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached LLM response.
        
        Args:
            operation_type: Type of LLM operation
            content: Content that was sent to LLM
            context: Additional context
            
        Returns:
            Cached response dict or None if not found
        """
        if not self.config.enabled:
            return None
            
        try:
            cache_key = self._generate_cache_key(operation_type, content, context)
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                self.stats['hits'] += 1
                result = json.loads(cached_data.decode('utf-8'))
                result['_cache_hit'] = True
                result['_cached_at'] = result.get('_cached_at')
                return result
            else:
                self.stats['misses'] += 1
                return None
                
        except Exception as e:
            self.stats['errors'] += 1
            print(f"⚠️ Cache get error: {e}")
            return None
    
    def set(self, operation_type: str, content: str, response: Dict[str, Any],
           context: Dict[str, Any] = None, ttl: int = None) -> bool:
        """
        Store LLM response in cache.
        
        Args:
            operation_type: Type of LLM operation
            content: Content that was sent to LLM
            response: LLM response to cache
            context: Additional context
            ttl: Time to live in seconds (uses default if None)
            
        Returns:
            True if stored successfully, False otherwise
        """
        if not self.config.enabled:
            return False
            
        try:
            cache_key = self._generate_cache_key(operation_type, content, context)
            
            # Add cache metadata
            cache_data = response.copy()
            cache_data['_cached_at'] = datetime.now(timezone.utc).isoformat()
            cache_data['_operation_type'] = operation_type
            
            # Determine TTL
            if ttl is None:
                ttl = self.config.ttl_settings.get(operation_type, self.config.default_ttl)
            
            # Store in Redis
            self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(cache_data, ensure_ascii=False)
            )
            
            self.stats['stores'] += 1
            return True
            
        except Exception as e:
            self.stats['errors'] += 1
            print(f"⚠️ Cache set error: {e}")
            return False
    
    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate cache entries matching a pattern.
        
        Args:
            pattern: Redis key pattern (e.g., "llm_cache:query_optimization:*")
            
        Returns:
            Number of keys deleted
        """
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            print(f"⚠️ Cache invalidation error: {e}")
            return 0
    
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
        """Reset cache statistics."""
        self.stats = {
            'hits': 0,
            'misses': 0,
            'stores': 0,
            'errors': 0
        }


class CachedLLMClient:
    """Wrapper for LLM clients that adds caching capabilities."""
    
    def __init__(self, llm_client, cache: LLMCache):
        """
        Initialize cached LLM client.
        
        Args:
            llm_client: Original LLM client (OpenAI or Ollama)
            cache: LLM cache instance
        """
        self.llm_client = llm_client
        self.cache = cache
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       operation_type: str = 'conversation',
                       cache_context: Dict[str, Any] = None,
                       bypass_cache: bool = False,
                       temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None,
                       **kwargs) -> Dict[str, Any]:
        """
        Generate chat completion with caching.
        
        Args:
            messages: Chat messages
            operation_type: Type of operation for cache key generation
            cache_context: Additional context for cache key
            bypass_cache: If True, skip caching entirely
            temperature: LLM temperature
            max_tokens: Max tokens
            **kwargs: Additional LLM parameters
            
        Returns:
            LLM response with cache metadata
        """
        # Skip cache entirely if bypass_cache is True
        if bypass_cache:
            response = self.llm_client.chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            response['_cache_hit'] = False
            response['_cache_bypassed'] = True
            return response
        
        # Create cache key from messages and parameters
        content = json.dumps(messages, sort_keys=True)
        context = {
            'temperature': temperature,
            'max_tokens': max_tokens,
            **(cache_context or {})
        }
        
        # Try to get from cache first
        cached_response = self.cache.get(operation_type, content, context)
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
        self.cache.set(operation_type, content, response, context)
        response['_cache_hit'] = False
        
        return response
    
    def test_connection(self) -> Dict[str, Any]:
        """Test connection (delegates to original client)."""
        return self.llm_client.test_connection()


def create_cached_llm_manager(llm_manager, redis_client: redis.Redis, 
                            cache_config: CacheConfig = None):
    """
    Create a cached version of an LLM manager.
    
    Args:
        llm_manager: Original LLM manager
        redis_client: Redis client for caching
        cache_config: Cache configuration
        
    Returns:
        LLM manager with cached clients
    """
    cache = LLMCache(redis_client, cache_config)
    
    # Wrap both tier clients with caching
    cached_tier1 = CachedLLMClient(llm_manager.get_tier1_client(), cache)
    cached_tier2 = CachedLLMClient(llm_manager.get_tier2_client(), cache)
    
    # Create a wrapper that maintains the same interface
    class CachedLLMManager:
        def __init__(self):
            self.tier1_config = llm_manager.tier1_config
            self.tier2_config = llm_manager.tier2_config
            self.cache = cache
        
        def get_tier1_client(self):
            return cached_tier1
        
        def get_tier2_client(self):
            return cached_tier2
        
        def test_all_connections(self):
            return llm_manager.test_all_connections()
        
        def get_cache_stats(self):
            return self.cache.get_stats()
        
        def clear_cache_stats(self):
            self.cache.clear_stats()
    
    return CachedLLMManager()

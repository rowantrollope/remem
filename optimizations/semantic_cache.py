#!/usr/bin/env python3
"""
Semantic LLM Cache using Redis VectorSet

This module provides semantic caching for LLM responses using vector similarity
instead of exact hash matching. This allows caching of semantically similar
queries that would produce similar responses.
"""

import json
import uuid
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone
import redis
import openai
import os
from dataclasses import dataclass


@dataclass
class SemanticCacheConfig:
    """Configuration for semantic LLM caching."""
    enabled: bool = True
    similarity_threshold: float = 0.85  # Minimum similarity for cache hit
    max_cache_entries: int = 10000  # Maximum cached entries per operation type
    default_ttl: int = 3600  # Default TTL in seconds
    vectorset_prefix: str = "llm_cache"  # Prefix for cache vectorsets
    embedding_model: str = "text-embedding-ada-002"
    
    # TTL settings for different operation types
    ttl_settings: Dict[str, int] = None
    
    # Similarity thresholds for different operation types
    similarity_thresholds: Dict[str, float] = None
    
    def __post_init__(self):
        if self.ttl_settings is None:
            self.ttl_settings = {
                'query_optimization': 7200,      # 2 hours - stable patterns
                'memory_relevance': 1800,        # 30 minutes - context dependent
                'context_analysis': 3600,        # 1 hour - relatively stable
                'memory_grounding': 1800,        # 30 minutes - context dependent
                'extraction_evaluation': 900,   # 15 minutes - user behavior dependent
                'conversation': 300,             # 5 minutes - highly dynamic
                'answer_generation': 1800        # 30 minutes - depends on context
            }
        
        if self.similarity_thresholds is None:
            self.similarity_thresholds = {
                'query_optimization': 0.90,     # High similarity for optimization
                'memory_relevance': 0.85,       # Medium-high for relevance
                'context_analysis': 0.88,       # High for context analysis
                'memory_grounding': 0.82,       # Medium for grounding (more flexible)
                'extraction_evaluation': 0.80,  # Lower for extraction (more variation)
                'conversation': 0.95,           # Very high for conversations (avoid wrong responses)
                'answer_generation': 0.87       # High for answer generation
            }


class SemanticLLMCache:
    """Semantic cache for LLM responses using Redis VectorSet."""
    
    def __init__(self, redis_client: redis.Redis, config: SemanticCacheConfig = None):
        """
        Initialize semantic LLM cache.
        
        Args:
            redis_client: Redis client instance (should use decode_responses=False for vectors)
            config: Cache configuration
        """
        self.redis_client = redis_client
        self.config = config or SemanticCacheConfig()
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Cache statistics
        self.stats = {
            'semantic_hits': 0,
            'semantic_misses': 0,
            'stores': 0,
            'errors': 0,
            'embedding_calls': 0
        }
        
        # Embedding dimension for text-embedding-ada-002
        self.EMBEDDING_DIM = 1536
    
    def _get_vectorset_name(self, operation_type: str) -> str:
        """Get vectorset name for operation type."""
        return f"{self.config.vectorset_prefix}_{operation_type}"
    
    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using OpenAI.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            self.stats['embedding_calls'] += 1
            response = self.openai_client.embeddings.create(
                model=self.config.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"⚠️ Embedding generation failed: {e}")
            raise
    
    def _create_cache_key(self, operation_type: str, content: str, context: Dict[str, Any] = None) -> str:
        """
        Create a cache key that includes operation context.
        
        Args:
            operation_type: Type of LLM operation
            content: Main content
            context: Additional context
            
        Returns:
            Cache key for embedding
        """
        # Combine content with relevant context for embedding
        cache_content = content
        
        if context:
            # Add relevant context that affects the response
            relevant_context = []
            for key, value in context.items():
                if key in ['temperature', 'max_tokens', 'model', 'system_prompt']:
                    relevant_context.append(f"{key}:{value}")
            
            if relevant_context:
                cache_content += " [CONTEXT: " + ", ".join(relevant_context) + "]"
        
        return cache_content
    
    def get(self, operation_type: str, content: str, 
           context: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve semantically similar cached LLM response.
        
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
            vectorset_name = self._get_vectorset_name(operation_type)
            cache_key = self._create_cache_key(operation_type, content, context)
            
            # Generate embedding for the query
            query_embedding = self._generate_embedding(cache_key)
            
            # Get similarity threshold for this operation type
            similarity_threshold = self.config.similarity_thresholds.get(
                operation_type, self.config.similarity_threshold
            )
            
            # Search for similar cached entries using VSIM
            embedding_values = [str(val) for val in query_embedding]
            cmd = ["VSIM", vectorset_name, "VALUES", str(self.EMBEDDING_DIM)] + embedding_values + ["WITHSCORES", "COUNT", "1"]
            
            try:
                result = self.redis_client.execute_command(*cmd)
            except redis.ResponseError:
                # Vectorset doesn't exist yet
                self.stats['semantic_misses'] += 1
                return None
            
            if not result or len(result) < 2:
                self.stats['semantic_misses'] += 1
                return None
            
            # Parse result: [element_id, score]
            element_id = result[0].decode('utf-8') if isinstance(result[0], bytes) else result[0]
            similarity_score = float(result[1])
            
            # Check if similarity meets threshold
            if similarity_score < similarity_threshold:
                self.stats['semantic_misses'] += 1
                return None
            
            # Get cached response metadata
            try:
                metadata_json = self.redis_client.execute_command("VGETATTR", vectorset_name, element_id)
                if not metadata_json:
                    self.stats['semantic_misses'] += 1
                    return None
                
                metadata_str = metadata_json.decode('utf-8') if isinstance(metadata_json, bytes) else metadata_json
                cached_data = json.loads(metadata_str)
                
                # Check if cache entry has expired
                if 'expires_at' in cached_data:
                    expires_at = datetime.fromisoformat(cached_data['expires_at'])
                    if datetime.now(timezone.utc) > expires_at:
                        # Entry expired, remove it
                        self.redis_client.execute_command("VREM", vectorset_name, element_id)
                        self.stats['semantic_misses'] += 1
                        return None
                
                # Cache hit!
                self.stats['semantic_hits'] += 1
                
                # Add cache metadata
                response = cached_data['response'].copy()
                response['_cache_hit'] = True
                response['_semantic_similarity'] = similarity_score
                response['_cached_at'] = cached_data.get('cached_at')
                response['_cache_key'] = cache_key
                
                print(f"🎯 Semantic cache HIT for {operation_type} (similarity: {similarity_score:.3f})")
                return response
                
            except Exception as e:
                print(f"⚠️ Error retrieving cached response: {e}")
                self.stats['semantic_misses'] += 1
                return None
                
        except Exception as e:
            self.stats['errors'] += 1
            print(f"⚠️ Semantic cache get error: {e}")
            return None
    
    def set(self, operation_type: str, content: str, response: Dict[str, Any],
           context: Dict[str, Any] = None, ttl: int = None) -> bool:
        """
        Store LLM response in semantic cache.
        
        Args:
            operation_type: Type of LLM operation
            content: Content that was sent to LLM
            response: LLM response to cache
            context: Additional context
            ttl: Time to live in seconds
            
        Returns:
            True if stored successfully
        """
        if not self.config.enabled:
            return False
        
        try:
            vectorset_name = self._get_vectorset_name(operation_type)
            cache_key = self._create_cache_key(operation_type, content, context)
            
            # Generate embedding for the cache key
            embedding = self._generate_embedding(cache_key)
            
            # Determine TTL
            if ttl is None:
                ttl = self.config.ttl_settings.get(operation_type, self.config.default_ttl)
            
            # Calculate expiration time
            expires_at = datetime.now(timezone.utc).timestamp() + ttl
            expires_at_iso = datetime.fromtimestamp(expires_at, timezone.utc).isoformat()
            
            # Create cache entry metadata
            cache_entry = {
                'operation_type': operation_type,
                'original_content': content,
                'cache_key': cache_key,
                'response': response,
                'cached_at': datetime.now(timezone.utc).isoformat(),
                'expires_at': expires_at_iso,
                'ttl': ttl,
                'context': context or {}
            }
            
            # Generate unique ID for this cache entry
            entry_id = str(uuid.uuid4())
            
            # Store in vectorset
            embedding_values = [str(val) for val in embedding]
            metadata_json = json.dumps(cache_entry, ensure_ascii=False)
            
            cmd = ["VADD", vectorset_name, "VALUES", str(self.EMBEDDING_DIM)] + embedding_values + [entry_id, "SETATTR", metadata_json]
            self.redis_client.execute_command(*cmd)
            
            self.stats['stores'] += 1
            print(f"💾 Stored semantic cache entry for {operation_type} (TTL: {ttl}s)")
            
            # Clean up expired entries periodically
            if self.stats['stores'] % 100 == 0:  # Every 100 stores
                self._cleanup_expired_entries(vectorset_name)
            
            return True
            
        except Exception as e:
            self.stats['errors'] += 1
            print(f"⚠️ Semantic cache set error: {e}")
            return False
    
    def _cleanup_expired_entries(self, vectorset_name: str):
        """Clean up expired cache entries from vectorset."""
        try:
            # This is a simplified cleanup - in production, you might want
            # to use VSCAN to iterate through entries more efficiently
            current_time = datetime.now(timezone.utc)
            
            # Get vectorset info to check if it exists
            try:
                self.redis_client.execute_command("VINFO", vectorset_name)
            except redis.ResponseError:
                return  # Vectorset doesn't exist
            
            print(f"🧹 Cleaning up expired entries in {vectorset_name}")
            
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")
    
    def invalidate_operation_type(self, operation_type: str) -> int:
        """
        Invalidate all cache entries for an operation type.
        
        Args:
            operation_type: Operation type to invalidate
            
        Returns:
            Number of entries removed
        """
        try:
            vectorset_name = self._get_vectorset_name(operation_type)
            
            # Delete the entire vectorset for this operation type
            result = self.redis_client.execute_command("DEL", vectorset_name)
            
            if result == 1:
                print(f"🗑️ Invalidated all cache entries for {operation_type}")
                return 1
            else:
                return 0
                
        except Exception as e:
            print(f"⚠️ Cache invalidation error: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get semantic cache statistics."""
        total_requests = self.stats['semantic_hits'] + self.stats['semantic_misses']
        hit_rate = (self.stats['semantic_hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self.stats,
            'hit_rate_percent': round(hit_rate, 2),
            'total_requests': total_requests,
            'cache_type': 'semantic_vectorset'
        }
    
    def clear_stats(self):
        """Reset cache statistics."""
        self.stats = {
            'semantic_hits': 0,
            'semantic_misses': 0,
            'stores': 0,
            'errors': 0,
            'embedding_calls': 0
        }


class SemanticCachedLLMClient:
    """Wrapper for LLM clients that adds semantic caching capabilities."""
    
    def __init__(self, llm_client, semantic_cache: SemanticLLMCache):
        """
        Initialize semantic cached LLM client.
        
        Args:
            llm_client: Original LLM client
            semantic_cache: Semantic cache instance
        """
        self.llm_client = llm_client
        self.semantic_cache = semantic_cache
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       operation_type: str = 'conversation',
                       cache_context: Dict[str, Any] = None,
                       temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None,
                       **kwargs) -> Dict[str, Any]:
        """
        Generate chat completion with semantic caching.
        
        Args:
            messages: Chat messages
            operation_type: Type of operation for cache
            cache_context: Additional context for cache
            temperature: LLM temperature
            max_tokens: Max tokens
            **kwargs: Additional LLM parameters
            
        Returns:
            LLM response with cache metadata
        """
        # Create cache content from messages
        content = json.dumps(messages, sort_keys=True)
        context = {
            'temperature': temperature,
            'max_tokens': max_tokens,
            **(cache_context or {})
        }
        
        # Try semantic cache first
        cached_response = self.semantic_cache.get(operation_type, content, context)
        if cached_response:
            return cached_response
        
        # Call original LLM client
        response = self.llm_client.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # Cache the response semantically
        self.semantic_cache.set(operation_type, content, response, context)
        response['_cache_hit'] = False
        response['_semantic_similarity'] = None
        
        return response
    
    def test_connection(self) -> Dict[str, Any]:
        """Test connection (delegates to original client)."""
        return self.llm_client.test_connection()

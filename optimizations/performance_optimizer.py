#!/usr/bin/env python3
"""
Performance Optimizer - Main integration module for LLM optimizations

This module provides the main interface for applying performance optimizations
to the memory system, including caching, batching, and merged operations.
"""

import os
import redis
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from optimizations.llm_cache import LLMCache, CacheConfig, create_cached_llm_manager
from optimizations.semantic_cache import SemanticLLMCache, SemanticCacheConfig, SemanticCachedLLMClient
from optimizations.optimized_processing import OptimizedMemoryProcessing
from optimizations.optimized_extraction import OptimizedMemoryExtraction


class PerformanceOptimizer:
    """Main performance optimization coordinator."""

    def __init__(self, redis_client: redis.Redis, cache_enabled: bool = True,
                 use_semantic_cache: bool = True):
        """
        Initialize performance optimizer.

        Args:
            redis_client: Redis client for caching
            cache_enabled: Whether to enable LLM response caching
            use_semantic_cache: Whether to use semantic (vectorset) caching instead of hash-based
        """
        self.redis_client = redis_client
        self.cache_enabled = cache_enabled
        self.use_semantic_cache = use_semantic_cache

        # Initialize cache configurations
        self.cache_config = CacheConfig(
            enabled=cache_enabled,
            default_ttl=3600,  # 1 hour default
            ttl_settings={
                'query_optimization': 7200,      # 2 hours - queries don't change often
                'query_optimization_batch': 7200, # 2 hours - batch queries
                'memory_relevance': 1800,        # 30 minutes - context dependent
                'memory_relevance_batch': 1800,  # 30 minutes - batch relevance
                'context_analysis': 3600,        # 1 hour - relatively stable
                'memory_grounding': 1800,        # 30 minutes - context dependent
                'extraction_evaluation': 900,   # 15 minutes - user behavior dependent
                'extraction_comprehensive': 1200, # 20 minutes - comprehensive extraction
                'extraction_batch': 1200,       # 20 minutes - batch extraction
                'conversation': 300,             # 5 minutes - highly dynamic
                'answer_generation': 1800        # 30 minutes - depends on memory context
            }
        )

        self.semantic_cache_config = SemanticCacheConfig(
            enabled=cache_enabled and use_semantic_cache,
            similarity_threshold=0.85,
            default_ttl=3600,
            ttl_settings=self.cache_config.ttl_settings,
            similarity_thresholds={
                'query_optimization': 0.90,     # High similarity for optimization
                'memory_relevance': 0.85,       # Medium-high for relevance
                'context_analysis': 0.88,       # High for context analysis
                'memory_grounding': 0.82,       # Medium for grounding
                'extraction_evaluation': 0.80,  # Lower for extraction
                'conversation': 0.95,           # Very high for conversations
                'answer_generation': 0.87       # High for answer generation
            }
        )

        # Initialize appropriate cache
        if use_semantic_cache and cache_enabled:
            # Create Redis client for vectorset operations (needs decode_responses=False)
            vector_redis_client = redis.Redis(
                host=redis_client.connection_pool.connection_kwargs.get('host', 'localhost'),
                port=redis_client.connection_pool.connection_kwargs.get('port', 6379),
                db=redis_client.connection_pool.connection_kwargs.get('db', 0),
                decode_responses=False  # Required for vector operations
            )
            self.cache = SemanticLLMCache(vector_redis_client, self.semantic_cache_config)
            print("✅ Semantic vectorset caching enabled")
        else:
            self.cache = LLMCache(redis_client, self.cache_config)
            print("✅ Hash-based caching enabled" if cache_enabled else "⚠️ Caching disabled")

        # Performance metrics
        self.metrics = {
            'llm_calls_saved': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'batch_operations': 0,
            'semantic_cache_enabled': use_semantic_cache and cache_enabled,
            'optimization_start_time': datetime.now(timezone.utc).isoformat()
        }
    
    def optimize_llm_manager(self, llm_manager):
        """
        Create an optimized version of the LLM manager with caching.

        Args:
            llm_manager: Original LLM manager

        Returns:
            Optimized LLM manager with caching
        """
        if not self.cache_enabled:
            return llm_manager

        if self.use_semantic_cache:
            # Create semantic cached LLM manager
            return self._create_semantic_cached_llm_manager(llm_manager)
        else:
            return create_cached_llm_manager(llm_manager, self.redis_client, self.cache_config)

    def _create_semantic_cached_llm_manager(self, llm_manager):
        """Create LLM manager with semantic caching."""
        # Wrap both tier clients with semantic caching
        cached_tier1 = SemanticCachedLLMClient(llm_manager.get_tier1_client(), self.cache)
        cached_tier2 = SemanticCachedLLMClient(llm_manager.get_tier2_client(), self.cache)

        # Store reference to cache for the inner class
        cache_instance = self.cache

        # Create wrapper that maintains the same interface
        class SemanticCachedLLMManager:
            def __init__(self):
                self.tier1_config = llm_manager.tier1_config
                self.tier2_config = llm_manager.tier2_config
                self.cache = cache_instance  # Fix: reference the actual cache instance

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

        return SemanticCachedLLMManager()
    
    def create_optimized_processing(self) -> OptimizedMemoryProcessing:
        """Create optimized memory processing instance."""
        return OptimizedMemoryProcessing(cache_enabled=self.cache_enabled)
    
    def create_optimized_extraction(self, memory_core) -> OptimizedMemoryExtraction:
        """Create optimized memory extraction instance."""
        return OptimizedMemoryExtraction(memory_core)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        cache_stats = self.cache.get_stats()

        # Handle different cache stat formats (semantic vs hash-based)
        hits = cache_stats.get('hits', cache_stats.get('semantic_hits', 0))
        misses = cache_stats.get('misses', cache_stats.get('semantic_misses', 0))

        return {
            'cache_enabled': self.cache_enabled,
            'cache_stats': cache_stats,
            'optimizer_metrics': self.metrics,
            'estimated_savings': {
                'llm_calls_avoided': hits,
                'estimated_cost_savings_usd': hits * 0.002,  # Rough estimate
                'estimated_time_savings_seconds': hits * 2.5  # Rough estimate
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def clear_cache(self, pattern: str = None, operation_type: str = None) -> Dict[str, Any]:
        """
        Clear cache entries.

        Args:
            pattern: Optional pattern to match (for hash-based cache)
            operation_type: Optional operation type to clear (for semantic cache)

        Returns:
            Dictionary with clearing results
        """
        if self.use_semantic_cache:
            # Semantic cache clearing
            if operation_type:
                deleted_count = self.cache.invalidate_operation_type(operation_type)
                return {
                    'success': True,
                    'cache_type': 'semantic',
                    'operation_type': operation_type,
                    'deleted_count': deleted_count,
                    'message': f'Cleared semantic cache for operation type: {operation_type}'
                }
            else:
                # Clear all operation types
                operation_types = [
                    'query_optimization', 'memory_relevance', 'context_analysis',
                    'memory_grounding', 'extraction_evaluation', 'conversation',
                    'answer_generation'
                ]
                total_deleted = 0
                for op_type in operation_types:
                    total_deleted += self.cache.invalidate_operation_type(op_type)

                return {
                    'success': True,
                    'cache_type': 'semantic',
                    'operation_types_cleared': operation_types,
                    'deleted_count': total_deleted,
                    'message': f'Cleared all semantic cache entries ({total_deleted} vectorsets)'
                }
        else:
            # Hash-based cache clearing
            if pattern:
                deleted_count = self.cache.invalidate_pattern(pattern)
                return {
                    'success': True,
                    'cache_type': 'hash',
                    'pattern': pattern,
                    'deleted_count': deleted_count,
                    'message': f'Cleared {deleted_count} cache entries matching pattern'
                }
            else:
                pattern = f"{self.cache_config.cache_prefix}:*"
                deleted_count = self.cache.invalidate_pattern(pattern)
                return {
                    'success': True,
                    'cache_type': 'hash',
                    'pattern': 'all',
                    'deleted_count': deleted_count,
                    'message': f'Cleared all {deleted_count} cache entries'
                }
    
    def update_cache_config(self, new_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update cache configuration.
        
        Args:
            new_config: New configuration parameters
            
        Returns:
            Updated configuration and status
        """
        old_enabled = self.cache_config.enabled
        
        # Update configuration
        if 'enabled' in new_config:
            self.cache_config.enabled = bool(new_config['enabled'])
            self.cache_enabled = self.cache_config.enabled
        
        if 'default_ttl' in new_config:
            self.cache_config.default_ttl = int(new_config['default_ttl'])
        
        if 'ttl_settings' in new_config:
            self.cache_config.ttl_settings.update(new_config['ttl_settings'])
        
        # Update cache instance
        self.cache.config = self.cache_config
        
        return {
            'success': True,
            'old_enabled': old_enabled,
            'new_enabled': self.cache_config.enabled,
            'updated_config': {
                'enabled': self.cache_config.enabled,
                'default_ttl': self.cache_config.default_ttl,
                'ttl_settings': self.cache_config.ttl_settings
            },
            'message': 'Cache configuration updated successfully'
        }
    
    def analyze_cache_effectiveness(self) -> Dict[str, Any]:
        """
        Analyze cache effectiveness and provide recommendations.

        Returns:
            Analysis results and recommendations
        """
        stats = self.cache.get_stats()
        total_requests = stats['total_requests']
        hit_rate = stats['hit_rate_percent']

        # Handle different cache stat formats (semantic vs hash-based)
        hits = stats.get('hits', stats.get('semantic_hits', 0))
        misses = stats.get('misses', stats.get('semantic_misses', 0))
        errors = stats.get('errors', 0)

        # Analyze effectiveness
        effectiveness = 'poor'
        if hit_rate >= 70:
            effectiveness = 'excellent'
        elif hit_rate >= 50:
            effectiveness = 'good'
        elif hit_rate >= 30:
            effectiveness = 'fair'

        # Generate recommendations
        recommendations = []

        if hit_rate < 30:
            recommendations.append("Consider increasing TTL values for stable operations")
            recommendations.append("Review cache key generation for better hit rates")

        if errors > 0:
            recommendations.append("Investigate cache errors to improve reliability")

        if total_requests < 10:
            recommendations.append("Insufficient data for meaningful analysis")

        # Calculate potential improvements
        potential_savings = 0
        if total_requests > 0:
            potential_calls_saved = total_requests - hits
            potential_savings = potential_calls_saved * 0.002  # Rough cost estimate

        return {
            'effectiveness': effectiveness,
            'hit_rate_percent': hit_rate,
            'total_requests': total_requests,
            'cache_hits': hits,
            'cache_misses': misses,
            'errors': errors,
            'recommendations': recommendations,
            'potential_improvements': {
                'additional_calls_that_could_be_cached': misses,
                'potential_additional_savings_usd': potential_savings,
                'current_savings_usd': hits * 0.002
            },
            'analysis_timestamp': datetime.now(timezone.utc).isoformat()
        }


def create_performance_optimizer(redis_host: str = None, redis_port: int = None,
                               redis_db: int = None, cache_enabled: bool = True,
                               use_semantic_cache: bool = True) -> PerformanceOptimizer:
    """
    Create a performance optimizer instance.

    Args:
        redis_host: Redis host (defaults to env var or localhost)
        redis_port: Redis port (defaults to env var or 6379)
        redis_db: Redis database (defaults to env var or 0)
        cache_enabled: Whether to enable caching
        use_semantic_cache: Whether to use semantic (vectorset) caching

    Returns:
        PerformanceOptimizer instance
    """
    # Get Redis connection details
    redis_host = redis_host or os.getenv("REDIS_HOST", "localhost")
    redis_port = redis_port or int(os.getenv("REDIS_PORT", "6379"))
    redis_db = redis_db or int(os.getenv("REDIS_DB", "0"))

    # Create Redis client
    redis_client = redis.Redis(
        host=redis_host,
        port=redis_port,
        db=redis_db,
        decode_responses=True  # For hash-based caching
    )

    return PerformanceOptimizer(redis_client, cache_enabled, use_semantic_cache)


# Global optimizer instance
_global_optimizer: Optional[PerformanceOptimizer] = None


def get_performance_optimizer() -> Optional[PerformanceOptimizer]:
    """Get the global performance optimizer instance."""
    return _global_optimizer


def init_performance_optimizer(redis_host: str = None, redis_port: int = None,
                             redis_db: int = None, cache_enabled: bool = True,
                             use_semantic_cache: bool = True) -> PerformanceOptimizer:
    """
    Initialize the global performance optimizer.

    Args:
        redis_host: Redis host
        redis_port: Redis port
        redis_db: Redis database
        cache_enabled: Whether to enable caching
        use_semantic_cache: Whether to use semantic (vectorset) caching

    Returns:
        PerformanceOptimizer instance
    """
    global _global_optimizer
    _global_optimizer = create_performance_optimizer(
        redis_host, redis_port, redis_db, cache_enabled, use_semantic_cache
    )
    return _global_optimizer


def optimize_memory_agent(memory_agent, cache_enabled: bool = True):
    """
    Apply performance optimizations to a memory agent.
    
    Args:
        memory_agent: Memory agent to optimize
        cache_enabled: Whether to enable caching
        
    Returns:
        Optimized memory agent
    """
    optimizer = get_performance_optimizer()
    if not optimizer:
        optimizer = init_performance_optimizer(cache_enabled=cache_enabled)
    
    # Replace processing and extraction modules with optimized versions
    if hasattr(memory_agent, 'processing'):
        memory_agent.processing = optimizer.create_optimized_processing()
    
    if hasattr(memory_agent, 'extraction'):
        memory_agent.extraction = optimizer.create_optimized_extraction(memory_agent.core)
    
    return memory_agent

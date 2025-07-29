#!/usr/bin/env python3
"""
Redis LangCache Client

This module provides a client for the Redis LangCache API for semantic prompt caching.
Uses the REST API endpoints for storing and retrieving cached LLM responses.
"""

import os
import json
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
                    attributes: Dict[str, Any] = None,
                    min_similarity: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Search for cached response using semantic similarity.

        Args:
            messages: Chat messages
            attributes: Additional attributes for scoping
            min_similarity: Minimum similarity threshold (0.0-1.0). If None, uses config default.

        Returns:
            Cached response dict or None if not found
        """
        try:
            prompt = self._create_prompt_key(messages)

            # Get minimum similarity threshold from config or parameter
            if min_similarity is None:
                try:
                    # Import here to avoid circular imports
                    import sys
                    if 'web_app' in sys.modules:
                        web_app = sys.modules['web_app']
                        if hasattr(web_app, 'app_config'):
                            config = web_app.app_config
                            min_similarity = config.get('langcache', {}).get('minimum_similarity', 0.95)
                        else:
                            min_similarity = 0.95
                    else:
                        min_similarity = 0.95
                except Exception:
                    min_similarity = 0.95

            # Prepare request data
            request_data = {
                "prompt": prompt,
                "similarityThreshold": min_similarity
            }

            # Add attributes if provided
            if attributes:
                request_data["attributes"] = attributes

            # DEBUG: Print the search query being sent to LangCache
            print(f"🔍 LANGCACHE SEARCH: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

            # Make API call
            response = requests.post(
                f"{self.base_url}/entries/search",
                headers=self.headers,
                json=request_data,
                timeout=10
            )

            # DEBUG: Print the response details
            try:
                response_json = response.json()
                # print(f"🔍 LANGCACHE SEARCH RESPONSE JSON: {json.dumps(response_json, indent=2)}")
            except:
                print(f"🔍 LANGCACHE SEARCH RESPONSE TEXT: {response.text}")
            
            if response.status_code == 200:
                result = response.json()
                # LangCache returns results in a 'data' array
                if result and 'data' in result and len(result['data']) > 0:
                    # Get the best match (first item in the array)
                    best_match = result['data'][0]
                    similarity_score = best_match.get('similarity', 'N/A')
                    self.stats['hits'] += 1
                    print(f"✅ LANGCACHE HIT: Found cached response (similarity: {similarity_score}, threshold: {min_similarity:.3f})")
                    return {
                        'content': best_match['response'],
                        '_cache_hit': True,
                        '_cached_at': best_match.get('created_at'),
                        '_similarity_score': best_match.get('similarity')
                    }
                else:
                    self.stats['misses'] += 1
                    print(f"❌ LANGCACHE MISS: No cached response found (threshold: {min_similarity:.3f})")
                    return None
            elif response.status_code == 404:
                # No matching cache entry found
                self.stats['misses'] += 1
                print(f"❌ LANGCACHE MISS: No matching cache entry (404)")
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
                   attributes: Dict[str, Any] = None,
                   ttl_minutes: Optional[float] = None) -> bool:
        """
        Store response in cache.

        Args:
            messages: Chat messages
            response: LLM response to cache
            attributes: Additional attributes
            ttl_minutes: Time to live in minutes. If None, uses config default.

        Returns:
            True if stored successfully
        """
        try:
            prompt = self._create_prompt_key(messages)

            # Get TTL from config or parameter
            if ttl_minutes is None:
                try:
                    # Import here to avoid circular imports
                    import sys
                    if 'web_app' in sys.modules:
                        web_app = sys.modules['web_app']
                        if hasattr(web_app, 'app_config'):
                            config = web_app.app_config
                            ttl_minutes = config.get('langcache', {}).get('ttl_minutes', 2)
                        else:
                            ttl_minutes = 2
                    else:
                        ttl_minutes = 2
                except Exception:
                    ttl_minutes = 2

            # Convert minutes to milliseconds for LangCache API
            ttl_millis = int(ttl_minutes * 60 * 1000)

            # Prepare request data
            request_data = {
                "prompt": prompt,
                "response": response,
                "ttlMillis": ttl_millis
            }

            # Add attributes if provided (but LangCache API is very strict about allowed attributes)
            if attributes:
                request_data["attributes"] = attributes

            # DEBUG: Print the store query being sent to LangCache
            print(f"💾 LANGCACHE STORE: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
            print(f"💾 LANGCACHE STORE URL: {self.base_url}/entries")
            print(f"💾 LANGCACHE STORE RESPONSE: {response[:100]}{'...' if len(response) > 100 else ''}")
            print(f"💾 LANGCACHE STORE TTL: {ttl_minutes} minutes ({ttl_millis} ms)")
            print(f"💾 LANGCACHE STORE REQUEST JSON: {json.dumps(request_data, indent=2)}")

            # Make API call
            api_response = requests.post(
                f"{self.base_url}/entries",
                headers=self.headers,
                json=request_data,
                timeout=10
            )

            # DEBUG: Print the response details
            print(f"💾 LANGCACHE STORE RESPONSE STATUS: {api_response.status_code}")
            print(f"💾 LANGCACHE STORE RESPONSE HEADERS: {dict(api_response.headers)}")
            try:
                store_response_json = api_response.json()
                print(f"💾 LANGCACHE STORE RESPONSE JSON: {json.dumps(store_response_json, indent=2)}")
            except:
                print(f"💾 LANGCACHE STORE RESPONSE TEXT: {api_response.text}")

            if api_response.status_code in [200, 201]:
                self.stats['stores'] += 1
                print(f"✅ LANGCACHE STORED: Successfully cached response")
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
            print(f"🚫 LANGCACHE DISABLED: Caching disabled for operation type '{operation_type}'")
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

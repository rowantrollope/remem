# Redis LangCache Integration

This document explains how to set up and use Redis LangCache for prompt caching in the memory system.

## Overview

Redis LangCache provides semantic prompt caching to reduce LLM API calls and improve response times. The system automatically caches responses for:

- **Memory Extraction**: Similar user inputs often generate similar memory extractions
- **Query Optimization**: Query preprocessing is highly repetitive 
- **Context Analysis**: Context analysis prompts have common patterns
- **Memory Grounding**: Memory grounding operations with similar contexts

## Setup

### 1. Environment Variables

Add the following environment variables to your `.env` file:

```bash
# Redis LangCache Configuration (optional - for prompt caching)
LANGCACHE_HOST=https://your-langcache-host.com
LANGCACHE_API_KEY=your_langcache_api_key_here
LANGCACHE_CACHE_ID=your_cache_id_here
```

### 2. How It Works

The system automatically:

1. **Checks cache first**: Before making LLM calls, searches for semantically similar cached responses
2. **Stores responses**: After successful LLM calls, stores the response for future use
3. **Uses operation types**: Categorizes cache entries by operation type for better organization
4. **Provides statistics**: Tracks cache hits, misses, and effectiveness

### 3. Operation Types

The following operation types are used for cache categorization:

- `memory_extraction`: Memory extraction from conversations
- `query_optimization`: User query validation and preprocessing  
- `embedding_optimization`: Query optimization for vector search
- `context_analysis`: Memory context analysis
- `memory_grounding`: Memory grounding with contextual information

### 4. Fallback Behavior

If LangCache is not configured or unavailable:
- The system automatically falls back to direct LLM calls
- No functionality is lost
- Performance may be slower due to lack of caching

### 5. Cache Statistics

Cache statistics are available through the LangCache client:

```python
# Get cache statistics
stats = langcache_client.get_stats()
print(f"Cache hit rate: {stats['hit_rate_percent']}%")
print(f"Total requests: {stats['total_requests']}")
```

## Benefits

- **Reduced API Costs**: Fewer LLM API calls mean lower costs
- **Faster Response Times**: Cached responses are returned immediately
- **Improved Reliability**: Less dependency on external LLM services
- **Semantic Matching**: Finds similar prompts even with different wording

## Configuration Notes

- LangCache is **optional** - the system works without it
- All three environment variables must be set for LangCache to be enabled
- If any environment variable is missing, the system will log a message and continue without caching
- Cache effectiveness depends on the similarity of prompts over time

## Monitoring

Monitor cache effectiveness by checking:
- Cache hit rates in application logs
- LLM API usage reduction
- Response time improvements
- Cache health status

The system logs cache initialization status and any errors during startup.

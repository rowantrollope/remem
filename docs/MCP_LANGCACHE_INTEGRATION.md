# MCP Server LangCache Integration

This document explains how LangCache caching is integrated into the Remem MCP Server to improve performance and reduce API costs.

## Overview

The MCP server now includes optional LangCache integration that caches responses from expensive operations like memory searches and question answering. This provides:

- **Faster responses** for repeated queries
- **Reduced API costs** by avoiding duplicate LLM calls
- **Better user experience** with near-instant responses for cached queries
- **Configurable caching** per operation type

## Architecture

```
MCP Client -> MCP Server -> LangCache Check -> [Cache Hit: Return] 
                        |                   |
                        -> [Cache Miss] -> Memory Agent -> LLM -> Store in Cache -> Return
```

## Setup

### 1. Environment Variables

Add these variables to your `.env` file:

```bash
# Required for LangCache
LANGCACHE_HOST=https://your-langcache-instance.com
LANGCACHE_API_KEY=your-langcache-api-key-here
LANGCACHE_CACHE_ID=your-cache-id-here

# Optional configuration
LANGCACHE_TTL_MINUTES=2
LANGCACHE_MIN_SIMILARITY=0.95
```

### 2. Cached Operations

The following MCP tools are automatically cached:

- **search_memories**: Caches memory search results
- **answer_question**: Caches question-answer pairs

Other operations like `store_memory` are not cached as they modify state.

### 3. Cache Management Tools

Two new MCP tools are available for cache management:

- **get_cache_stats**: View cache hit rates and health status
- **clear_cache_stats**: Reset cache statistics counters

## Usage Examples

### Basic Usage with Claude Desktop

Once configured, caching works transparently:

```
User: "Search for memories about coffee"
[First call: ~2 seconds, calls LLM]

User: "Search for memories about coffee" 
[Second call: ~0.1 seconds, from cache]
```

### Cache Statistics

```
User: "Get cache statistics"

Response:
{
  "cache_stats": {
    "hits": 5,
    "misses": 3,
    "stores": 3,
    "errors": 0,
    "hit_rate_percent": 62.5,
    "total_requests": 8
  },
  "health": {
    "healthy": true
  },
  "cache_enabled": true
}
```

## Configuration

### Cache Behavior

- **TTL**: Cached responses expire after 2 minutes by default
- **Similarity Threshold**: 0.95 similarity required for cache hits
- **Operation Types**: Only read operations are cached
- **Fallback**: If caching fails, operations continue normally

### Disabling Caching

Caching can be disabled by:

1. **Environment**: Remove LangCache environment variables
2. **Per Operation**: Modify `is_cache_enabled_for_operation()` in `langcache_client.py`
3. **Global**: Set `LANGCACHE_ENABLED=false`

## Testing

Run the caching test script:

```bash
python test_mcp_caching.py
```

This will:
- Test cache hits and misses
- Measure performance improvements
- Verify cache statistics
- Test cache management functions

## Troubleshooting

### Cache Not Working

1. **Check Environment Variables**: Ensure all required variables are set
2. **Verify LangCache Service**: Test connection with `get_cache_stats`
3. **Check Logs**: Look for cache-related messages in stderr
4. **Test Manually**: Use the test script to isolate issues

### Performance Issues

1. **Monitor Hit Rate**: Use `get_cache_stats` to check effectiveness
2. **Adjust Similarity**: Lower threshold for more cache hits
3. **Check TTL**: Ensure TTL is appropriate for your use case

### Common Error Messages

- `"LangCache not initialized"`: Environment variables not set
- `"Cache error"`: Network or API issues with LangCache service
- `"Caching disabled for operation"`: Operation type excluded from caching

## Implementation Details

### Cache Key Generation

Cache keys are generated from:
- Function name
- All function parameters
- MD5 hash for consistency

### Decorator Pattern

The `@cached_tool` decorator wraps MCP tool functions:

```python
@mcp.tool()
@cached_tool(operation_type="memory_search")
async def search_memories(...):
    # Function implementation
```

### Error Handling

- Cache failures don't break functionality
- Operations fall back to normal execution
- Errors are logged but don't propagate

## Best Practices

1. **Monitor Cache Hit Rates**: Aim for >50% hit rate for cached operations
2. **Adjust TTL**: Balance freshness vs. performance
3. **Use Appropriate Similarity**: Too high = few hits, too low = stale results
4. **Test Regularly**: Use the test script to verify caching behavior

## Related Files

- `mcp_server.py`: Main server with caching integration
- `clients/langcache_client.py`: LangCache client implementation
- `test_mcp_caching.py`: Caching test script
- `.env.langcache.example`: Example configuration

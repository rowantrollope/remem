# LangCache Configuration Guide

This document explains how to configure and control LangCache caching for individual LLM operations.

## 🎛️ **Configurable Cache Types**

The following LLM calls can be individually controlled for caching:

### 1. **`memory_extraction`** (Tier 1 LLM)
- **What it does**: Extracts memories from user conversations
- **When it's called**: During conversation processing when new memories need to be extracted
- **Why cache it**: Similar user inputs often generate similar memory extractions
- **High cache hit potential**: ⭐⭐⭐⭐⭐

### 2. **`query_optimization`** (Tier 2 LLM)  
- **What it does**: Validates and preprocesses user queries
- **When it's called**: Before searching memories to optimize the query
- **Why cache it**: Query preprocessing patterns are highly repetitive
- **High cache hit potential**: ⭐⭐⭐⭐⭐

### 3. **`embedding_optimization`** (Tier 2 LLM)
- **What it does**: Optimizes queries for vector similarity search
- **When it's called**: When optimizing search queries for better embedding matches
- **Why cache it**: Similar optimization patterns occur frequently
- **High cache hit potential**: ⭐⭐⭐⭐

### 4. **`context_analysis`** (Tier 1 LLM)
- **What it does**: Analyzes memory context and dependencies
- **When it's called**: During memory grounding and context analysis
- **Why cache it**: Context analysis prompts have common patterns
- **High cache hit potential**: ⭐⭐⭐⭐

### 5. **`memory_grounding`** (Tier 1 LLM)
- **What it does**: Grounds memories with contextual information
- **When it's called**: When storing memories with temporal/spatial context
- **Why cache it**: Similar grounding operations occur for similar contexts
- **High cache hit potential**: ⭐⭐⭐

## ⚙️ **Configuration Methods**

### Method 1: API Configuration (Recommended)

Use the `/api/config` endpoint to update cache settings:

```bash
curl -X POST http://localhost:5001/api/config \
  -H "Content-Type: application/json" \
  -d '{
    "langcache": {
      "enabled": true,
      "cache_types": {
        "memory_extraction": true,
        "query_optimization": true,
        "embedding_optimization": false,
        "context_analysis": true,
        "memory_grounding": false
      }
    }
  }'
```

### Method 2: Default Configuration

The default configuration in `web_app.py`:

```python
"langcache": {
    "enabled": True,  # Master switch for all caching
    "cache_types": {
        "memory_extraction": True,
        "query_optimization": True,
        "embedding_optimization": True,
        "context_analysis": True,
        "memory_grounding": True
    }
}
```

## 🔧 **Configuration Options**

### Master Switch
- **`langcache.enabled`**: Global on/off switch for all caching
- When `false`, all cache types are disabled regardless of individual settings

### Individual Cache Types
- **`langcache.cache_types.memory_extraction`**: Enable/disable memory extraction caching
- **`langcache.cache_types.query_optimization`**: Enable/disable query optimization caching
- **`langcache.cache_types.embedding_optimization`**: Enable/disable embedding optimization caching
- **`langcache.cache_types.context_analysis`**: Enable/disable context analysis caching
- **`langcache.cache_types.memory_grounding`**: Enable/disable memory grounding caching

## 📊 **Monitoring Cache Usage**

### Check Current Configuration
```bash
curl http://localhost:5001/api/config
```

### Response Metadata
When caching is disabled for an operation, the LLM response includes:
```json
{
  "content": "LLM response...",
  "_cache_hit": false,
  "_cache_disabled": true
}
```

When caching is enabled:
```json
{
  "content": "LLM response...",
  "_cache_hit": true,
  "_cached_at": "2024-01-01T12:00:00Z",
  "_similarity_score": 0.95
}
```

## 🎯 **Recommended Settings**

### For Development
```json
{
  "langcache": {
    "enabled": true,
    "cache_types": {
      "memory_extraction": false,     // Disable for testing memory extraction
      "query_optimization": true,     // Keep enabled (high hit rate)
      "embedding_optimization": true, // Keep enabled (high hit rate)
      "context_analysis": false,      // Disable for testing context analysis
      "memory_grounding": false       // Disable for testing grounding
    }
  }
}
```

### For Production
```json
{
  "langcache": {
    "enabled": true,
    "cache_types": {
      "memory_extraction": true,      // Enable all for maximum efficiency
      "query_optimization": true,
      "embedding_optimization": true,
      "context_analysis": true,
      "memory_grounding": true
    }
  }
}
```

### For Cost Optimization
```json
{
  "langcache": {
    "enabled": true,
    "cache_types": {
      "memory_extraction": true,      // High-cost Tier 1 operations
      "query_optimization": true,     // High hit rate
      "embedding_optimization": true, // High hit rate
      "context_analysis": false,      // Lower hit rate
      "memory_grounding": false       // Lower hit rate
    }
  }
}
```

## 🔍 **Testing Configuration**

Run the configuration test script:
```bash
python3 test_langcache_config.py
```

This will test:
- Configuration loading and modification
- Individual cache type controls
- Master switch functionality
- CachedLLMClient behavior

## 📝 **Environment Variables**

Required for LangCache to work:
```bash
LANGCACHE_HOST=https://your-langcache-host.com
LANGCACHE_API_KEY=your_api_key_here
LANGCACHE_CACHE_ID=your_cache_id_here
```

If any of these are missing, the system will show "LangCache not configured" and work without caching.

## 🚨 **Important Notes**

1. **Configuration is runtime**: Changes via API take effect immediately, no restart required
2. **Graceful fallback**: If LangCache is unavailable, operations continue without caching
3. **Operation-specific**: Each cache type can be controlled independently
4. **Master override**: The master `enabled` switch overrides all individual settings
5. **No performance impact**: Disabling caching has no negative performance impact

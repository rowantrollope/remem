# Performance Optimization Guide

This guide provides comprehensive strategies for optimizing the Memory Agent API performance by reducing LLM calls and implementing intelligent caching.

## Overview

The Memory Agent API has been analyzed for performance bottlenecks, and several optimization strategies have been implemented to reduce latency and API costs.

## Current Performance Issues

### High-Frequency Endpoints with Multiple LLM Calls

1. **Memory-enabled chat sessions** (`/api/agent/session/<session_id>` POST)
   - **3-4 LLM calls per message**:
     - Query optimization (Tier 2 LLM)
     - Memory relevance filtering (OpenAI direct)
     - Main conversation response (Tier 1 LLM)
     - Memory extraction evaluation (Tier 2 LLM)

2. **Memory storage** (`/api/memory` POST)
   - **2-3 LLM calls per memory**:
     - Context dependency analysis (Tier 1 LLM)
     - Memory grounding (Tier 1 LLM)

3. **K-line operations** (`/api/klines/recall`, `/api/klines/ask`)
   - **2-3 LLM calls per request**:
     - Query optimization (processing)
     - Memory relevance filtering (OpenAI direct)
     - Answer generation (Tier 1 LLM)

## Optimization Strategies Implemented

### 1. Semantic LLM Response Caching

**Implementation**: Redis VectorSet-based semantic caching
**Location**: `optimizations/semantic_cache.py`

**Features**:
- **Semantic Similarity**: Uses vector embeddings to find semantically similar queries
- **Intelligent Matching**: Caches responses for similar queries, not just exact matches
- **Operation-Specific Thresholds**: Different similarity thresholds per operation type
- **Vector Storage**: Leverages existing Redis VectorSet infrastructure
- **Embedding-Based Keys**: Uses OpenAI embeddings for semantic comparison

**Advantages over Hash-Based Caching**:
- Catches semantically similar queries: "Italian restaurants I've been to" ≈ "restaurants in Italy I visited"
- Much higher cache hit rates for natural language variations
- Contextually aware caching based on meaning, not just text

**Semantic Similarity Thresholds**:
```python
{
    'query_optimization': 0.90,     # High similarity for optimization
    'memory_relevance': 0.85,       # Medium-high for relevance
    'context_analysis': 0.88,       # High for context analysis
    'memory_grounding': 0.82,       # Medium for grounding (more flexible)
    'extraction_evaluation': 0.80,  # Lower for extraction (more variation)
    'conversation': 0.95,           # Very high for conversations (avoid wrong responses)
    'answer_generation': 0.87       # High for answer generation
}
```

**Cache TTL Settings**:
```python
{
    'query_optimization': 7200,      # 2 hours - queries don't change often
    'memory_relevance': 1800,        # 30 minutes - context dependent
    'context_analysis': 3600,        # 1 hour - relatively stable
    'memory_grounding': 1800,        # 30 minutes - context dependent
    'extraction_evaluation': 900,   # 15 minutes - user behavior dependent
    'conversation': 300,             # 5 minutes - highly dynamic
    'answer_generation': 1800        # 30 minutes - depends on memory context
}
```

**Expected Impact**: 60-85% reduction in LLM calls (higher than hash-based due to semantic matching)

### 2. Merged LLM Operations

**Implementation**: Combined multiple LLM calls into single prompts
**Location**: `optimizations/optimized_processing.py`, `optimizations/optimized_extraction.py`

**Optimizations**:
- **Memory extraction**: Combined evaluation + extraction in single call
- **Batch relevance filtering**: Process multiple memories in one call
- **Query optimization batching**: Handle multiple queries together

**Expected Impact**: 50-60% reduction in LLM calls for batch operations

### 3. Intelligent Query Optimization

**Implementation**: Reduced LLM dependency for query optimization
**Features**:
- Heuristic-based optimization for simple queries
- LLM optimization only for complex queries
- Cached optimization results

**Expected Impact**: 30-40% reduction in query optimization calls

## Performance Monitoring

### API Endpoints

1. **Get Performance Metrics**
   ```
   GET /api/performance/metrics
   ```
   Returns cache statistics, hit rates, and estimated savings

2. **Clear Cache**
   ```
   POST /api/performance/cache/clear
   Body: {
     "operation_type": "query_optimization"  # For semantic cache
     // OR
     "pattern": "llm_cache:query_optimization:*"  # For hash-based cache
   }
   ```

3. **Analyze Cache Effectiveness**
   ```
   GET /api/performance/cache/analyze
   ```
   Returns effectiveness analysis and recommendations

### Configuration

Performance optimizations can be configured via `/api/config`:

```json
{
  "performance": {
    "cache_enabled": true,
    "use_semantic_cache": true,
    "optimization_enabled": true,
    "batch_processing_enabled": true,
    "cache_default_ttl": 3600,
    "semantic_similarity_threshold": 0.85,
    "cache_ttl_settings": {
      "query_optimization": 7200,
      "memory_relevance": 1800,
      "context_analysis": 3600
    },
    "semantic_similarity_thresholds": {
      "query_optimization": 0.90,
      "memory_relevance": 0.85,
      "context_analysis": 0.88,
      "conversation": 0.95
    }
  }
}
```

## Implementation Priority

### High Impact, Low Effort (Immediate)

1. **Enable Semantic LLM Response Caching**
   - Set `performance.cache_enabled: true` and `performance.use_semantic_cache: true` in config
   - Expected: 60-85% call reduction for semantically similar operations
   - Implementation: Already integrated

2. **Use Batch Memory Filtering**
   - Automatically enabled with optimizations
   - Expected: 50% reduction in relevance filtering calls
   - Implementation: Already integrated

### Medium Impact, Medium Effort (Next Phase)

3. **Optimize Memory Extraction**
   - Combined evaluation and extraction
   - Expected: 50% reduction in extraction calls
   - Implementation: Available in optimized modules

4. **Smart Query Preprocessing**
   - Reduce LLM dependency for simple queries
   - Expected: 30% reduction in query optimization calls
   - Implementation: Available in optimized processing

### High Impact, High Effort (Future)

5. **Predictive Caching**
   - Pre-cache likely queries based on user patterns
   - Expected: Additional 20-30% improvement
   - Implementation: Future enhancement

6. **Memory Clustering**
   - Group similar memories to reduce search space
   - Expected: 25% improvement in search performance
   - Implementation: Future enhancement

## Performance Estimates

### Before Optimization
- **Chat session**: 3-4 LLM calls per message (~8-12 seconds)
- **Memory storage**: 2-3 LLM calls per memory (~4-8 seconds)
- **K-line operations**: 2-3 LLM calls per request (~6-10 seconds)

### After Optimization (Conservative Estimates)
- **Chat session**: 1-2 LLM calls per message (~3-6 seconds) - **50-60% improvement**
- **Memory storage**: 1-2 LLM calls per memory (~2-4 seconds) - **50-60% improvement**
- **K-line operations**: 1-2 LLM calls per request (~3-5 seconds) - **50-60% improvement**

### Cost Savings
- **API Cost Reduction**: 40-70% for cached operations
- **Latency Reduction**: 50-60% average response time improvement
- **Throughput Increase**: 2-3x more requests per second capacity

## Monitoring and Tuning

### Key Metrics to Monitor

1. **Cache Hit Rate**: Target >60% for stable operations
2. **Average Response Time**: Monitor per endpoint
3. **LLM Call Frequency**: Track calls per operation type
4. **Error Rates**: Ensure optimizations don't increase errors

### Tuning Recommendations

1. **Adjust TTL Values**: Based on cache hit rates and data freshness needs
2. **Monitor Memory Usage**: Redis cache memory consumption
3. **A/B Testing**: Compare performance with/without optimizations
4. **User Experience**: Ensure response quality isn't degraded

## Troubleshooting

### Common Issues

1. **Low Cache Hit Rate (<30%)**
   - Check TTL settings (may be too low)
   - Verify cache key generation consistency
   - Review query patterns for variability

2. **High Memory Usage**
   - Reduce TTL values for less critical operations
   - Implement cache size limits
   - Monitor Redis memory usage

3. **Degraded Response Quality**
   - Review merged prompt effectiveness
   - Adjust batch sizes for optimal quality
   - Monitor user feedback

### Performance Testing

Use the provided test scripts to validate optimizations:

```bash
# Test cache effectiveness
python tests/test_cache_performance.py

# Test batch operations
python tests/test_batch_optimization.py

# Load testing
python tests/test_load_performance.py
```

## Future Enhancements

1. **Machine Learning-based Caching**: Predict optimal cache strategies
2. **Dynamic TTL Adjustment**: Automatically adjust based on usage patterns
3. **Distributed Caching**: Scale across multiple Redis instances
4. **Advanced Batching**: Intelligent grouping of operations
5. **Response Streaming**: Reduce perceived latency for long operations

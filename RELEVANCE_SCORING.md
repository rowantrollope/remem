# Memory Relevance Scoring System

## Overview

The enhanced memory system now includes sophisticated relevance scoring that combines vector similarity with temporal recency and usage frequency to provide more contextually appropriate memory retrieval for agentic GenAI applications.

## Key Features

### 1. Enhanced Memory Storage
Each memory now includes three additional fields:
- `created_at`: ISO timestamp of when the memory was first created
- `last_accessed_at`: ISO timestamp of when the memory was last retrieved
- `access_count`: Integer counter tracking retrieval frequency

### 2. Configurable Relevance Scoring
The relevance scoring algorithm combines three components with tunable weights:
- **Vector Similarity** (default weight: 0.7): Traditional semantic similarity
- **Temporal Recency** (default weight: 0.2): How recent the memory is
- **Usage Frequency** (default weight: 0.1): How often the memory is accessed

### 3. Automatic Access Tracking
Every memory retrieval automatically:
- Increments the `access_count`
- Updates `last_accessed_at` timestamp
- Recalculates relevance scores for sorting

## Relevance Scoring Algorithm

### Formula
```
relevance_score = (vector_score × vector_weight) + 
                 (temporal_component × temporal_weight) + 
                 (usage_component × usage_weight)
```

### Temporal Component
Combines creation recency and access recency with exponential decay:
```
creation_recency = e^(-days_since_creation / recency_decay_days)
access_recency = e^(-days_since_access / access_decay_days)
temporal_factor = (creation_recency × 0.3) + (access_recency × 0.7)
```

### Usage Component
Uses logarithmic scaling to prevent excessive boost:
```
usage_factor = log(1 + access_count) × usage_boost_factor
```

## Configuration Parameters

### Core Weights
- `vector_weight` (default: 0.7): Weight for vector similarity score
- `temporal_weight` (default: 0.2): Weight for temporal recency component  
- `usage_weight` (default: 0.1): Weight for usage frequency component

### Temporal Parameters
- `recency_decay_days` (default: 30.0): Days for creation recency to decay to ~37%
- `access_decay_days` (default: 7.0): Days for last access recency to decay to ~37%
- `max_temporal_boost` (default: 0.3): Maximum boost from temporal factors

### Usage Parameters
- `usage_boost_factor` (default: 0.1): Multiplier for access count boost
- `max_usage_boost` (default: 0.2): Maximum boost from usage factors

## API Changes

### Memory Objects
All memory objects returned by search operations now include:
```json
{
  "id": "memory-uuid",
  "text": "memory content",
  "score": 0.85,
  "relevance_score": 0.92,
  "created_at": "2024-06-19T10:30:00Z",
  "last_accessed_at": "2024-06-19T15:45:00Z",
  "access_count": 5
}
```

### K-lines API
The `/api/klines/recall` endpoint automatically uses relevance scoring:
- Memories are sorted by `relevance_score` (highest first)
- Access tracking is updated for all retrieved memories
- Response includes both `score` and `relevance_score` fields

### Configuration Management
```python
# Get current configuration
config = memory_core.get_relevance_config()

# Update configuration
updated_config = memory_core.update_relevance_config(
    vector_weight=0.6,
    temporal_weight=0.3,
    usage_weight=0.1
)
```

## Use Cases

### Travel Agent Scenario
```python
# User preferences get boosted by usage frequency
memory_core.store_memory("I prefer window seats on flights")
memory_core.store_memory("I am vegetarian")

# Frequent searches for dining boost food preferences
for _ in range(5):
    memory_core.search_memories("restaurant recommendations")

# Later searches will prioritize frequently accessed preferences
results = memory_core.search_memories("book a flight and dinner")
# "vegetarian" preference will rank higher due to usage frequency
```

### Temporal Relevance
```python
# Recent memories get temporal boost
memory_core.store_memory("I'm currently in Tokyo")  # Recent
time.sleep(days=30)
memory_core.store_memory("I was in Paris last month")  # Older

# Search will favor recent location
results = memory_core.search_memories("current location")
# Tokyo memory ranks higher due to recency
```

## Backward Compatibility

### Existing Memories
- Memories without new fields are handled gracefully
- Missing `created_at` falls back to `timestamp` field
- Missing `last_accessed_at` defaults to `created_at`
- Missing `access_count` defaults to 0

### Migration
```python
# Automatic migration during retrieval
result = memory_core.migrate_legacy_memories()
# Returns migration status and count
```

## Performance Considerations

### Access Tracking Overhead
- Each search operation updates Redis metadata
- Uses Redis `VSETATTR` command for atomic updates
- Minimal performance impact for typical usage

### Scoring Computation
- Relevance calculation is O(1) per memory
- Exponential and logarithmic functions are computationally efficient
- Sorting by relevance score is O(n log n) where n is result count

## Best Practices

### Configuration Tuning
1. **Vector-Heavy Applications**: Increase `vector_weight` to 0.8-0.9
2. **Temporal-Sensitive Applications**: Increase `temporal_weight` to 0.3-0.4
3. **Usage-Pattern Learning**: Increase `usage_weight` to 0.2-0.3

### Decay Parameters
1. **Short-Term Memory**: Set `access_decay_days` to 1-3 days
2. **Long-Term Memory**: Set `recency_decay_days` to 60-90 days
3. **Balanced Approach**: Use defaults (7 and 30 days respectively)

### Usage Monitoring
```python
# Monitor memory access patterns
memories = memory_core.search_memories("query")
for memory in memories:
    print(f"Access count: {memory['access_count']}")
    print(f"Last accessed: {memory['last_accessed_at']}")
```

## Testing

Run the comprehensive test suite:
```bash
python3 test_relevance_scoring.py
```

This tests:
- Configuration system
- Memory storage with new fields
- Relevance scoring algorithm
- API integration
- Backward compatibility

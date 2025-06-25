# Minimum Similarity Threshold Feature

## Overview

The memory system now supports a configurable minimum similarity threshold for all memory retrieval operations. This feature allows you to filter out memories that don't meet a specified similarity score, ensuring only highly relevant memories are returned.

## Default Value

- **Default threshold**: `0.7` (70% similarity)
- This provides a good balance between relevance and recall

## Supported Methods

The `min_similarity` parameter has been added to the following methods:

### Core Methods
- `MemoryCore.search_memories(query, top_k=10, filterBy=None, memory_type=None, min_similarity=0.9)`
- `MemoryAgent.search_memories(query, top_k=10, filterBy=None, memory_type=None, min_similarity=0.9)`
- `MemoryAgent.answer_question(question, top_k=5, filterBy=None, min_similarity=0.9)`
- `MemoryAgent.recall_memories(query, top_k=10, min_similarity=0.9)`

### API Endpoints
- `POST /api/klines/recall` - Add `min_similarity` to request body
- `POST /api/klines/ask` - Add `min_similarity` to request body  
- `POST /api/agent/session/<session_id>` - Add `min_similarity` to request body

### Internal Functions
- `_handle_memory_enabled_message()` - Updated to accept and use `min_similarity`

## Usage Examples

### Python API
```python
# Search with default threshold (0.7)
memories = memory_agent.search_memories("Italian food")

# Search with custom threshold (0.9)
memories = memory_agent.search_memories("Italian food", min_similarity=0.9)

# Answer question with strict threshold (0.95)
answer = memory_agent.answer_question("What do I like to eat?", min_similarity=0.95)
```

### REST API
```json
// K-line recall with custom threshold
POST /api/klines/recall
{
    "query": "Italian food preferences",
    "top_k": 5,
    "min_similarity": 0.8
}

// Agent session with custom threshold
POST /api/agent/session/abc123
{
    "message": "What do I like to eat?",
    "top_k": 10,
    "min_similarity": 0.85
}
```

## How It Works

1. **Vector Search**: The system first performs vector similarity search using Redis VSIM
2. **Score Filtering**: Results are filtered to only include memories with `score >= min_similarity`
3. **Logging**: When filtering occurs, the system logs: `"Similarity filtering: X → Y memories (min_similarity: Z)"`
4. **LLM Filtering**: After similarity filtering, LLM-based relevance filtering is still applied

## Benefits

- **Quality Control**: Ensures only highly relevant memories are considered
- **Performance**: Reduces the number of memories sent to LLM filtering
- **Customization**: Different use cases can use different thresholds
- **Consistency**: Applied uniformly across all memory retrieval operations

## Threshold Guidelines

- **0.95-1.0**: Very strict - only nearly identical content
- **0.9-0.95**: Strict - highly relevant content
- **0.8-0.9**: Moderate - somewhat relevant content
- **0.7-0.8**: Balanced - broadly related content (default)
- **0.5-0.7**: Very loose - tangentially related content
- **0.0-0.5**: No filtering - all results (not recommended)

## Backward Compatibility

- All existing code continues to work unchanged
- Default value of 0.7 provides balanced relevance and recall
- Optional parameter - can be omitted in API calls

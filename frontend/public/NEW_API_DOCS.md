# Memory Agent API Documentation

## Overview

The Memory Agent API provides a clean, RESTful interface for developers building agents that need sophisticated memory capabilities. The API is designed with two primary use cases:

1. **Developer Memory APIs** - Core memory operations for agent developers
2. **Chat Application API** - Conversational interface for demo/UI applications

## Base URL

```
http://localhost:5001
```

## Authentication

Currently no authentication is required. The API is designed for local development and testing.

---

## Developer Memory APIs

These endpoints provide core memory operations for developers building agents.

### 1. Store Memory

Store a new memory with optional contextual grounding.

**POST** `/api/memory/{vectorset_name}`

**Parameters:**
- `vectorset_name` (string, required): Name of the vectorset (e.g., "memories")

**Request Body:**
```json
{
  "text": "I went to Mario's Italian Restaurant and had amazing pasta carbonara",
  "apply_grounding": true
}
```

**Body Parameters:**
- `text` (string, required): Memory text to store
- `apply_grounding` (boolean, optional): Whether to apply contextual grounding (default: true)

**Response:**
```json
{
  "success": true,
  "memory_id": "memory:550e8400-e29b-41d4-a716-446655440000",
  "message": "Memory stored successfully"
}
```

### 2. Search Memories

Search for memories using vector similarity.

**POST** `/api/memory/{vectorset_name}/search`

**Parameters:**
- `vectorset_name` (string, required): Name of the vectorset (e.g., "memories")

**Request Body:**
```json
{
  "query": "Italian restaurant food",
  "top_k": 5,
  "filter": "optional_filter_expression"
}
```

**Body Parameters:**
- `query` (string, required): Search query text
- `top_k` (integer, optional): Number of results to return (default: 5)
- `filter` (string, optional): Filter expression for Redis VSIM command

**Response:**
```json
{
  "success": true,
  "query": "Italian restaurant food",
  "memories": [
    {
      "id": "memory:550e8400-e29b-41d4-a716-446655440000",
      "text": "I went to Mario's Italian Restaurant and had amazing pasta carbonara",
      "score": 0.952,
      "formatted_time": "2024-01-15 18:30:00",
      "tags": ["restaurant", "food", "Mario's"]
    }
  ],
  "count": 1
}
```

### 3. Answer Question (Advanced)

Answer questions using sophisticated confidence analysis and structured responses.

**⭐ This endpoint calls `memory_agent.answer_question()` directly for the highest quality responses.**

**POST** `/api/memory/{vectorset_name}/ask`

**Parameters:**
- `vectorset_name` (string, required): Name of the vectorset (e.g., "memories")

**Request Body:**
```json
{
  "question": "What did I eat at Mario's restaurant?",
  "top_k": 5,
  "filter": "optional_filter_expression"
}
```

**Body Parameters:**
- `question` (string, required): Question to answer
- `top_k` (integer, optional): Number of memories to retrieve for context (default: 5)
- `filter` (string, optional): Filter expression for Redis VSIM command

**Response:**
```json
{
  "success": true,
  "question": "What did I eat at Mario's restaurant?",
  "type": "answer",
  "answer": "You had amazing pasta carbonara at Mario's Italian Restaurant.",
  "confidence": "high",
  "reasoning": "Memory directly mentions pasta carbonara at Mario's with specific details",
  "supporting_memories": [
    {
      "id": "memory:550e8400-e29b-41d4-a716-446655440000",
      "text": "I went to Mario's Italian Restaurant and had amazing pasta carbonara",
      "relevance_score": 95.2,
      "timestamp": "2024-01-15 18:30:00",
      "tags": ["restaurant", "food", "Mario's"]
    }
  ]
}
```

**Confidence Levels:**
- `high`: Memories directly and clearly answer the question with specific, relevant information
- `medium`: Memories provide good information but may be incomplete or somewhat indirect  
- `low`: Memories are tangentially related or don't provide enough information to answer confidently

### 4. Get Memory Info

Get statistics about stored memories and system information.

**GET** `/api/memory/{vectorset_name}`

**Parameters:**
- `vectorset_name` (string, required): Name of the vectorset (e.g., "memories")

**Response:**
```json
{
  "success": true,
  "memory_count": 42,
  "vector_dimension": 1536,
  "vectorset_name": "memories",
  "embedding_model": "text-embedding-ada-002",
  "redis_host": "localhost",
  "redis_port": 6379,
  "timestamp": "2024-01-15T18:30:00Z"
}
```

### 5. Context Management

Set and get current context for memory grounding.

#### Set Context

**POST** `/api/memory/{vectorset_name}/context`

**Parameters:**
- `vectorset_name` (string, required): Name of the vectorset (e.g., "memories")

**Request Body:**
```json
{
  "location": "Jakarta, Indonesia",
  "activity": "working on Redis project",
  "people_present": ["John", "Sarah"],
  "weather": "sunny",
  "mood": "focused"
}
```

**Parameters:**
- `location` (string, optional): Current location
- `activity` (string, optional): Current activity  
- `people_present` (array, optional): List of people present
- Additional fields will be stored as environment context

**Response:**
```json
{
  "success": true,
  "message": "Context updated successfully",
  "context": {
    "location": "Jakarta, Indonesia",
    "activity": "working on Redis project", 
    "people_present": ["John", "Sarah"],
    "environment": {
      "weather": "sunny",
      "mood": "focused"
    }
  }
}
```

#### Get Context

**GET** `/api/memory/{vectorset_name}/context`

**Parameters:**
- `vectorset_name` (string, required): Name of the vectorset (e.g., "memories")

**Response:**
```json
{
  "success": true,
  "context": {
    "temporal": {
      "date": "2024-01-15",
      "time": "18:30:00"
    },
    "spatial": {
      "location": "Jakarta, Indonesia",
      "activity": "working on Redis project"
    },
    "social": {
      "people_present": ["John", "Sarah"]
    },
    "environmental": {
      "weather": "sunny",
      "mood": "focused"
    }
  }
}
```

### 6. Delete Memory

Delete a specific memory by ID.

**DELETE** `/api/memory/{vectorset_name}/{memory_id}`

**Parameters:**
- `vectorset_name` (string, required): Name of the vectorset (e.g., "memories")
- `memory_id` (string, required): ID of the memory to delete

**Response:**
```json
{
  "success": true,
  "message": "Memory 550e8400-e29b-41d4-a716-446655440000 deleted successfully",
  "memory_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### 7. Clear All Memories

Delete all memories from the system.

**DELETE** `/api/memory/{vectorset_name}/all`

**Parameters:**
- `vectorset_name` (string, required): Name of the vectorset (e.g., "memories")

**Response:**
```json
{
  "success": true,
  "message": "Successfully cleared all memories",
  "memories_deleted": 42,
  "vectorset_existed": true
}
```

---

## Chat Application API

This endpoint provides a conversational interface using the LangGraph workflow for complex multi-step reasoning.

### Chat Interface

**POST** `/api/chat`

**Request Body:**
```json
{
  "message": "What restaurants have I been to and what did I eat at each?"
}
```

**Parameters:**
- `message` (string, required): User message/question

**Response:**
```json
{
  "success": true,
  "message": "What restaurants have I been to and what did I eat at each?",
  "response": "Based on your memories, you've been to Mario's Italian Restaurant where you had pasta carbonara. The LangGraph workflow analyzed your memories and found this information with high confidence."
}
```

**Key Differences from `/api/memory/{vectorset_name}/ask`:**
- Uses full LangGraph workflow with tool orchestration
- Can perform multi-step reasoning and complex conversations
- More flexible but potentially higher latency
- Best for conversational UIs and complex queries

---

## System APIs

### Health Check

**GET** `/api/health`

**Response:**
```json
{
  "status": "healthy",
  "service": "LangGraph Memory Agent API", 
  "timestamp": "2024-01-15T18:30:00Z"
}
```

---

## Error Responses

All endpoints return consistent error responses:

```json
{
  "error": "Description of the error",
  "success": false
}
```

Common HTTP status codes:
- `400` - Bad Request (missing required parameters)
- `404` - Not Found (memory ID not found)
- `500` - Internal Server Error

---

## Usage Examples

### Basic Memory Operations

```bash
# Store a memory
curl -X POST http://localhost:5001/api/memory/memories \
  -H "Content-Type: application/json" \
  -d '{"text": "I love pizza with pepperoni"}'

# Search for memories
curl -X POST http://localhost:5001/api/memory/memories/search \
  -H "Content-Type: application/json" \
  -d '{"query": "pizza", "top_k": 3}'

# Answer a question
curl -X POST http://localhost:5001/api/memory/memories/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What kind of pizza do I like?"}'
```

### Context-Aware Memory Storage

```bash
# Set context first
curl -X POST http://localhost:5001/api/memory/memories/context \
  -H "Content-Type: application/json" \
  -d '{
    "location": "New York",
    "activity": "dining",
    "people_present": ["Alice", "Bob"]
  }'

# Store memory (will be grounded with context)
curl -X POST http://localhost:5001/api/memory/memories \
  -H "Content-Type: application/json" \
  -d '{"text": "We had an amazing dinner here"}'
```

### Chat Interface

```bash
# Complex conversational query
curl -X POST http://localhost:5001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me about all my restaurant experiences and what I learned about food preferences"}'
```

---

## Developer Notes

### Memory vs Answer vs Chat

**Use `/api/memory/{vectorset_name}/search`** when you need:
- Raw vector similarity search results
- Multiple memory candidates for further processing
- Building your own confidence analysis

**Use `/api/memory/{vectorset_name}/ask`** when you need:
- High-quality question answering with confidence scores
- Structured responses with supporting evidence
- Single-step question answering

**Use `/api/chat`** when you need:
- Multi-step reasoning and complex conversations
- Tool orchestration and workflow management
- Conversational UI interfaces

### Performance Considerations

- `/api/memory/{vectorset_name}/search`: Fastest, single vector search
- `/api/memory/{vectorset_name}/ask`: Medium, includes LLM analysis for confidence
- `/api/chat`: Slowest, full LangGraph workflow with potential multiple LLM calls

### Memory Grounding

When `apply_grounding: true` (default), the system will:
- Convert relative time references ("yesterday", "last week") to absolute dates
- Resolve location context ("here", "this place") using current context
- Add people context based on current social setting

Set `apply_grounding: false` for raw memory storage without context resolution.

### Filter Expressions

The `filter` parameter supports Redis VectorSet filter syntax:

```bash
# Filter by exact match
"filter": ".category == \"work\""

# Filter by range
"filter": ".priority >= 5"

# Multiple conditions
"filter": ".category == \"work\" and .priority >= 5"

# Array containment
"filter": ".tags in [\"important\", \"urgent\"]"
```

### API Migration Guide

**IMPORTANT: All endpoints now require explicit vectorset name in URL path**

**Old API → New API:**
- `/api/remember` → `/api/memory/{vectorset_name}` (POST)
- `/api/recall` → `/api/memory/{vectorset_name}/search` (POST)
- `/api/ask` → `/api/memory/{vectorset_name}/ask` (POST) or `/api/chat` (for conversations)
- `/api/memory-info` → `/api/memory/{vectorset_name}` (GET)
- `/api/context` → `/api/memory/{vectorset_name}/context` (GET/POST)
- `/api/delete/{id}` → `/api/memory/{vectorset_name}/{id}` (DELETE)
- `/api/delete-all` → `/api/memory/{vectorset_name}/all` (DELETE)

**Example Migration:**
- OLD: `POST /api/memory/search`
- NEW: `POST /api/memory/memories/search`

Use "memories" as the default vectorset name for existing functionality. All request/response formats remain the same, only URLs changed.

# Minsky Memory Agent API Documentation

## Overview

This API implements a cognitive memory architecture inspired by Marvin Minsky's "Society of Mind" theory. The system is organized around two fundamental concepts:

- **Memories**: Atomic memory units (fundamental knowledge structures)
- **K-lines**: Mental states constructed by activating and connecting relevant Memories

## Three-Layer Architecture

### 1. NEME API - Fundamental Memory Operations
Direct manipulation of atomic memories (Memories) - the building blocks of knowledge.

### 2. K-LINE API - Reflective Operations  
Mental state construction and reasoning by combining Memories into coherent cognitive structures.

### 3. AGENT API - High-Level Orchestration
Full conversational agents that orchestrate both Memories and K-lines for sophisticated interactions.

### 4. ASYNC MEMORY API - Background Processing
Asynchronous memory processing system for improved performance and hierarchical memory structures.

---

## NEME API

### Store Atomic Memory
**POST** `/api/memory`

Store a fundamental memory unit with optional contextual grounding.

```json
{
  "text": "User prefers window seats on flights",
  "apply_grounding": true
}
```

**Response:**
```json
{
  "success": true,
  "memory_id": "uuid-here",
  "original_text": "User prefers window seats on flights",
  "final_text": "User prefers window seats on flights (as of 2024-06-18)",
  "grounding_applied": true,
  "tags": ["preference", "travel"],
  "created_at": "2024-06-18T12:00:00Z"
}
```

### Search Atomic Memories
**POST** `/api/memory/search`

Find relevant Memories using vector similarity search.

```json
{
  "query": "travel preferences",
  "top_k": 5,
  "filter": "tags:preference"
}
```

**Response:**
```json
{
  "success": true,
  "query": "travel preferences",
  "memories": [
    {
      "id": "uuid-here",
      "text": "User prefers window seats on flights",
      "relevance_score": 95.2,
      "tags": ["preference", "travel"],
      "timestamp": "2024-06-18 12:00:00"
    }
  ],
  "count": 1
}
```

### Get Memory Statistics
**GET** `/api/memory`

Retrieve system information and memory count.

### Delete Specific Memory
**DELETE** `/api/memory/{memory_id}`

Remove a specific Neme by ID.

### Clear All Memories
**DELETE** `/api/memory`

Remove all stored Memories from the system.

### Context Management
**POST** `/api/memory/context` - Set current context for grounding
**GET** `/api/memory/context` - Get current context information

---

## K-LINE API

### Construct Mental State
**POST** `/api/klines/recall`

Build a mental state (K-line) by activating relevant Memories using enhanced relevance scoring.

```json
{
  "query": "travel planning",
  "top_k": 5
}
```

**Response:**
```json
{
  "success": true,
  "query": "travel planning",
  "mental_state": "Formatted mental state with relevant memories...",
  "memories": [
    {
      "id": "uuid-here",
      "text": "User prefers window seats on flights",
      "score": 0.85,
      "relevance_score": 0.92,
      "created_at": "2024-06-18T12:00:00Z",
      "last_accessed_at": "2024-06-19T10:30:00Z",
      "access_count": 5
    }
  ],
  "memory_count": 3
}
```

**Enhanced Relevance Scoring:**
- `score`: Original vector similarity score (0.0-1.0)
- `relevance_score`: Enhanced score combining vector similarity, temporal recency, and usage frequency
- `created_at`: ISO timestamp of memory creation
- `last_accessed_at`: ISO timestamp of last retrieval
- `access_count`: Number of times memory has been accessed

Memories are automatically sorted by `relevance_score` (highest first) and access tracking is updated on each retrieval.

### Question Answering with Reasoning
**POST** `/api/klines/answer`

Answer questions using K-line construction and sophisticated reasoning.

```json
{
  "question": "What seat should I book for my flight?",
  "top_k": 5
}
```

**Response:**
```json
{
  "success": true,
  "question": "What seat should I book for my flight?",
  "answer": "Based on your preferences, I recommend booking a window seat.",
  "confidence": "I'm fairly confident",
  "reasoning": "Your stored preferences indicate you prefer window seats on flights.",
  "supporting_memories": [...]
}
```

### Extract Memories from Conversation
**POST** `/api/klines/extract`

Intelligently extract valuable information from conversations and store as new Memories.

```json
{
  "raw_input": "User: I really love Italian food, especially pasta. My wife is vegetarian though.",
  "context_prompt": "Extract user preferences and family information",
  "apply_grounding": true
}
```

---

## ASYNC MEMORY API

### Store Raw Memory for Background Processing
**POST** `/api/memory/{vectorstore_name}/store_raw`

Store complete chat session data for asynchronous memory processing.

```json
{
  "session_data": "Complete chat session text...",
  "session_id": "optional-session-id",
  "metadata": {
    "user_id": "user123",
    "session_type": "code_review"
  }
}
```

**Response:**
```json
{
  "success": true,
  "raw_memory_id": "uuid-here",
  "queued_at": "2024-01-15T10:30:00Z",
  "estimated_processing_time": "2-5 minutes",
  "queue_position": 3
}
```

### Get Processing Status
**GET** `/api/memory/{vectorstore_name}/processing_status`

Check background processor status and queue information.

### Get Memory Hierarchy
**GET** `/api/memory/{vectorstore_name}/hierarchy`

Retrieve hierarchical memory data with filtering options.

**Query Parameters:**
- `session_id`: Filter by session ID
- `memory_type`: Filter by type ('discrete', 'summary', 'raw', 'all')
- `start_date`: Start date filter (ISO format)
- `end_date`: End date filter (ISO format)
- `limit`: Maximum results (1-1000, default: 50)

### Get Session Details
**GET** `/api/memory/{vectorstore_name}/session/{session_id}`

Get complete information about a specific session.

### Manual Cleanup
**POST** `/api/memory/{vectorstore_name}/cleanup`

Manually trigger cleanup of expired memory data.

### Memory Statistics
**GET** `/api/memory/{vectorstore_name}/stats`

Get comprehensive memory statistics including hierarchy breakdown.

---

## AGENT API

### Full Conversational Agent
**POST** `/api/agent/chat`

Complete cognitive architecture with memory integration.

```json
{
  "message": "Help me plan a dinner for my wife and me",
  "system_prompt": "You are a helpful travel assistant" // Optional custom system prompt
}
```

### Create Agent Session
**POST** `/api/agent/session`

Create persistent conversational session with memory capabilities.

```json
{
  "system_prompt": "You are a helpful travel assistant",
  "config": {
    "use_memory": true,
    "model": "gpt-3.5-turbo",
    "temperature": 0.7
  }
}
```

### Chat with Session
**POST** `/api/agent/session/{session_id}`

Send messages to an agent session with full cognitive architecture.

```json
{
  "message": "What restaurants do you recommend?",
  "top_k": 10,
  "store_memory": true,
  "stream": false
}
```

Parameters:
- `message` (string, required): The user's message
- `top_k` (integer, optional): Number of memories to search and return (default: 10)
- `store_memory` (boolean, optional): Whether to extract and store memories (default: true)
- `stream` (boolean, optional): Whether to stream the response (default: false)

### Session Management
**GET** `/api/agent/session/{session_id}` - Get session info
**DELETE** `/api/agent/session/{session_id}` - Delete session
**GET** `/api/agent/sessions` - List all sessions

---

## Conceptual Framework

### Minsky's Theory Applied

1. **Memories** represent atomic knowledge units:
   - "User likes Italian food"
   - "Wife is vegetarian" 
   - "Prefers window seats"

2. **K-lines** activate relevant Memories for specific tasks:
   - Question: "Plan dinner" → Activates food preferences + dietary restrictions
   - Creates mental state: "Italian food + vegetarian options"
   - Generates answer: "Try vegetarian Italian restaurant"

3. **Agent** orchestrates the complete process:
   - Searches relevant Memories
   - Constructs appropriate K-lines
   - Applies reasoning and language generation
   - Extracts new Memories from interactions

### Progressive Complexity

- **Level 1 (Memories)**: Simple storage and retrieval
- **Level 2 (K-lines)**: Reasoning and mental state construction  
- **Level 3 (Agent)**: Full conversational intelligence

This architecture provides both theoretical grounding and practical utility for building sophisticated memory-enabled AI agents.

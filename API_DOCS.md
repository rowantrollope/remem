# Memory Agent API Reference

A comprehensive API reference for the Memory Agent system, including both the Python SDK and REST API.

## Table of Contents

- [Quick Start](#quick-start)
- [Python SDK](#python-sdk)
  - [Core Methods](#core-methods)
  - [Data Structures](#data-structures)
- [REST API](#rest-api)
  - [Endpoints](#endpoints)
  - [Request/Response Examples](#requestresponse-examples)
- [CLI Commands](#cli-commands)
- [Context System](#context-system)
- [Advanced Usage](#advanced-usage)
- [Error Handling](#error-handling)
- [Requirements](#requirements)

---

## Quick Start

### Python SDK
```python
from memory_agent import MemoryAgent

# Initialize
agent = MemoryAgent()

# Set context
agent.set_context(location="Tokyo, Japan", activity="vacation")

# Store memory with grounding
memory_id = agent.store_memory("The food here is amazing")

# Search memories
memories = agent.search_memories("food in Tokyo")

# Ask questions
response = agent.answer_question("What did you think of Tokyo food?")
```

### REST API
```bash
# Base URL: http://localhost:5001

# Store a memory
curl -X POST http://localhost:5001/api/remember \
  -H "Content-Type: application/json" \
  -d '{"memory": "The food here is amazing", "apply_grounding": true}'

# Search memories
curl -X POST http://localhost:5001/api/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "food in Tokyo", "top_k": 3}'

# Ask a question
curl -X POST http://localhost:5001/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What did you think of Tokyo food?"}'
```

---

## Python SDK

### Core Methods

#### `MemoryAgent(redis_host="localhost", redis_port=6381, redis_db=0)`
Initialize the memory agent with Redis connection.

**Parameters:**
- `redis_host` (str): Redis server hostname
- `redis_port` (int): Redis server port
- `redis_db` (int): Redis database number

#### `set_context(**kwargs)`
Set current context for memory grounding.

**Parameters:**
- `location` (str): Current location (e.g., "Tokyo, Japan", "Home office")
- `activity` (str): Current activity (e.g., "vacation", "working", "meeting")
- `people_present` (List[str]): People currently present
- `**kwargs`: Additional environment context (weather, mood, etc.)

**Example:**
```python
agent.set_context(
    location="Jakarta, Indonesia",
    activity="business trip",
    people_present=["Sarah", "Mike"],
    weather="hot and humid",
    mood="excited"
)
```

#### `store_memory(memory_text, apply_grounding=True)`
Store a memory with optional contextual grounding.

**Parameters:**
- `memory_text` (str): The memory text to store
- `apply_grounding` (bool): Whether to apply contextual grounding (default: True)

**Returns:** Memory UUID (str)

**Examples:**
```python
# With grounding (default)
id1 = agent.store_memory("It's really hot outside")
# Stores: "The weather in Jakarta, Indonesia is very hot"

# Without grounding
id2 = agent.store_memory("It's really hot outside", apply_grounding=False)
# Stores: "It's really hot outside"
```

#### `search_memories(query, top_k=3, filterBy=None)`
Search for relevant memories using vector similarity.

**Parameters:**
- `query` (str): Search query
- `top_k` (int): Number of results to return (default: 3)
- `filterBy` (str, optional): Filter expression for Redis VectorSet FILTER syntax

**Returns:** List of memory dictionaries

**Filter Examples:**
```python
# Filter by category
memories = agent.search_memories("work", filterBy='.category == "work"')

# Filter by priority and year
memories = agent.search_memories("project", filterBy='.priority == "high" and .year >= 2024')

# Filter using IN operator
memories = agent.search_memories("meeting", filterBy='.attendees in ["Sarah", "Mike"]')
```

#### `answer_question(question, top_k=5, filterBy=None)`
Answer a question using relevant memories and OpenAI.

**Parameters:**
- `question` (str): Question to answer
- `top_k` (int): Number of memories to use for context (default: 5)
- `filterBy` (str, optional): Filter expression for Redis VectorSet FILTER syntax

**Returns:** Response dictionary

#### `get_memory_info()`
Get statistics about stored memories.

**Returns:** Dictionary with memory count, vector info, etc.

#### `delete_memory(memory_id)`
Delete a specific memory by ID.

**Parameters:**
- `memory_id` (str): UUID of memory to delete

**Returns:** Boolean success status

#### `clear_all_memories()`
Delete all stored memories.

**Returns:** Dictionary with operation results

### Data Structures

#### Memory Dictionary
```python
{
    "id": "uuid-string",
    "text": "grounded or original text",
    "original_text": "original memory text",
    "grounded_text": "context-resolved text",
    "tags": ["extracted", "tags"],
    "timestamp": 1234567890.0,
    "score": 0.95,  # similarity score (0-1)
    "formatted_time": "2024-01-15 14:30",
    "grounding_applied": True,
    "grounding_info": {
        "dependencies_resolved": {...},
        "changes_made": [...],
        "unresolved_references": [...]
    },
    "context_snapshot": {
        "location": "Tokyo, Japan",
        "activity": "vacation",
        "people_present": ["Sarah"],
        "environment": {...}
    }
}
```

#### Answer Response Dictionary
```python
{
    "type": "answer" | "help",
    "answer": "The answer text",
    "confidence": "high" | "medium" | "low",
    "reasoning": "Brief explanation of the answer",
    "supporting_memories": [
        {
            "id": "memory-uuid",
            "text": "memory text",
            "relevance_score": 85.2,
            "timestamp": "2024-01-15 14:30",
            "tags": ["food", "Tokyo"]
        }
    ]
}
```

---

## REST API

### Base URL
```
http://localhost:5001
```

### Endpoints

#### `POST /api/remember`
Store a new memory.

**Request Body:**
```json
{
    "memory": "The food here is amazing",
    "apply_grounding": true
}
```

**Response:**
```json
{
    "success": true,
    "memory_id": "uuid-string",
    "message": "Memory stored successfully",
    "grounding_applied": true
}
```

#### `POST /api/recall`
Search for memories.

**Request Body:**
```json
{
    "query": "food in Tokyo",
    "top_k": 3,
    "filterBy": ".category == \"work\""
}
```

**Parameters:**
- `query` (str): Search query text
- `top_k` (int, optional): Number of results to return (default: 3)
- `filterBy` (str, optional): Redis VectorSet filter expression

**Filter Expression Examples:**
- `.category == "work"` - Filter by exact category match
- `.priority == "high" and .year >= 2024` - Combined conditions
- `.tags in ["important", "urgent"]` - Array containment
- `.score > 0.8 or .priority == "critical"` - Logical OR

**Response:**
```json
{
    "success": true,
    "memories": [
        {
            "id": "uuid-string",
            "text": "The food in Tokyo, Japan is amazing",
            "score": 0.95,
            "formatted_time": "2024-01-15 14:30",
            "tags": ["food", "Tokyo"]
        }
    ],
    "count": 1
}
```

#### `POST /api/ask`
Ask a question based on stored memories.

**Request Body:**
```json
{
    "question": "What did you think of Tokyo food?",
    "top_k": 5,
    "filterBy": ".category == \"personal\""
}
```

**Parameters:**
- `question` (str): Question to answer
- `top_k` (int, optional): Number of memories to use for context (default: 5)
- `filterBy` (str, optional): Redis VectorSet filter expression

**Response:**
```json
{
    "success": true,
    "question": "What did you think of Tokyo food?",
    "type": "answer",
    "answer": "Based on your memories, you found the food in Tokyo to be amazing.",
    "confidence": "high",
    "reasoning": "Direct memory match about Tokyo food experience",
    "supporting_memories": [...]
}
```

#### `GET /api/status`
Check system status.

**Response:**
```json
{
    "status": "ready",
    "timestamp": "2024-01-15T14:30:00Z"
}
```

#### `GET /api/memory-info`
Get memory statistics.

**Response:**
```json
{
    "success": true,
    "memory_count": 42,
    "vector_dimension": 1536,
    "vectorset_name": "memories",
    "embedding_model": "text-embedding-ada-002",
    "redis_host": "localhost",
    "redis_port": 6381,
    "timestamp": "2024-01-15T14:30:00Z"
}
```

#### `POST /api/context`
Set current context for memory grounding.

**Request Body:**
```json
{
    "location": "Tokyo, Japan",
    "activity": "vacation",
    "people_present": ["Sarah", "Mike"],
    "weather": "sunny",
    "mood": "excited"
}
```

**Response:**
```json
{
    "success": true,
    "message": "Context updated successfully",
    "context": {
        "location": "Tokyo, Japan",
        "activity": "vacation",
        "people_present": ["Sarah", "Mike"],
        "environment": {
            "weather": "sunny",
            "mood": "excited"
        }
    }
}
```

#### `GET /api/context`
Get current context information.

**Response:**
```json
{
    "success": true,
    "context": {
        "location": "Tokyo, Japan",
        "activity": "vacation",
        "people_present": ["Sarah", "Mike"],
        "environment": {
            "weather": "sunny",
            "mood": "excited"
        },
        "timestamp": "2024-01-15T14:30:00Z"
    }
}
```

#### `DELETE /api/delete/{memory_id}`
Delete a specific memory.

**Response:**
```json
{
    "success": true,
    "message": "Memory uuid-string deleted successfully",
    "memory_id": "uuid-string"
}
```

#### `DELETE /api/delete-all`
Clear all memories.

**Response:**
```json
{
    "success": true,
    "message": "All memories cleared successfully",
    "memories_deleted": 42,
    "vectorset_existed": true
}
```

### Request/Response Examples

#### Complete Memory Storage Flow
```bash
# 1. Set context
curl -X POST http://localhost:5001/api/context \
  -H "Content-Type: application/json" \
  -d '{
    "location": "Paris, France",
    "activity": "business meeting",
    "people_present": ["Jean", "Marie"]
  }'

# 2. Store memory with grounding
curl -X POST http://localhost:5001/api/remember \
  -H "Content-Type: application/json" \
  -d '{
    "memory": "The presentation went really well",
    "apply_grounding": true
  }'

# 3. Search for related memories
curl -X POST http://localhost:5001/api/recall \
  -H "Content-Type: application/json" \
  -d '{
    "query": "business meeting presentation",
    "top_k": 5
  }'

# 4. Ask a question
curl -X POST http://localhost:5001/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How did the meeting in Paris go?"
  }'
```

---

## CLI Commands

### Interactive Mode
```bash
# Start the memory agent CLI
python memory_agent.py

# Available commands:
Memory Agent> remember The weather is nice today
Memory Agent> remember-raw The weather is nice today  # without grounding
Memory Agent> search weather
Memory Agent> ask What's the weather like?
Memory Agent> info
Memory Agent> context location="New York" activity="work"
Memory Agent> clear-all
Memory Agent> quit
```

### Direct Commands
```bash
# Store a memory
python memory_agent.py remember "Had lunch at the new cafe"

# Search memories
python memory_agent.py search "lunch cafe"

# Ask a question
python memory_agent.py ask "Where did I eat lunch?"

# Get memory info
python memory_agent.py info

# Clear all memories
python memory_agent.py clear-all
```

---

## Filter System

The Memory Agent supports advanced filtering using Redis VectorSet filter expressions. This allows you to combine vector similarity search with scalar filtering based on memory attributes.

### Filter Expression Syntax

Filter expressions use JavaScript-like syntax:

#### Operators
- **Arithmetic:** `+`, `-`, `*`, `/`, `%`, `**`
- **Comparison:** `==`, `!=`, `>`, `<`, `>=`, `<=`
- **Logical:** `and`, `or`, `not` (or `&&`, `||`, `!`)
- **Containment:** `in`
- **Grouping:** Parentheses `()`

#### Data Types
- **Numbers:** `42`, `3.14`
- **Strings:** `"work"`, `"high priority"`
- **Booleans:** `true`, `false` (converted to 1 or 0)
- **Arrays:** `["item1", "item2"]` (for `in` operator)

#### Field Access
Use dot notation to access memory attributes: `.category`, `.priority`, `.year`

> **Note:** Only top-level fields are supported (e.g., `.genre`, but not `.movie.genre`)

### Filter Examples

#### Basic Filtering
```python
# Filter by category
agent.search_memories("project", filterBy='.category == "work"')

# Filter by priority
agent.search_memories("task", filterBy='.priority == "high"')

# Filter by year range
agent.search_memories("meeting", filterBy='.year >= 2024 and .year < 2025')
```

#### Advanced Filtering
```python
# Combine multiple conditions
agent.search_memories("project",
    filterBy='.category == "work" and .priority != "low" and .year >= 2024')

# Use IN operator with arrays
agent.search_memories("meeting",
    filterBy='.attendees in ["Sarah", "Mike", "John"]')

# Mathematical operations
agent.search_memories("review",
    filterBy='(.year - 2020) ** 2 < 16 and .rating / 2 > 4')

# Logical OR conditions
agent.search_memories("task",
    filterBy='.priority == "critical" or (.category == "work" and .urgent == true)')
```

#### Setting Memory Attributes

To use filters, memories need to have attributes. You can set them when storing or update them later:

```python
# Store memory and set attributes
memory_id = agent.store_memory("Completed the quarterly report")

# Add attributes using Redis VectorSet commands
import json
agent.redis_client.execute_command(
    "VSETATTR", agent.VECTORSET_KEY, memory_id,
    json.dumps({
        "category": "work",
        "priority": "high",
        "year": 2024,
        "project": "Q4 Analysis",
        "completed": True
    })
)
```

### REST API Filter Examples

```bash
# Filter work-related memories
curl -X POST http://localhost:5001/api/recall \
  -H "Content-Type: application/json" \
  -d '{
    "query": "project meeting",
    "top_k": 5,
    "filterBy": ".category == \"work\""
  }'

# Filter high-priority items from recent years
curl -X POST http://localhost:5001/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What important work did I do recently?",
    "top_k": 10,
    "filterBy": ".priority == \"high\" and .year >= 2024"
  }'

# Filter by multiple categories
curl -X POST http://localhost:5001/api/recall \
  -H "Content-Type: application/json" \
  -d '{
    "query": "learning development",
    "filterBy": ".category in [\"education\", \"training\", \"development\"]"
  }'
```

### Performance Tips

- **Missing attributes** are treated as non-matching
- Use **FILTER-EF** parameter to tune recall vs performance (not exposed in current API)
- **Combine multiple attributes** for fine-grained filtering
- **Index frequently filtered fields** for better performance

---

## Context System

The Memory Agent includes a sophisticated contextual grounding system that automatically resolves relative references in memories.

### Context Categories

#### Temporal References
- "today", "yesterday", "last week"
- "this morning", "earlier"
- "now", "currently"

#### Spatial References
- "here", "there", "this place"
- "nearby", "around here"
- "upstairs", "downstairs"

#### Personal References
- "he", "she", "they"
- "my friend", "the guy"
- "everyone", "someone"

#### Environmental References
- "this weather", "it's hot"
- "the atmosphere", "the vibe"
- "the situation"

#### Demonstrative References
- "this", "that", "these", "those"
- "the one", "the thing"

### Context Resolution Examples

```python
# Set context
agent.set_context(
    location="Tokyo, Japan",
    activity="vacation",
    people_present=["Sarah", "Mike"],
    weather="rainy"
)

# Store memory with relative references
agent.store_memory("It's really wet outside today")
# Resolves to: "The weather in Tokyo, Japan is very rainy today (2024-01-15)"

agent.store_memory("Sarah and Mike loved this place")
# Resolves to: "Sarah and Mike loved Tokyo, Japan"

agent.store_memory("We should come back here next year")
# Resolves to: "We should come back to Tokyo, Japan next year"
```

### Context Persistence
- Context persists across memory operations
- Update context as your situation changes
- Context snapshots are stored with each memory
- Historical context is preserved for each memory

---

## Advanced Usage

### Batch Operations
```python
# Store multiple memories with context
agent.set_context(location="Conference Center", activity="tech conference")

memories = [
    "The keynote speaker was excellent",
    "Learned about new AI frameworks",
    "Met interesting people from startups",
    "The networking lunch was valuable"
]

memory_ids = []
for memory in memories:
    memory_id = agent.store_memory(memory)
    memory_ids.append(memory_id)
```

### Custom Context Tracking
```python
# Track complex environmental context
agent.set_context(
    location="Home Office",
    activity="deep work session",
    people_present=[],
    lighting="dim",
    music="ambient",
    focus_level="high",
    project="memory agent development",
    tools_used=["VSCode", "Redis", "OpenAI"]
)

agent.store_memory("Made significant progress on the API documentation")
```

### Memory Analysis
```python
# Get detailed memory information
info = agent.get_memory_info()
print(f"Total memories: {info['memory_count']}")
print(f"Vector dimension: {info['vector_dimension']}")

# Search with different similarity thresholds
memories = agent.search_memories("Tokyo food", top_k=10)
high_relevance = [m for m in memories if m['score'] > 0.8]
medium_relevance = [m for m in memories if 0.6 < m['score'] <= 0.8]
```

### Question Answering Strategies
```python
# Different question types
factual_response = agent.answer_question("What restaurants did I visit in Tokyo?")
opinion_response = agent.answer_question("What did I think of the Tokyo restaurants?")
comparison_response = agent.answer_question("How did Tokyo food compare to Paris food?")
recommendation_response = agent.answer_question("What Tokyo restaurants would you recommend?")
```

---

## Error Handling

### Common Error Responses

#### API Errors
```json
{
    "error": "Memory text is required",
    "success": false
}
```

```json
{
    "error": "Memory agent not initialized",
    "success": false
}
```

#### Python SDK Exceptions
```python
try:
    agent = MemoryAgent()
except redis.ConnectionError:
    print("Redis server not available")
except Exception as e:
    print(f"Initialization failed: {e}")

try:
    memory_id = agent.store_memory("test memory")
except Exception as e:
    print(f"Failed to store memory: {e}")
```

### HTTP Status Codes
- `200` - Success
- `400` - Bad Request (missing required fields)
- `404` - Not Found (memory ID not found)
- `500` - Internal Server Error

---

## Requirements

### System Requirements
- Python 3.8+
- Redis server with RedisSearch module (Redis Stack recommended)
- OpenAI API key

### Python Dependencies
```txt
redis>=5.0.0
openai>=1.0.0
python-dotenv>=1.0.0
numpy>=1.24.0
flask>=2.3.0
flask-cors>=4.0.0
```

### Environment Variables
```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (defaults shown)
REDIS_HOST=localhost
REDIS_PORT=6381
REDIS_DB=0
```

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your OpenAI API key

# Start Redis (if using Docker)
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest

# Run the memory agent
python memory_agent.py

# Or start the web API
python web_app.py
```

### Redis Setup
See `setup_redis.md` for detailed Redis installation and configuration instructions.

---

## Support

For issues, questions, or contributions, please refer to the project repository or documentation.
# Set context
context location="Tokyo, Japan" activity="vacation" people="Alice,Bob"

# View current context
context-info

# Store memory with grounding
remember "The sushi here is incredible"

# Store memory without grounding  
remember-raw "The sushi here is incredible"

# Search memories
recall "sushi in Tokyo"

# Ask questions
ask "What did you think of the food?"

# Get system info
info

# Delete memory
delete <memory-id>
```

## Context Types

### Temporal References
**Detected:** today, yesterday, now, this morning, last week, recently
**Resolved to:** Specific dates and times

```python
# Input: "I had a great meeting today"
# Output: "I had a great meeting on January 15, 2024"
```

### Spatial References
**Detected:** here, outside, this place, nearby, upstairs, local
**Resolved to:** Specific locations

```python
# Input: "It's really crowded here"
# Output: "Tokyo, Japan is really crowded"
```

### Personal References
**Detected:** this guy, my boss, the meeting, that person
**Resolved to:** Specific names when determinable

```python
# Input: "Sarah had great ideas in the meeting"
# Context: people_present=["Sarah", "Mike"]
# Output: "Sarah had great ideas in the team meeting"
```

### Environmental References
**Detected:** this weather, the current situation, right now
**Resolved to:** Specific conditions with location/time

```python
# Input: "This weather is terrible"
# Output: "The weather in Jakarta, Indonesia on January 15, 2024 is terrible"
```

## Advanced Usage

### Context Snapshots
Each grounded memory includes a context snapshot:

```python
memory = agent.search_memories("weather")[0]
context = memory["context_snapshot"]
print(context["temporal"]["date"])  # "Monday, January 15, 2024"
print(context["spatial"]["location"])  # "Jakarta, Indonesia"
```

### Grounding Information
View what changes were made during grounding:

```python
memory = agent.search_memories("hot")[0]
if memory["grounding_applied"]:
    changes = memory["grounding_info"]["changes_made"]
    for change in changes:
        print(f"{change['original']} → {change['replacement']} ({change['type']})")
```

### Batch Operations
```python
# Set context once
agent.set_context(location="Conference Center", activity="tech conference")

# Store multiple memories
memories = [
    "The keynote speaker was amazing",
    "This venue has great WiFi", 
    "I met some interesting people here"
]

for memory in memories:
    agent.store_memory(memory)
```

## Best Practices

1. **Set context before storing memories** for optimal grounding
2. **Use specific locations** ("Tokyo, Japan" vs "here")
3. **Update context when changing locations/activities**
4. **Include relevant people** in social context
5. **Review grounded memories** to ensure accuracy

## Error Handling

```python
try:
    agent = MemoryAgent()
    agent.store_memory("Test memory")
except redis.ConnectionError:
    print("Redis not available")
except Exception as e:
    print(f"Error: {e}")
```

## Requirements

- Redis 8 - with VectorSet
- OpenAI API key in environment
- Python packages: `redis>=5.0.0`, `openai>=1.0.0`, `python-dotenv>=1.0.0`

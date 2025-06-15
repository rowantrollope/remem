# Memory Agent REST API Documentation

Base URL: `http://localhost:5001`

## Authentication
No authentication required for local development.

## Content Type
All requests should use `Content-Type: application/json`

---

## Endpoints

### 1. Store Memory

**POST** `/api/remember`

Store a new memory with optional contextual grounding.

#### Request Body
```json
{
  "memory": "It's really hot outside",
  "apply_grounding": true
}
```

#### Parameters
- `memory` (string, required): The memory text to store
- `apply_grounding` (boolean, optional): Whether to apply contextual grounding (default: true)

#### Response
```json
{
  "success": true,
  "memory_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Memory stored successfully",
  "grounding_applied": true
}
```

#### Error Response
```json
{
  "error": "Memory text is required"
}
```

---

### 2. Search Memories

**POST** `/api/recall`

Search for relevant memories using vector similarity.

#### Request Body
```json
{
  "query": "weather in Jakarta",
  "top_k": 5
}
```

#### Parameters
- `query` (string, required): Search query
- `top_k` (integer, optional): Number of results to return (default: 3)

#### Response
```json
{
  "success": true,
  "memories": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "text": "The weather in Jakarta, Indonesia is very hot",
      "original_text": "It's really hot outside",
      "grounded_text": "The weather in Jakarta, Indonesia is very hot",
      "tags": ["weather"],
      "timestamp": 1705123456.789,
      "score": 0.95,
      "formatted_time": "2024-01-15 14:30",
      "grounding_applied": true,
      "grounding_info": {
        "dependencies_found": {
          "spatial": ["outside"],
          "environmental": ["hot"]
        },
        "changes_made": [
          {
            "original": "outside",
            "replacement": "in Jakarta, Indonesia",
            "type": "spatial"
          }
        ],
        "unresolved_references": []
      },
      "context_snapshot": {
        "temporal": {
          "date": "Monday, January 15, 2024",
          "time": "02:30 PM",
          "iso_date": "2024-01-15T14:30:00"
        },
        "spatial": {
          "location": "Jakarta, Indonesia",
          "activity": "traveling"
        },
        "social": {
          "people_present": ["travel companion"]
        },
        "environmental": {
          "weather": "hot and humid"
        }
      }
    }
  ],
  "count": 1
}
```

---

### 3. Ask Question

**POST** `/api/ask`

Ask a question and get an AI-generated answer based on stored memories.

#### Request Body
```json
{
  "question": "What was the weather like in Jakarta?"
}
```

#### Parameters
- `question` (string, required): Question to answer

#### Response
```json
{
  "success": true,
  "question": "What was the weather like in Jakarta?",
  "type": "answer",
  "answer": "The weather in Jakarta was very hot and humid when you visited.",
  "confidence": "high",
  "reasoning": "Multiple memories directly describe the hot weather conditions in Jakarta",
  "supporting_memories": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "text": "The weather in Jakarta, Indonesia is very hot",
      "relevance_score": 95.5,
      "timestamp": "2024-01-15 14:30",
      "tags": ["weather"]
    }
  ]
}
```

#### Help Response (for invalid questions)
```json
{
  "success": true,
  "question": "hello",
  "type": "help",
  "answer": "Ask me to remember anything, for example: 'What color is Molly?' or 'Tell me about my meeting with Sarah last week.'",
  "confidence": "n/a",
  "supporting_memories": []
}
```

---

### 4. Set Context

**POST** `/api/context`

Set current context for memory grounding.

#### Request Body
```json
{
  "location": "Jakarta, Indonesia",
  "activity": "business trip",
  "people_present": ["Sarah", "Mike"],
  "weather": "hot and humid",
  "temperature": "32°C",
  "mood": "excited"
}
```

#### Parameters
- `location` (string, optional): Current location
- `activity` (string, optional): Current activity
- `people_present` (array, optional): List of people currently present
- Any additional fields will be stored as environmental context

#### Response
```json
{
  "success": true,
  "message": "Context updated successfully",
  "context": {
    "location": "Jakarta, Indonesia",
    "activity": "business trip",
    "people_present": ["Sarah", "Mike"],
    "environment": {
      "weather": "hot and humid",
      "temperature": "32°C",
      "mood": "excited"
    }
  }
}
```

---

### 5. Get Context

**GET** `/api/context`

Get current context information.

#### Response
```json
{
  "success": true,
  "context": {
    "temporal": {
      "date": "Monday, January 15, 2024",
      "time": "02:30 PM",
      "iso_date": "2024-01-15T14:30:00",
      "day_of_week": "Monday",
      "month": "January",
      "year": 2024
    },
    "spatial": {
      "location": "Jakarta, Indonesia",
      "activity": "business trip"
    },
    "social": {
      "people_present": ["Sarah", "Mike"]
    },
    "environmental": {
      "weather": "hot and humid",
      "temperature": "32°C",
      "mood": "excited"
    }
  }
}
```

---

### 6. System Status

**GET** `/api/status`

Check system status and health.

#### Response
```json
{
  "status": "ready",
  "timestamp": "2024-01-15T14:30:00.123456"
}
```

#### Error Response
```json
{
  "status": "not_initialized",
  "timestamp": "2024-01-15T14:30:00.123456"
}
```

---

### 7. Memory Information

**GET** `/api/memory-info`

Get statistics about stored memories.

#### Response
```json
{
  "success": true,
  "memory_count": 42,
  "vector_dimension": 1536,
  "vectorset_name": "memories",
  "embedding_model": "text-embedding-ada-002",
  "redis_host": "localhost",
  "redis_port": 6381,
  "timestamp": "2024-01-15T14:30:00.123456",
  "vectorset_info": {
    "algorithm": "FLAT",
    "dimension": "1536",
    "distance_metric": "COSINE"
  }
}
```

---

### 8. Delete Memory

**DELETE** `/api/delete/{memory_id}`

Delete a specific memory by ID.

#### URL Parameters
- `memory_id` (string, required): UUID of the memory to delete

#### Response
```json
{
  "success": true,
  "message": "Memory 550e8400-e29b-41d4-a716-446655440000 deleted successfully",
  "memory_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

#### Error Response
```json
{
  "success": false,
  "error": "Memory 550e8400-e29b-41d4-a716-446655440000 not found or could not be deleted"
}
```

---

## Error Codes

- **400 Bad Request**: Missing required parameters or invalid input
- **404 Not Found**: Memory not found (for delete operations)
- **500 Internal Server Error**: System error (Redis connection, OpenAI API, etc.)

## Example Usage

### JavaScript/Fetch
```javascript
// Set context
await fetch('/api/context', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    location: 'Tokyo, Japan',
    activity: 'vacation',
    weather: 'sunny'
  })
});

// Store memory
const response = await fetch('/api/remember', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    memory: 'The sushi here is incredible',
    apply_grounding: true
  })
});

// Search memories
const searchResponse = await fetch('/api/recall', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: 'sushi in Tokyo',
    top_k: 3
  })
});
```

### Python/Requests
```python
import requests

# Set context
requests.post('http://localhost:5001/api/context', json={
    'location': 'Tokyo, Japan',
    'activity': 'vacation',
    'weather': 'sunny'
})

# Store memory
response = requests.post('http://localhost:5001/api/remember', json={
    'memory': 'The sushi here is incredible',
    'apply_grounding': True
})

# Ask question
answer = requests.post('http://localhost:5001/api/ask', json={
    'question': 'What did you think of the food in Tokyo?'
})
```

### cURL
```bash
# Set context
curl -X POST http://localhost:5001/api/context \
  -H "Content-Type: application/json" \
  -d '{"location": "Tokyo, Japan", "activity": "vacation"}'

# Store memory
curl -X POST http://localhost:5001/api/remember \
  -H "Content-Type: application/json" \
  -d '{"memory": "The sushi here is incredible"}'

# Search memories
curl -X POST http://localhost:5001/api/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "sushi", "top_k": 3}'
```

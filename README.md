# LangGraph Memory Agent

A sophisticated Redis-based memory system that provides intelligent memory storage, retrieval, and analysis using LangGraph workflow orchestration and OpenAI embeddings. This project focuses entirely on advanced memory capabilities for AI agents.

## 🧠 Features

- **LangGraph Workflow**: Intelligent tool orchestration for complex memory operations
- **Vector Memory Storage**: Semantic similarity search using Redis VectorSet API
- **Contextual Grounding**: Automatic resolution of relative time/location references
- **Question Answering**: AI-powered answers with confidence indicators and supporting evidence
- **Memory Management**: Store, search, update, and delete memories with full CRUD operations
- **Web Interface**: Clean, minimalist web UI with responsive design
- **CLI Interface**: Simple command-line interface for storing and recalling memories
- **Context Management**: Set and track current context for memory grounding

## Setup

### 1. Install Redis Stack (includes RedisSearch)

```bash
# Using Docker (recommended)
docker run -d --name redis-stack -p 6381:6381 redis/redis-stack:latest

# Or using Homebrew (macOS)
brew tap redis-stack/redis-stack
brew install redis-stack
redis-stack-server --port 6381
```

### 2. Create Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Copy the example environment file and add your OpenAI API key:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Run the Memory Agent

Make sure your virtual environment is activated:

```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Interactive Chat Mode

```bash
python main.py
```

#### Single Query Mode

```bash
python main.py "Remember that I like pizza"
```

#### Web API Server

```bash
python web_app.py
```

Then visit http://localhost:5001 for the web interface.

## Project Structure

- `main.py` - Entry point for running the memory agent
- `langgraph_memory_agent.py` - LangGraph workflow orchestration for memory operations
- `memory_agent.py` - Core memory system with Redis and OpenAI embeddings
- `tools.py` - Memory tools for the LangGraph agent
- `web_app.py` - Flask web API server
- `requirements.txt` - Python dependencies
- `.env.example` - Environment variables template

## How It Works

The memory agent uses a sophisticated multi-layer architecture:

### 1. Core Memory Layer (`memory_agent.py`)
- **Vector Storage**: Uses Redis VectorSet for semantic memory storage
- **Embeddings**: OpenAI text-embedding-ada-002 for vector representations
- **Contextual Grounding**: Converts relative references (today, here) to absolute ones
- **Confidence Analysis**: Sophisticated question answering with confidence scoring

### 2. Tool Layer (`tools.py`)
- **Memory Tools**: LangChain tools that wrap core memory operations
- **Tool Integration**: Seamless integration with LangGraph workflow

### 3. Workflow Layer (`langgraph_memory_agent.py`)
- **LangGraph Orchestration**: Intelligent tool selection and multi-step reasoning
- **Workflow Nodes**: Analyzer → Memory Agent → Tools → Synthesizer
- **Complex Reasoning**: Handles multi-step questions and sophisticated analysis

### 4. API Layer (`web_app.py`)
- **Developer APIs**: Core memory operations for agent developers
- **Chat API**: Conversational interface using LangGraph workflow
- **Web Interface**: Clean, responsive UI for memory management

## Available Memory Tools

- `store_memory` - Store new memories with contextual grounding
- `search_memories` - Find relevant memories using vector similarity search
- `answer_with_confidence` - Answer questions with sophisticated confidence analysis
- `format_memory_results` - Format memory search results for display
- `set_context` - Set current context (location, activity, people) for better memory grounding
- `get_memory_stats` - Get statistics about stored memories
- `analyze_question_type` - Analyze what type of question the user is asking

## Requirements

- Python 3.8+
- Redis Stack (with RedisSearch module)
- OpenAI API key
- Internet connection for API calls

---

## API Documentation

The Memory Agent provides a comprehensive RESTful API designed for two primary use cases:

1. **Developer Memory APIs** - Core memory operations for agent developers
2. **Chat Application API** - Conversational interface for demo/UI applications

### Base URL

```
http://localhost:5001
```

### Authentication

Currently no authentication is required. The API is designed for local development and testing.

---

### Developer Memory APIs

These endpoints provide core memory operations for developers building agents.

#### 1. Store Memory

Store a new memory with optional contextual grounding.

**POST** `/api/memory`

**Request Body:**
```json
{
  "text": "I went to Mario's Italian Restaurant and had amazing pasta carbonara",
  "apply_grounding": true
}
```

**Parameters:**
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

#### 2. Search Memories

Search for memories using vector similarity.

**POST** `/api/memory/search`

**Request Body:**
```json
{
  "query": "Italian restaurant food",
  "top_k": 5,
  "filter": "optional_filter_expression"
}
```

**Parameters:**
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

#### 3. Answer Question (Advanced)

Answer questions using sophisticated confidence analysis and structured responses.

**⭐ This endpoint calls `memory_agent.answer_question()` directly for the highest quality responses.**

**POST** `/api/memory/answer`

**Request Body:**
```json
{
  "question": "What did I eat at Mario's restaurant?",
  "top_k": 5,
  "filter": "optional_filter_expression"
}
```

**Parameters:**
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

#### 4. Get Memory Info

Get statistics about stored memories and system information.

**GET** `/api/memory`

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

#### 5. Context Management

Set and get current context for memory grounding.

##### Set Context

**POST** `/api/memory/context`

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

##### Get Context

**GET** `/api/memory/context`

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

#### 6. Delete Memory

Delete a specific memory by ID.

**DELETE** `/api/memory/{memory_id}`

**Response:**
```json
{
  "success": true,
  "message": "Memory 550e8400-e29b-41d4-a716-446655440000 deleted successfully",
  "memory_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

#### 7. Clear All Memories

Delete all memories from the system.

**DELETE** `/api/memory`

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

### Chat Application API

This endpoint provides a conversational interface using the LangGraph workflow for complex multi-step reasoning.

#### Chat Interface

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

**Key Differences from `/api/memory/answer`:**
- Uses full LangGraph workflow with tool orchestration
- Can perform multi-step reasoning and complex conversations
- More flexible but potentially higher latency
- Best for conversational UIs and complex queries

---

### System APIs

#### Health Check

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

### Error Responses

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

### API Usage Examples

#### Basic Memory Operations

```bash
# Store a memory
curl -X POST http://localhost:5001/api/memory \
  -H "Content-Type: application/json" \
  -d '{"text": "I love pizza with pepperoni"}'

# Search for memories
curl -X POST http://localhost:5001/api/memory/search \
  -H "Content-Type: application/json" \
  -d '{"query": "pizza", "top_k": 3}'

# Answer a question
curl -X POST http://localhost:5001/api/memory/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "What kind of pizza do I like?"}'
```

#### Context-Aware Memory Storage

```bash
# Set context first
curl -X POST http://localhost:5001/api/memory/context \
  -H "Content-Type: application/json" \
  -d '{
    "location": "New York",
    "activity": "dining",
    "people_present": ["Alice", "Bob"]
  }'

# Store memory (will be grounded with context)
curl -X POST http://localhost:5001/api/memory \
  -H "Content-Type: application/json" \
  -d '{"text": "We had an amazing dinner here"}'
```

#### Chat Interface

```bash
# Complex conversational query
curl -X POST http://localhost:5001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me about all my restaurant experiences and what I learned about food preferences"}'
```

### Command Line Interface Examples

```bash
# Store memories
Memory Agent> remember "I went to the mall to visit New Planet store"
Memory Agent> remember "Had lunch at Olive Garden with Sarah"

# Search memories
Memory Agent> recall "I'm going to the mall"
Memory Agent> recall "Where did I eat?"

# Ask questions
Memory Agent> ask "How much did I spend on my laptop?"
Memory Agent> ask "What did I eat with Sarah?"
```

### Web Interface

```bash
# Start the web UI
python web_app.py

# Open browser to http://localhost:5001
# Use the clean, modern interface to:
# - Store memories with the text area
# - Search memories with similarity scores
# - Ask questions and get AI-powered answers
```

---

## Developer Notes

### Memory vs Answer vs Chat

**Use `/api/memory/search`** when you need:
- Raw vector similarity search results
- Multiple memory candidates for further processing
- Building your own confidence analysis

**Use `/api/memory/answer`** when you need:
- High-quality question answering with confidence scores
- Structured responses with supporting evidence
- Single-step question answering

**Use `/api/chat`** when you need:
- Multi-step reasoning and complex conversations
- Tool orchestration and workflow management
- Conversational UI interfaces

### Performance Considerations

- `/api/memory/search`: Fastest, single vector search
- `/api/memory/answer`: Medium, includes LLM analysis for confidence
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

**Old API → New API:**
- `/api/remember` → `/api/memory`
- `/api/recall` → `/api/memory/search`
- `/api/ask` → `/api/memory/answer` (for structured responses) or `/api/chat` (for conversations)
- `/api/memory-info` → `/api/memory` (GET)
- `/api/context` → `/api/memory/context`
- `/api/delete/{id}` → `/api/memory/{id}` (DELETE)
- `/api/delete-all` → `/api/memory` (DELETE)

The new API provides cleaner, more RESTful endpoints with better separation of concerns between developer operations and chat applications.

### Testing

Run the test script to verify everything works:

```bash
python test_memory_agent.py
```

See `setup_redis.md` for detailed Redis setup instructions.

---

## License

MIT License

# Asynchronous Memory Processing System

## Overview

The Asynchronous Memory Processing System is designed to improve chat application performance by moving memory extraction and processing to background workers. This creates a hierarchical memory structure with better scalability and reduced latency.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   Chat Client   │───▶│  Raw Memory API  │───▶│   Redis Queue       │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
                                                           │
                                                           ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  Memory APIs    │◀───│ Background       │◀───│  Job Processor      │
│  (Retrieval)    │    │ Memory Processor │    │  (Scheduled)        │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
```

## Key Components

### 1. Raw Memory Storage API
**Endpoint:** `POST /api/memory/{vectorstore_name}/store_raw`

Accepts complete chat session histories and queues them for background processing.

**Request:**
```json
{
  "session_data": "Complete chat session text...",
  "session_id": "optional-session-id",
  "metadata": {
    "user_id": "user123",
    "session_type": "code_review",
    "timestamp": "2024-01-15T10:30:00Z"
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

### 2. Background Memory Processor
**Module:** `memory/async_processor.py`

Processes queued raw memories and creates hierarchical memory structures.

**Features:**
- Configurable processing intervals
- Automatic memory extraction using existing LLM prompts
- Session summary generation
- Cross-reference creation between memories and sessions
- Error handling and retry logic
- Data retention policies

**Usage:**
```bash
# Run continuously
python memory/async_processor.py

# Run once for testing
python memory/async_processor.py --run-once

# Custom configuration
python memory/async_processor.py --interval 30 --retention-days 60
```

### 3. Memory Hierarchy Structure

#### Discrete Memories
- Individual facts, preferences, and insights
- Stored in existing vectorsets with embeddings
- Example: "User prefers 4-space indentation", "User likes window seats"

#### Session Summaries
- High-level overviews of conversations
- Stored as: `{vectorstore_name}:session_summary:{session_id}`
- Include pointers to extracted memories

#### Raw Transcripts
- Complete chat logs stored temporarily
- Stored as: `{vectorstore_name}:raw_memory:{uuid}`
- Subject to configurable retention policies

### 4. Data Structures

#### Redis Keys
- **Raw Memory:** `{vectorstore_name}:raw_memory:{uuid}`
- **Processing Queue:** `RAW_MEMORY_QUEUE` (sorted set with timestamps)
- **Session Summary:** `{vectorstore_name}:session_summary:{session_id}`
- **Session Memories:** `{vectorstore_name}:session_memories:{session_id}` (set of memory IDs)
- **Processing Stats:** `processing_stats:{date}`
- **Processor Status:** `background_processor:status`

#### Raw Memory Record
```json
{
  "raw_memory_id": "uuid",
  "session_id": "session-uuid",
  "session_data": "complete chat text",
  "metadata": {...},
  "vectorstore_name": "vectorstore",
  "created_at": "2024-01-15T10:30:00Z",
  "status": "queued|processed|error",
  "processing_attempts": 0
}
```

## API Endpoints

### Processing Status
**GET** `/api/memory/{vectorstore_name}/processing_status`

Returns background processor status and queue information.

### Memory Hierarchy
**GET** `/api/memory/{vectorstore_name}/hierarchy`

Retrieves hierarchical memory data with filtering options.

**Query Parameters:**
- `session_id`: Filter by session ID
- `memory_type`: Filter by type ('discrete', 'summary', 'raw', 'all')
- `start_date`: Start date filter (ISO format)
- `end_date`: End date filter (ISO format)
- `limit`: Maximum results (1-1000, default: 50)

### Session Details
**GET** `/api/memory/{vectorstore_name}/session/{session_id}`

Get complete information about a specific session.

### Memory Statistics
**GET** `/api/memory/{vectorstore_name}/stats`

Comprehensive memory statistics including hierarchy breakdown.

### Manual Cleanup
**POST** `/api/memory/{vectorstore_name}/cleanup`

Manually trigger cleanup of expired data.

**Query Parameters:**
- `retention_days`: Days to retain raw transcripts (default: 30)

## Data Retention System

### Retention Policies
- **Raw Transcripts:** Configurable (default: 30 days)
- **Processed Sessions:** Long-term retention (default: 365 days)
- **Discrete Memories:** Permanent (managed by existing memory system)

### Cleanup Process
- Runs automatically during background processing
- Only deletes processed raw memories older than retention period
- Maintains references between memories and sessions
- Configurable cleanup intervals

## Performance Benefits

### Before (Synchronous)
- Memory extraction during chat sessions
- Blocking API calls
- Latency issues with complex conversations
- Resource contention

### After (Asynchronous)
- Immediate response to chat sessions
- Background processing with queuing
- Scalable worker architecture
- Better resource utilization

## Usage Examples

### 1. Store Raw Memory
```python
import requests

response = requests.post(
    "http://localhost:5001/api/memory/user123/store_raw",
    json={
        "session_data": "User: I prefer window seats...\nAssistant: Noted!...",
        "session_id": "session-123",
        "metadata": {"user_id": "user123", "session_type": "travel_planning"}
    }
)
```

### 2. Check Processing Status
```python
response = requests.get(
    "http://localhost:5001/api/memory/user123/processing_status"
)
status = response.json()
print(f"Queue size: {status['queue_size']}")
```

### 3. Retrieve Memory Hierarchy
```python
response = requests.get(
    "http://localhost:5001/api/memory/user123/hierarchy",
    params={"session_id": "session-123", "memory_type": "all"}
)
hierarchy = response.json()
```

## Testing

Run the comprehensive test suite:
```bash
python tests/test_async_memory.py
```

This tests:
- Raw memory storage
- Background processing
- Memory hierarchy retrieval
- Session details
- Statistics and cleanup

## Configuration

### Environment Variables
- `REDIS_HOST`: Redis server host (default: localhost)
- `REDIS_PORT`: Redis server port (default: 6379)
- `REDIS_DB`: Redis database number (default: 0)
- `OPENAI_API_KEY`: Required for memory extraction

### Background Processor Options
- `--interval`: Processing interval in seconds (default: 60)
- `--retention-days`: Raw transcript retention (default: 30)
- `--run-once`: Run single processing cycle and exit

## Monitoring

### Health Checks
- Processor heartbeat in Redis
- Queue size monitoring
- Processing statistics
- Error tracking

### Metrics
- Processed memories per day
- Queue processing time
- Memory extraction success rate
- Storage usage statistics

## Migration from Synchronous System

1. **Deploy async APIs** alongside existing endpoints
2. **Start background processor** with appropriate configuration
3. **Update chat clients** to use `/store_raw` endpoint
4. **Monitor processing** and adjust intervals as needed
5. **Gradually migrate** existing synchronous endpoints

## Troubleshooting

### Common Issues
- **Queue not processing:** Check if background processor is running
- **Memory extraction fails:** Verify OpenAI API key and LLM configuration
- **High memory usage:** Adjust retention policies and cleanup intervals
- **Slow processing:** Increase processor instances or reduce interval

### Debug Commands
```bash
# Check queue status
redis-cli ZCARD RAW_MEMORY_QUEUE

# View processor status
redis-cli HGETALL background_processor:status

# Manual processing
python memory/async_processor.py --run-once

# View processing stats
redis-cli HGETALL processing_stats:2024-01-15
```

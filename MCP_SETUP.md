# Memory Agent MCP Server Setup Guide

This guide will help you set up and configure the Memory Agent MCP (Model Context Protocol) server to work with Cursor and other MCP clients.

## 🚀 Quick Setup

### 1. Install Dependencies

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Start Redis

The memory agent requires Redis with the RedisSearch module. The easiest way is using Docker:

```bash
# Start Redis 8 with RedisSearch
docker run -d --name redis-memory -p 6379:6379 redis:8
```

Or if you have Redis installed locally:
```bash
redis-server --port 6379
```

### 3. Set Environment Variables

Create a `.env` file in the project root:

```bash
# Required: OpenAI API key for embeddings and LLM operations
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Redis configuration (defaults to localhost:6379)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
```

### 4. Test the MCP Server

Test the server in stdio mode (used by MCP clients):

```bash
python mcp_server.py stdio
```

Or test with HTTP mode for debugging:

```bash
python mcp_server.py
# Server will run on http://localhost:8000
```

## 🎯 Connecting to Cursor

### Method 1: Direct Configuration

Add this to your Cursor settings (⌘ + , → Extensions → MCP):

```json
{
  "mcpServers": {
    "memory-agent": {
      "command": "python",
      "args": ["mcp_server.py", "stdio"],
      "cwd": "/absolute/path/to/your/project",
      "env": {
        "OPENAI_API_KEY": "your_openai_api_key_here",
        "REDIS_HOST": "localhost",
        "REDIS_PORT": "6379"
      }
    }
  }
}
```

### Method 2: Using the Configuration File

1. Copy `mcp_config.json` to your desired location
2. Update the `cwd` path to point to your project directory
3. Set your OpenAI API key in the environment variables
4. Import the configuration in Cursor

## 🧠 Available MCP Tools

Once connected, Cursor will have access to these memory tools:

### Core Memory Operations
- **store_memory**: Store new memories with contextual grounding
- **search_memories**: Find relevant memories using vector similarity
- **answer_question**: Answer questions using K-line reasoning with confidence analysis
- **recall_memories**: Construct mental states from relevant memories

### Context Management
- **set_context**: Set current location, activity, and environmental context
- **get_memory_stats**: View memory system statistics and current context

### Advanced Operations
- **extract_and_store_memories**: Extract significant information from conversations
- **chat_with_memory**: Have conversations that automatically manage memory
- **delete_memory**: Remove specific memories
- **clear_all_memories**: Clear all stored memories (with confirmation)

### MCP Resources
- **memory://stats**: Memory system statistics
- **memory://context**: Current context settings
- **memory://recent**: Recently stored memories

### MCP Prompts
- **memory_guided_conversation**: Start conversations with memory context
- **memory_summary**: Generate summaries of memories about specific topics

## 🔧 Advanced Configuration

### Custom Vectorset

To use a custom vectorset name (useful for multiple projects):

```python
# In your code
from memory.agent import LangGraphMemoryAgent
agent = LangGraphMemoryAgent(vectorset_key="my_project_memories")
```

### Redis Configuration

For remote Redis or custom configuration:

```bash
# .env file
REDIS_HOST=your-redis-host.com
REDIS_PORT=6380
REDIS_DB=1
REDIS_PASSWORD=your_password  # if needed
```

### OpenAI Model Configuration

The system uses GPT-3.5-turbo by default. To change models, modify the LangGraphMemoryAgent initialization in `mcp_server.py`:

```python
memory_agent = LangGraphMemoryAgent(
    model_name="gpt-4",  # or "gpt-4-turbo"
    temperature=0.1
)
```

## 🧪 Testing the Setup

### 1. Test Memory Storage

```bash
# In Cursor, use the MCP tool:
store_memory: "I prefer Italian restaurants and am allergic to shellfish"
```

### 2. Test Memory Search

```bash
# Search for relevant memories:
search_memories: "restaurant preferences"
```

### 3. Test Question Answering

```bash
# Ask questions about stored memories:
answer_question: "What are my dietary restrictions?"
```

### 4. Test Conversation with Memory

```bash
# Have a conversation that uses memory context:
chat_with_memory: "Can you recommend a good restaurant for dinner?"
```

## 🐛 Troubleshooting

### Redis Connection Issues

```bash
# Check if Redis is running
redis-cli ping
# Should return: PONG

# Check Redis modules
redis-cli MODULE LIST
# Should show: search, json (for Redis Stack)
```

### OpenAI API Issues

```bash
# Test your API key
python -c "
import openai
import os
from dotenv import load_dotenv
load_dotenv()
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
print('API key is valid!')
"
```

### MCP Connection Issues

1. Ensure the `cwd` path in your MCP configuration points to the correct project directory
2. Check that Python can find all dependencies from that directory
3. Verify environment variables are set correctly
4. Check Cursor's MCP logs for error messages

### Memory Agent Issues

```bash
# Test the memory agent directly
python main.py "Remember that I like pizza"
```

## 🎯 Usage Examples in Cursor

Once connected, you can use the memory system in Cursor:

```
"Store this memory: I'm working on a React project with TypeScript and prefer using hooks over class components"

"Search my memories for: React preferences"

"What do you remember about my coding preferences?"

"Set my current context: location='San Francisco office' activity='coding a new feature' people_present=['Sarah', 'Mike']"
```

## 📚 Memory Architecture

The system is based on Marvin Minsky's Society of Mind theory:

- **Nemes**: Atomic memory units (individual memories)
- **K-lines**: Mental states constructed from relevant memories
- **Contextual Grounding**: Automatic addition of time/location context
- **Vector Similarity**: Semantic search using OpenAI embeddings
- **Confidence Analysis**: Sophisticated reasoning about answer quality

## 🔒 Privacy & Security

- All memories are stored locally in your Redis instance
- No data is sent to external services except OpenAI for embeddings/reasoning
- You have full control over memory storage and deletion
- Context information helps with memory organization but can be disabled

---

## 🆘 Need Help?

1. Check that Redis is running and accessible
2. Verify your OpenAI API key is valid and has credits
3. Ensure all Python dependencies are installed
4. Check the MCP server logs for error messages
5. Test the memory agent directly with `python main.py`

For more advanced usage, see the main project README.md for detailed API documentation.
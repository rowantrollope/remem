# Remem Memory Agent MCP Server

This is a Model Context Protocol (MCP) server implementation for the Remem Memory Agent. It exposes the sophisticated memory system capabilities as standardized tools that can be used by any MCP-compatible client like Claude Desktop, IDEs, or other AI applications.

## What is MCP?

Model Context Protocol (MCP) is an open protocol that standardizes how applications provide context to LLMs. Think of MCP like a USB-C port for AI applications - it provides a standardized way to connect AI models to different data sources and tools.

## Features

The Remem MCP Server provides the following tools:

### Core Memory Operations
- **store_memory**: Store new memories (Memories) with contextual grounding
- **search_memories**: Search for relevant memories using vector similarity
- **get_memory_stats**: Get statistics about the memory system
- **answer_question**: Answer questions using the memory system with confidence scoring

### Advanced Operations
- **extract_and_store_memories**: Extract and store memories from conversations or text
- **set_context**: Set contextual information for memory grounding

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key and Redis configuration
   ```

3. **Start Redis** (if not already running):
   ```bash
   docker run -d --name redis -p 6379:6379 redis:8
   ```

## Usage with Claude Desktop

### 1. Install Claude Desktop
Download and install Claude Desktop from [claude.ai/download](https://claude.ai/download).

### 2. Configure Claude Desktop
Create or edit the Claude Desktop configuration file:

**macOS/Linux**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

Use the example configuration:

```json
{
  "mcpServers": {
    "remem-memory": {
      "command": "python",
      "args": ["/ABSOLUTE/PATH/TO/remem/mcp_server.py"],
      "env": {
        "OPENAI_API_KEY": "your-openai-api-key-here",
        "REDIS_HOST": "localhost",
        "REDIS_PORT": "6379",
        "REDIS_DB": "0"
      }
    }
  }
}
```

**Important**: Replace `/ABSOLUTE/PATH/TO/remem/` with the actual absolute path to your remem directory.

### 3. Restart Claude Desktop
After saving the configuration, restart Claude Desktop completely.

### 4. Test the Integration
Look for the "Search and tools" icon in Claude Desktop. You should see the remem memory tools listed.

Try these example commands:
- "Store this memory: I prefer coffee over tea"
- "What do I prefer to drink?"
- "Search my memories for food preferences"
- "What's the current state of my memory system?"

## Tool Reference

### store_memory
Store a new memory in the system.

**Parameters**:
- `text` (required): The memory content to store
- `vectorstore_name` (required): Vectorstore name
- `apply_grounding` (optional): Apply contextual grounding (default: true)

**Example**: "Store this memory: I met John at the coffee shop yesterday"

### search_memories
Search for relevant memories using vector similarity.

**Parameters**:
- `query` (required): Search query text
- `vectorstore_name` (required): Vectorstore name
- `top_k` (optional): Number of results (default: 5)
- `min_similarity` (optional): Minimum similarity threshold (default: 0.7)

**Example**: "Search my memories for coffee shops"

### get_memory_stats
Get statistics about the memory system.

**Parameters**:
- `vectorstore_name` (required): Vectorstore name

**Example**: "What's the current state of my memory system?"

### answer_question
Answer a question using the memory system with confidence scoring.

**Parameters**:
- `question` (required): The question to answer
- `vectorstore_name` (required): Vectorstore name
- `top_k` (optional): Number of memories to consider (default: 5)
- `confidence_threshold` (optional): Minimum confidence threshold (default: 0.7)

**Example**: "What restaurants have I been to?"

### extract_and_store_memories
Extract and store memories from conversation text.

**Parameters**:
- `conversation_text` (required): Text to extract memories from
- `vectorstore_name` (required): Vectorstore name
- `apply_grounding` (optional): Apply contextual grounding (default: true)

**Example**: "Extract memories from this conversation: [conversation text]"

### set_context
Set contextual information for memory grounding.

**Parameters**:
- `vectorstore_name` (required): Vectorstore name
- `location` (optional): Current location
- `activity` (optional): Current activity
- `people_present` (optional): People present (comma-separated)
- `environment` (optional): Environmental context

**Example**: "Set my context: location=home, activity=working, environment=quiet"

## Architecture

The MCP server acts as a bridge between MCP clients and the Remem memory system:

```
MCP Client (Claude Desktop) <-> MCP Server <-> Remem Memory Agent <-> Redis VectorSet
```

The server uses the FastMCP framework for easy tool definition and automatic schema generation from Python type hints and docstrings.

## Troubleshooting

### Server not showing up in Claude Desktop
1. Check the configuration file syntax
2. Ensure the path to `mcp_server.py` is absolute
3. Verify environment variables are set correctly
4. Check Claude Desktop logs: `~/Library/Logs/Claude/mcp*.log`

### Tool calls failing
1. Ensure Redis is running and accessible
2. Verify OpenAI API key is valid
3. Check server logs for error messages
4. Test the memory agent directly with `python cli.py`

### Memory operations not working
1. Verify Redis connection settings
2. Check if the vectorstore exists and is accessible
3. Ensure sufficient OpenAI API credits
4. Test with simple operations first

## Development

To extend the MCP server with additional tools:

1. Add new tool functions using the `@mcp.tool()` decorator
2. Use proper type hints and docstrings for automatic schema generation
3. Handle errors gracefully and return informative messages
4. Test with Claude Desktop or other MCP clients

## Related Documentation

- [Model Context Protocol Documentation](https://modelcontextprotocol.io/)
- [Remem Memory Agent Documentation](./README.md)
- [FastMCP Documentation](https://github.com/modelcontextprotocol/python-sdk)

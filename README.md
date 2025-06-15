# AI Agent Collection

This repository contains two AI agent implementations:

1. **LangGraph Basic Agent** - A tool-using conversational agent with OpenAI
2. **Memory Agent** - A Redis-based memory system with vector embeddings

## 🤖 LangGraph Basic Agent

A basic LangGraph agent implementation using OpenAI's GPT models. This agent can use tools and maintain conversation state through a graph-based workflow.

## Features

- **LangGraph Integration**: Uses LangGraph for workflow management
- **OpenAI Integration**: Powered by OpenAI's GPT models
- **Tool Support**: Includes sample tools (weather, calculator, web search)
- **Interactive Chat**: Command-line chat interface
- **Extensible**: Easy to add new tools and modify behavior

## Setup

### 1. Create Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Copy the example environment file and add your OpenAI API key:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Run the Agent

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
python main.py "What's the weather like in New York?"
```

## Project Structure

- `main.py` - Entry point for running the agent
- `agent.py` - Main LangGraph agent implementation
- `tools.py` - Custom tools for the agent
- `requirements.txt` - Python dependencies
- `.env.example` - Environment variables template

## How It Works

The agent uses LangGraph to create a workflow with two main nodes:

1. **Agent Node**: Calls the OpenAI model to generate responses
2. **Tools Node**: Executes any tools requested by the model

The workflow follows this pattern:
1. User provides input
2. Agent processes input and decides whether to use tools
3. If tools are needed, they are executed
4. Results are fed back to the agent
5. Agent provides final response

## Available Tools

- `get_weather(city)` - Get weather information (mock implementation)
- `calculate(expression)` - Perform mathematical calculations
- `search_web(query)` - Search the web (mock implementation)

## Customization

### Adding New Tools

1. Create a new tool function in `tools.py`:

```python
@tool
def my_new_tool(param: str) -> str:
    """Description of what the tool does."""
    # Your implementation here
    return "result"
```

2. Add it to the `AVAILABLE_TOOLS` list in `tools.py`

### Modifying the Agent

The agent behavior can be customized by modifying the `LangGraphAgent` class in `agent.py`. You can:

- Change the system prompt
- Modify the workflow graph
- Add new nodes or edges
- Customize the model parameters

## Requirements

- Python 3.8+
- OpenAI API key
- Internet connection for API calls

---

## 🧠 Memory Agent

A Redis-based memory system that can store and retrieve memories using OpenAI embeddings for vector similarity search.

### Features

- **Memory Storage**: Store memories with automatic tag extraction and timestamping
- **Vector Search**: Find relevant memories using semantic similarity
- **Question Answering**: AI-powered answers with confidence indicators
- **Redis VectorSet**: Uses Redis VectorSet API for efficient vector operations
- **Web Interface**: Clean, minimalist web UI with responsive design
- **CLI Interface**: Simple command-line interface for storing and recalling memories

### Setup

1. **Install Redis Stack** (includes RedisSearch):
   ```bash
   # Using Docker (recommended)
   docker run -d --name redis-stack -p 6379:6379 redis/redis-stack:latest

   # Or using Homebrew (macOS)
   brew tap redis-stack/redis-stack
   brew install redis-stack
   redis-stack-server
   ```

2. **Install Dependencies**:
   ```bash
   source venv/bin/activate
   pip install redis numpy
   ```

3. **Run the Memory Agent**:
   ```bash
   python memory_agent.py
   ```

### Usage Examples

#### Command Line Interface
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

#### Web Interface
```bash
# Start the web UI
python web_app.py

# Open browser to http://localhost:5001
# Use the clean, modern interface to:
# - Store memories with the text area
# - Search memories with similarity scores
# - Ask questions and get AI-powered answers
```

### Testing

Run the test script to verify everything works:

```bash
python test_memory_agent.py
```

See `setup_redis.md` for detailed Redis setup instructions.

---

## License

MIT License

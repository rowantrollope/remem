# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a sophisticated Agent Memory API powered by Redis VectorSet with multiple interfaces: CLI, Web API, LangGraph workflow, and Model Context Protocol (MCP) server. The system provides intelligent memory storage, retrieval, and analysis using Redis VectorSet and configurable embeddings (OpenAI, Ollama).

## Development Commands

### Essential Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
make test
# or: python -m pytest tests/ -v

# Code quality checks
make lint           # Linting with flake8
make type-check     # Type checking with mypy
make format         # Format with black and isort
make check          # Run all quality checks

# Start Redis (required for all operations)
make redis-start    # Start Redis 8 with Docker
# or: docker run -d --name remem-redis -p 6379:6379 redis:8

# Run applications
make run            # Web API server on port 5001
make run-cli        # CLI interface
make run-mcp        # MCP server
make run-dev        # Development mode with auto-reload
```

### Frontend Development
```bash
cd frontend/
npm run dev         # Next.js dev server with turbopack
npm run build       # Production build
npm run lint        # ESLint
```

### Environment Setup
```bash
# Copy environment template and configure
cp .env.example .env
# Edit .env with OPENAI_API_KEY and other settings
```

## Architecture Overview

### Core System
- **memory/core.py**: Core memory operations with Redis VectorSet
- **memory/agent.py**: Main LangGraph memory agent workflow
- **memory/tools.py**: LangChain tools for memory operations
- **api/**: FastAPI web server with REST endpoints
- **cli.py**: Command-line interface
- **web_app.py**: Web API server entry point
- **mcp_server.py**: Model Context Protocol server

### Key Components
- **Vector Storage**: Redis VectorSet for semantic similarity search
- **Embeddings**: Configurable providers (OpenAI text-embedding-3-small, Ollama nomic-embed-text)
- **LangGraph Workflow**: Multi-step reasoning with tool orchestration
- **Contextual Grounding**: Automatic resolution of relative time/location references
- **Configuration Management**: Runtime configuration via API endpoints

### Frontend
- **Next.js application** in `frontend/` directory
- **React components** with Radix UI and Tailwind CSS
- **TypeScript** with comprehensive type definitions

## API Structure

### Memory Operations (require vectorstore_name path parameter)
- `POST /api/memory/{vectorstore_name}` - Store memory
- `POST /api/memory/{vectorstore_name}/search` - Search memories
- `POST /api/memory/{vectorstore_name}/answer` - Answer questions with confidence
- `GET /api/memory/{vectorstore_name}` - Get memory stats
- `DELETE /api/memory/{vectorstore_name}/{memory_id}` - Delete memory

### Chat & K-lines
- `POST /api/chat` - LangGraph conversational interface
- `POST /api/klines/{vectorstore_name}/recall` - Mental state construction
- `POST /api/klines/{vectorstore_name}/ask` - K-line reasoning

### Configuration
- `GET /api/config` - Get current configuration
- `PUT /api/config` - Update configuration
- `POST /api/config/test` - Test configuration changes

## Testing

```bash
# Core functionality tests
python tests/test_memory_agent.py
python tests/test_api.py
python tests/test_config_api.py

# CLI integration tests
python tests/test_cli_integration.py

# Run all tests
make test
```

## Dependencies

- **Redis 8+** with VectorSet support (critical requirement)
- **Python 3.10+** with specific package versions in pyproject.toml
- **OpenAI API key** for embeddings (or configure Ollama alternative)
- **Node.js** for frontend development

## Configuration

The system supports runtime configuration changes via API. Key configurable components:
- Redis connection settings
- Embedding provider (OpenAI/Ollama) and models
- LangGraph model selection and parameters
- Memory agent search and grounding settings

## Entry Points

1. **CLI**: `python cli.py` - Interactive memory operations
2. **Web API**: `python web_app.py` - REST API + web interface on port 5001
3. **MCP Server**: `python mcp_server.py` - Universal AI integration
4. **LangGraph**: Import `LangGraphMemoryAgent` for programmatic use
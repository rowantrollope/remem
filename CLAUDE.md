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
The architecture uses a layered modular design with clean separation of concerns:

#### Memory Layer (`memory/`)
- **core.py**: Redis VectorSet operations and low-level memory management
- **core_agent.py**: Layered memory agent with processing, extraction, and reasoning
- **agent.py**: OpenAI SDK-based conversational agent with direct function calling
- **tools.py**: OpenAI function calling tools for memory operations
- **processing.py**: Memory data processing and transformation
- **extraction.py**: Memory extraction from conversations
- **reasoning.py**: Advanced reasoning operations
- **async_processor.py**: Asynchronous memory processing capabilities

#### API Layer (`api/`)
- **app.py**: FastAPI application factory and configuration
- **startup.py**: Application initialization and service setup
- **dependencies.py**: Dependency injection for services
- **routers/**: Modular API endpoints
  - **memory.py**: Basic memory CRUD operations
  - **klines.py**: K-line mental state operations (inspired by Minsky)
  - **agent.py**: Full conversational agent with session management
  - **config.py**: Runtime configuration management
  - **health.py**: Health check endpoints
- **services/**: Business logic layer
  - **memory_service.py**: Memory operation orchestration
  - **agent_service.py**: Agent conversation handling
  - **config_service.py**: Configuration management
- **models/**: Pydantic data models for API contracts
- **core/**: Shared utilities, exceptions, and constants

#### Application Entry Points
- **web_app.py**: Main web API server (FastAPI)
- **cli.py**: Command-line interface
- **mcp_server.py**: Model Context Protocol server

#### Supporting Systems
- **llm/**: LLM configuration and model management
- **embedding/**: Embedding provider abstraction (OpenAI/Ollama)
- **clients/**: External service integrations (LangCache)

### Key Features
- **Modular Architecture**: Clean separation between memory, API, and business logic layers
- **Vector Storage**: Redis VectorSet for semantic similarity search
- **Configurable Embeddings**: Support for OpenAI and Ollama embedding providers
- **K-lines Architecture**: Mental state construction inspired by Minsky's cognitive theory
- **Session Management**: Persistent conversational context
- **Async Processing**: Background memory operations
- **Runtime Configuration**: Dynamic system reconfiguration via API

### Frontend
- **Next.js application** in `frontend/` directory
- **React components** with Radix UI and Tailwind CSS
- **TypeScript** with comprehensive type definitions
- **Real-time chat interface** with memory integration

## API Structure

### Memory Operations (require vectorstore_name path parameter)
- `POST /api/memory/{vectorstore_name}` - Store memory
- `POST /api/memory/{vectorstore_name}/search` - Search memories
- `GET /api/memory/{vectorstore_name}` - Get memory stats
- `DELETE /api/memory/{vectorstore_name}/{memory_id}` - Delete specific memory
- `DELETE /api/memory/{vectorstore_name}/all` - Delete all memories
- `POST /api/memory/{vectorstore_name}/context` - Store contextual memories
- `GET /api/memory/{vectorstore_name}/context` - Get contextual information

### Agent Operations
- `POST /api/agent/chat` - Full conversational agent with memory integration
- `POST /api/agent/{vectorstore_name}/session` - Create new chat session
- `GET /api/agent/{vectorstore_name}/session/{session_id}` - Get session details
- `DELETE /api/agent/{vectorstore_name}/session/{session_id}` - Delete session
- `GET /api/agent/{vectorstore_name}/sessions` - List all sessions

### K-lines Operations (Mental State Construction)
- `POST /api/memory/{vectorstore_name}/ask` - K-line reasoning with confidence scoring

### Configuration Management
- `GET /api/config` - Get current configuration
- `PUT /api/config` - Update configuration
- `POST /api/config/test` - Test configuration changes
- `POST /api/config/reload` - Reload configuration

### Health & Monitoring
- `GET /api/health` - System health check

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
- ChatOpenAI model selection and parameters
- Memory agent search and grounding settings

## Memory Usage Instructions for Claude Code

When working with this codebase, Claude Code has access to the REMEM-MEMORY system via MCP. Use these instructions for when and how to leverage memory capabilities.

### Vectorstore Naming Convention
- `code_agent:global` - Universal coding preferences, patterns, and cross-project insights
- `code_agent:remem` - This project's specific context, decisions, and patterns
- `code_agent:[project]` - Other project-specific contexts when working across codebases

### When to Consult Memory

**Session Start:**
- Search `coding_memory:global` for coding preferences and general patterns
- Search `coding_memory:remem` for project-specific context and previous decisions
- Use findings to inform responses throughout the session

**During Development:**
- Before making architectural decisions, check for relevant prior context
- When user mentions unfamiliar project concepts, search for background
- For debugging, look for similar issues previously encountered
- Balance utility with performance - use targeted searches

### Memory Storage Strategy - Store Valuable Insights Only

**What TO Store:**
- User-stated preferences ONLY ("I prefer...", "Always use...", "Don't...")
- Specific architectural decisions the user has made for this project
- Performance insights discovered through actual implementation work
- Configuration choices the user has explicitly selected
- User-requested coding patterns or standards

**What NOT to Store:**
- Explanations or educational content provided to the user
- Comparisons between technologies (unless user chose one)
- Code reviews, bug fixes, or routine development work
- Standard patterns without explicit user preference
- Process descriptions or how-to information
- General technical knowledge or framework explanations

**Storage Examples:**
- ✅ "User prefers detailed comments above each function" (explicit preference)
- ✅ "User chose Redis VectorSet over alternatives for performance reasons" (user decision)
- ❌ "Explained OpenAI SDK vs LangGraph differences" (educational content)
- ❌ "Provided comparison between agent architectures" (explanation)
- ❌ "Fixed type errors in memory module" (routine work)

### Best Practices
1. **Be Selective:** Only store insights that improve future development
2. **Focus on Decisions:** Prioritize architectural choices and user preferences
3. **Project Context:** Remember this is both a memory tool AND a project using memory
4. **Be Specific:** Include enough context for future reference
5. **Stay Organized:** Use appropriate vectorstore names consistently

## Entry Points

1. **CLI**: `python cli.py` - Interactive memory operations
2. **Web API**: `python web_app.py` - REST API + web interface on port 5001
3. **MCP Server**: `python mcp_server.py` - Universal AI integration
4. **Programmatic**: Import `MemoryAgentChat` for direct use
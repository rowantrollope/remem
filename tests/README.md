# Test Suite

This directory contains comprehensive tests for the remem memory system.

## Core Test Files

### CLI Tests
- **`test_cli.py`** - Basic CLI functionality tests (command processing, help, interactive mode, memory operations, error handling, vectorstore selection, environment variables, input validation, performance)
- **`test_cli_integration.py`** - Integration tests requiring real API access (full conversation flows, concurrent instances, real LLM interactions, end-to-end persistence)

### API Tests
- **`test_api.py`** - Basic API endpoint testing
- **`test_chat_api.py`** - Chat API functionality for frontend developers
- **`test_new_api.py`** - Minsky-inspired three-layer API architecture (Memory/K-line/Agent APIs)
- **`test_config_api.py`** - Configuration API testing

### Memory System Tests
- **`test_memory_agent.py`** - Core memory agent functionality
- **`test_async_memory.py`** - Asynchronous memory processing
- **`test_mcp_server.py`** - MCP server functionality

### Performance & Caching Tests
- **`test_performance.py`** - Performance optimizer testing
- **`test_performance_optimizations.py`** - Performance optimization validation
- **`test_langcache.py`** - LangCache client integration
- **`test_langcache_config.py`** - LangCache configuration controls

### Specialized Tests
- **`test_llm_config.py`** - LLM configuration testing
- **`test_langgraph_memory_agent.py`** - LangGraph memory agent integration

### Test Runner
- **`run_cli_tests.py`** - Test runner script (prerequisite checking, selective execution, comprehensive reporting)

## Prerequisites

### Required
- **OpenAI API Key**: Set `OPENAI_API_KEY` environment variable
- **Redis Server**: Running on localhost:6379 (or configured via environment)
- **Python Packages**: 
  - `redis`
  - `python-dotenv`
  - Local memory modules

### Optional
- **pexpect**: For interactive session testing (`pip install pexpect`)

## Running Tests

### Quick Start
```bash
# Run all tests
python run_cli_tests.py

# Run only basic tests (no API calls)
python run_cli_tests.py --basic

# Run only integration tests (requires API)
python run_cli_tests.py --integration
```

### Individual Test Files
```bash
# Run basic tests directly
python tests/test_cli.py

# Run integration tests directly
python tests/test_cli_integration.py
```

### Environment Setup
```bash
# Copy example environment file
cp .env.example .env

# Edit .env to add your OpenAI API key
# OPENAI_API_KEY=your_key_here

# Ensure Redis is running
redis-server

# Install optional dependencies
pip install pexpect
```

## Test Categories

### 1. Command Line Interface Tests
- Help command variations (`help`, `--help`, `-h`)
- Single query processing
- Argument parsing
- Exit codes

### 2. Interactive Mode Tests
- Command recognition (`/help`, `/stats`, `/profile`, etc.)
- Memory commands (`remember`, `remember-raw`)
- Session management
- Graceful exit

### 3. Memory System Tests
- Memory storage with and without grounding
- Memory retrieval and search
- Vectorstore isolation
- Statistics reporting

### 4. Error Handling Tests
- Invalid inputs
- Missing dependencies
- Network failures
- Timeout scenarios

### 5. Performance Tests
- Rapid memory operations
- Concurrent access
- Large input handling
- Response time measurement

### 6. Integration Tests
- Full conversation flows
- Multi-step memory building
- Real API interactions
- Concurrent CLI instances

## Test Output

Tests provide detailed output including:
- ✅ Success indicators
- ❌ Failure indicators  
- ⚠️ Warning indicators
- Performance metrics
- Error details

### Example Output
```
🚀 Comprehensive CLI End-to-End Test Suite
============================================================

🧪 Testing Command Line Help
----------------------------------------
Testing: python main.py help
✅ Help command 'help' works
✅ Help content is complete
✅ Interactive command '/help' documented
...

📊 Results: 8 passed, 0 failed
⏱️ Total time: 45.23s

🎉 All CLI tests passed!
```

## Troubleshooting

### Common Issues

**Redis Connection Failed**
```bash
# Start Redis server
redis-server

# Check Redis is running
redis-cli ping
```

**OpenAI API Key Missing**
```bash
# Check environment variable
echo $OPENAI_API_KEY

# Set in .env file
echo "OPENAI_API_KEY=your_key_here" >> .env
```

**Import Errors**
```bash
# Install required packages
pip install -r requirements.txt

# Install optional packages
pip install pexpect
```

**Tests Timeout**
- Check network connectivity
- Verify OpenAI API key is valid
- Ensure Redis is responsive

### Debugging Tests

Enable debug output:
```bash
# Run with debug environment
MEMORY_DEBUG=true python tests/test_cli.py

# Run with verbose output
MEMORY_VERBOSE=true python tests/test_cli.py
```

## Contributing

When adding new tests:

1. **Basic tests** go in `test_cli.py`
   - No external API calls
   - Fast execution
   - Mock external dependencies when possible

2. **Integration tests** go in `test_cli_integration.py`
   - Real API interactions
   - End-to-end scenarios
   - Longer execution time acceptable

3. **Update test runner** if needed
   - Add new test categories
   - Update prerequisite checks
   - Enhance reporting

### Test Guidelines

- Use descriptive test names
- Include both positive and negative test cases
- Provide clear success/failure indicators
- Clean up test data when possible
- Handle timeouts gracefully
- Document any special requirements

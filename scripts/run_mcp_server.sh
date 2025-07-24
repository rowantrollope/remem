#!/bin/bash
# Wrapper script to run MCP server with virtual environment

# Get the directory where this script is located and go to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Activate virtual environment and run the server
source .venv/bin/activate && python3 mcp_server.py

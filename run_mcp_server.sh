#!/bin/bash
# Wrapper script to run MCP server with virtual environment

# Change to the project directory
cd /Users/rowantrollope/git/remem

# Activate virtual environment and run the server
source venv/bin/activate && python3 mcp_server.py

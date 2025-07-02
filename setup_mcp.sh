#!/bin/bash

# Memory Agent MCP Server Setup Script
# Creates virtual environment and installs all dependencies

echo "🧠 Memory Agent MCP Server Setup"
echo "================================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate
echo "✅ Virtual environment activated"

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install core dependencies
echo "📦 Installing core dependencies..."
pip install -r requirements.txt

# Install MCP dependencies
echo "📦 Installing MCP dependencies..."
pip install mcp fastapi uvicorn pydantic

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To use the MCP server:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Start Redis: docker run -d --name redis-memory -p 6379:6379 redis:8"
echo "3. Create .env file with your OPENAI_API_KEY"
echo "4. Test the server: python start_mcp_server.py stdio"
echo ""
echo "For Cursor integration, see MCP_SETUP.md"
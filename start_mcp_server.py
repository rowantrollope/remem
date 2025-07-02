#!/usr/bin/env python3
"""
MCP Server Startup Script for Memory Agent

Simple startup script that handles dependencies and provides clear error messages.
"""

import sys
import os
import subprocess

def check_dependencies():
    """Check if required dependencies are installed."""
    missing_deps = []
    
    try:
        import dotenv
    except ImportError:
        missing_deps.append("python-dotenv")
    
    try:
        import redis
    except ImportError:
        missing_deps.append("redis")
    
    try:
        import openai
    except ImportError:
        missing_deps.append("openai")
    
    try:
        import langchain
    except ImportError:
        missing_deps.append("langchain")
    
    try:
        import langgraph
    except ImportError:
        missing_deps.append("langgraph")
    
    return missing_deps

def check_mcp_dependencies():
    """Check if MCP dependencies are installed."""
    missing_mcp = []
    
    try:
        import mcp
    except ImportError:
        missing_mcp.append("mcp")
    
    try:
        import uvicorn
    except ImportError:
        missing_mcp.append("uvicorn")
    
    try:
        import fastapi
    except ImportError:
        missing_mcp.append("fastapi")
    
    try:
        import pydantic
    except ImportError:
        missing_mcp.append("pydantic")
    
    return missing_mcp

def check_redis_connection():
    """Check if Redis is accessible."""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        return True
    except Exception as e:
        return str(e)

def check_openai_key():
    """Check if OpenAI API key is available."""
    from dotenv import load_dotenv
    load_dotenv()
    
    return os.getenv("OPENAI_API_KEY") is not None

def main():
    """Main startup function."""
    print("🧠 Memory Agent MCP Server Startup")
    print("=" * 50)
    
    # Check basic dependencies
    missing_deps = check_dependencies()
    if missing_deps:
        print("❌ Missing core dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\n💡 Install with: pip install -r requirements.txt")
        return 1
    
    # Check MCP dependencies
    missing_mcp = check_mcp_dependencies()
    if missing_mcp:
        print("❌ Missing MCP dependencies:")
        for dep in missing_mcp:
            print(f"   - {dep}")
        print("\n💡 Install with: pip install mcp fastapi uvicorn pydantic")
        return 1
    
    print("✅ All dependencies found")
    
    # Check Redis connection
    redis_status = check_redis_connection()
    if redis_status is not True:
        print(f"❌ Redis connection failed: {redis_status}")
        print("\n💡 Start Redis with: docker run -d --name redis-memory -p 6379:6379 redis:8")
        return 1
    
    print("✅ Redis connection successful")
    
    # Check OpenAI API key
    if not check_openai_key():
        print("❌ OpenAI API key not found")
        print("💡 Create a .env file with: OPENAI_API_KEY=your_key_here")
        return 1
    
    print("✅ OpenAI API key found")
    
    # Check if memory agent can be imported
    try:
        from memory.agent import LangGraphMemoryAgent
        print("✅ Memory agent imported successfully")
    except Exception as e:
        print(f"❌ Failed to import memory agent: {e}")
        return 1
    
    # Start the MCP server
    print("\n🚀 Starting MCP server...")
    
    # Determine mode from arguments
    if len(sys.argv) > 1 and sys.argv[1] == "stdio":
        print("📡 Using stdio transport (for MCP clients)")
    else:
        print("🌐 Using HTTP transport for testing")
        print("💡 For MCP clients, use: python start_mcp_server.py stdio")
    
    try:
        # Import and run the actual MCP server
        from mcp_server import main
        import asyncio
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 MCP server stopped")
        return 0
    except Exception as e:
        print(f"❌ Server error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
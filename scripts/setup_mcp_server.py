#!/usr/bin/env python3
"""
Setup script for the Remem MCP Server

This script helps users set up the MCP server for use with Claude Desktop
and other MCP-compatible clients.
"""

import os
import sys
import json
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 10):
        print("❌ Python 3.10 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_redis():
    """Check if Redis is accessible."""
    try:
        import redis
        r = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=int(os.getenv('REDIS_DB', 0))
        )
        r.ping()
        print("✅ Redis connection successful")
        return True
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print("Please ensure Redis is running:")
        print("  docker run -d --name redis -p 6379:6379 redis:8")
        return False

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_env_file():
    """Check if .env file exists and has required variables."""
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found")
        print("Please copy .env.example to .env and set your OpenAI API key")
        return False
    
    # Check for required environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in .env file")
        return False
    
    print("✅ Environment variables configured")
    return True

def get_claude_config_path():
    """Get the Claude Desktop configuration file path."""
    system = platform.system()
    if system == "Darwin":  # macOS
        return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    elif system == "Windows":
        return Path.home() / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json"
    else:  # Linux
        print("⚠️  Claude Desktop is not available on Linux")
        print("You can use the MCP server with other MCP-compatible clients")
        return None

def create_claude_config():
    """Create or update Claude Desktop configuration."""
    config_path = get_claude_config_path()
    if not config_path:
        return False
    
    # Get absolute path to the MCP server
    server_path = Path(__file__).parent.absolute() / "mcp_server.py"
    
    # Create the configuration
    config = {
        "mcpServers": {
            "remem-memory": {
                "command": "python",
                "args": [str(server_path)],
                "env": {
                    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
                    "REDIS_HOST": os.getenv("REDIS_HOST", "localhost"),
                    "REDIS_PORT": os.getenv("REDIS_PORT", "6379"),
                    "REDIS_DB": os.getenv("REDIS_DB", "0")
                }
            }
        }
    }
    
    # Check if config file already exists
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                existing_config = json.load(f)
            
            # Merge configurations
            if "mcpServers" not in existing_config:
                existing_config["mcpServers"] = {}
            
            existing_config["mcpServers"]["remem-memory"] = config["mcpServers"]["remem-memory"]
            config = existing_config
            
        except json.JSONDecodeError:
            print("⚠️  Existing Claude config file is invalid, creating new one")
    
    # Create directory if it doesn't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write the configuration
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Claude Desktop configuration created at: {config_path}")
        print("Please restart Claude Desktop to load the new configuration")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create Claude config: {e}")
        return False

def test_mcp_server():
    """Test the MCP server functionality."""
    print("🧪 Testing MCP server...")
    try:
        # Import and test the server
        from mcp_server import init_memory_agent
        
        if init_memory_agent("setup_test"):
            print("✅ MCP server test successful")
            return True
        else:
            print("❌ MCP server test failed")
            return False
            
    except Exception as e:
        print(f"❌ MCP server test failed: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 Remem MCP Server Setup")
    print("=" * 40)
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    if not check_env_file():
        print("\n📝 Please set up your .env file first:")
        print("1. Copy .env.example to .env")
        print("2. Add your OpenAI API key")
        print("3. Configure Redis settings if needed")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Check Redis
    if not check_redis():
        print("\n🔧 Please start Redis and try again")
        sys.exit(1)
    
    # Test the server
    if not test_mcp_server():
        print("\n❌ Server test failed. Please check your configuration.")
        sys.exit(1)
    
    # Create Claude Desktop configuration
    if get_claude_config_path():
        if create_claude_config():
            print("\n🎉 Setup completed successfully!")
            print("\nNext steps:")
            print("1. Restart Claude Desktop")
            print("2. Look for the 'Search and tools' icon")
            print("3. Try: 'Store this memory: I love coffee'")
            print("4. Try: 'What do I like to drink?'")
        else:
            print("\n⚠️  MCP server is ready, but Claude config creation failed")
            print("Please manually configure Claude Desktop using the example config")
    else:
        print("\n✅ MCP server is ready!")
        print("Use it with any MCP-compatible client")
    
    print(f"\nServer location: {Path(__file__).parent.absolute() / 'mcp_server.py'}")

if __name__ == "__main__":
    main()

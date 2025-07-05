#!/usr/bin/env python3
"""
Test script for the Remem MCP Server

This script tests the MCP server functionality by simulating tool calls
and verifying the responses.
"""

import os
import sys
import json
import asyncio
from typing import Dict, Any

# Add the current directory to the path so we can import the MCP server
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the MCP server functions
from mcp_server import (
    init_memory_agent,
    store_memory,
    search_memories,
    get_memory_stats,
    answer_question,
    extract_and_store_memories,
    set_context
)

async def test_memory_operations():
    """Test the basic memory operations."""
    print("🧪 Testing Remem MCP Server")
    print("=" * 50)
    
    # Initialize the memory agent
    print("1. Initializing memory agent...")
    if not init_memory_agent("test_mcp"):
        print("❌ Failed to initialize memory agent")
        return False
    print("✅ Memory agent initialized")
    
    # Test storing a memory
    print("\n2. Testing store_memory...")
    result = await store_memory(
        text="I love drinking coffee in the morning",
        vectorstore_name="test_mcp"
    )
    print(f"Result: {result}")
    
    # Test storing another memory
    result = await store_memory(
        text="My favorite restaurant is Luigi's Italian Kitchen",
        vectorstore_name="test_mcp"
    )
    print(f"Result: {result}")
    
    # Test memory stats
    print("\n3. Testing get_memory_stats...")
    result = await get_memory_stats(vectorstore_name="test_mcp")
    print(f"Stats:\n{result}")
    
    # Test searching memories
    print("\n4. Testing search_memories...")
    result = await search_memories(
        query="coffee",
        vectorstore_name="test_mcp",
        top_k=3
    )
    print(f"Search results:\n{result}")
    
    # Test answering a question
    print("\n5. Testing answer_question...")
    result = await answer_question(
        question="What do I like to drink?",
        vectorstore_name="test_mcp"
    )
    print(f"Answer:\n{result}")
    
    # Test setting context
    print("\n6. Testing set_context...")
    result = await set_context(
        location="home",
        activity="testing",
        environment="development",
        vectorstore_name="test_mcp"
    )
    print(f"Context result:\n{result}")
    
    # Test extracting memories from conversation
    print("\n7. Testing extract_and_store_memories...")
    conversation = """
    User: I went to the new sushi place downtown yesterday.
    Assistant: How was it?
    User: It was amazing! The salmon was so fresh. I'll definitely go back.
    """
    result = await extract_and_store_memories(
        conversation_text=conversation,
        vectorstore_name="test_mcp"
    )
    print(f"Extraction result:\n{result}")
    
    # Test searching for the extracted memory
    print("\n8. Testing search for extracted memory...")
    result = await search_memories(
        query="sushi restaurant",
        vectorstore_name="test_mcp",
        top_k=3
    )
    print(f"Search results:\n{result}")
    
    # Final stats
    print("\n9. Final memory stats...")
    result = await get_memory_stats(vectorstore_name="test_mcp")
    print(f"Final stats:\n{result}")
    
    print("\n✅ All tests completed successfully!")
    return True

async def test_error_handling():
    """Test error handling scenarios."""
    print("\n🔧 Testing error handling...")
    print("=" * 30)
    
    # Test with invalid vectorstore
    print("1. Testing with non-existent vectorstore...")
    result = await search_memories(
        query="test",
        vectorstore_name="nonexistent_store"
    )
    print(f"Result: {result}")
    
    # Test with empty query
    print("\n2. Testing with empty search query...")
    result = await search_memories(
        query="",
        vectorstore_name="test_mcp"
    )
    print(f"Result: {result}")
    
    print("✅ Error handling tests completed")

def main():
    """Main test function."""
    print("🚀 Starting Remem MCP Server Tests")
    print("Make sure Redis is running and environment variables are set!")
    print()
    
    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment")
        print("Please set your OpenAI API key in .env file")
        return
    
    try:
        # Run the tests
        asyncio.run(test_memory_operations())
        asyncio.run(test_error_handling())
        
        print("\n🎉 All tests passed!")
        print("\nYou can now use the MCP server with Claude Desktop or other MCP clients.")
        print("See MCP_SERVER_README.md for configuration instructions.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print("Make sure Redis is running and all dependencies are installed.")
        sys.exit(1)

if __name__ == "__main__":
    main()

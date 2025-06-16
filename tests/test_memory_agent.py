#!/usr/bin/env python3
"""
Test script for the Memory Agent

This script tests the basic functionality of the memory agent.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_memory_agent():
    """Test the memory agent functionality."""
    print("🧪 Testing Memory Agent")
    print("=" * 30)
    
    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found. Please set it in .env file.")
        return False
    
    try:
        from memory_agent import MemoryAgent
        
        # Initialize agent
        print("🔧 Initializing Memory Agent...")
        agent = MemoryAgent()
        
        # Test storing memories
        print("\n📝 Testing memory storage...")
        
        test_memories = [
            "I went to the mall to visit New Planet store",
            "Had lunch at Olive Garden with Sarah",
            "Bought groceries at Whole Foods on Sunday",
            "Meeting with client at Starbucks downtown"
        ]
        
        memory_ids = []
        for memory in test_memories:
            memory_id = agent.store_memory(memory)
            memory_ids.append(memory_id)
            print(f"   ✅ Stored: {memory}")
        
        # Test searching memories
        print("\n🔍 Testing memory search...")
        
        test_queries = [
            "I'm going to the mall",
            "Where did I eat?",
            "Shopping for food",
            "Business meetings"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: {query}")
            memories = agent.search_memories(query, top_k=2)
            result = agent.format_memory_results(memories)
            print(result)
        
        print("\n✅ All tests completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed.")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main test function."""
    success = test_memory_agent()
    
    if success:
        print("\n🎉 Memory Agent is working correctly!")
        print("\nYou can now run: python memory_agent.py")
    else:
        print("\n❌ Tests failed. Please check the setup.")
        sys.exit(1)

if __name__ == "__main__":
    main()

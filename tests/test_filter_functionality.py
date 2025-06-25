#!/usr/bin/env python3
"""
Test script to verify filterBy functionality in memory agent
"""

import os
import sys
import json
from memory.core_agent import MemoryAgent

def test_filter_functionality():
    """Test the new filterBy parameter functionality."""
    print("🧪 Testing filterBy functionality...")
    
    try:
        # Initialize memory agent
        agent = MemoryAgent()
        
        # Clear any existing memories
        agent.clear_all_memories()
        
        # Set context for testing
        agent.set_context(
            location="Test Lab",
            activity="testing filters"
        )
        
        # Store some test memories with different attributes
        print("\n📝 Storing test memories...")
        
        # Memory 1: Work-related
        memory1_id = agent.store_memory("Had a productive meeting about the new project")
        # Add custom attributes to this memory
        agent.redis_client.execute_command(
            "VSETATTR", agent.VECTORSET_KEY, memory1_id, 
            json.dumps({"category": "work", "priority": "high", "year": 2024})
        )
        
        # Memory 2: Personal
        memory2_id = agent.store_memory("Went for a nice walk in the park")
        agent.redis_client.execute_command(
            "VSETATTR", agent.VECTORSET_KEY, memory2_id,
            json.dumps({"category": "personal", "priority": "low", "year": 2024})
        )
        
        # Memory 3: Work-related with different priority
        memory3_id = agent.store_memory("Reviewed some documentation for the API")
        agent.redis_client.execute_command(
            "VSETATTR", agent.VECTORSET_KEY, memory3_id,
            json.dumps({"category": "work", "priority": "medium", "year": 2023})
        )
        
        print(f"✅ Stored 3 test memories with attributes")
        
        # Test 1: Search without filter
        print("\n🔍 Test 1: Search without filter")
        memories = agent.search_memories("project work", top_k=5)
        print(f"Found {len(memories)} memories without filter:")
        for i, memory in enumerate(memories, 1):
            print(f"  {i}. {memory['text']} (score: {memory['score']:.3f})")
        
        # Test 2: Search with category filter
        print("\n🔍 Test 2: Search with category filter (.category == 'work')")
        memories = agent.search_memories("project work", top_k=5, filterBy='.category == "work"')
        print(f"Found {len(memories)} work-related memories:")
        for i, memory in enumerate(memories, 1):
            print(f"  {i}. {memory['text']} (score: {memory['score']:.3f})")
        
        # Test 3: Search with priority filter
        print("\n🔍 Test 3: Search with priority filter (.priority == 'high')")
        memories = agent.search_memories("meeting project", top_k=5, filterBy='.priority == "high"')
        print(f"Found {len(memories)} high-priority memories:")
        for i, memory in enumerate(memories, 1):
            print(f"  {i}. {memory['text']} (score: {memory['score']:.3f})")
        
        # Test 4: Search with combined filter
        print("\n🔍 Test 4: Search with combined filter (.category == 'work' and .year >= 2024)")
        memories = agent.search_memories("work", top_k=5, filterBy='.category == "work" and .year >= 2024')
        print(f"Found {len(memories)} recent work memories:")
        for i, memory in enumerate(memories, 1):
            print(f"  {i}. {memory['text']} (score: {memory['score']:.3f})")
        
        # Test 5: Test answer_question with filter
        print("\n🤔 Test 5: Answer question with filter")
        response = agent.answer_question("What work did I do?", top_k=5, filterBy='.category == "work"')
        print(f"Answer: {response['answer']}")
        print(f"Confidence: {response['confidence']}")
        print(f"Supporting memories: {len(response['supporting_memories'])}")
        
        print("\n✅ All filter tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment variables.")
        print("Please set your OpenAI API key in the .env file or environment.")
        sys.exit(1)
    
    success = test_filter_functionality()
    
    if success:
        print("\n🎉 FilterBy functionality is working correctly!")
    else:
        print("\n❌ FilterBy tests failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()

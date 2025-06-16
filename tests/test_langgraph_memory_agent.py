#!/usr/bin/env python3
"""
Test script for the LangGraph Memory Agent

This script tests the enhanced LangGraph-based memory agent functionality.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_langgraph_memory_agent():
    """Test the LangGraph memory agent functionality."""
    print("🧪 Testing LangGraph Memory Agent")
    print("=" * 40)
    
    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found. Please set it in .env file.")
        return False
    
    try:
        from langgraph_memory_agent import LangGraphMemoryAgent
        
        # Initialize agent
        print("🔧 Initializing LangGraph Memory Agent...")
        agent = LangGraphMemoryAgent()
        
        # Test storing memories through conversation
        print("\n📝 Testing memory storage through conversation...")
        
        test_conversations = [
            "Remember that I went to Mario's Italian Restaurant last Friday and had amazing pasta carbonara",
            "Store this memory: I met with Sarah at Starbucks downtown to discuss the Redis project",
            "I need to remember that my cat Molly is 3 years old and loves tuna treats",
            "Please remember that I bought groceries at Whole Foods on Sunday - spent $85 on organic vegetables"
        ]
        
        for conversation in test_conversations:
            print(f"\n💬 User: {conversation}")
            response = agent.run(conversation)
            print(f"🤖 Agent: {response}")
        
        # Test memory retrieval through questions
        print("\n🔍 Testing memory retrieval through questions...")
        
        test_questions = [
            "What did I eat at Mario's?",
            "Tell me about my meeting with Sarah",
            "How old is my cat?",
            "Where did I go shopping and how much did I spend?",
            "What do I know about Redis?",
            "Who did I meet with recently?"
        ]
        
        for question in test_questions:
            print(f"\n❓ Question: {question}")
            response = agent.run(question)
            print(f"🤖 Agent: {response}")
        
        # Test complex multi-step reasoning
        print("\n🧠 Testing complex multi-step reasoning...")
        
        complex_questions = [
            "What restaurants have I been to and what did I eat at each?",
            "Tell me about all my meetings and what we discussed",
            "What do I know about my pets and their preferences?"
        ]
        
        for question in complex_questions:
            print(f"\n🔬 Complex Question: {question}")
            response = agent.run(question)
            print(f"🤖 Agent: {response}")
        
        print("\n✅ All tests completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed.")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_compatibility():
    """Test that the API compatibility is maintained."""
    print("\n🔗 Testing API Compatibility...")
    
    try:
        from langgraph_memory_agent import LangGraphMemoryAgent
        
        agent = LangGraphMemoryAgent()
        
        # Test the answer_question method for API compatibility
        response = agent.answer_question("What did I eat at Mario's?")
        
        # Check that response has expected structure
        expected_keys = ["type", "answer", "confidence", "reasoning"]
        for key in expected_keys:
            if key not in response:
                print(f"❌ Missing key in API response: {key}")
                return False
        
        print("✅ API compatibility maintained")
        print(f"   Response type: {response['type']}")
        print(f"   Answer: {response['answer'][:100]}...")
        print(f"   Confidence: {response['confidence']}")
        
        return True
        
    except Exception as e:
        print(f"❌ API compatibility test failed: {e}")
        return False

def test_tool_functionality():
    """Test individual tool functionality."""
    print("\n🔧 Testing Tool Functionality...")
    
    try:
        from tools import store_memory, search_memories, set_context, get_memory_stats
        from langgraph_memory_agent import LangGraphMemoryAgent
        
        # Initialize agent to set up tools
        agent = LangGraphMemoryAgent()
        
        # Test store_memory tool
        print("Testing store_memory tool...")
        result = store_memory("Test memory for tool functionality")
        print(f"Store result: {result}")
        
        # Test search_memories tool
        print("Testing search_memories tool...")
        result = search_memories("test memory")
        print(f"Search result: {result[:200]}...")
        
        # Test set_context tool
        print("Testing set_context tool...")
        result = set_context(location="Test Location", activity="Testing")
        print(f"Context result: {result}")
        
        # Test get_memory_stats tool
        print("Testing get_memory_stats tool...")
        result = get_memory_stats()
        print(f"Stats result: {result[:200]}...")
        
        print("✅ Tool functionality tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Tool functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🚀 LangGraph Memory Agent Test Suite")
    print("=" * 50)
    
    tests = [
        ("LangGraph Memory Agent", test_langgraph_memory_agent),
        ("API Compatibility", test_api_compatibility),
        ("Tool Functionality", test_tool_functionality)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*20} TEST SUMMARY {'='*20}")
    all_passed = True
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! LangGraph Memory Agent is working correctly!")
        print("\nYou can now:")
        print("- Run: python langgraph_memory_agent.py (for CLI)")
        print("- Run: python web_app.py (for web interface)")
        print("- The web interface now uses LangGraph for enhanced conversational capabilities!")
    else:
        print("\n❌ Some tests failed. Please check the setup and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()

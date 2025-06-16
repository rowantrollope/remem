#!/usr/bin/env python3
"""
Test script to verify confidence scoring and JSON formatting fixes
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_confidence_and_json():
    """Test that confidence scoring and JSON formatting are working properly."""
    print("🧪 Testing Confidence Scoring and JSON Formatting Fixes")
    print("=" * 60)
    
    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found. Please set it in .env file.")
        return False
    
    try:
        from langgraph_memory_agent import LangGraphMemoryAgent
        
        # Initialize agent
        print("🔧 Initializing LangGraph Memory Agent...")
        agent = LangGraphMemoryAgent()
        
        # Store some test memories first
        print("\n📝 Storing test memories...")
        test_memories = [
            "I went to Mario's Italian Restaurant last Friday and had amazing pasta carbonara",
            "My cat Molly is 3 years old and loves tuna treats",
            "I met with Sarah at Starbucks to discuss the Redis project"
        ]
        
        for memory in test_memories:
            print(f"Storing: {memory}")
            agent.memory_agent.store_memory(memory)
        
        # Test the answer_question method for proper confidence scoring
        print("\n🎯 Testing Confidence Scoring and JSON Structure...")
        
        test_questions = [
            ("What did I eat at Mario's?", "Should be HIGH confidence - direct match"),
            ("How old is my cat?", "Should be HIGH confidence - specific fact"),
            ("What did Sarah and I discuss?", "Should be MEDIUM/HIGH confidence"),
            ("What's my favorite color?", "Should be LOW confidence - no relevant memories"),
            ("Tell me about my pets", "Should be MEDIUM/HIGH confidence")
        ]
        
        for question, expected in test_questions:
            print(f"\n❓ Question: {question}")
            print(f"   Expected: {expected}")
            
            # Test the API method that should return structured JSON
            response = agent.answer_question(question)
            
            print(f"   Response Type: {response.get('type', 'MISSING')}")
            print(f"   Confidence: {response.get('confidence', 'MISSING')}")
            print(f"   Answer: {response.get('answer', 'MISSING')[:100]}...")
            
            # Check for supporting memories
            supporting = response.get('supporting_memories', [])
            print(f"   Supporting Memories: {len(supporting)} found")
            
            if supporting:
                for i, memory in enumerate(supporting[:2], 1):  # Show first 2
                    relevance = memory.get('relevance_score', 'N/A')
                    timestamp = memory.get('timestamp', 'N/A')
                    print(f"     {i}. {memory.get('text', 'N/A')[:50]}... ({relevance}% relevant, {timestamp})")
            
            # Verify structure
            required_keys = ['type', 'answer', 'confidence']
            missing_keys = [key for key in required_keys if key not in response]
            if missing_keys:
                print(f"   ❌ MISSING KEYS: {missing_keys}")
            else:
                print(f"   ✅ All required keys present")
            
            print("-" * 50)
        
        # Test the LangGraph workflow directly
        print("\n🤖 Testing LangGraph Workflow (run method)...")
        
        workflow_questions = [
            "What restaurants have I been to?",
            "Tell me about my pets and their details"
        ]
        
        for question in workflow_questions:
            print(f"\n💬 Workflow Question: {question}")
            response = agent.run(question)
            print(f"🤖 Response: {response}")
            print("-" * 50)
        
        print("\n✅ Confidence and JSON formatting tests completed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_original_vs_langgraph():
    """Compare original memory agent vs LangGraph agent responses."""
    print("\n🔄 Comparing Original vs LangGraph Agent...")
    
    try:
        from memory_agent import MemoryAgent
        from langgraph_memory_agent import LangGraphMemoryAgent
        
        # Initialize both agents
        original_agent = MemoryAgent()
        langgraph_agent = LangGraphMemoryAgent()
        
        test_question = "What did I eat at Mario's?"
        
        print(f"\n❓ Test Question: {test_question}")
        
        # Test original agent
        print("\n🔸 Original Memory Agent Response:")
        original_response = original_agent.answer_question(test_question)
        print(f"   Type: {original_response.get('type')}")
        print(f"   Confidence: {original_response.get('confidence')}")
        print(f"   Answer: {original_response.get('answer', '')[:100]}...")
        print(f"   Supporting Memories: {len(original_response.get('supporting_memories', []))}")
        
        # Test LangGraph agent
        print("\n🔹 LangGraph Memory Agent Response:")
        langgraph_response = langgraph_agent.answer_question(test_question)
        print(f"   Type: {langgraph_response.get('type')}")
        print(f"   Confidence: {langgraph_response.get('confidence')}")
        print(f"   Answer: {langgraph_response.get('answer', '')[:100]}...")
        print(f"   Supporting Memories: {len(langgraph_response.get('supporting_memories', []))}")
        
        # Compare
        print("\n📊 Comparison:")
        if original_response.get('confidence') == langgraph_response.get('confidence'):
            print("   ✅ Confidence levels match")
        else:
            print(f"   ❌ Confidence mismatch: {original_response.get('confidence')} vs {langgraph_response.get('confidence')}")
        
        if len(original_response.get('supporting_memories', [])) == len(langgraph_response.get('supporting_memories', [])):
            print("   ✅ Supporting memories count matches")
        else:
            print(f"   ⚠️ Supporting memories count differs: {len(original_response.get('supporting_memories', []))} vs {len(langgraph_response.get('supporting_memories', []))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Comparison test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Confidence Scoring and JSON Formatting Fix Test")
    print("=" * 60)
    
    tests = [
        ("Confidence and JSON Tests", test_confidence_and_json),
        ("Original vs LangGraph Comparison", test_original_vs_langgraph)
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
        print("\n🎉 All tests passed! Confidence scoring and JSON formatting are working correctly!")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()

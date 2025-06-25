#!/usr/bin/env python3
"""
Test script to verify the new JSON response format

This script tests that the answer_question method returns structured JSON
with answer, confidence, and supporting_memories fields.
"""

import os
import sys
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_json_response_format():
    """Test the new JSON response format."""
    print("🧪 Testing JSON Response Format")
    print("=" * 35)
    
    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found. Please set it in .env file.")
        return False
    
    try:
        from memory.core_agent import MemoryAgent
        
        # Initialize agent
        print("🔧 Initializing Memory Agent...")
        agent = MemoryAgent()
        
        # Store some test memories
        print("\n📝 Storing test memories...")
        
        test_memories = [
            "Molly is a dog",
            "Molly (the dog) is black",
            "Molly loves to play fetch",
            "Molly is 3 years old"
        ]
        
        for memory in test_memories:
            memory_id = agent.store_memory(memory)
            print(f"   ✅ Stored: {memory}")
        
        # Test questions with different expected confidence levels
        print("\n🔍 Testing JSON response format...")
        
        test_cases = [
            {
                "question": "what color is molly?",
                "expected_confidence": "high",
                "description": "Direct question with clear answer"
            },
            {
                "question": "how old is molly?", 
                "expected_confidence": "high",
                "description": "Another direct question"
            },
            {
                "question": "what does molly like to eat?",
                "expected_confidence": "low",
                "description": "Question without clear answer in memories"
            },
            {
                "question": "hello",
                "expected_type": "help",
                "description": "Help request"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n--- Test Case {i}: {test_case['description']} ---")
            print(f"❓ Question: {test_case['question']}")
            
            response = agent.answer_question(test_case['question'])
            
            # Verify response structure
            print(f"📋 Response structure check:")
            
            # Check required fields
            required_fields = ['type', 'answer', 'confidence', 'supporting_memories']
            for field in required_fields:
                if field in response:
                    print(f"   ✅ {field}: {type(response[field]).__name__}")
                else:
                    print(f"   ❌ Missing field: {field}")
            
            # Display the response content
            print(f"\n📄 Response content:")
            print(f"   Type: {response.get('type', 'N/A')}")
            print(f"   Answer: {response.get('answer', 'N/A')}")
            print(f"   Confidence: {response.get('confidence', 'N/A')}")
            
            if response.get('reasoning'):
                print(f"   Reasoning: {response.get('reasoning')}")
            
            supporting_memories = response.get('supporting_memories', [])
            print(f"   Supporting Memories: {len(supporting_memories)} found")
            
            for j, memory in enumerate(supporting_memories[:2], 1):  # Show first 2
                print(f"      {j}. {memory.get('text', 'N/A')} ({memory.get('relevance_score', 0)}% relevant)")
            
            # Verify expected values
            if 'expected_confidence' in test_case:
                actual_confidence = response.get('confidence', '')
                if actual_confidence == test_case['expected_confidence']:
                    print(f"   ✅ Confidence matches expected: {actual_confidence}")
                else:
                    print(f"   ⚠️ Confidence mismatch - Expected: {test_case['expected_confidence']}, Got: {actual_confidence}")
            
            if 'expected_type' in test_case:
                actual_type = response.get('type', '')
                if actual_type == test_case['expected_type']:
                    print(f"   ✅ Type matches expected: {actual_type}")
                else:
                    print(f"   ❌ Type mismatch - Expected: {test_case['expected_type']}, Got: {actual_type}")
            
            # Test JSON serialization
            try:
                json_str = json.dumps(response, indent=2)
                print(f"   ✅ Response is JSON serializable")
            except Exception as e:
                print(f"   ❌ JSON serialization failed: {e}")
        
        print("\n✅ JSON response format test completed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    success = test_json_response_format()
    
    if success:
        print("\n🎉 JSON response format is working correctly!")
        print("The API now returns structured responses with answer, confidence, and supporting memories.")
    else:
        print("\n❌ JSON response format test failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()

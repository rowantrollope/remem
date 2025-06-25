#!/usr/bin/env python3
"""
Test script to verify the validation fix for entity-based questions

This script specifically tests that questions like "what color is molly?" 
are properly recognized as memory questions.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_validation_fix():
    """Test the validation fix for entity-based questions."""
    print("🧪 Testing Validation Fix for Entity-Based Questions")
    print("=" * 55)
    
    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found. Please set it in .env file.")
        return False
    
    try:
        from memory.core_agent import MemoryAgent
        
        # Initialize agent
        print("🔧 Initializing Memory Agent...")
        agent = MemoryAgent()
        
        # First, store some test memories about Molly
        print("\n📝 Storing test memories about Molly...")
        
        test_memories = [
            "Molly is a dog",
            "Molly (the dog) is black",
            "Molly loves to play fetch",
            "Molly is 3 years old"
        ]
        
        memory_ids = []
        for memory in test_memories:
            memory_id = agent.store_memory(memory)
            memory_ids.append(memory_id)
            print(f"   ✅ Stored: {memory}")
        
        # Test entity-based questions that should be recognized as memory questions
        print("\n🔍 Testing entity-based questions...")
        
        test_questions = [
            "what color is molly?",
            "What color is Molly?",
            "How old is Molly?",
            "What does Molly like to do?",
            "Tell me about Molly",
            "What kind of animal is Molly?"
        ]
        
        for question in test_questions:
            print(f"\n❓ Question: {question}")
            
            # Test the validation logic directly
            validation_result = agent._validate_and_preprocess_question(question)
            print(f"   🔍 Validation type: {validation_result['type']}")
            print(f"   📝 Content: {validation_result['content']}")
            
            if validation_result['type'] == 'search':
                print("   ✅ Correctly identified as a memory question!")
                
                # Now test the full answer_question flow
                answer = agent.answer_question(question)
                print(f"   🤖 Answer: {answer}")
            else:
                print("   ❌ Incorrectly identified as help request!")
                print(f"   📄 Help message: {validation_result['content']}")
        
        # Test some questions that should still be help requests
        print("\n🔍 Testing questions that should be help requests...")
        
        help_questions = [
            "hello",
            "how are you?",
            "what's the weather like?",
            "what is the capital of France?",
            "asdfghjkl"
        ]
        
        for question in help_questions:
            print(f"\n❓ Question: {question}")
            validation_result = agent._validate_and_preprocess_question(question)
            print(f"   🔍 Validation type: {validation_result['type']}")
            
            if validation_result['type'] == 'help':
                print("   ✅ Correctly identified as help request!")
            else:
                print("   ⚠️ Unexpectedly identified as memory question!")
        
        print("\n✅ Validation fix test completed!")
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

def main():
    """Main test function."""
    success = test_validation_fix()
    
    if success:
        print("\n🎉 Validation fix is working correctly!")
        print("Entity-based questions like 'what color is molly?' should now work properly.")
    else:
        print("\n❌ Validation fix test failed. Please check the implementation.")
        sys.exit(1)

if __name__ == "__main__":
    main()

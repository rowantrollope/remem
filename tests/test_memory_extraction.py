#!/usr/bin/env python3
"""
Test script for the improved LLM-based memory extraction system.

This script demonstrates how the new system can intelligently identify
information worth remembering vs. information that should be ignored.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory.core_agent import MemoryAgent

def test_memory_extraction():
    """Test the improved memory extraction system with various types of input."""
    
    print("🧠 Testing LLM-based Memory Extraction System")
    print("=" * 60)
    
    # Initialize memory agent
    try:
        agent = MemoryAgent()
        print("✅ Memory agent initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize memory agent: {e}")
        return
    
    # Test cases with different types of information
    test_cases = [
        {
            "name": "Personal Preferences",
            "input": "I really prefer window seats when I fly because I like to look out during takeoff. Also, I'm a vegetarian so I always need to request the vegetarian meal option.",
            "should_extract": True
        },
        {
            "name": "Family Information",
            "input": "My wife Sarah is allergic to shellfish, and we have two kids - Emma who is 8 and Jake who is 12. We live in San Francisco.",
            "should_extract": True
        },
        {
            "name": "Budget Constraints",
            "input": "Our budget for this vacation is around $3000 total, and we can't spend more than $150 per night on hotels.",
            "should_extract": True
        },
        {
            "name": "Conversational Filler",
            "input": "Hi there! How are you doing today? The weather is nice, isn't it?",
            "should_extract": False
        },
        {
            "name": "Temporary Information",
            "input": "I have a meeting at 3 PM today and then I need to pick up groceries. The traffic is pretty bad right now.",
            "should_extract": False
        },
        {
            "name": "General Questions",
            "input": "What's the best way to get from the airport to downtown? How long does it usually take?",
            "should_extract": False
        },
        {
            "name": "Mixed Content",
            "input": "Thanks for the restaurant recommendations! I should mention that I'm into Michelin star restaurants and fine dining. The weather today is cloudy but that's okay.",
            "should_extract": True  # Should extract the dining preference, ignore weather
        },
        {
            "name": "Goals and Plans",
            "input": "I'm planning to learn Spanish this year and want to take a trip to Spain in the fall. I've always wanted to visit Barcelona.",
            "should_extract": True
        }
    ]
    
    print("\n🔍 Testing Memory Extraction on Various Inputs:")
    print("-" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print(f"Input: \"{test_case['input']}\"")
        print(f"Expected: {'Should extract' if test_case['should_extract'] else 'Should NOT extract'}")
        
        try:
            # Test the extraction
            result = agent.extract_and_store_memories(
                raw_input=f"User: {test_case['input']}",
                context_prompt="Extract user preferences, constraints, and important personal information from travel conversations.",
                apply_grounding=False  # Skip grounding for testing
            )
            
            extracted_count = result.get('total_extracted', 0)
            extracted_memories = result.get('extracted_memories', [])
            
            if extracted_count > 0:
                print(f"✅ Extracted {extracted_count} memories:")
                for memory in extracted_memories:
                    print(f"   - \"{memory.get('extracted_text', 'N/A')}\" (confidence: {memory.get('confidence', 'N/A')})")
                
                # Check if this matches expectation
                if test_case['should_extract']:
                    print("   ✅ CORRECT: Extraction matched expectation")
                else:
                    print("   ❌ INCORRECT: Should not have extracted anything")
            else:
                print("❌ No memories extracted")
                
                # Check if this matches expectation
                if not test_case['should_extract']:
                    print("   ✅ CORRECT: No extraction matched expectation")
                else:
                    print("   ❌ INCORRECT: Should have extracted memories")
                    
        except Exception as e:
            print(f"❌ Error during extraction: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 Test completed! Review the results above to see how well")
    print("   the LLM-based system identifies extractable information.")
    
    # Clean up test memories
    try:
        print("\n🧹 Cleaning up test memories...")
        cleanup_result = agent.clear_all_memories()
        if cleanup_result['success']:
            print(f"✅ Cleaned up {cleanup_result['memories_deleted']} test memories")
        else:
            print(f"⚠️ Cleanup warning: {cleanup_result.get('detail') or cleanup_result.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"⚠️ Cleanup failed: {e}")

if __name__ == "__main__":
    test_memory_extraction()

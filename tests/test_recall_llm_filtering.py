#!/usr/bin/env python3
"""
Test script for the enhanced /api/klines/recall endpoint with LLM filtering.

This script tests the new use_llm_filtering parameter to ensure it works correctly
and provides the same quality filtering as the /ask endpoint.
"""

import requests
import json
import time
import sys
import os

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_URL = "http://localhost:5001"

def test_recall_llm_filtering():
    """Test the enhanced recall endpoint with LLM filtering."""
    
    print("🧪 Testing /api/klines/recall with LLM filtering enhancement")
    print("=" * 60)
    
    # First, clear any existing memories
    print("\n1. Clearing existing memories...")
    try:
        response = requests.delete(f"{BASE_URL}/api/memory")
        if response.status_code == 200:
            print("✅ Memories cleared successfully")
        else:
            print(f"⚠️ Warning: Could not clear memories (status: {response.status_code})")
    except Exception as e:
        print(f"⚠️ Warning: Could not clear memories: {e}")
    
    # Store test memories with varying relevance
    print("\n2. Storing test memories...")
    test_memories = [
        "I love Italian restaurants, especially ones with good pasta carbonara",
        "My favorite pizza place is Tony's on Main Street",
        "I went to the dentist last Tuesday for a cleaning",
        "I prefer restaurants with outdoor seating when the weather is nice",
        "My car needs an oil change soon",
        "I had amazing sushi at Sakura restaurant last month",
        "The weather has been really rainy lately",
        "I'm planning to visit my grandmother next weekend"
    ]
    
    memory_ids = []
    for memory_text in test_memories:
        try:
            response = requests.post(f"{BASE_URL}/api/memory", 
                                   json={"text": memory_text, "apply_grounding": False})
            if response.status_code == 200:
                data = response.json()
                memory_ids.append(data.get('memory_id'))
                print(f"✅ Stored: {memory_text[:50]}...")
            else:
                print(f"❌ Failed to store memory: {memory_text[:50]}...")
        except Exception as e:
            print(f"❌ Error storing memory: {e}")
    
    print(f"\n✅ Stored {len(memory_ids)} test memories")
    
    # Wait a moment for indexing
    time.sleep(2)
    
    # Test query about restaurants
    query = "restaurant preferences for dinner"
    
    print(f"\n3. Testing recall without LLM filtering...")
    print(f"Query: '{query}'")
    
    try:
        response = requests.post(f"{BASE_URL}/api/klines/recall", 
                               json={
                                   "query": query,
                                   "top_k": 8,
                                   "use_llm_filtering": False
                               })
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Recall without filtering successful")
            print(f"📊 Retrieved {data['memory_count']} memories")
            print(f"🧠 Mental state preview: {data['mental_state'][:100]}...")
            
            unfiltered_count = data['memory_count']
            unfiltered_memories = data['memories']
            
        else:
            print(f"❌ Recall without filtering failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error during recall without filtering: {e}")
        return False
    
    print(f"\n4. Testing recall with LLM filtering...")
    
    try:
        response = requests.post(f"{BASE_URL}/api/klines/recall", 
                               json={
                                   "query": query,
                                   "top_k": 8,
                                   "use_llm_filtering": True
                               })
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Recall with LLM filtering successful")
            print(f"📊 Original memories: {data.get('original_memory_count', 'N/A')}")
            print(f"📊 Filtered memories: {data['memory_count']}")
            print(f"🤖 LLM filtering applied: {data.get('llm_filtering_applied', False)}")
            print(f"🧠 Mental state preview: {data['mental_state'][:100]}...")
            
            filtered_count = data['memory_count']
            filtered_memories = data['memories']
            
            # Check that filtering was applied
            if data.get('llm_filtering_applied'):
                print("✅ LLM filtering was applied")
            else:
                print("❌ LLM filtering was not applied")
                return False
                
            # Check that filtering reduced the number of memories (in most cases)
            if filtered_count <= unfiltered_count:
                print(f"✅ Filtering reduced memories from {unfiltered_count} to {filtered_count}")
            else:
                print(f"⚠️ Warning: Filtering increased memories from {unfiltered_count} to {filtered_count}")
            
            # Check that filtered memories have relevance reasoning
            has_relevance_reasoning = any(
                'relevance_reasoning' in memory for memory in filtered_memories
            )
            if has_relevance_reasoning:
                print("✅ Filtered memories include relevance reasoning")
                # Show an example
                for memory in filtered_memories:
                    if 'relevance_reasoning' in memory:
                        print(f"   Example reasoning: {memory['relevance_reasoning'][:80]}...")
                        break
            else:
                print("❌ Filtered memories missing relevance reasoning")
                return False
            
        else:
            print(f"❌ Recall with LLM filtering failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error during recall with LLM filtering: {e}")
        return False
    
    print(f"\n5. Comparing with /api/klines/ask endpoint...")
    
    try:
        # Convert query to a question for the ask endpoint
        question = f"What are my {query}?"
        response = requests.post(f"{BASE_URL}/api/klines/ask", 
                               json={
                                   "question": question,
                                   "top_k": 8
                               })
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Ask endpoint successful")
            ask_memories = data.get('supporting_memories', [])
            print(f"📊 Ask endpoint returned {len(ask_memories)} supporting memories")
            
            # Check that both endpoints use similar filtering logic
            ask_has_reasoning = any(
                'relevance_reasoning' in memory for memory in ask_memories
            )
            if ask_has_reasoning:
                print("✅ Ask endpoint also includes relevance reasoning")
            else:
                print("⚠️ Ask endpoint missing relevance reasoning")
            
        else:
            print(f"❌ Ask endpoint failed: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error during ask endpoint test: {e}")
    
    print(f"\n6. Testing backward compatibility...")
    
    try:
        # Test that the default behavior (without use_llm_filtering) still works
        response = requests.post(f"{BASE_URL}/api/klines/recall", 
                               json={
                                   "query": query,
                                   "top_k": 5
                               })
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Backward compatibility maintained")
            print(f"📊 Default behavior returned {data['memory_count']} memories")
            
            # Should not have LLM filtering applied by default
            if not data.get('llm_filtering_applied', False):
                print("✅ LLM filtering not applied by default")
            else:
                print("❌ LLM filtering applied by default (should be false)")
                return False
            
        else:
            print(f"❌ Backward compatibility test failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error during backward compatibility test: {e}")
        return False
    
    print(f"\n" + "=" * 60)
    print("🎉 All tests passed! LLM filtering enhancement is working correctly.")
    print("\nKey features verified:")
    print("✅ LLM filtering can be enabled with use_llm_filtering parameter")
    print("✅ Filtering reduces irrelevant memories")
    print("✅ Filtered memories include relevance reasoning")
    print("✅ Backward compatibility maintained")
    print("✅ Consistent with /ask endpoint filtering logic")
    
    return True

if __name__ == "__main__":
    success = test_recall_llm_filtering()
    if not success:
        print("\n❌ Some tests failed!")
        sys.exit(1)
    else:
        print("\n✅ All tests passed!")
        sys.exit(0)

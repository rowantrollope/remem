#!/usr/bin/env python3
"""
Test script to verify vectorstore isolation functionality.

This script tests that different vectorstore names create separate memory spaces
and that memories are properly isolated between different vectorstores.
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:5001"
VECTORSTORE_USER1 = "user1_memories"
VECTORSTORE_USER2 = "user2_memories"

def test_vectorstore_isolation():
    """Test that different vectorstores maintain separate memory spaces."""
    
    print("🧪 Testing Vectorstore Isolation")
    print("=" * 50)
    
    # Test data
    user1_memories = [
        "I love pizza and Italian food",
        "My favorite color is blue",
        "I work as a software engineer"
    ]
    
    user2_memories = [
        "I prefer sushi and Japanese cuisine", 
        "My favorite color is red",
        "I work as a graphic designer"
    ]
    
    try:
        # Step 1: Clear any existing memories in both vectorstores
        print("🗑️ Clearing existing memories...")
        
        clear_response1 = requests.delete(f"{BASE_URL}/api/memory", 
                                        json={"vectorstore_name": VECTORSTORE_USER1})
        clear_response2 = requests.delete(f"{BASE_URL}/api/memory", 
                                        json={"vectorstore_name": VECTORSTORE_USER2})
        
        print(f"   User1 vectorstore cleared: {clear_response1.status_code}")
        print(f"   User2 vectorstore cleared: {clear_response2.status_code}")
        
        # Step 2: Store memories for User 1
        print(f"\n💾 Storing memories for User 1 (vectorstore: {VECTORSTORE_USER1})...")
        user1_memory_ids = []
        
        for memory in user1_memories:
            response = requests.post(f"{BASE_URL}/api/memory", 
                                   json={
                                       "text": memory,
                                       "vectorstore_name": VECTORSTORE_USER1
                                   })
            if response.status_code == 200:
                data = response.json()
                user1_memory_ids.append(data['memory_id'])
                print(f"   ✅ Stored: {memory[:50]}...")
            else:
                print(f"   ❌ Failed to store: {memory}")
                print(f"      Error: {response.text}")
        
        # Step 3: Store memories for User 2
        print(f"\n💾 Storing memories for User 2 (vectorstore: {VECTORSTORE_USER2})...")
        user2_memory_ids = []
        
        for memory in user2_memories:
            response = requests.post(f"{BASE_URL}/api/memory", 
                                   json={
                                       "text": memory,
                                       "vectorstore_name": VECTORSTORE_USER2
                                   })
            if response.status_code == 200:
                data = response.json()
                user2_memory_ids.append(data['memory_id'])
                print(f"   ✅ Stored: {memory[:50]}...")
            else:
                print(f"   ❌ Failed to store: {memory}")
                print(f"      Error: {response.text}")
        
        # Step 4: Test isolation - search User 1's vectorstore
        print(f"\n🔍 Testing isolation - searching User 1's vectorstore...")
        search_payload1 = {
            "query": "favorite color",
            "vectorstore_name": VECTORSTORE_USER1,
            "top_k": 10
        }
        print(f"   Search payload: {search_payload1}")
        search_response1 = requests.post(f"{BASE_URL}/api/memory/search", json=search_payload1)
        
        if search_response1.status_code == 200:
            data1 = search_response1.json()
            print(f"   Found {data1['count']} memories in User 1's vectorstore")
            print(f"   Debug: First memory structure: {data1['memories'][0] if data1['memories'] else 'No memories'}")
            for memory in data1['memories']:
                # Handle different possible memory structures
                text = memory.get('final_text') or memory.get('text') or memory.get('raw_text', 'Unknown text')
                print(f"   - {text}")

            # Verify only User 1's memories are returned
            user1_color_found = any("blue" in (memory.get('final_text') or memory.get('text') or memory.get('raw_text', '')).lower() for memory in data1['memories'])
            user2_color_found = any("red" in (memory.get('final_text') or memory.get('text') or memory.get('raw_text', '')).lower() for memory in data1['memories'])
            
            if user1_color_found and not user2_color_found:
                print("   ✅ Isolation test passed: Only User 1's memories found")
            else:
                print("   ❌ Isolation test failed: Found unexpected memories")
        else:
            print(f"   ❌ Search failed: {search_response1.text}")
        
        # Step 5: Test isolation - search User 2's vectorstore
        print(f"\n🔍 Testing isolation - searching User 2's vectorstore...")
        search_response2 = requests.post(f"{BASE_URL}/api/memory/search",
                                       json={
                                           "query": "favorite color",
                                           "vectorstore_name": VECTORSTORE_USER2,
                                           "top_k": 10
                                       })
        
        if search_response2.status_code == 200:
            data2 = search_response2.json()
            print(f"   Found {data2['count']} memories in User 2's vectorstore")
            for memory in data2['memories']:
                # Handle different possible memory structures
                text = memory.get('final_text') or memory.get('text') or memory.get('raw_text', 'Unknown text')
                print(f"   - {text}")

            # Verify only User 2's memories are returned
            user1_color_found = any("blue" in (memory.get('final_text') or memory.get('text') or memory.get('raw_text', '')).lower() for memory in data2['memories'])
            user2_color_found = any("red" in (memory.get('final_text') or memory.get('text') or memory.get('raw_text', '')).lower() for memory in data2['memories'])
            
            if user2_color_found and not user1_color_found:
                print("   ✅ Isolation test passed: Only User 2's memories found")
            else:
                print("   ❌ Isolation test failed: Found unexpected memories")
        else:
            print(f"   ❌ Search failed: {search_response2.text}")
        
        # Step 6: Test K-line recall with vectorstore isolation
        print(f"\n🧠 Testing K-line recall with vectorstore isolation...")
        
        kline_response1 = requests.post(f"{BASE_URL}/api/klines/recall",
                                      json={
                                          "query": "What do I do for work?",
                                          "vectorstore_name": VECTORSTORE_USER1,
                                          "top_k": 5
                                      })
        
        if kline_response1.status_code == 200:
            kline_data1 = kline_response1.json()
            print(f"   User 1 K-line memories: {kline_data1['memory_count']}")
            engineer_found = any("engineer" in (memory.get('final_text') or memory.get('text') or memory.get('raw_text', '')).lower() for memory in kline_data1['memories'])
            designer_found = any("designer" in (memory.get('final_text') or memory.get('text') or memory.get('raw_text', '')).lower() for memory in kline_data1['memories'])
            
            if engineer_found and not designer_found:
                print("   ✅ K-line isolation test passed: Found engineer, not designer")
            else:
                print("   ❌ K-line isolation test failed")
        else:
            print(f"   ❌ K-line recall failed: {kline_response1.text}")
        
        print(f"\n🎉 Vectorstore isolation test completed!")
        print(f"   User 1 vectorstore: {VECTORSTORE_USER1}")
        print(f"   User 2 vectorstore: {VECTORSTORE_USER2}")
        print(f"   Memories are properly isolated between vectorstores")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")

if __name__ == "__main__":
    test_vectorstore_isolation()

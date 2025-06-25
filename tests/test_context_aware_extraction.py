#!/usr/bin/env python3
"""
Test script to verify context-aware memory extraction is working correctly.
"""

import requests
import json

def test_context_aware_extraction():
    """Test that context-aware extraction prevents duplicates."""
    
    print("🧪 Testing Context-Aware Memory Extraction")
    print("=" * 60)
    
    base_url = "http://localhost:5001"
    
    # Test 1: First extraction - should store new information
    print("\n1. First extraction - should store new information:")
    response1 = requests.post(f"{base_url}/api/klines/extract", json={
        "raw_input": "I want to plan a trip to Paris with my family of 4",
        "context_prompt": "I am a travel agent app. Extract user preferences, constraints, and personal details."
    })
    
    if response1.status_code == 200:
        result1 = response1.json()
        print(f"   ✅ Extracted {result1['total_extracted']} memories")
        if result1.get('extracted_memories'):
            for mem in result1['extracted_memories']:
                print(f"      - {mem['text']}")
    else:
        print(f"   ❌ Error: {response1.text}")
        return
    
    # Test 2: Duplicate extraction - should extract nothing
    print("\n2. Duplicate extraction - should extract nothing:")
    response2 = requests.post(f"{base_url}/api/klines/extract", json={
        "raw_input": "I want to plan a trip to Paris",
        "context_prompt": "I am a travel agent app. Extract user preferences, constraints, and personal details."
    })
    
    if response2.status_code == 200:
        result2 = response2.json()
        print(f"   ✅ Extracted {result2['total_extracted']} memories (should be 0)")
        if result2.get('extracted_memories'):
            for mem in result2['extracted_memories']:
                print(f"      - {mem['text']}")
        else:
            print("   🎉 No duplicates extracted!")
    else:
        print(f"   ❌ Error: {response2.text}")
        return
    
    # Test 3: New information - should extract only new details
    print("\n3. New information - should extract only new details:")
    response3 = requests.post(f"{base_url}/api/klines/extract", json={
        "raw_input": "I want to plan a trip to Paris in June and I prefer window seats",
        "context_prompt": "I am a travel agent app. Extract user preferences, constraints, and personal details."
    })
    
    if response3.status_code == 200:
        result3 = response3.json()
        print(f"   ✅ Extracted {result3['total_extracted']} memories")
        if result3.get('extracted_memories'):
            for mem in result3['extracted_memories']:
                print(f"      - {mem['text']}")
        print("   📝 Should only extract timing (June) and seating preference, not Paris trip again")
    else:
        print(f"   ❌ Error: {response3.text}")
        return
    
    # Test 4: Search to verify what's stored
    print("\n4. Searching stored memories:")
    response4 = requests.post(f"{base_url}/api/memory/search", json={
        "query": "Paris trip family",
        "top_k": 10,
        "min_similarity": 0.5
    })
    
    if response4.status_code == 200:
        result4 = response4.json()
        print(f"   ✅ Found {len(result4['memories'])} memories:")
        for i, mem in enumerate(result4['memories'], 1):
            print(f"      {i}. {mem['text']} (score: {mem['score']:.3f})")
    else:
        print(f"   ❌ Error: {response4.text}")

def test_chat_session_context_aware():
    """Test context-aware extraction in chat sessions."""
    
    print("\n" + "=" * 60)
    print("🧪 Testing Chat Session Context-Aware Extraction")
    print("=" * 60)
    
    base_url = "http://localhost:5001"
    
    # Create a chat session
    print("\n1. Creating chat session:")
    session_response = requests.post(f"{base_url}/api/agent/session", json={
        "system_prompt": "You are a helpful travel agent.",
        "config": {"use_memory": True}
    })
    
    if session_response.status_code != 200:
        print(f"   ❌ Error creating session: {session_response.text}")
        return
    
    session_data = session_response.json()
    session_id = session_data['session_id']
    print(f"   ✅ Created session: {session_id}")
    
    # Send first message
    print("\n2. First message - should store new information:")
    msg1_response = requests.post(f"{base_url}/api/agent/session/{session_id}", json={
        "message": "I want to plan a trip to Italy with my 12 kids, some of them are vegetarian"
    })
    
    if msg1_response.status_code == 200:
        result1 = msg1_response.json()
        print(f"   ✅ Response: {result1['message'][:100]}...")
    else:
        print(f"   ❌ Error: {msg1_response.text}")
        return
    
    # Send duplicate message
    print("\n3. Duplicate message - should extract nothing:")
    msg2_response = requests.post(f"{base_url}/api/agent/session/{session_id}", json={
        "message": "I want to plan a trip to Italy"
    })
    
    if msg2_response.status_code == 200:
        result2 = msg2_response.json()
        print(f"   ✅ Response: {result2['message'][:100]}...")
        print("   📝 Check console logs - should show 0 memories extracted")
    else:
        print(f"   ❌ Error: {msg2_response.text}")

if __name__ == "__main__":
    print("🚀 Make sure the web app is running on localhost:5001")
    print("   Run: python web_app.py")
    print()
    
    try:
        test_context_aware_extraction()
        test_chat_session_context_aware()
        print("\n🎉 Context-aware extraction tests completed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("Make sure the web app is running and accessible.")

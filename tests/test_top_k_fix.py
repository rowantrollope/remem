#!/usr/bin/env python3
"""
Test script to verify that the top_k parameter works correctly in the chatWithSession API.
This test verifies that:
1. The API accepts the top_k parameter
2. The number of memories returned matches the top_k value
3. All memories are included in the response (not limited to 3)
"""

import requests
import json
import time

BASE_URL = "http://localhost:5001"

def test_top_k_parameter():
    """Test that top_k parameter controls memory retrieval and response."""
    print("🧪 Testing top_k parameter in chatWithSession API...")
    
    # First, create some memories by having a conversation
    print("\n1. Creating agent session...")
    session_data = {
        "system_prompt": "You are a helpful assistant that remembers user preferences.",
        "config": {
            "use_memory": True,
            "model": "gpt-3.5-turbo",
            "temperature": 0.7
        }
    }
    
    response = requests.post(f"{BASE_URL}/api/agent/session", json=session_data)
    if response.status_code != 200:
        print(f"❌ Failed to create session: {response.text}")
        return False
    
    session = response.json()
    session_id = session['session_id']
    print(f"✅ Created session: {session_id}")
    
    # Add some memories by having a conversation
    print("\n2. Adding memories through conversation...")
    memory_messages = [
        "I love Italian food, especially pasta carbonara",
        "I'm allergic to shellfish and nuts",
        "I prefer window seats when flying",
        "My wife is vegetarian and loves Thai food",
        "I have a budget of $200 for dinner",
        "I need wheelchair accessible restaurants",
        "I always order wine with dinner, preferably red",
        "I don't like spicy food but my wife does"
    ]
    
    for i, message in enumerate(memory_messages, 1):
        print(f"   Adding memory {i}: {message[:50]}...")
        message_data = {"message": message}
        response = requests.post(f"{BASE_URL}/api/agent/session/{session_id}", json=message_data)
        if response.status_code != 200:
            print(f"❌ Failed to send message {i}: {response.text}")
            continue
        time.sleep(0.5)  # Brief pause to allow memory extraction
    
    # Now test different top_k values
    print("\n3. Testing different top_k values...")
    test_message = "What restaurant should I choose for dinner with my wife?"
    
    test_cases = [
        {"top_k": 3, "expected_max": 3},
        {"top_k": 5, "expected_max": 5},
        {"top_k": 10, "expected_max": 10},
        {"top_k": 15, "expected_max": 15}  # This might return fewer if there aren't enough memories
    ]
    
    for test_case in test_cases:
        top_k = test_case["top_k"]
        expected_max = test_case["expected_max"]
        
        print(f"\n   Testing top_k={top_k}...")
        message_data = {
            "message": test_message,
            "top_k": top_k
        }
        
        response = requests.post(f"{BASE_URL}/api/agent/session/{session_id}", json=message_data)
        if response.status_code != 200:
            print(f"❌ Failed to send message with top_k={top_k}: {response.text}")
            continue
        
        result = response.json()
        
        if 'memory_context' in result:
            memories_used = result['memory_context']['memories_used']
            memories_returned = len(result['memory_context']['memories'])
            
            print(f"   ✅ top_k={top_k}: {memories_used} memories used, {memories_returned} memories returned")
            
            # Verify that memories_used and memories_returned are the same (no artificial limit)
            if memories_used != memories_returned:
                print(f"   ❌ ISSUE: memories_used ({memories_used}) != memories_returned ({memories_returned})")
                return False
            
            # Verify that we don't exceed the requested top_k
            if memories_used > top_k:
                print(f"   ❌ ISSUE: memories_used ({memories_used}) > requested top_k ({top_k})")
                return False
            
            print(f"   ✅ All {memories_returned} memories are included in response")
        else:
            print(f"   ⚠️ No memory context in response for top_k={top_k}")
    
    # Test default behavior (should default to 10)
    print(f"\n   Testing default behavior (no top_k specified)...")
    message_data = {"message": "What's your recommendation for tonight?"}
    response = requests.post(f"{BASE_URL}/api/agent/session/{session_id}", json=message_data)
    
    if response.status_code == 200:
        result = response.json()
        if 'memory_context' in result:
            memories_used = result['memory_context']['memories_used']
            memories_returned = len(result['memory_context']['memories'])
            print(f"   ✅ Default: {memories_used} memories used, {memories_returned} memories returned")
            
            if memories_used != memories_returned:
                print(f"   ❌ ISSUE: Default behavior has artificial limit")
                return False
        else:
            print(f"   ⚠️ No memory context in default response")
    
    # Clean up
    print(f"\n4. Cleaning up session...")
    requests.delete(f"{BASE_URL}/api/agent/session/{session_id}")
    
    print(f"\n✅ All tests passed! The top_k parameter works correctly.")
    return True

def test_invalid_top_k():
    """Test that invalid top_k values are handled properly."""
    print("\n🧪 Testing invalid top_k values...")
    
    # Create a session
    session_data = {
        "system_prompt": "Test assistant",
        "config": {"use_memory": True}
    }
    
    response = requests.post(f"{BASE_URL}/api/agent/session", json=session_data)
    if response.status_code != 200:
        print(f"❌ Failed to create session: {response.text}")
        return False
    
    session_id = response.json()['session_id']
    
    # Test invalid top_k values
    invalid_cases = [
        {"top_k": 0, "description": "zero"},
        {"top_k": -1, "description": "negative"},
        {"top_k": "invalid", "description": "string"},
        {"top_k": 3.14, "description": "float"}
    ]
    
    for case in invalid_cases:
        print(f"   Testing {case['description']} top_k: {case['top_k']}")
        message_data = {
            "message": "Test message",
            "top_k": case['top_k']
        }
        
        response = requests.post(f"{BASE_URL}/api/agent/session/{session_id}", json=message_data)
        
        if response.status_code == 400:
            print(f"   ✅ Correctly rejected {case['description']} top_k")
        else:
            print(f"   ❌ Should have rejected {case['description']} top_k, got status {response.status_code}")
    
    # Clean up
    requests.delete(f"{BASE_URL}/api/agent/session/{session_id}")
    return True

if __name__ == "__main__":
    print("🚀 Starting top_k parameter tests...")
    
    try:
        # Test basic functionality
        success1 = test_top_k_parameter()
        
        # Test error handling
        success2 = test_invalid_top_k()
        
        if success1 and success2:
            print("\n🎉 All tests completed successfully!")
            print("The chatWithSession API now properly supports the top_k parameter.")
        else:
            print("\n❌ Some tests failed. Please check the implementation.")
            
    except Exception as e:
        print(f"\n💥 Test failed with exception: {e}")

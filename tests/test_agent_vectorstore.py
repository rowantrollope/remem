#!/usr/bin/env python3
"""
Test script for agent session vectorstore_name parameter functionality.
Tests that agent sessions can use different vectorstores for memory operations.
"""

import requests
import json
import time

BASE_URL = "http://localhost:5001"

def test_agent_session_with_custom_vectorstore():
    """Test agent session with custom vectorstore name."""
    print("🧪 Testing agent session with custom vectorstore...")
    
    # Create session with custom vectorstore
    vectorstore_name = "test_user_123"
    session_data = {
        "system_prompt": "You are a helpful travel agent. Remember user preferences.",
        "config": {
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "use_memory": True
        }
    }

    response = requests.post(f"{BASE_URL}/api/agent/{vectorstore_name}/session", json=session_data)
    if response.status_code != 200:
        print(f"❌ Failed to create session: {response.text}")
        return None
    
    session = response.json()
    session_id = session['session_id']
    print(f"✅ Created session with custom vectorstore: {session_id}")
    print(f"   Memory enabled: {session.get('use_memory', 'not specified')}")
    
    # Send a message with memory content
    message_data = {
        "message": "I prefer window seats on flights and I'm vegetarian",
        "store_memory": True
    }
    response = requests.post(f"{BASE_URL}/api/agent/{vectorstore_name}/session/{session_id}", json=message_data)
    
    if response.status_code != 200:
        print(f"❌ Failed to send message: {response.text}")
        return None
    
    result = response.json()
    print(f"✅ Message sent successfully")
    print(f"   Response: {result['message'][:100]}...")
    print(f"   Memories found: {result.get('memories_found', 0)}")
    
    # Send another message to test memory retrieval
    message_data = {
        "message": "What do you know about my travel preferences?",
        "store_memory": False
    }
    response = requests.post(f"{BASE_URL}/api/agent/{vectorstore_name}/session/{session_id}", json=message_data)
    
    if response.status_code != 200:
        print(f"❌ Failed to send second message: {response.text}")
        return None
    
    result = response.json()
    print(f"✅ Second message sent successfully")
    print(f"   Response: {result['message'][:100]}...")
    print(f"   Memories found: {result.get('memories_found', 0)}")
    
    return session_id

def test_agent_session_different_vectorstore():
    """Test agent session with different vectorstore."""
    print("\n🧪 Testing agent session with different vectorstore...")

    # Create session with different vectorstore
    vectorstore_name = "test_user_456"
    session_data = {
        "system_prompt": "You are a helpful assistant.",
        "config": {
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "use_memory": True
        }
    }

    response = requests.post(f"{BASE_URL}/api/agent/{vectorstore_name}/session", json=session_data)
    if response.status_code != 200:
        print(f"❌ Failed to create session: {response.text}")
        return None

    session = response.json()
    session_id = session['session_id']
    print(f"✅ Created session with vectorstore {vectorstore_name}: {session_id}")

    # Send a message
    message_data = {
        "message": "I like coffee and work from home",
        "store_memory": True
    }
    response = requests.post(f"{BASE_URL}/api/agent/{vectorstore_name}/session/{session_id}", json=message_data)

    if response.status_code != 200:
        print(f"❌ Failed to send message: {response.text}")
        return None

    result = response.json()
    print(f"✅ Message sent successfully")
    print(f"   Response: {result['message'][:100]}...")
    print(f"   Memories found: {result.get('memories_found', 0)}")

    return session_id

def test_memory_isolation():
    """Test that different vectorstores isolate memories."""
    print("\n🧪 Testing memory isolation between vectorstores...")
    
    # Create two sessions with different vectorstores
    vectorstore1 = "isolation_test_1"
    vectorstore2 = "isolation_test_2"

    session1_data = {
        "system_prompt": "You are a helpful assistant.",
        "config": {"use_memory": True}
    }

    session2_data = {
        "system_prompt": "You are a helpful assistant.",
        "config": {"use_memory": True}
    }

    # Create first session and store a memory
    response1 = requests.post(f"{BASE_URL}/api/agent/{vectorstore1}/session", json=session1_data)
    if response1.status_code != 200:
        print(f"❌ Failed to create session 1: {response1.text}")
        return
    
    session1_id = response1.json()['session_id']
    print(f"✅ Created session 1: {session1_id}")
    
    # Store memory in session 1
    message1 = {
        "message": "My favorite color is blue and I live in San Francisco",
        "store_memory": True
    }
    requests.post(f"{BASE_URL}/api/agent/{vectorstore1}/session/{session1_id}", json=message1)
    print("✅ Stored memory in session 1")
    
    # Create second session
    response2 = requests.post(f"{BASE_URL}/api/agent/{vectorstore2}/session", json=session2_data)
    if response2.status_code != 200:
        print(f"❌ Failed to create session 2: {response2.text}")
        return
    
    session2_id = response2.json()['session_id']
    print(f"✅ Created session 2: {session2_id}")
    
    # Query session 2 for session 1's memory
    message2 = {
        "message": "What is my favorite color?",
        "store_memory": False
    }
    response = requests.post(f"{BASE_URL}/api/agent/{vectorstore2}/session/{session2_id}", json=message2)
    
    if response.status_code == 200:
        result = response.json()
        memories_found = result.get('memories_found', 0)
        print(f"✅ Session 2 query completed")
        print(f"   Memories found: {memories_found}")
        print(f"   Response: {result['message'][:100]}...")
        
        if memories_found == 0:
            print("✅ Memory isolation working correctly - no cross-vectorstore access")
        else:
            print("⚠️ Memory isolation may not be working - found memories from other vectorstore")
    else:
        print(f"❌ Failed to query session 2: {response.text}")

def main():
    """Run all tests."""
    print("🚀 Starting agent vectorstore tests...")
    print("=" * 50)
    
    try:
        # Test custom vectorstore in session creation
        session1 = test_agent_session_with_custom_vectorstore()
        
        # Test different vectorstore
        session2 = test_agent_session_different_vectorstore()
        
        # Test memory isolation
        test_memory_isolation()
        
        print("\n" + "=" * 50)
        print("✅ All tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

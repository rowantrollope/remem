#!/usr/bin/env python3
"""
Test script for the new memory-enabled chat API functionality.
Tests both memory-enabled and memory-disabled sessions.
"""

import requests
import json
import time

BASE_URL = "http://localhost:5001"

def test_memory_disabled_session():
    """Test standard chat session without memory."""
    print("🧪 Testing memory-disabled session...")
    
    # Create session without memory
    session_data = {
        "system_prompt": "You are a helpful travel agent. Help users plan trips.",
        "config": {
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 500,
            "use_memory": False
        }
    }
    
    response = requests.post(f"{BASE_URL}/api/agent/session", json=session_data)
    if response.status_code != 200:
        print(f"❌ Failed to create session: {response.text}")
        return None
    
    session = response.json()
    session_id = session['session_id']
    print(f"✅ Created memory-disabled session: {session_id}")
    print(f"   Memory enabled: {session.get('use_memory', 'not specified')}")
    
    # Send a message
    message_data = {"message": "I want to plan a trip to Japan"}
    response = requests.post(f"{BASE_URL}/api/agent/session/{session_id}", json=message_data)
    
    if response.status_code != 200:
        print(f"❌ Failed to send message: {response.text}")
        return None
    
    result = response.json()
    print(f"✅ Message sent successfully")
    print(f"   Response: {result['message'][:100]}...")
    print(f"   Memory context included: {'memory_context' in result}")
    
    return session_id

def test_memory_enabled_session():
    """Test chat session with memory enabled."""
    print("\n🧪 Testing memory-enabled session...")
    
    # Create session with memory
    session_data = {
        "system_prompt": "You are a helpful travel agent. Help users plan trips and remember their preferences.",
        "config": {
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 500,
            "use_memory": True
        }
    }
    
    response = requests.post(f"{BASE_URL}/api/agent/session", json=session_data)
    if response.status_code != 200:
        print(f"❌ Failed to create session: {response.text}")
        return None
    
    session = response.json()
    session_id = session['session_id']
    print(f"✅ Created memory-enabled session: {session_id}")
    print(f"   Memory enabled: {session.get('use_memory', 'not specified')}")
    
    # Send messages to test memory functionality
    messages = [
        "I want to plan a trip to Japan",
        "I prefer luxury hotels and I'm vegetarian",
        "My budget is around $5000 and I love museums"
    ]
    
    for i, message in enumerate(messages, 1):
        print(f"\n--- Message {i} ---")
        message_data = {"message": message}
        response = requests.post(f"{BASE_URL}/api/agent/session/{session_id}", json=message_data)
        
        if response.status_code != 200:
            print(f"❌ Failed to send message: {response.text}")
            continue
        
        result = response.json()
        print(f"User: {message}")
        print(f"Assistant: {result['message'][:150]}...")
        
        if 'memory_context' in result:
            print(f"🧠 Memory context: {result['memory_context']['memories_used']} memories used")
        else:
            print("🧠 No memory context in response")
        
        time.sleep(1)  # Brief pause between messages
    
    return session_id

def test_session_info():
    """Test retrieving session information."""
    print("\n🧪 Testing session info retrieval...")
    
    # Create a memory-enabled session
    session_data = {
        "system_prompt": "You are a test assistant.",
        "config": {"use_memory": True}
    }
    
    response = requests.post(f"{BASE_URL}/api/agent/session", json=session_data)
    if response.status_code != 200:
        print(f"❌ Failed to create session: {response.text}")
        return
    
    session_id = response.json()['session_id']
    
    # Get session info
    response = requests.get(f"{BASE_URL}/api/agent/session/{session_id}")
    if response.status_code != 200:
        print(f"❌ Failed to get session info: {response.text}")
        return
    
    session_info = response.json()
    print(f"✅ Retrieved session info")
    print(f"   Session ID: {session_info['session_id']}")
    print(f"   Memory enabled: {session_info.get('use_memory', 'not specified')}")
    
    if 'memory_info' in session_info:
        memory_info = session_info['memory_info']
        print(f"   Memory info: threshold={memory_info['extraction_threshold']}, buffer_size={memory_info['buffer_size']}")

def main():
    """Run all tests."""
    print("🚀 Starting Chat API Memory Tests")
    print("=" * 50)
    
    try:
        # Test health endpoint first
        response = requests.get(f"{BASE_URL}/api/health")
        if response.status_code != 200:
            print(f"❌ Server not responding: {response.text}")
            return
        
        print("✅ Server is healthy")
        
        # Run tests
        test_memory_disabled_session()
        test_memory_enabled_session()
        test_session_info()
        
        print("\n" + "=" * 50)
        print("🎉 All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure web_app.py is running.")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test script to verify the store_memory parameter functionality in the chat session endpoint.
"""

import requests
import json
import time

BASE_URL = "http://localhost:5001"

def test_store_memory_functionality():
    """Test the store_memory parameter in chat sessions."""
    print("🧪 Testing store_memory parameter functionality")
    print("=" * 60)
    
    # Test 1: Create a memory-enabled session
    print("\n1️⃣ Creating memory-enabled chat session...")
    session_data = {
        "system_prompt": "You are a helpful travel assistant.",
        "config": {
            "use_memory": True,
            "model": "gpt-3.5-turbo",
            "temperature": 0.7
        }
    }
    
    response = requests.post(f"{BASE_URL}/api/agent/session", json=session_data)
    if response.status_code != 200:
        print(f"❌ Failed to create session: {response.text}")
        return
    
    session_id = response.json()["session_id"]
    print(f"✅ Created session: {session_id}")
    
    # Test 2: Send messages with store_memory=true (should extract memories)
    print(f"\n2️⃣ Testing store_memory=true...")
    messages_with_memory = [
        "I prefer window seats when flying",
        "My wife is vegetarian", 
        "We have a budget of $5000 for our trip",
        "I love visiting historical sites",
        "We are planning to visit Paris next month",
        "My family has 4 members including 2 teenagers"
    ]
    
    for i, message in enumerate(messages_with_memory, 1):
        print(f"   Sending message {i}: '{message}'")
        msg_data = {
            "message": message,
            "store_memory": True
        }
        response = requests.post(f"{BASE_URL}/api/chat/session/{session_id}", json=msg_data)
        if response.status_code == 200:
            print(f"   ✅ Message {i} sent successfully")
        else:
            print(f"   ❌ Message {i} failed: {response.text}")
        time.sleep(0.5)  # Small delay to see console output
    
    # Test 3: Send messages with store_memory=false (should NOT extract memories)
    print(f"\n3️⃣ Testing store_memory=false...")
    messages_without_memory = [
        "I also like aisle seats sometimes",
        "My husband prefers spicy food",
        "We might extend our budget to $6000",
        "I enjoy modern art museums too"
    ]
    
    for i, message in enumerate(messages_without_memory, 1):
        print(f"   Sending message {i}: '{message}' (store_memory=false)")
        msg_data = {
            "message": message,
            "store_memory": False
        }
        response = requests.post(f"{BASE_URL}/api/chat/session/{session_id}", json=msg_data)
        if response.status_code == 200:
            print(f"   ✅ Message {i} sent successfully")
        else:
            print(f"   ❌ Message {i} failed: {response.text}")
        time.sleep(0.5)  # Small delay to see console output
    
    # Test 4: Check session info
    print(f"\n4️⃣ Checking session information...")
    response = requests.get(f"{BASE_URL}/api/chat/session/{session_id}")
    if response.status_code == 200:
        session_info = response.json()
        print(f"   ✅ Session has {session_info['message_count']} messages")
        if 'memory_info' in session_info:
            print(f"   📊 Memory info: {session_info['memory_info']}")
    else:
        print(f"   ❌ Failed to get session info: {response.text}")
    
    # Test 5: Check stored memories
    print(f"\n5️⃣ Checking stored memories...")
    response = requests.get(f"{BASE_URL}/api/memory/info")
    if response.status_code == 200:
        memory_info = response.json()
        print(f"   📊 Total memories in system: {memory_info.get('total_memories', 'unknown')}")
    else:
        print(f"   ❌ Failed to get memory info: {response.text}")
    
    print(f"\n✅ Test completed! Check the console output above for:")
    print(f"   - Session ID logging in all messages")
    print(f"   - Memory extraction happening for store_memory=true")
    print(f"   - Memory extraction being skipped for store_memory=false")
    print(f"   - Clear console messages about extraction decisions")

if __name__ == "__main__":
    try:
        # Check if server is running
        response = requests.get(f"{BASE_URL}/api/health", timeout=5)
        if response.status_code == 200:
            test_store_memory_functionality()
        else:
            print("❌ Server health check failed")
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure web_app.py is running.")
    except Exception as e:
        print(f"❌ Error: {e}")

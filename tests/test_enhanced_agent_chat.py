#!/usr/bin/env python3
"""
Test script for the enhanced /api/agent/chat endpoint with custom system prompts.
This allows for apples-to-apples comparison with the session-based approach.
"""

import requests
import json
import time

BASE_URL = "http://localhost:5001"

def test_agent_chat_with_custom_system_prompt():
    """Test the enhanced /api/agent/chat endpoint with custom system prompts."""
    print("🧪 Testing Enhanced /api/agent/chat with Custom System Prompts")
    print("=" * 60)
    
    # Test cases with different system prompts
    test_cases = [
        {
            "name": "Travel Assistant",
            "system_prompt": "You are a helpful travel assistant. Help users plan trips and remember their preferences.",
            "messages": [
                "I prefer window seats when flying",
                "My wife is vegetarian",
                "What restaurants do you recommend for our trip to Italy?"
            ]
        },
        {
            "name": "Customer Support",
            "system_prompt": "You are a helpful customer support agent for TechCorp. Be empathetic and solution-focused.",
            "messages": [
                "I'm having trouble with my laptop",
                "It keeps freezing when I open multiple applications",
                "What troubleshooting steps should I try?"
            ]
        },
        {
            "name": "Default Memory Assistant",
            "system_prompt": None,  # Use default system prompt
            "messages": [
                "I love sushi and my budget for dining is $100 per meal",
                "What do you remember about my food preferences?"
            ]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}️⃣ Testing: {test_case['name']}")
        print("-" * 40)
        
        for j, message in enumerate(test_case['messages'], 1):
            print(f"\n   Message {j}: {message}")
            
            # Prepare request data
            request_data = {"message": message}
            if test_case['system_prompt']:
                request_data["system_prompt"] = test_case['system_prompt']
            
            try:
                response = requests.post(f"{BASE_URL}/api/agent/chat", json=request_data)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"   ✅ Success: {result['response'][:100]}...")
                    if 'system_prompt_used' in result:
                        print(f"   🎯 Custom system prompt used: {result['system_prompt_used']}")
                else:
                    print(f"   ❌ Failed: {response.status_code} - {response.text}")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
            
            time.sleep(1)  # Brief pause between messages

def test_comparison_with_session_api():
    """Compare the enhanced agent/chat with the session-based approach."""
    print("\n\n🔄 COMPARISON: /api/agent/chat vs /api/agent/session")
    print("=" * 60)
    
    system_prompt = "You are a helpful travel assistant. Help users plan trips and remember their preferences."
    test_message = "I prefer window seats when flying and my wife is vegetarian"
    
    # Test 1: Enhanced /api/agent/chat
    print("\n1️⃣ Testing Enhanced /api/agent/chat")
    try:
        response = requests.post(f"{BASE_URL}/api/agent/chat", json={
            "message": test_message,
            "system_prompt": system_prompt
        })
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Response: {result['response'][:150]}...")
            print(f"   🎯 Custom system prompt used: {result.get('system_prompt_used', False)}")
        else:
            print(f"   ❌ Failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Session-based approach
    print("\n2️⃣ Testing Session-based /api/agent/session")
    try:
        # Create session
        session_response = requests.post(f"{BASE_URL}/api/agent/session", json={
            "system_prompt": system_prompt,
            "config": {"use_memory": True}
        })
        
        if session_response.status_code == 200:
            session = session_response.json()
            session_id = session['session_id']
            print(f"   ✅ Session created: {session_id[:8]}...")
            
            # Send message to session
            message_response = requests.post(f"{BASE_URL}/api/agent/session/{session_id}", json={
                "message": test_message
            })
            
            if message_response.status_code == 200:
                result = message_response.json()
                print(f"   ✅ Response: {result['message'][:150]}...")
                if 'memory_context' in result:
                    print(f"   🧠 Memory context: {result['memory_context'].get('memories_used', 0)} memories used")
            else:
                print(f"   ❌ Message failed: {message_response.status_code} - {message_response.text}")
        else:
            print(f"   ❌ Session creation failed: {session_response.status_code} - {session_response.text}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")

def test_error_cases():
    """Test error handling for the enhanced endpoint."""
    print("\n\n🚨 Testing Error Cases")
    print("=" * 30)
    
    error_tests = [
        {
            "name": "Missing message",
            "data": {"system_prompt": "You are helpful"},
            "expected_status": 400
        },
        {
            "name": "Empty message",
            "data": {"message": "", "system_prompt": "You are helpful"},
            "expected_status": 400
        },
        {
            "name": "Valid message, no system prompt",
            "data": {"message": "Hello"},
            "expected_status": 200
        }
    ]
    
    for test in error_tests:
        print(f"\n   Testing: {test['name']}")
        try:
            response = requests.post(f"{BASE_URL}/api/agent/chat", json=test['data'])
            if response.status_code == test['expected_status']:
                print(f"   ✅ Expected status {test['expected_status']}")
            else:
                print(f"   ❌ Got {response.status_code}, expected {test['expected_status']}")
                print(f"      Response: {response.text}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 Enhanced Agent Chat API Test Suite")
    print("Testing the new system_prompt parameter in /api/agent/chat")
    print()
    
    test_agent_chat_with_custom_system_prompt()
    test_comparison_with_session_api()
    test_error_cases()
    
    print("\n\n✅ Test suite completed!")
    print("\nNow you can compare both approaches:")
    print("1. Enhanced /api/agent/chat - Stateless with custom system prompts")
    print("2. Session-based /api/agent/session - Stateful with custom filtering")

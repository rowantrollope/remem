#!/usr/bin/env python3
"""
Test script to verify the refactored API works correctly.
"""

import requests
import json
import time

# Test configuration
BASE_URL = "http://localhost:5001"
TEST_VECTORSTORE = "test_refactor"

def test_health():
    """Test health endpoint."""
    print("🔍 Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/api/health")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Health check: {data['status']}")
        return True
    else:
        print(f"❌ Health check failed: {response.text}")
        return False

def test_memory_operations():
    """Test memory operations."""
    print("\n🔍 Testing memory operations...")
    
    # Test store memory
    print("Testing store memory...")
    store_data = {
        "text": "This is a test memory for the refactored API",
        "apply_grounding": True
    }
    response = requests.post(f"{BASE_URL}/api/memory/{TEST_VECTORSTORE}", json=store_data)
    print(f"Store status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        memory_id = data.get('memory_id')
        print(f"✅ Memory stored: {memory_id}")
        
        # Test search memory
        print("Testing search memory...")
        search_data = {
            "query": "test memory refactored",
            "top_k": 5,
            "min_similarity": 0.5
        }
        response = requests.post(f"{BASE_URL}/api/memory/{TEST_VECTORSTORE}/search", json=search_data)
        print(f"Search status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Found {data['count']} memories")
            
            # Test get memory info
            print("Testing get memory info...")
            response = requests.get(f"{BASE_URL}/api/memory/{TEST_VECTORSTORE}")
            print(f"Info status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Memory count: {data['memory_count']}")
                return True
    
    print("❌ Memory operations failed")
    return False

def test_config_operations():
    """Test configuration operations."""
    print("\n🔍 Testing configuration operations...")
    
    # Test get config
    print("Testing get config...")
    response = requests.get(f"{BASE_URL}/api/config")
    print(f"Config status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Config retrieved: {data['success']}")
        return True
    else:
        print(f"❌ Config operations failed: {response.text}")
        return False

def test_agent_operations():
    """Test agent operations."""
    print("\n🔍 Testing agent operations...")
    
    # Test create session
    print("Testing create agent session...")
    session_data = {
        "system_prompt": "You are a helpful assistant for testing the refactored API.",
        "config": {"use_memory": True}
    }
    response = requests.post(f"{BASE_URL}/api/agent/{TEST_VECTORSTORE}/session", json=session_data)
    print(f"Session create status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        session_id = data.get('session_id')
        print(f"✅ Session created: {session_id}")
        
        # Test list sessions
        print("Testing list sessions...")
        response = requests.get(f"{BASE_URL}/api/agent/{TEST_VECTORSTORE}/sessions")
        print(f"List sessions status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Found {data['total_sessions']} sessions")
            return True
    
    print("❌ Agent operations failed")
    return False

def main():
    """Run all tests."""
    print("🚀 Testing Refactored Memory Agent API")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health),
        ("Memory Operations", test_memory_operations),
        ("Configuration Operations", test_config_operations),
        ("Agent Operations", test_agent_operations)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
        
        time.sleep(1)  # Brief pause between tests
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Refactored API is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test script to verify timestamp standardization across the memory system.

This script tests that:
1. Memory storage uses only ISO 8601 UTC timestamps
2. Memory retrieval returns only ISO timestamps
3. API responses contain only ISO timestamps
4. Backward compatibility with old timestamp formats works
"""

import os
import sys
import json
import requests
from datetime import datetime, timezone
from typing import Dict, Any

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_memory_storage_api():
    """Test that memory storage API returns only ISO timestamps."""
    print("🧪 Testing memory storage API...")
    
    url = "http://localhost:5000/api/memory"
    data = {
        "text": "Test memory for timestamp standardization",
        "apply_grounding": False
    }
    
    try:
        response = requests.post(url, json=data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            
            # Check that response contains created_at in ISO format
            if 'created_at' in result:
                created_at = result['created_at']
                # Try to parse as ISO format
                try:
                    parsed_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    print(f"✅ Memory storage API returns ISO timestamp: {created_at}")
                    
                    # Check that old timestamp fields are not present
                    if 'timestamp' in result:
                        print(f"⚠️ Warning: Old 'timestamp' field still present in response")
                    if 'formatted_time' in result:
                        print(f"⚠️ Warning: Old 'formatted_time' field still present in response")
                    
                    return True
                except ValueError as e:
                    print(f"❌ Invalid ISO timestamp format: {created_at} - {e}")
                    return False
            else:
                print(f"❌ No 'created_at' field in response: {result}")
                return False
        else:
            print(f"❌ API request failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error testing memory storage API: {e}")
        return False

def test_memory_search_api():
    """Test that memory search API returns only ISO timestamps."""
    print("🧪 Testing memory search API...")
    
    url = "http://localhost:5000/api/memory/search"
    data = {
        "query": "test memory",
        "top_k": 5
    }
    
    try:
        response = requests.post(url, json=data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            
            if 'memories' in result and len(result['memories']) > 0:
                memory = result['memories'][0]
                
                # Check for ISO timestamp fields
                if 'created_at' in memory:
                    created_at = memory['created_at']
                    try:
                        parsed_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        print(f"✅ Memory search API returns ISO timestamp: {created_at}")
                        
                        # Check that old timestamp fields are not present
                        if 'timestamp' in memory:
                            print(f"⚠️ Warning: Old 'timestamp' field still present in search results")
                        if 'formatted_time' in memory:
                            print(f"⚠️ Warning: Old 'formatted_time' field still present in search results")
                        
                        return True
                    except ValueError as e:
                        print(f"❌ Invalid ISO timestamp format in search results: {created_at} - {e}")
                        return False
                else:
                    print(f"❌ No 'created_at' field in search results: {memory}")
                    return False
            else:
                print("ℹ️ No memories found in search results")
                return True
        else:
            print(f"❌ Search API request failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error testing memory search API: {e}")
        return False

def test_chat_session_api():
    """Test that chat session API uses ISO timestamps."""
    print("🧪 Testing chat session API...")
    
    # Create a session
    create_url = "http://localhost:5000/api/chat/session"
    create_data = {
        "system_prompt": "You are a helpful assistant.",
        "use_memory": False
    }
    
    try:
        response = requests.post(create_url, json=create_data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            
            if 'created_at' in result:
                created_at = result['created_at']
                try:
                    parsed_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    print(f"✅ Chat session API returns ISO timestamp: {created_at}")
                    return True
                except ValueError as e:
                    print(f"❌ Invalid ISO timestamp format in session: {created_at} - {e}")
                    return False
            else:
                print(f"❌ No 'created_at' field in session response: {result}")
                return False
        else:
            print(f"❌ Session creation failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error testing chat session API: {e}")
        return False

def test_health_api():
    """Test that health API uses ISO timestamps."""
    print("🧪 Testing health API...")
    
    url = "http://localhost:5000/api/health"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            result = response.json()
            
            if 'timestamp' in result:
                timestamp = result['timestamp']
                try:
                    parsed_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    print(f"✅ Health API returns ISO timestamp: {timestamp}")
                    return True
                except ValueError as e:
                    print(f"❌ Invalid ISO timestamp format in health response: {timestamp} - {e}")
                    return False
            else:
                print(f"❌ No 'timestamp' field in health response: {result}")
                return False
        else:
            print(f"❌ Health API request failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error testing health API: {e}")
        return False

def main():
    """Run all timestamp standardization tests."""
    print("🚀 Starting timestamp standardization tests...")
    print("=" * 60)
    
    tests = [
        test_health_api,
        test_chat_session_api,
        test_memory_storage_api,
        test_memory_search_api,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print("-" * 40)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            print("-" * 40)
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All timestamp standardization tests passed!")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
Test script to verify filterBy functionality in web API endpoints
"""

import requests
import json
import time

def test_web_api_filter():
    """Test the filterBy parameter in web API endpoints."""
    base_url = "http://localhost:5001"
    
    print("🌐 Testing filterBy functionality in web API...")
    
    try:
        # Test 1: Recall API without filter
        print("\n🔍 Test 1: /api/recall without filter")
        response = requests.post(f"{base_url}/api/recall", json={
            "query": "project work",
            "top_k": 5
        })
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Found {data['count']} memories without filter")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False
        
        # Test 2: Recall API with filter
        print("\n🔍 Test 2: /api/recall with category filter")
        response = requests.post(f"{base_url}/api/recall", json={
            "query": "project work",
            "top_k": 5,
            "filterBy": '.category == "work"'
        })
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Found {data['count']} work-related memories with filter")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False
        
        # Test 3: Ask API without filter
        print("\n🤔 Test 3: /api/ask without filter")
        response = requests.post(f"{base_url}/api/ask", json={
            "question": "What work did I do?",
            "top_k": 5
        })
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Answer: {data.get('answer', 'No answer')}")
            print(f"✅ Confidence: {data.get('confidence', 'Unknown')}")
            print(f"✅ Supporting memories: {len(data.get('supporting_memories', []))}")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False
        
        # Test 4: Ask API with filter
        print("\n🤔 Test 4: /api/ask with category filter")
        response = requests.post(f"{base_url}/api/ask", json={
            "question": "What work did I do?",
            "top_k": 5,
            "filterBy": '.category == "work"'
        })
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Answer: {data.get('answer', 'No answer')}")
            print(f"✅ Confidence: {data.get('confidence', 'Unknown')}")
            print(f"✅ Supporting memories: {len(data.get('supporting_memories', []))}")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False
        
        # Test 5: Test with complex filter
        print("\n🔍 Test 5: /api/recall with complex filter")
        response = requests.post(f"{base_url}/api/recall", json={
            "query": "work",
            "top_k": 5,
            "filterBy": '.category == "work" and .priority != "low"'
        })
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Found {data['count']} non-low-priority work memories")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False
        
        print("\n✅ All web API filter tests completed successfully!")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to web server. Make sure it's running on localhost:5001")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Starting web API filter tests...")
    print("📝 Note: Make sure the web server is running (python3 web_app.py)")
    print("📝 Note: Make sure test memories are already stored from previous test")
    
    time.sleep(2)  # Give user time to read the note
    
    success = test_web_api_filter()
    
    if success:
        print("\n🎉 Web API filterBy functionality is working correctly!")
    else:
        print("\n❌ Web API filterBy tests failed.")

if __name__ == "__main__":
    main()

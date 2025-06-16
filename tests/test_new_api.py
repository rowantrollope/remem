#!/usr/bin/env python3
"""
Test script for the new Memory Agent API structure

This script tests all the new API endpoints to ensure they work correctly.
"""

import requests
import json
import time

BASE_URL = "http://localhost:5001"

def test_api_endpoint(method, endpoint, data=None, expected_status=200):
    """Test an API endpoint and return the response."""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data, headers={"Content-Type": "application/json"})
        elif method == "DELETE":
            response = requests.delete(url)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        print(f"{method} {endpoint}")
        print(f"Status: {response.status_code}")
        
        if response.status_code == expected_status:
            print("✅ SUCCESS")
        else:
            print(f"❌ FAILED - Expected {expected_status}, got {response.status_code}")
        
        try:
            response_json = response.json()
            print(f"Response: {json.dumps(response_json, indent=2)}")
        except:
            print(f"Response: {response.text}")
        
        print("-" * 50)
        return response
    
    except requests.exceptions.ConnectionError:
        print(f"❌ CONNECTION ERROR - Is the server running at {BASE_URL}?")
        print("-" * 50)
        return None
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print("-" * 50)
        return None

def main():
    """Test all the new API endpoints."""
    print("🧪 Testing New Memory Agent API Structure")
    print("=" * 60)
    
    # Test health check first
    print("1. Testing Health Check")
    health_response = test_api_endpoint("GET", "/api/health")
    if not health_response or health_response.status_code != 200:
        print("❌ Server not healthy. Please start the web app first:")
        print("   python web_app.py")
        return
    
    # Test memory storage
    print("2. Testing Memory Storage")
    store_response = test_api_endpoint("POST", "/api/memory", {
        "text": "I went to Mario's Italian Restaurant and had amazing pasta carbonara",
        "apply_grounding": True
    })
    
    memory_id = None
    if store_response and store_response.status_code == 200:
        try:
            memory_id = store_response.json().get("memory_id")
            print(f"📝 Stored memory with ID: {memory_id}")
        except:
            pass
    
    # Store another memory for testing
    print("3. Testing Memory Storage (Second Memory)")
    test_api_endpoint("POST", "/api/memory", {
        "text": "My cat Molly is 3 years old and loves tuna treats",
        "apply_grounding": True
    })
    
    # Test memory search
    print("4. Testing Memory Search")
    test_api_endpoint("POST", "/api/memory/search", {
        "query": "Italian restaurant food",
        "top_k": 5
    })
    
    # Test advanced question answering
    print("5. Testing Advanced Question Answering")
    test_api_endpoint("POST", "/api/memory/answer", {
        "question": "What did I eat at Mario's restaurant?",
        "top_k": 5
    })
    
    # Test memory info
    print("6. Testing Memory Info")
    test_api_endpoint("GET", "/api/memory")
    
    # Test context management
    print("7. Testing Context Setting")
    test_api_endpoint("POST", "/api/memory/context", {
        "location": "Jakarta, Indonesia",
        "activity": "testing API",
        "people_present": ["Alice", "Bob"],
        "weather": "sunny"
    })
    
    print("8. Testing Context Retrieval")
    test_api_endpoint("GET", "/api/memory/context")
    
    # Test chat interface
    print("9. Testing Chat Interface")
    test_api_endpoint("POST", "/api/chat", {
        "message": "What do I know about my pets and food preferences?"
    })
    
    # Test memory deletion (if we have a memory ID)
    if memory_id:
        print("10. Testing Memory Deletion")
        test_api_endpoint("DELETE", f"/api/memory/{memory_id}")
    
    # Test clear all memories
    print("11. Testing Clear All Memories")
    test_api_endpoint("DELETE", "/api/memory")
    
    print("🎉 API Testing Complete!")
    print("\nAPI Summary:")
    print("✅ Developer Memory APIs:")
    print("   - POST /api/memory (store)")
    print("   - POST /api/memory/search (vector search)")
    print("   - POST /api/memory/answer (advanced Q&A)")
    print("   - GET /api/memory (info)")
    print("   - POST/GET /api/memory/context (context management)")
    print("   - DELETE /api/memory/{id} (delete specific)")
    print("   - DELETE /api/memory (clear all)")
    print("✅ Chat Application API:")
    print("   - POST /api/chat (conversational interface)")
    print("✅ System APIs:")
    print("   - GET /api/health (health check)")

def test_error_cases():
    """Test error cases to ensure proper error handling."""
    print("\n🔍 Testing Error Cases")
    print("=" * 40)
    
    # Test missing required parameters
    print("1. Testing Missing Text Parameter")
    test_api_endpoint("POST", "/api/memory", {}, expected_status=400)
    
    print("2. Testing Missing Query Parameter")
    test_api_endpoint("POST", "/api/memory/search", {}, expected_status=400)
    
    print("3. Testing Missing Question Parameter")
    test_api_endpoint("POST", "/api/memory/answer", {}, expected_status=400)
    
    print("4. Testing Missing Message Parameter")
    test_api_endpoint("POST", "/api/chat", {}, expected_status=400)
    
    print("5. Testing Invalid Memory ID")
    test_api_endpoint("DELETE", "/api/memory/invalid-id", expected_status=404)

if __name__ == "__main__":
    main()
    
    # Ask if user wants to test error cases
    test_errors = input("\nDo you want to test error cases? (y/n): ").lower().strip()
    if test_errors == 'y':
        test_error_cases()
    
    print("\n📚 For complete API documentation, see: NEW_API_DOCS.md")

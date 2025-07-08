#!/usr/bin/env python3
"""
Test script for the Minsky-inspired Memory Agent API

This script tests the three-layer API architecture:
- NEME API: Fundamental memory operations
- K-LINE API: Reflective operations and reasoning
- AGENT API: High-level orchestration

Based on Marvin Minsky's Society of Mind theory.
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
    """Test the Minsky-inspired three-layer API architecture."""
    print("🧠 Testing Minsky Memory Agent API")
    print("🔬 Three-Layer Architecture: Memories → K-lines → Agent")
    print("=" * 60)
    
    # Test health check first
    print("1. Testing Health Check")
    health_response = test_api_endpoint("GET", "/api/health")
    if not health_response or health_response.status_code != 200:
        print("❌ Server not healthy. Please start the web app first:")
        print("   python web_app.py")
        return
    
    # ===== NEME API TESTS =====
    print("\n🧠 NEME API - Fundamental Memory Operations")
    print("=" * 50)

    # Test atomic memory storage
    print("2. Testing Neme Storage (Atomic Memory)")
    store_response = test_api_endpoint("POST", "/api/memory", {
        "text": "I went to Mario's Italian Restaurant and had amazing pasta carbonara",
        "apply_grounding": True
    })

    memory_id = None
    if store_response and store_response.status_code == 200:
        try:
            memory_id = store_response.json().get("memory_id")
            print(f"📝 Stored Neme with ID: {memory_id}")
        except:
            pass

    # Store another atomic memory for testing
    print("3. Testing Neme Storage (Second Atomic Memory)")
    test_api_endpoint("POST", "/api/memory", {
        "text": "My cat Molly is 3 years old and loves tuna treats",
        "apply_grounding": True
    })
    
    # Test atomic memory search
    print("4. Testing Neme Search (Vector Similarity)")
    test_api_endpoint("POST", "/api/memory/search", {
        "query": "Italian restaurant food",
        "top_k": 5
    })

    # Test memory info
    print("5. Testing Neme System Info")
    test_api_endpoint("GET", "/api/memory")

    # Test context management for grounding
    print("6. Testing Neme Context Setting")
    test_api_endpoint("POST", "/api/memory/context", {
        "location": "Jakarta, Indonesia",
        "activity": "testing API",
        "people_present": ["Alice", "Bob"],
        "weather": "sunny"
    })

    print("7. Testing Neme Context Retrieval")
    test_api_endpoint("GET", "/api/memory/context")

    # ===== K-LINE API TESTS =====
    print("\n🧠 K-LINE API - Mental State Construction & Reasoning")
    print("=" * 50)

    # Test mental state construction
    print("8. Testing K-line Construction (Mental State)")
    test_api_endpoint("POST", "/api/klines/recall", {
        "query": "food and pets",
        "top_k": 5
    })

    # Test question answering with reasoning
    print("9. Testing K-line Reasoning (Question Answering)")
    test_api_endpoint("POST", "/api/klines/answer", {
        "question": "What did I eat at Mario's restaurant?",
        "top_k": 5
    })

    # Test memory extraction
    print("10. Testing K-line Extraction (Experience → Memory)")
    test_api_endpoint("POST", "/api/klines/extract", {
        "raw_input": "User: I love sushi and my wife is vegetarian. We went to a great Japanese place last week.",
        "context_prompt": "Extract dining preferences and family information",
        "apply_grounding": True
    })
    
    # ===== AGENT API TESTS =====
    print("\n🤖 AGENT API - Full Cognitive Architecture")
    print("=" * 50)

    # Test full conversational agent
    print("11. Testing Agent Chat (Full Orchestration)")
    test_api_endpoint("POST", "/api/agent/chat", {
        "message": "What do I know about my pets and food preferences?"
    })

    # Test agent session creation
    print("12. Testing Agent Session Creation")
    session_response = test_api_endpoint("POST", "/api/agent/session", {
        "system_prompt": "You are a helpful assistant with memory of user preferences",
        "config": {
            "use_memory": True,
            "model": "gpt-3.5-turbo",
            "temperature": 0.7
        }
    })

    session_id = None
    if session_response and session_response.status_code == 200:
        try:
            session_id = session_response.json().get("session_id")
            print(f"🆕 Created agent session: {session_id}")
        except:
            pass

    # Test agent session conversation
    if session_id:
        print("13. Testing Agent Session Conversation")
        test_api_endpoint("POST", f"/api/agent/session/{session_id}", {
            "message": "Remember that I prefer Italian food and have a cat named Molly"
        })

    # ===== CLEANUP TESTS =====
    print("\n🧹 CLEANUP - Testing Deletion Operations")
    print("=" * 50)

    # Test memory deletion (if we have a memory ID)
    if memory_id:
        print("14. Testing Neme Deletion")
        test_api_endpoint("DELETE", f"/api/memory/{memory_id}")

    # Test clear all memories
    print("15. Testing Clear All Memories")
    test_api_endpoint("DELETE", "/api/memory")
    
    print("🎉 Minsky Memory Agent API Testing Complete!")
    print("\n📊 Three-Layer Architecture Summary:")
    print("🧠 NEME API (Fundamental Memory Operations):")
    print("   - POST /api/memory (store atomic memory)")
    print("   - POST /api/memory/search (vector similarity)")
    print("   - GET /api/memory (system info)")
    print("   - POST/GET /api/memory/context (context management)")
    print("   - DELETE /api/memory/{id} (delete specific)")
    print("   - DELETE /api/memory (clear all)")
    print("\n🧠 K-LINE API (Mental State Construction):")
    print("   - POST /api/klines/recall (construct mental state)")
    print("   - POST /api/klines/answer (reasoning & Q&A)")
    print("   - POST /api/klines/extract (experience → memory)")
    print("\n🤖 AGENT API (Full Cognitive Architecture):")
    print("   - POST /api/agent/chat (conversational interface)")
    print("   - POST /api/agent/session (create session)")
    print("   - POST /api/agent/session/{id} (session conversation)")
    print("\n✅ System APIs:")
    print("   - GET /api/health (health check)")
    print("\n📖 Based on Marvin Minsky's Society of Mind theory")

def test_error_cases():
    """Test error cases to ensure proper error handling across all API layers."""
    print("\n🔍 Testing Error Cases - Three-Layer Architecture")
    print("=" * 50)

    # NEME API error cases
    print("NEME API Error Cases:")
    print("1. Testing Missing Text Parameter")
    test_api_endpoint("POST", "/api/memory", {}, expected_status=400)

    print("2. Testing Missing Query Parameter")
    test_api_endpoint("POST", "/api/memory/search", {}, expected_status=400)

    print("3. Testing Invalid Neme ID")
    test_api_endpoint("DELETE", "/api/memory/invalid-id", expected_status=404)

    # K-LINE API error cases
    print("\nK-LINE API Error Cases:")
    print("4. Testing Missing Question Parameter")
    test_api_endpoint("POST", "/api/klines/answer", {}, expected_status=400)

    print("5. Testing Missing Raw Input for Extraction")
    test_api_endpoint("POST", "/api/klines/extract", {}, expected_status=400)

    # AGENT API error cases
    print("\nAGENT API Error Cases:")
    print("6. Testing Missing Message Parameter")
    test_api_endpoint("POST", "/api/agent/chat", {}, expected_status=400)

    print("7. Testing Invalid Session ID")
    test_api_endpoint("POST", "/api/agent/session/invalid-session", {
        "message": "test"
    }, expected_status=404)

if __name__ == "__main__":
    main()
    
    # Ask if user wants to test error cases
    test_errors = input("\nDo you want to test error cases? (y/n): ").lower().strip()
    if test_errors == 'y':
        test_error_cases()
    
    print("\n📚 For complete API documentation, see:")
    print("   - API_DOCUMENTATION.md (Minsky framework & API reference)")
    print("   - EXAMPLES.md (Three-layer usage examples)")
    print("\n🧠 Minsky's Society of Mind: Memories + K-lines = Intelligence")

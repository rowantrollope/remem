#!/usr/bin/env python3
"""
Test the improved /api/chat with automatic memory extraction
"""

import requests
import json
import time

def test_chat_with_auto_extraction():
    """Test the improved chat API that automatically extracts memories."""
    
    base_url = "http://localhost:5001"
    
    # Test conversation that should trigger memory extraction
    travel_conversation = [
        "Hi, I'm planning a trip to Paris",
        "I prefer window seats when flying",
        "My wife is vegetarian so we'll need vegetarian meal options",
        "Our budget is around $8000 for the whole trip",
        "We have two kids aged 8 and 12",
        "I have mobility issues so we need accessible accommodations"
    ]
    
    print("🧳 Testing Travel Agent Chat with Auto Memory Extraction")
    print("=" * 60)
    
    # Set context for travel agent
    try:
        context_response = requests.post(f"{base_url}/api/memory/context", 
            json={
                "activity": "travel_planning",
                "location": "Travel Agency Office"
            },
            timeout=10
        )
        print(f"✅ Context set: {context_response.status_code}")
    except Exception as e:
        print(f"⚠️ Could not set context: {e}")
    
    # Send messages to chat API
    for i, message in enumerate(travel_conversation, 1):
        print(f"\n--- Message {i} ---")
        print(f"User: {message}")
        
        try:
            # Send to chat API
            chat_response = requests.post(f"{base_url}/api/chat",
                json={"message": message},
                timeout=15
            )
            
            if chat_response.status_code == 200:
                chat_result = chat_response.json()
                print(f"Assistant: {chat_result['response']}")
            else:
                print(f"❌ Chat error: {chat_response.status_code} - {chat_response.text}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Server not running. Start with: python3 web_app.py")
            return
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Small delay between messages
        time.sleep(1)
    
    # Check what memories were extracted
    print(f"\n{'='*60}")
    print("🧠 Checking Extracted Memories")
    print("=" * 60)
    
    try:
        # Get memory info
        memory_info_response = requests.get(f"{base_url}/api/memory", timeout=10)
        if memory_info_response.status_code == 200:
            memory_info = memory_info_response.json()
            print(f"📊 Total memories stored: {memory_info.get('memory_count', 0)}")
        
        # Search for travel-related memories
        search_response = requests.post(f"{base_url}/api/memory/search",
            json={
                "query": "travel preferences and constraints",
                "top_k": 10
            },
            timeout=10
        )
        
        if search_response.status_code == 200:
            search_result = search_response.json()
            memories = search_result.get('memories', [])
            
            print(f"\n🔍 Found {len(memories)} travel-related memories:")
            for i, memory in enumerate(memories, 1):
                print(f"   {i}. {memory['text']}")
                print(f"      Relevance: {memory['relevance_score']:.1f}% | Time: {memory['timestamp']}")
                if memory.get('tags'):
                    print(f"      Tags: {', '.join(memory['tags'])}")
                print()
        
    except Exception as e:
        print(f"❌ Error checking memories: {e}")

def test_memory_aware_responses():
    """Test that the chat API uses previously extracted memories."""
    
    base_url = "http://localhost:5001"
    
    print("\n🤖 Testing Memory-Aware Responses")
    print("=" * 40)
    
    # Ask questions that should use extracted memories
    memory_questions = [
        "What are my seating preferences?",
        "Tell me about my dietary restrictions",
        "What's my travel budget?",
        "Do I have any accessibility needs?"
    ]
    
    for question in memory_questions:
        print(f"\nUser: {question}")
        
        try:
            chat_response = requests.post(f"{base_url}/api/chat",
                json={"message": question},
                timeout=10
            )
            
            if chat_response.status_code == 200:
                chat_result = chat_response.json()
                print(f"Assistant: {chat_result['response']}")
            else:
                print(f"❌ Error: {chat_response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Run all tests."""
    print("🚀 Testing Improved Chat API with Automatic Memory Extraction")
    print("=" * 70)
    
    # Test 1: Chat with automatic extraction
    test_chat_with_auto_extraction()
    
    # Test 2: Memory-aware responses
    test_memory_aware_responses()
    
    print(f"\n{'='*70}")
    print("✅ Testing Complete!")
    print("\nKey Improvements:")
    print("• Chat API now automatically extracts memories during conversation")
    print("• No need for separate /api/memory/extract calls")
    print("• Real-time learning while providing immediate responses")
    print("• Memory-aware responses using previously learned information")

if __name__ == "__main__":
    main()

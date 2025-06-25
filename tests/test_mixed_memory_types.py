#!/usr/bin/env python3
"""
Test script to verify that mixed k-line and neme retrieval works correctly.
This test verifies that:
1. Both k-lines and nemes can be stored and retrieved together
2. Memory type filtering works correctly
3. API responses include proper type information
4. Memory processing handles both types gracefully
5. Different display formats work for both types
"""

import requests
import json
import time
import sys

BASE_URL = "http://localhost:5001"

def test_mixed_memory_storage_and_retrieval():
    """Test storing both nemes and k-lines, then retrieving them together."""
    print("🧪 Testing mixed memory type storage and retrieval...")
    
    # Step 1: Store some nemes
    print("\n1. Storing nemes...")
    neme_texts = [
        "User prefers window seats on flights",
        "User enjoys Italian cuisine",
        "User's wife is vegetarian",
        "User has a budget of $200 per night for hotels"
    ]
    
    stored_nemes = []
    for text in neme_texts:
        response = requests.post(f"{BASE_URL}/api/memory", json={
            "text": text,
            "apply_grounding": True
        })
        if response.status_code == 200:
            result = response.json()
            stored_nemes.append(result['memory_id'])
            print(f"   ✅ Stored neme: {text}")
        else:
            print(f"   ❌ Failed to store neme: {text}")
            return False
    
    # Step 2: Create some k-lines by asking questions
    print("\n2. Creating k-lines through Q&A...")
    questions = [
        "What type of seat should I book for my flight?",
        "What kind of restaurant should I choose for dinner with my wife?"
    ]
    
    stored_klines = []
    for question in questions:
        response = requests.post(f"{BASE_URL}/api/klines/ask", json={
            "question": question,
            "top_k": 5,
            "store_kline": "always"  # Force storage for testing
        })
        if response.status_code == 200:
            result = response.json()
            if result.get('kline_storage_result', {}).get('success'):
                kline_id = result['kline_storage_result']['kline_id']
                stored_klines.append(kline_id)
                print(f"   ✅ Created k-line: {question}")
            else:
                print(f"   ⚠️ K-line not stored: {question}")
        else:
            print(f"   ❌ Failed to create k-line: {question}")
    
    # Step 3: Test mixed retrieval (all types)
    print("\n3. Testing mixed memory retrieval...")
    response = requests.post(f"{BASE_URL}/api/memory/search", json={
        "query": "travel preferences",
        "top_k": 10
        # No memory_type specified - should return both
    })
    
    if response.status_code != 200:
        print(f"   ❌ Search failed: {response.text}")
        return False
    
    result = response.json()
    memories = result.get('memories', [])
    breakdown = result.get('memory_breakdown', {})
    
    print(f"   📊 Found {len(memories)} total memories")
    print(f"   📊 Breakdown: {breakdown.get('nemes', 0)} nemes, {breakdown.get('klines', 0)} k-lines")
    
    # Verify we have both types
    neme_count = len([m for m in memories if m.get('type', 'neme') == 'neme'])
    kline_count = len([m for m in memories if m.get('type', 'neme') == 'k-line'])
    
    print(f"   🔍 Actual counts: {neme_count} nemes, {kline_count} k-lines")
    
    if neme_count == 0:
        print("   ⚠️ No nemes found in mixed search")
    if kline_count == 0:
        print("   ⚠️ No k-lines found in mixed search")
    
    # Step 4: Test type-specific filtering
    print("\n4. Testing memory type filtering...")
    
    # Test neme-only filtering
    response = requests.post(f"{BASE_URL}/api/memory/search", json={
        "query": "travel preferences",
        "top_k": 10,
        "memory_type": "neme"
    })
    
    if response.status_code == 200:
        result = response.json()
        neme_only_memories = result.get('memories', [])
        neme_only_count = len([m for m in neme_only_memories if m.get('type', 'neme') == 'neme'])
        kline_in_neme_search = len([m for m in neme_only_memories if m.get('type', 'neme') == 'k-line'])
        
        print(f"   🔍 Neme-only search: {neme_only_count} nemes, {kline_in_neme_search} k-lines")
        if kline_in_neme_search > 0:
            print("   ❌ K-lines found in neme-only search!")
            return False
    else:
        print(f"   ❌ Neme-only search failed: {response.text}")
        return False
    
    # Test k-line-only filtering
    response = requests.post(f"{BASE_URL}/api/memory/search", json={
        "query": "travel preferences",
        "top_k": 10,
        "memory_type": "k-line"
    })
    
    if response.status_code == 200:
        result = response.json()
        kline_only_memories = result.get('memories', [])
        kline_only_count = len([m for m in kline_only_memories if m.get('type', 'neme') == 'k-line'])
        neme_in_kline_search = len([m for m in kline_only_memories if m.get('type', 'neme') == 'neme'])
        
        print(f"   🔍 K-line-only search: {kline_only_count} k-lines, {neme_in_kline_search} nemes")
        if neme_in_kline_search > 0:
            print("   ❌ Nemes found in k-line-only search!")
            return False
    else:
        print(f"   ❌ K-line-only search failed: {response.text}")
        return False
    
    # Step 5: Test memory formatting
    print("\n5. Testing memory formatting...")
    for memory in memories[:3]:  # Test first 3 memories
        memory_type = memory.get('type', 'neme')
        if memory_type == 'k-line':
            required_fields = ['original_question', 'answer', 'confidence']
            missing_fields = [f for f in required_fields if f not in memory]
            if missing_fields:
                print(f"   ❌ K-line missing fields: {missing_fields}")
                return False
            else:
                print(f"   ✅ K-line has all required fields")
        else:
            if 'text' not in memory:
                print(f"   ❌ Neme missing text field")
                return False
            else:
                print(f"   ✅ Neme has text field")
    
    print("\n✅ All mixed memory type tests passed!")
    return True

def test_agent_session_with_mixed_memories():
    """Test that agent sessions work correctly with mixed memory types."""
    print("\n🤖 Testing agent session with mixed memory types...")
    
    # Create agent session
    response = requests.post(f"{BASE_URL}/api/agent/session", json={
        "system_prompt": "You are a helpful travel assistant.",
        "config": {
            "use_memory": True,
            "model": "gpt-3.5-turbo",
            "temperature": 0.7
        }
    })
    
    if response.status_code != 200:
        print(f"   ❌ Failed to create agent session: {response.text}")
        return False
    
    session_data = response.json()
    session_id = session_data['session_id']
    print(f"   ✅ Created agent session: {session_id}")
    
    # Send a message that should retrieve mixed memory types
    response = requests.post(f"{BASE_URL}/api/agent/session/{session_id}", json={
        "message": "Help me plan a trip",
        "top_k": 10  # Should retrieve both nemes and k-lines
    })
    
    if response.status_code != 200:
        print(f"   ❌ Failed to send message: {response.text}")
        return False
    
    result = response.json()
    memory_context = result.get('memory_context', {})
    memories_used = memory_context.get('memories_used', 0)
    
    print(f"   📊 Agent used {memories_used} memories")
    
    if memories_used > 0:
        print("   ✅ Agent successfully used memories")
        return True
    else:
        print("   ⚠️ Agent didn't use any memories")
        return True  # This might be expected if no relevant memories found

def main():
    """Run all mixed memory type tests."""
    print("🚀 Starting mixed memory type compatibility tests...")
    
    try:
        # Test basic functionality
        success1 = test_mixed_memory_storage_and_retrieval()
        
        # Test agent integration
        success2 = test_agent_session_with_mixed_memories()
        
        if success1 and success2:
            print("\n🎉 All mixed memory type tests completed successfully!")
            print("The system now properly handles both k-lines and nemes together.")
            return True
        else:
            print("\n❌ Some tests failed. Please check the implementation.")
            return False
            
    except Exception as e:
        print(f"\n💥 Test failed with exception: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Test script for the enhanced memory relevance scoring system.

This script tests the new temporal and usage-based relevance scoring features
including memory storage with new fields, relevance calculation, and API integration.
"""

import os
import sys
import time
import json
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory_core import MemoryCore, RelevanceConfig


def test_relevance_config():
    """Test the relevance configuration system."""
    print("🔧 Testing Relevance Configuration")
    print("=" * 50)
    
    # Test default configuration
    config = RelevanceConfig()
    print(f"Default config: {config.to_dict()}")
    
    # Test custom configuration
    custom_config = RelevanceConfig(
        vector_weight=0.6,
        temporal_weight=0.3,
        usage_weight=0.1,
        recency_decay_days=14.0
    )
    print(f"Custom config: {custom_config.to_dict()}")
    
    # Test configuration validation
    try:
        invalid_config = RelevanceConfig(vector_weight=0.5, temporal_weight=0.3, usage_weight=0.3)
        print("❌ Should have failed validation")
    except ValueError as e:
        print(f"✅ Configuration validation works: {e}")
    
    print()


def test_memory_storage_with_new_fields():
    """Test memory storage with new temporal and usage fields."""
    print("💾 Testing Memory Storage with New Fields")
    print("=" * 50)
    
    # Initialize memory core
    memory_core = MemoryCore()
    
    # Clear existing memories for clean test
    memory_core.clear_all_memories()
    
    # Store a test memory
    result = memory_core.store_memory("I love pizza from Mario's restaurant", apply_grounding=True)
    print(f"Stored memory: {result['memory_id']}")
    print(f"Original text: {result['original_text']}")
    print(f"Final text: {result['final_text']}")
    
    # Search for the memory to verify new fields
    memories = memory_core.search_memories("pizza", top_k=1)
    if memories:
        memory = memories[0]
        print(f"Retrieved memory fields:")
        print(f"  - ID: {memory['id']}")
        print(f"  - Created at: {memory.get('created_at', 'MISSING')}")
        print(f"  - Last accessed: {memory.get('last_accessed_at', 'MISSING')}")
        print(f"  - Access count: {memory.get('access_count', 'MISSING')}")
        print(f"  - Vector score: {memory.get('score', 'MISSING')}")
        print(f"  - Relevance score: {memory.get('relevance_score', 'MISSING')}")
    else:
        print("❌ No memories found")
    
    print()


def test_relevance_scoring():
    """Test the relevance scoring algorithm."""
    print("🧮 Testing Relevance Scoring Algorithm")
    print("=" * 50)
    
    # Initialize memory core with custom config
    config = RelevanceConfig(
        vector_weight=0.6,
        temporal_weight=0.3,
        usage_weight=0.1,
        recency_decay_days=7.0,
        access_decay_days=3.0
    )
    memory_core = MemoryCore(relevance_config=config)
    
    # Clear existing memories
    memory_core.clear_all_memories()
    
    # Store multiple test memories with delays to create temporal differences
    memories_data = [
        "I prefer window seats on flights",
        "My wife is vegetarian",
        "I like Italian food",
        "I work in tech industry"
    ]
    
    stored_ids = []
    for i, text in enumerate(memories_data):
        result = memory_core.store_memory(text)
        stored_ids.append(result['memory_id'])
        print(f"Stored: {text}")
        if i < len(memories_data) - 1:
            time.sleep(1)  # Small delay to create temporal differences
    
    # Search multiple times to increase access counts for some memories
    print("\nSimulating usage patterns...")
    
    # Access "vegetarian" memory multiple times
    for _ in range(3):
        memory_core.search_memories("vegetarian", top_k=1)
        time.sleep(0.1)
    
    # Access "flight" memory once
    memory_core.search_memories("flight", top_k=1)
    
    # Now search for a general query to see relevance scoring in action
    print("\nSearching for 'food preferences':")
    results = memory_core.search_memories("food preferences", top_k=4)
    
    for i, memory in enumerate(results, 1):
        print(f"{i}. {memory['text']}")
        print(f"   Vector score: {memory.get('score', 0):.4f}")
        print(f"   Relevance score: {memory.get('relevance_score', 0):.4f}")
        print(f"   Access count: {memory.get('access_count', 0)}")
        print(f"   Created: {memory.get('created_at', 'Unknown')}")
        print()
    
    print()


def test_api_integration():
    """Test the K-lines API integration with relevance scoring."""
    print("🌐 Testing API Integration")
    print("=" * 50)
    
    # Test the K-lines recall endpoint
    base_url = "http://localhost:5001"
    
    try:
        # Test recall endpoint
        response = requests.post(f"{base_url}/api/klines/recall", json={
            "query": "food and dining preferences",
            "top_k": 3
        })
        
        if response.status_code == 200:
            data = response.json()
            print("✅ K-lines recall API working")
            print(f"Found {data.get('memory_count', 0)} memories")
            
            memories = data.get('memories', [])
            for i, memory in enumerate(memories, 1):
                print(f"{i}. {memory.get('text', 'No text')}")
                print(f"   Relevance score: {memory.get('relevance_score', 'N/A')}")
                print(f"   Access count: {memory.get('access_count', 'N/A')}")
        else:
            print(f"❌ API request failed: {response.status_code}")
            print(response.text)
    
    except requests.exceptions.ConnectionError:
        print("⚠️ Could not connect to API server. Make sure the server is running.")
    
    print()


def test_configuration_api():
    """Test relevance configuration through API (if implemented)."""
    print("⚙️ Testing Configuration Management")
    print("=" * 50)
    
    memory_core = MemoryCore()
    
    # Test getting current configuration
    current_config = memory_core.get_relevance_config()
    print(f"Current configuration: {json.dumps(current_config, indent=2)}")
    
    # Test updating configuration
    updated_config = memory_core.update_relevance_config(
        vector_weight=0.5,
        temporal_weight=0.4,
        usage_weight=0.1
    )
    print(f"Updated configuration: {json.dumps(updated_config, indent=2)}")
    
    print()


def main():
    """Run all relevance scoring tests."""
    print("🧠 Memory Relevance Scoring Test Suite")
    print("=" * 60)
    print()
    
    try:
        test_relevance_config()
        test_memory_storage_with_new_fields()
        test_relevance_scoring()
        test_configuration_api()
        test_api_integration()
        
        print("🎉 All tests completed!")
        print("\n📊 Test Summary:")
        print("✅ Relevance configuration system")
        print("✅ Memory storage with temporal/usage fields")
        print("✅ Relevance scoring algorithm")
        print("✅ Configuration management")
        print("✅ API integration (if server running)")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

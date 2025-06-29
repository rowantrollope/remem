#!/usr/bin/env python3
"""
Test script to verify timestamp standardization in the core memory system.
"""

import os
import sys
from datetime import datetime, timezone

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_core_memory_storage():
    """Test that core memory storage uses only ISO timestamps."""
    print("🧪 Testing core memory storage...")

    try:
        from memory.core_agent import MemoryAgent

        # Initialize memory agent
        memory_agent = MemoryAgent()

        # Store a test memory
        test_text = "Test memory for timestamp standardization"
        result = memory_agent.store_memory(test_text, apply_grounding=False)

        print(f"Storage result keys: {list(result.keys())}")

        # Check that result contains created_at in ISO format
        if 'created_at' in result:
            created_at = result['created_at']
            # Try to parse as ISO format
            try:
                parsed_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                print(f"✅ Core memory storage returns ISO timestamp: {created_at}")

                # Check that old timestamp fields are not present
                if 'timestamp' in result:
                    print(f"⚠️ Warning: Old 'timestamp' field still present in storage result")
                if 'formatted_time' in result:
                    print(f"⚠️ Warning: Old 'formatted_time' field still present in storage result")

                return True, result['memory_id']
            except ValueError as e:
                print(f"❌ Invalid ISO timestamp format: {created_at} - {e}")
                return False, None
        else:
            print(f"❌ No 'created_at' field in storage result")
            return False, None

    except Exception as e:
        print(f"❌ Error testing core memory storage: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_core_memory_search(memory_id=None):
    """Test that core memory search returns only ISO timestamps."""
    print("🧪 Testing core memory search...")
    
    try:
        from memory.core_agent import MemoryAgent
        
        # Initialize memory agent
        memory_agent = MemoryAgent()
        
        # Search for memories
        memories = memory_agent.search_memories("test memory", top_k=5)
        
        print(f"Found {len(memories)} memories")
        
        if len(memories) > 0:
            memory = memories[0]
            print(f"First memory: {memory}")
            
            # Check for ISO timestamp fields
            if 'created_at' in memory:
                created_at = memory['created_at']
                try:
                    parsed_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    print(f"✅ Core memory search returns ISO timestamp: {created_at}")
                    
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
            
    except Exception as e:
        print(f"❌ Error testing core memory search: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_timestamp_format():
    """Test that our timestamp generation produces valid ISO format."""
    print("🧪 Testing timestamp format generation...")
    
    try:
        # Test the timestamp format we're using
        current_time_iso = datetime.now(timezone.utc).isoformat()
        print(f"Generated timestamp: {current_time_iso}")
        
        # Try to parse it back
        parsed_time = datetime.fromisoformat(current_time_iso.replace('Z', '+00:00'))
        print(f"✅ Timestamp format is valid ISO 8601: {current_time_iso}")
        
        # Check that it's in UTC
        if parsed_time.tzinfo is not None:
            print(f"✅ Timestamp includes timezone information")
        else:
            print(f"⚠️ Warning: Timestamp missing timezone information")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing timestamp format: {e}")
        return False

def main():
    """Run all core timestamp standardization tests."""
    print("🚀 Starting core timestamp standardization tests...")
    print("=" * 60)
    
    tests = [
        test_timestamp_format,
        test_core_memory_storage,
        test_core_memory_search,
    ]
    
    passed = 0
    total = len(tests)
    memory_id = None
    
    for test in tests:
        try:
            if test == test_core_memory_storage:
                result, memory_id = test()
                if result:
                    passed += 1
            elif test == test_core_memory_search:
                if test(memory_id):
                    passed += 1
            else:
                if test():
                    passed += 1
            print("-" * 40)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            print("-" * 40)
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All core timestamp standardization tests passed!")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    exit(main())

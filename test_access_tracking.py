#!/usr/bin/env python3
"""
Simple test to verify access tracking is working correctly.
"""

import os
import sys
import time
import json

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory_core import MemoryCore


def test_access_tracking():
    """Test that access tracking updates correctly."""
    print("🔍 Testing Access Tracking")
    print("=" * 40)
    
    # Initialize memory core
    memory_core = MemoryCore()
    
    # Clear existing memories
    memory_core.clear_all_memories()
    
    # Store a test memory
    result = memory_core.store_memory("I love testing access tracking")
    memory_id = result['memory_id']
    print(f"Stored memory: {memory_id}")
    
    # Search for the memory to check initial state
    memories = memory_core.search_memories("testing", top_k=1)
    if memories:
        memory = memories[0]
        print(f"Initial state:")
        print(f"  Access count: {memory.get('access_count', 'MISSING')}")
        print(f"  Last accessed: {memory.get('last_accessed_at', 'MISSING')}")
        print(f"  Relevance score: {memory.get('relevance_score', 'MISSING'):.4f}")
        
        # Wait a moment and search again
        time.sleep(1)
        print(f"\nAfter second search:")
        memories2 = memory_core.search_memories("testing", top_k=1)
        if memories2:
            memory2 = memories2[0]
            print(f"  Access count: {memory2.get('access_count', 'MISSING')}")
            print(f"  Last accessed: {memory2.get('last_accessed_at', 'MISSING')}")
            print(f"  Relevance score: {memory2.get('relevance_score', 'MISSING'):.4f}")
            
            # Check if access count increased
            if memory2.get('access_count', 0) > memory.get('access_count', 0):
                print("✅ Access tracking is working!")
            else:
                print("❌ Access tracking not working")
                
                # Let's check the Redis metadata directly
                print("\nChecking Redis metadata directly:")
                try:
                    metadata_json = memory_core.redis_client.execute_command("VGETATTR", memory_core.VECTORSET_KEY, memory_id)
                    if metadata_json:
                        metadata_str = metadata_json.decode('utf-8') if isinstance(metadata_json, bytes) else metadata_json
                        metadata = json.loads(metadata_str)
                        print(f"Direct Redis metadata:")
                        print(f"  Access count: {metadata.get('access_count', 'MISSING')}")
                        print(f"  Last accessed: {metadata.get('last_accessed_at', 'MISSING')}")
                except Exception as e:
                    print(f"Error checking Redis metadata: {e}")
        else:
            print("❌ No memories found in second search")
    else:
        print("❌ No memories found in initial search")


def test_manual_access_update():
    """Test manual access update to verify VSETATTR works."""
    print("\n🔧 Testing Manual Access Update")
    print("=" * 40)
    
    memory_core = MemoryCore()
    
    # Store a test memory
    result = memory_core.store_memory("Manual update test")
    memory_id = result['memory_id']
    print(f"Stored memory: {memory_id}")
    
    # Try to manually update the access tracking
    try:
        # Get current metadata
        metadata_json = memory_core.redis_client.execute_command("VGETATTR", memory_core.VECTORSET_KEY, memory_id)
        if metadata_json:
            metadata_str = metadata_json.decode('utf-8') if isinstance(metadata_json, bytes) else metadata_json
            metadata = json.loads(metadata_str)
            
            print(f"Original metadata access_count: {metadata.get('access_count', 'MISSING')}")
            
            # Update access count manually
            metadata["access_count"] = 99
            metadata["last_accessed_at"] = "2024-06-19T12:00:00Z"
            
            # Update metadata in Redis
            updated_metadata_json = json.dumps(metadata)
            result = memory_core.redis_client.execute_command("VSETATTR", memory_core.VECTORSET_KEY, memory_id, updated_metadata_json)
            print(f"VSETATTR result: {result}")
            
            # Verify the update
            metadata_json2 = memory_core.redis_client.execute_command("VGETATTR", memory_core.VECTORSET_KEY, memory_id)
            if metadata_json2:
                metadata_str2 = metadata_json2.decode('utf-8') if isinstance(metadata_json2, bytes) else metadata_json2
                metadata2 = json.loads(metadata_str2)
                print(f"Updated metadata access_count: {metadata2.get('access_count', 'MISSING')}")
                
                if metadata2.get('access_count') == 99:
                    print("✅ VSETATTR is working correctly!")
                else:
                    print("❌ VSETATTR did not update the metadata")
            else:
                print("❌ Could not retrieve updated metadata")
        else:
            print("❌ Could not retrieve original metadata")
            
    except Exception as e:
        print(f"❌ Error in manual update test: {e}")


if __name__ == "__main__":
    test_access_tracking()
    test_manual_access_update()

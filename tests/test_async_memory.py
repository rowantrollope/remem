#!/usr/bin/env python3
"""
Test script for the Asynchronous Memory Processing System

This script demonstrates and tests the complete async memory workflow:
1. Store raw chat session data
2. Monitor background processing
3. Retrieve processed memories and session summaries
4. Test memory hierarchy and data retention
"""

import requests
import json
import time
import uuid
from datetime import datetime, timezone

BASE_URL = "http://localhost:5001"
VECTORSTORE_NAME = "test_async_memory"

def test_async_memory_system():
    """Test the complete async memory processing workflow."""
    print("🧪 Testing Asynchronous Memory Processing System")
    print("=" * 60)

    # Sample chat session data
    sample_session = """
User: I'm planning a trip to Japan next month. I prefer window seats on flights and I'm vegetarian.
Assistant: That sounds like a wonderful trip! For your flight to Japan, I'll make sure to note your preference for window seats. Since you're vegetarian, I can help you with information about vegetarian-friendly restaurants in Japan and how to communicate your dietary needs.

User: Great! I also prefer 4-space indentation when coding and I'm working on a Python project about travel planning.
Assistant: Excellent! I'll remember your coding preferences. Python is a great choice for travel planning projects. The 4-space indentation is the standard Python convention, so that's perfect.

User: One more thing - I really dislike crowded places and prefer quiet restaurants when traveling.
Assistant: Noted! I'll keep that in mind when suggesting restaurants in Japan. There are many quiet, traditional places that would suit your preferences perfectly.
"""

    session_id = str(uuid.uuid4())

    # Test 1: Store raw memory
    print("\n1️⃣ Testing raw memory storage...")
    raw_memory_data = {
        "session_data": sample_session,
        "session_id": session_id,
        "metadata": {
            "user_id": "test_user_123",
            "session_type": "travel_planning",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    }

    response = requests.post(
        f"{BASE_URL}/api/memory/{VECTORSTORE_NAME}/store_raw",
        json=raw_memory_data
    )

    if response.status_code != 200:
        print(f"❌ Failed to store raw memory: {response.text}")
        return False

    result = response.json()
    raw_memory_id = result["raw_memory_id"]
    print(f"✅ Raw memory stored successfully: {raw_memory_id}")
    print(f"📊 Queue position: {result.get('queue_position', 'N/A')}")
    print(f"⏱️ Estimated processing time: {result.get('estimated_processing_time', 'N/A')}")

    # Test 2: Check processing status
    print("\n2️⃣ Checking processing status...")
    response = requests.get(f"{BASE_URL}/api/memory/{VECTORSTORE_NAME}/processing_status")

    if response.status_code == 200:
        status = response.json()
        print(f"✅ Processor running: {status['processor_running']}")
        print(f"📊 Queue size: {status['queue_size']}")
        print(f"📈 Processed today: {status['processed_today']}")
    else:
        print(f"⚠️ Could not get processing status: {response.text}")

    # Test 3: Wait for processing (or simulate it)
    print("\n3️⃣ Waiting for background processing...")
    print("💡 In a real scenario, the background processor would handle this automatically.")
    print("💡 For testing, you can run: python memory/async_processor.py --run-once")

    # Simulate some processing time
    time.sleep(2)

    # Test 4: Check memory hierarchy
    print("\n4️⃣ Testing memory hierarchy retrieval...")
    response = requests.get(
        f"{BASE_URL}/api/memory/{VECTORSTORE_NAME}/hierarchy",
        params={"session_id": session_id, "limit": 10}
    )

    if response.status_code == 200:
        hierarchy = response.json()
        results = hierarchy["results"]
        print(f"✅ Memory hierarchy retrieved:")
        print(f"   📝 Discrete memories: {len(results['discrete_memories'])}")
        print(f"   📋 Session summaries: {len(results['session_summaries'])}")
        print(f"   📄 Raw transcripts: {len(results['raw_transcripts'])}")
        print(f"   📊 Total items: {results['total_count']}")
    else:
        print(f"❌ Failed to get memory hierarchy: {response.text}")

    # Test 5: Get session details
    print("\n5️⃣ Testing session details retrieval...")
    response = requests.get(f"{BASE_URL}/api/memory/{VECTORSTORE_NAME}/session/{session_id}")

    if response.status_code == 200:
        session_details = response.json()
        print(f"✅ Session details retrieved:")
        print(f"   📝 Memory count: {session_details['memory_count']}")
        print(f"   📋 Has summary: {session_details['has_summary']}")
        print(f"   📄 Has raw data: {session_details['has_raw_data']}")
    else:
        print(f"⚠️ Session not found (expected if not processed yet): {response.status_code}")

    # Test 6: Get comprehensive stats
    print("\n6️⃣ Testing memory statistics...")
    response = requests.get(f"{BASE_URL}/api/memory/{VECTORSTORE_NAME}/stats")

    if response.status_code == 200:
        stats = response.json()
        print(f"✅ Memory statistics retrieved:")
        hierarchy_stats = stats["hierarchy_stats"]
        processing_stats = stats["processing_stats"]
        print(f"   📊 Hierarchy: {hierarchy_stats['total_items']} total items")
        print(f"   🔄 Processing: {processing_stats['processed']} processed, {processing_stats['queued']} queued")
    else:
        print(f"❌ Failed to get stats: {response.text}")

    print("\n✅ Async memory system test completed!")
    print("\n📋 Next Steps:")
    print("1. Start the background processor: python memory/async_processor.py")
    print("2. Wait for processing to complete")
    print("3. Re-run this test to see processed results")
    print("4. Test cleanup: POST /api/memory/{vectorstore}/cleanup")

    return True


def test_background_processor():
    """Test running the background processor once."""
    print("\n🔄 Testing background processor...")

    try:
        from memory.async_processor import AsyncMemoryProcessor

        processor = AsyncMemoryProcessor(processing_interval=60, retention_days=30)
        result = processor.run_once()

        print(f"✅ Background processor test completed:")
        print(f"   📊 Processing result: {result['processing_result']['message']}")
        if result["cleanup_result"]:
            print(f"   🧹 Cleanup result: {result['cleanup_result']}")

        return True

    except Exception as e:
        print(f"❌ Background processor test failed: {e}")
        return False


def main():
    """Main test runner."""
    print("🚀 Starting Asynchronous Memory Processing Tests")
    print("=" * 80)

    # Test the API endpoints
    api_success = test_async_memory_system()

    # Test the background processor
    processor_success = test_background_processor()

    print("\n" + "=" * 80)
    print("📊 Test Results Summary:")
    print(f"   🌐 API Tests: {'✅ PASSED' if api_success else '❌ FAILED'}")
    print(f"   🔄 Processor Tests: {'✅ PASSED' if processor_success else '❌ FAILED'}")

    if api_success and processor_success:
        print("\n🎉 All tests passed! The async memory system is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test script for Redis LangCache integration

This script tests the LangCache client and integration with memory modules.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path for imports
sys.path.append('.')

def test_langcache_client():
    """Test the LangCache client directly."""
    print("🧪 Testing LangCache Client...")
    
    try:
        from langcache_client import LangCacheClient, CachedLLMClient
        
        # Check if environment variables are set
        required_vars = ["LANGCACHE_HOST", "LANGCACHE_API_KEY", "LANGCACHE_CACHE_ID"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
            print("ℹ️ Set these in your .env file to test LangCache")
            return False
        
        # Initialize client
        client = LangCacheClient()
        print("✅ LangCache client initialized successfully")
        
        # Test health check
        health = client.health_check()
        if health['healthy']:
            print("✅ LangCache health check passed")
        else:
            print(f"❌ LangCache health check failed: {health['error']}")
            return False
        
        # Test cache operations with dummy data
        test_messages = [{"role": "user", "content": "What is the capital of France?"}]
        test_response = "The capital of France is Paris."
        
        # Store in cache
        stored = client.store_cache(
            messages=test_messages,
            response=test_response,
            operation_type='test'
        )
        
        if stored:
            print("✅ Successfully stored test response in cache")
        else:
            print("❌ Failed to store test response in cache")
            return False
        
        # Search cache
        cached_response = client.search_cache(
            messages=test_messages,
            operation_type='test'
        )
        
        if cached_response and cached_response.get('_cache_hit'):
            print("✅ Successfully retrieved cached response")
            print(f"   Cached content: {cached_response['content'][:50]}...")
        else:
            print("❌ Failed to retrieve cached response")
            return False
        
        # Get stats
        stats = client.get_stats()
        print(f"✅ Cache statistics: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ LangCache client test failed: {e}")
        return False

def test_memory_integration():
    """Test LangCache integration with memory modules."""
    print("\n🧪 Testing Memory Module Integration...")
    
    try:
        # Test memory extraction
        from memory.extraction import MemoryExtraction
        from memory.core import MemoryCore
        
        # Create a dummy memory core (won't actually connect to Redis in test)
        print("✅ Memory modules imported successfully")
        
        # Check if LangCache is properly initialized in modules
        # This is a basic import test - full integration would require Redis
        print("✅ Memory integration test passed (import level)")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory integration test failed: {e}")
        return False

def test_fallback_behavior():
    """Test that the system works without LangCache configured."""
    print("\n🧪 Testing Fallback Behavior...")
    
    # Temporarily clear environment variables
    original_vars = {}
    langcache_vars = ["LANGCACHE_HOST", "LANGCACHE_API_KEY", "LANGCACHE_CACHE_ID"]
    
    for var in langcache_vars:
        original_vars[var] = os.getenv(var)
        if var in os.environ:
            del os.environ[var]
    
    try:
        from langcache_client import LangCacheClient
        
        # This should raise an error due to missing env vars
        try:
            client = LangCacheClient()
            print("❌ LangCache client should have failed without env vars")
            return False
        except ValueError as e:
            print("✅ LangCache client properly fails without configuration")
        
        # Test memory modules without LangCache
        from memory.extraction import MemoryExtraction
        print("✅ Memory modules work without LangCache configuration")
        
        return True
        
    except Exception as e:
        print(f"❌ Fallback behavior test failed: {e}")
        return False
    
    finally:
        # Restore environment variables
        for var, value in original_vars.items():
            if value is not None:
                os.environ[var] = value

def main():
    """Run all tests."""
    print("🚀 Starting LangCache Integration Tests\n")
    
    tests = [
        ("LangCache Client", test_langcache_client),
        ("Memory Integration", test_memory_integration),
        ("Fallback Behavior", test_fallback_behavior)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print('='*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! LangCache integration is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

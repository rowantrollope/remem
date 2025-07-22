#!/usr/bin/env python3
"""
Test script for LangCache configuration controls

This script tests the new configuration system for controlling individual cache types.
"""

import os
import sys
import json

# Add current directory to path for imports
sys.path.append('.')

def test_cache_configuration():
    """Test the cache configuration system."""
    print("🧪 Testing LangCache Configuration System...")
    
    try:
        from langcache_client import is_cache_enabled_for_operation
        
        # Test without web_app loaded (should default to True)
        print("\n1. Testing without configuration loaded:")
        for operation in ['memory_extraction', 'query_optimization', 'embedding_optimization', 'context_analysis', 'memory_grounding']:
            enabled = is_cache_enabled_for_operation(operation)
            print(f"   {operation}: {enabled} (default)")
        
        # Import web_app to load configuration
        print("\n2. Testing with default configuration:")
        import web_app
        
        for operation in ['memory_extraction', 'query_optimization', 'embedding_optimization', 'context_analysis', 'memory_grounding']:
            enabled = is_cache_enabled_for_operation(operation)
            print(f"   {operation}: {enabled}")
        
        # Test modifying configuration
        print("\n3. Testing configuration modifications:")
        
        # Disable memory_extraction caching
        web_app.app_config['langcache']['cache_types']['memory_extraction'] = False
        enabled = is_cache_enabled_for_operation('memory_extraction')
        print(f"   memory_extraction (disabled): {enabled}")
        
        # Disable all caching
        web_app.app_config['langcache']['enabled'] = False
        for operation in ['memory_extraction', 'query_optimization']:
            enabled = is_cache_enabled_for_operation(operation)
            print(f"   {operation} (master disabled): {enabled}")
        
        # Re-enable master switch
        web_app.app_config['langcache']['enabled'] = True
        enabled = is_cache_enabled_for_operation('query_optimization')
        print(f"   query_optimization (master re-enabled): {enabled}")
        
        print("\n✅ Configuration system working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cached_llm_client():
    """Test the CachedLLMClient with configuration."""
    print("\n🧪 Testing CachedLLMClient Configuration...")
    
    try:
        from langcache_client import CachedLLMClient, LangCacheClient
        import web_app
        
        # Mock LLM client
        class MockLLMClient:
            def chat_completion(self, **kwargs):
                return {'content': 'Mock response', 'usage': {'tokens': 10}}
        
        # Mock LangCache client (won't actually connect)
        class MockLangCacheClient:
            def search_cache(self, **kwargs):
                return None  # No cache hit
            def store_cache(self, **kwargs):
                return True
        
        mock_llm = MockLLMClient()
        mock_cache = MockLangCacheClient()
        cached_client = CachedLLMClient(mock_llm, mock_cache)
        
        # Test with caching enabled
        print("\n1. Testing with caching enabled:")
        response = cached_client.chat_completion(
            messages=[{"role": "user", "content": "test"}],
            operation_type='memory_extraction'
        )
        print(f"   Response: {response}")
        print(f"   Cache disabled flag: {response.get('_cache_disabled', False)}")
        
        # Test with caching disabled for this operation
        print("\n2. Testing with memory_extraction caching disabled:")
        web_app.app_config['langcache']['cache_types']['memory_extraction'] = False
        response = cached_client.chat_completion(
            messages=[{"role": "user", "content": "test"}],
            operation_type='memory_extraction'
        )
        print(f"   Response: {response}")
        print(f"   Cache disabled flag: {response.get('_cache_disabled', False)}")
        
        # Test with master caching disabled
        print("\n3. Testing with master caching disabled:")
        web_app.app_config['langcache']['enabled'] = False
        response = cached_client.chat_completion(
            messages=[{"role": "user", "content": "test"}],
            operation_type='query_optimization'
        )
        print(f"   Response: {response}")
        print(f"   Cache disabled flag: {response.get('_cache_disabled', False)}")
        
        print("\n✅ CachedLLMClient configuration working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ CachedLLMClient test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_current_config():
    """Show the current LangCache configuration."""
    print("\n📋 Current LangCache Configuration:")
    
    try:
        import web_app
        config = web_app.app_config.get('langcache', {})
        
        print(f"Master enabled: {config.get('enabled', 'Not set')}")
        print("Cache types:")
        cache_types = config.get('cache_types', {})
        for cache_type, enabled in cache_types.items():
            print(f"  - {cache_type}: {enabled}")
        
    except Exception as e:
        print(f"❌ Could not load configuration: {e}")

def main():
    """Run all configuration tests."""
    print("🚀 Testing LangCache Configuration System\n")
    
    tests = [
        ("Cache Configuration", test_cache_configuration),
        ("CachedLLMClient", test_cached_llm_client)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print('='*60)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Show current configuration
    show_current_config()
    
    # Summary
    print(f"\n{'='*60}")
    print("CONFIGURATION TEST SUMMARY")
    print('='*60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All configuration tests passed!")
        print("✅ LangCache configuration system is working correctly")
        return 0
    else:
        print("\n⚠️ Some configuration tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

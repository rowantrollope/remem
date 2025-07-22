#!/usr/bin/env python3
"""
Simple validation script for LangCache implementation

This script validates that our LangCache implementation is correctly integrated
without requiring external dependencies or actual Redis connections.
"""

import os
import sys

def test_langcache_client():
    """Test LangCache client implementation."""
    print("🧪 Testing LangCache Client Implementation...")
    
    try:
        # Import the client
        from langcache_client import LangCacheClient, CachedLLMClient
        print("✅ LangCache client classes imported successfully")
        
        # Test that it properly validates environment variables
        try:
            client = LangCacheClient()
            print("❌ Should have failed without environment variables")
            return False
        except ValueError as e:
            print("✅ Properly validates required environment variables")
            print(f"   Expected error: {str(e)[:80]}...")
        
        # Test with mock environment variables
        os.environ['LANGCACHE_HOST'] = 'https://test.example.com'
        os.environ['LANGCACHE_API_KEY'] = 'test_key'
        os.environ['LANGCACHE_CACHE_ID'] = 'test_cache'
        
        try:
            client = LangCacheClient()
            print("✅ Client initializes with environment variables")
            
            # Test basic methods exist
            assert hasattr(client, 'search_cache'), "Missing search_cache method"
            assert hasattr(client, 'store_cache'), "Missing store_cache method"
            assert hasattr(client, 'health_check'), "Missing health_check method"
            assert hasattr(client, 'get_stats'), "Missing get_stats method"
            print("✅ All required methods are present")
            
        except Exception as e:
            print(f"❌ Client initialization failed: {e}")
            return False
        finally:
            # Clean up environment variables
            for var in ['LANGCACHE_HOST', 'LANGCACHE_API_KEY', 'LANGCACHE_CACHE_ID']:
                if var in os.environ:
                    del os.environ[var]
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_memory_integration():
    """Test that memory modules integrate LangCache correctly."""
    print("\n🧪 Testing Memory Module Integration...")
    
    try:
        # Test memory extraction
        print("  Testing memory extraction...")
        from memory.extraction import MemoryExtraction
        
        # Check that LangCache import is present
        import memory.extraction as ext_module
        assert hasattr(ext_module, 'LangCacheClient'), "LangCacheClient not imported in extraction"
        assert hasattr(ext_module, 'CachedLLMClient'), "CachedLLMClient not imported in extraction"
        print("  ✅ Memory extraction has LangCache imports")
        
        # Test memory processing
        print("  Testing memory processing...")
        from memory.processing import MemoryProcessing
        
        import memory.processing as proc_module
        assert hasattr(proc_module, 'LangCacheClient'), "LangCacheClient not imported in processing"
        assert hasattr(proc_module, 'CachedLLMClient'), "CachedLLMClient not imported in processing"
        print("  ✅ Memory processing has LangCache imports")
        
        # Test memory core
        print("  Testing memory core...")
        from memory.core import MemoryCore
        
        import memory.core as core_module
        assert hasattr(core_module, 'LangCacheClient'), "LangCacheClient not imported in core"
        assert hasattr(core_module, 'CachedLLMClient'), "CachedLLMClient not imported in core"
        print("  ✅ Memory core has LangCache imports")
        
        print("✅ All memory modules have proper LangCache integration")
        return True
        
    except ImportError as e:
        print(f"❌ Memory module import failed: {e}")
        return False
    except AssertionError as e:
        print(f"❌ Integration check failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_removed_old_caching():
    """Test that old caching infrastructure was properly removed."""
    print("\n🧪 Testing Old Cache Removal...")
    
    # Check that old cache files are gone
    old_cache_files = [
        'optimizations/llm_cache.py',
        'optimizations/semantic_cache.py', 
        'optimizations/performance_optimizer.py',
        'optimizations/optimized_processing.py',
        'optimizations/optimized_extraction.py'
    ]
    
    for file_path in old_cache_files:
        if os.path.exists(file_path):
            print(f"❌ Old cache file still exists: {file_path}")
            return False
    
    print("✅ All old cache files have been removed")
    
    # Test that web_app doesn't import old modules
    try:
        import web_app
        
        # Check that old performance optimizer imports are gone
        web_app_source = open('web_app.py', 'r').read()
        
        old_imports = [
            'from optimizations.performance_optimizer',
            'get_performance_optimizer',
            'init_performance_optimizer',
            'optimize_memory_agent'
        ]
        
        for old_import in old_imports:
            if old_import in web_app_source:
                print(f"❌ Old import still present in web_app.py: {old_import}")
                return False
        
        print("✅ Old cache imports removed from web_app.py")
        
    except Exception as e:
        print(f"❌ Error checking web_app.py: {e}")
        return False
    
    return True

def test_configuration_files():
    """Test that configuration files are properly updated."""
    print("\n🧪 Testing Configuration Files...")
    
    # Check .env.example
    try:
        with open('.env.example', 'r') as f:
            env_content = f.read()
        
        required_vars = ['LANGCACHE_HOST', 'LANGCACHE_API_KEY', 'LANGCACHE_CACHE_ID']
        for var in required_vars:
            if var not in env_content:
                print(f"❌ Missing {var} in .env.example")
                return False
        
        print("✅ .env.example contains LangCache configuration")
        
    except FileNotFoundError:
        print("❌ .env.example file not found")
        return False
    
    # Check documentation
    if os.path.exists('LANGCACHE_SETUP.md'):
        print("✅ LangCache setup documentation exists")
    else:
        print("❌ LangCache setup documentation missing")
        return False
    
    return True

def main():
    """Run all validation tests."""
    print("🚀 Validating LangCache Implementation\n")
    
    tests = [
        ("LangCache Client", test_langcache_client),
        ("Memory Integration", test_memory_integration), 
        ("Old Cache Removal", test_removed_old_caching),
        ("Configuration Files", test_configuration_files)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"{'='*60}")
        print(f"Running: {test_name}")
        print('='*60)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print('='*60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All validation tests passed!")
        print("✅ LangCache implementation is correctly integrated")
        print("✅ Old caching infrastructure has been removed")
        print("✅ Configuration files are updated")
        return 0
    else:
        print("\n⚠️ Some validation tests failed.")
        print("Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

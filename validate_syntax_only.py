#!/usr/bin/env python3
"""
Syntax-only validation for LangCache implementation

This script validates syntax and basic structure without requiring external dependencies.
"""

import os
import ast
import sys

def validate_file_syntax(file_path):
    """Validate Python syntax of a file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Parse the AST to check syntax
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error reading file: {e}"

def check_langcache_integration():
    """Check that LangCache is properly integrated."""
    print("🧪 Checking LangCache Integration...")
    
    # Files that should have LangCache integration
    files_to_check = [
        'langcache_client.py',
        'memory/extraction.py',
        'memory/processing.py', 
        'memory/core.py'
    ]
    
    all_good = True
    
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            print(f"❌ Missing file: {file_path}")
            all_good = False
            continue
        
        # Check syntax
        syntax_ok, error = validate_file_syntax(file_path)
        if not syntax_ok:
            print(f"❌ Syntax error in {file_path}: {error}")
            all_good = False
            continue
        
        # Check for LangCache imports (except for langcache_client.py itself)
        if file_path != 'langcache_client.py':
            with open(file_path, 'r') as f:
                content = f.read()
            
            if 'langcache_client' not in content:
                print(f"❌ Missing LangCache import in {file_path}")
                all_good = False
            else:
                print(f"✅ {file_path} has LangCache integration")
        else:
            print(f"✅ {file_path} syntax is valid")
    
    return all_good

def check_old_cache_removal():
    """Check that old caching files are removed."""
    print("\n🧪 Checking Old Cache Removal...")
    
    # Files that should be removed
    old_files = [
        'optimizations/llm_cache.py',
        'optimizations/semantic_cache.py',
        'optimizations/performance_optimizer.py',
        'optimizations/optimized_processing.py',
        'optimizations/optimized_extraction.py'
    ]
    
    all_removed = True
    for file_path in old_files:
        if os.path.exists(file_path):
            print(f"❌ Old cache file still exists: {file_path}")
            all_removed = False
    
    if all_removed:
        print("✅ All old cache files have been removed")
    
    # Check web_app.py for old imports
    if os.path.exists('web_app.py'):
        syntax_ok, error = validate_file_syntax('web_app.py')
        if not syntax_ok:
            print(f"❌ Syntax error in web_app.py: {error}")
            all_removed = False
        else:
            print("✅ web_app.py syntax is valid")
            
            with open('web_app.py', 'r') as f:
                content = f.read()
            
            # Check for old imports
            old_patterns = [
                'from optimizations.performance_optimizer',
                'get_performance_optimizer',
                'init_performance_optimizer'
            ]
            
            found_old = False
            for pattern in old_patterns:
                if pattern in content:
                    print(f"❌ Old import pattern found in web_app.py: {pattern}")
                    found_old = True
                    all_removed = False
            
            if not found_old:
                print("✅ Old cache imports removed from web_app.py")
    
    return all_removed

def check_configuration():
    """Check configuration files."""
    print("\n🧪 Checking Configuration...")
    
    all_good = True
    
    # Check .env.example
    if os.path.exists('.env.example'):
        with open('.env.example', 'r') as f:
            content = f.read()
        
        required_vars = ['LANGCACHE_HOST', 'LANGCACHE_API_KEY', 'LANGCACHE_CACHE_ID']
        missing_vars = []
        
        for var in required_vars:
            if var not in content:
                missing_vars.append(var)
        
        if missing_vars:
            print(f"❌ Missing variables in .env.example: {', '.join(missing_vars)}")
            all_good = False
        else:
            print("✅ .env.example contains all LangCache variables")
    else:
        print("❌ .env.example file not found")
        all_good = False
    
    # Check documentation
    if os.path.exists('LANGCACHE_SETUP.md'):
        print("✅ LangCache setup documentation exists")
    else:
        print("❌ LangCache setup documentation missing")
        all_good = False
    
    return all_good

def check_core_files():
    """Check that core files have valid syntax."""
    print("\n🧪 Checking Core File Syntax...")
    
    core_files = [
        'web_app.py',
        'memory/agent.py',
        'memory/core.py',
        'memory/extraction.py',
        'memory/processing.py'
    ]
    
    all_good = True
    
    for file_path in core_files:
        if os.path.exists(file_path):
            syntax_ok, error = validate_file_syntax(file_path)
            if syntax_ok:
                print(f"✅ {file_path} syntax is valid")
            else:
                print(f"❌ {file_path} syntax error: {error}")
                all_good = False
        else:
            print(f"⚠️ {file_path} not found (may be optional)")
    
    return all_good

def main():
    """Run all syntax validation tests."""
    print("🚀 LangCache Implementation Syntax Validation\n")
    
    tests = [
        ("Core File Syntax", check_core_files),
        ("LangCache Integration", check_langcache_integration),
        ("Old Cache Removal", check_old_cache_removal),
        ("Configuration", check_configuration)
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
            print(f"❌ Test {test_name} failed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("SYNTAX VALIDATION SUMMARY")
    print('='*60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All syntax validation tests passed!")
        print("✅ LangCache implementation syntax is correct")
        print("✅ Old caching infrastructure has been properly removed")
        print("✅ Integration points are in place")
        print("\nNext steps:")
        print("1. Install required dependencies (requests)")
        print("2. Configure LangCache environment variables")
        print("3. Test with actual LangCache instance")
        return 0
    else:
        print("\n⚠️ Some syntax validation tests failed.")
        print("Fix the issues above before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

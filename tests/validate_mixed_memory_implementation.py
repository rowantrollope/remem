#!/usr/bin/env python3
"""
Validation script to check that mixed memory type implementation is correct.
This script analyzes the code structure without requiring external dependencies.
"""

import sys
import os
import re
import ast

def check_file_exists(filepath):
    """Check if a file exists and return its content."""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return f.read()
    return None

def validate_memory_core_changes():
    """Validate that memory_core.py has the necessary changes for mixed types."""
    print("🔍 Validating memory_core.py changes...")
    
    content = check_file_exists('memory_core.py')
    if not content:
        print("   ❌ memory_core.py not found")
        return False
    
    checks = [
        ('search_memories method has memory_type parameter', r'def search_memories\(.*memory_type.*\)'),
        ('memory type filtering logic', r'if memory_type:'),
        ('k-line type detection', r'memory_type == [\'"]k-line[\'"]'),
        ('neme type detection', r'memory_type == [\'"]neme[\'"]'),
        ('type field in memory object', r'[\'"]type[\'"]:\s*memory_type'),
        ('k-line specific fields', r'original_question.*answer.*confidence'),
        ('should_store_kline method', r'def should_store_kline\('),
        ('store_kline method', r'def store_kline\(')
    ]
    
    passed = 0
    for check_name, pattern in checks:
        if re.search(pattern, content, re.DOTALL):
            print(f"   ✅ {check_name}")
            passed += 1
        else:
            print(f"   ❌ {check_name}")
    
    print(f"   📊 Memory core validation: {passed}/{len(checks)} checks passed")
    return passed == len(checks)

def validate_memory_agent_changes():
    """Validate that memory_agent.py has the necessary changes."""
    print("\n🔍 Validating memory_agent.py changes...")
    
    content = check_file_exists('memory_agent.py')
    if not content:
        print("   ❌ memory_agent.py not found")
        return False
    
    checks = [
        ('search_memories has memory_type parameter', r'def search_memories\(.*memory_type.*\)'),
        ('should_store_kline method exists', r'def should_store_kline\('),
        ('should_store_kline delegates to core', r'return self\.core\.should_store_kline'),
        ('store_kline method exists', r'def store_kline\('),
        ('store_kline delegates to core', r'return self\.core\.store_kline')
    ]
    
    passed = 0
    for check_name, pattern in checks:
        if re.search(pattern, content, re.DOTALL):
            print(f"   ✅ {check_name}")
            passed += 1
        else:
            print(f"   ❌ {check_name}")
    
    print(f"   📊 Memory agent validation: {passed}/{len(checks)} checks passed")
    return passed == len(checks)

def validate_memory_processing_changes():
    """Validate that memory_processing.py handles mixed types."""
    print("\n🔍 Validating memory_processing.py changes...")
    
    content = check_file_exists('memory_processing.py')
    if not content:
        print("   ❌ memory_processing.py not found")
        return False
    
    checks = [
        ('format_memory_results handles k-lines', r'memory_type == [\'"]k-line[\'"]'),
        ('k-line formatting with question/answer', r'original_question.*answer'),
        ('neme formatting fallback', r'memory\.get\([\'"]text[\'"]'),
        ('type detection logic', r'memory\.get\([\'"]type[\'"]')
    ]
    
    passed = 0
    for check_name, pattern in checks:
        if re.search(pattern, content, re.DOTALL):
            print(f"   ✅ {check_name}")
            passed += 1
        else:
            print(f"   ❌ {check_name}")
    
    print(f"   📊 Memory processing validation: {passed}/{len(checks)} checks passed")
    return passed == len(checks)

def validate_web_app_changes():
    """Validate that web_app.py handles mixed types in API responses."""
    print("\n🔍 Validating web_app.py changes...")
    
    content = check_file_exists('web_app.py')
    if not content:
        print("   ❌ web_app.py not found")
        return False
    
    checks = [
        ('memory_type parameter in search endpoint', r'memory_type.*=.*data\.get\([\'"]memory_type[\'"]'),
        ('memory_type validation', r'memory_type.*not in.*neme.*k-line'),
        ('memory_breakdown in response', r'memory_breakdown'),
        ('Memories and klines counting', r'Memories.*=.*klines.*='),
        ('enhanced memory context for k-lines', r'if memory_type == [\'"]k-line[\'"]'),
        ('k-line formatting in context', r'original_question.*answer.*confidence')
    ]
    
    passed = 0
    for check_name, pattern in checks:
        if re.search(pattern, content, re.DOTALL):
            print(f"   ✅ {check_name}")
            passed += 1
        else:
            print(f"   ❌ {check_name}")
    
    print(f"   📊 Web app validation: {passed}/{len(checks)} checks passed")
    return passed == len(checks)

def validate_documentation_updates():
    """Validate that documentation includes mixed type examples."""
    print("\n🔍 Validating documentation updates...")
    
    content = check_file_exists('docs/EXAMPLES.md')
    if not content:
        print("   ❌ docs/EXAMPLES.md not found")
        return False
    
    checks = [
        ('memory_type parameter example', r'memory_type.*:.*[\'"]neme[\'"]'),
        ('k-line only search example', r'memory_type.*:.*[\'"]k-line[\'"]'),
        ('mixed memory handling example', r'memory\.type.*===.*k-line'),
        ('memory breakdown usage', r'memory_breakdown\.Memories.*memory_breakdown\.klines')
    ]
    
    passed = 0
    for check_name, pattern in checks:
        if re.search(pattern, content, re.DOTALL):
            print(f"   ✅ {check_name}")
            passed += 1
        else:
            print(f"   ❌ {check_name}")
    
    print(f"   📊 Documentation validation: {passed}/{len(checks)} checks passed")
    return passed == len(checks)

def validate_api_consistency():
    """Check that API endpoints consistently handle memory types."""
    print("\n🔍 Validating API consistency...")
    
    web_content = check_file_exists('web_app.py')
    if not web_content:
        print("   ❌ web_app.py not found")
        return False
    
    # Check that both search endpoints support memory_type
    search_endpoints = re.findall(r'@app\.route\([\'"][^\'"]*/search[\'"].*?\ndef\s+(\w+)', web_content, re.DOTALL)
    recall_endpoints = re.findall(r'@app\.route\([\'"][^\'"]*/recall[\'"].*?\ndef\s+(\w+)', web_content, re.DOTALL)
    
    print(f"   📊 Found {len(search_endpoints)} search endpoints: {search_endpoints}")
    print(f"   📊 Found {len(recall_endpoints)} recall endpoints: {recall_endpoints}")
    
    # Check that memory_type parameter is documented in docstrings
    memory_type_docs = len(re.findall(r'memory_type.*str.*optional.*Filter by memory type', web_content, re.DOTALL))
    print(f"   📊 Found {memory_type_docs} endpoints with memory_type documentation")

    # Also check for memory_type parameter usage in the code
    memory_type_usage = len(re.findall(r'memory_type.*=.*data\.get\([\'"]memory_type[\'"]', web_content))
    print(f"   📊 Found {memory_type_usage} endpoints using memory_type parameter")

    if memory_type_docs >= 1 and memory_type_usage >= 2:  # Should be documented and used
        print("   ✅ API endpoints consistently support memory_type parameter")
        return True
    else:
        print("   ❌ Not all API endpoints support memory_type parameter")
        return False

def main():
    """Run all validation checks."""
    print("🚀 Validating mixed memory type implementation...")
    print("This checks the code structure without requiring external dependencies.\n")
    
    validations = [
        validate_memory_core_changes,
        validate_memory_agent_changes,
        validate_memory_processing_changes,
        validate_web_app_changes,
        validate_documentation_updates,
        validate_api_consistency
    ]
    
    passed_validations = 0
    total_validations = len(validations)
    
    for validation in validations:
        try:
            if validation():
                passed_validations += 1
        except Exception as e:
            print(f"   ❌ Validation failed with error: {e}")
    
    print(f"\n📊 Overall validation results: {passed_validations}/{total_validations} validations passed")
    
    if passed_validations == total_validations:
        print("\n🎉 All validations passed!")
        print("The mixed memory type implementation appears to be complete and correct.")
        print("\nKey features implemented:")
        print("✅ Memory core supports both Memories and k-lines")
        print("✅ Type-specific filtering (memory_type parameter)")
        print("✅ Proper memory object structure with type field")
        print("✅ Enhanced formatting for different memory types")
        print("✅ API endpoints support memory type filtering")
        print("✅ Documentation includes mixed type examples")
        return True
    else:
        print(f"\n❌ {total_validations - passed_validations} validations failed.")
        print("Please review the implementation for missing features.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

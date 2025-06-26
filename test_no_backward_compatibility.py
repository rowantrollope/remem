#!/usr/bin/env python3
"""
Test to verify that backward compatibility code has been removed.
"""

import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_no_backward_compatibility_in_code():
    """Test that backward compatibility code has been removed from source files."""
    print("🧪 Testing that backward compatibility code has been removed...")
    
    files_to_check = [
        'memory/core.py',
        'memory/agent.py',
        'memory/extraction.py',
        'web_app.py'
    ]
    
    backward_compat_patterns = [
        'formatted_time',  # Old formatted time field
        'fromtimestamp',  # Converting from Unix timestamp
        'fallback to timestamp',  # Specific fallback logic for old timestamps
    ]
    
    issues_found = []
    
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            continue
            
        with open(file_path, 'r') as f:
            content = f.read().lower()
            
        for pattern in backward_compat_patterns:
            if pattern in content:
                # Count occurrences
                count = content.count(pattern)
                issues_found.append(f"{file_path}: '{pattern}' found {count} times")
    
    if issues_found:
        print("⚠️ Potential backward compatibility code still found:")
        for issue in issues_found:
            print(f"  - {issue}")
        return False
    else:
        print("✅ No backward compatibility patterns found in source code")
        return True

def test_memory_storage_clean():
    """Test that memory storage only returns clean ISO timestamps."""
    print("🧪 Testing that memory storage returns only clean timestamps...")
    
    try:
        from memory.core_agent import MemoryAgent
        
        # Initialize memory agent
        memory_agent = MemoryAgent()
        
        # Store a test memory
        result = memory_agent.store_memory("Clean timestamp test", apply_grounding=False)
        
        # Check that only expected fields are present
        expected_fields = {'memory_id', 'original_text', 'final_text', 'grounding_applied', 'tags', 'created_at'}
        actual_fields = set(result.keys())
        
        # Check for unwanted old fields
        unwanted_fields = {'timestamp', 'formatted_time'}
        found_unwanted = actual_fields.intersection(unwanted_fields)
        
        if found_unwanted:
            print(f"❌ Found unwanted old timestamp fields: {found_unwanted}")
            return False
        
        # Check that we have the expected field
        if 'created_at' not in actual_fields:
            print(f"❌ Missing expected 'created_at' field")
            return False
        
        print(f"✅ Memory storage returns only clean fields: {actual_fields}")
        return True
        
    except Exception as e:
        print(f"❌ Error testing memory storage: {e}")
        return False

def main():
    """Run all tests to verify backward compatibility removal."""
    print("🚀 Testing removal of backward compatibility code...")
    print("=" * 60)
    
    tests = [
        test_no_backward_compatibility_in_code,
        test_memory_storage_clean,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print("-" * 40)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            print("-" * 40)
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All backward compatibility removal tests passed!")
        return 0
    else:
        print("⚠️ Some tests failed. Backward compatibility code may still exist.")
        return 1

if __name__ == "__main__":
    exit(main())

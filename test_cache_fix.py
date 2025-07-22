#!/usr/bin/env python3
"""
Test script to verify cache configuration fix
"""

from langcache_client import is_cache_enabled_for_operation, LangCacheClient

def test_cache_configuration():
    """Test that memory extraction operations are disabled for caching."""

    print("Testing cache configuration...")

    # Test that memory extraction operations are disabled
    disabled_operations = ['memory_extraction', 'memory_grounding', 'context_analysis', 'query_optimization']
    enabled_operations = ['embedding_optimization']

    print("\nOperations that should be DISABLED for caching:")
    for operation in disabled_operations:
        enabled = is_cache_enabled_for_operation(operation)
        status = "✅ DISABLED" if not enabled else "❌ ENABLED (should be disabled)"
        print(f"  {operation}: {status}")

    print("\nOperations that should be ENABLED for caching:")
    for operation in enabled_operations:
        enabled = is_cache_enabled_for_operation(operation)
        status = "✅ ENABLED" if enabled else "❌ DISABLED (should be enabled)"
        print(f"  {operation}: {status}")

    # Check if all disabled operations are actually disabled
    all_disabled = all(not is_cache_enabled_for_operation(op) for op in disabled_operations)
    all_enabled = all(is_cache_enabled_for_operation(op) for op in enabled_operations)

    print(f"\nOverall result:")
    if all_disabled and all_enabled:
        print("✅ Cache configuration is correct!")
        return True
    else:
        print("❌ Cache configuration needs fixing")
        return False

def test_prompt_key_generation():
    """Test that prompt keys are now plain text user content."""

    print("\n" + "="*50)
    print("Testing prompt key generation...")

    # Create client with dummy values to avoid environment variable requirement
    client = LangCacheClient(host="dummy", api_key="dummy", cache_id="dummy")

    # Test cases
    test_cases = [
        {
            "messages": [{"role": "user", "content": "What do I like to do?"}],
            "expected": "What do I like to do?"
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Tell me about my preferences"}
            ],
            "expected": "Tell me about my preferences"
        },
        {
            "messages": [{"role": "user", "content": "A" * 1100}],  # Long content
            "expected_starts_with": "A" * 1000,
            "expected_ends_with": "..."
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest case {i}:")
        print(f"  Input messages: {len(test_case['messages'])} message(s)")

        prompt_key = client._create_prompt_key(test_case['messages'])
        print(f"  Generated key: '{prompt_key[:50]}{'...' if len(prompt_key) > 50 else ''}'")
        print(f"  Key length: {len(prompt_key)} characters")

        if 'expected' in test_case:
            if prompt_key == test_case['expected']:
                print(f"  ✅ Matches expected: '{test_case['expected']}'")
            else:
                print(f"  ❌ Expected: '{test_case['expected']}', Got: '{prompt_key}'")
        elif 'expected_starts_with' in test_case:
            if prompt_key.startswith(test_case['expected_starts_with']) and prompt_key.endswith(test_case['expected_ends_with']):
                print(f"  ✅ Correctly truncated long content")
            else:
                print(f"  ❌ Truncation not working as expected")

        # Check length is within LangCache limit
        if len(prompt_key) <= 1024:
            print(f"  ✅ Within 1024 character limit")
        else:
            print(f"  ❌ Exceeds 1024 character limit!")

if __name__ == "__main__":
    test_cache_configuration()
    test_prompt_key_generation()

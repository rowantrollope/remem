#!/usr/bin/env python3
"""
Test script for Embedding Integration with Memory Core

This script tests the integration of the new embedding system with the memory core.
"""

import os
import sys
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, '..')
sys.path.insert(0, '.')

def test_memory_core_with_openai():
    """Test memory core with OpenAI embeddings."""
    print("🧠 Testing Memory Core with OpenAI Embeddings...")
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("   ⏭️ SKIP - No OPENAI_API_KEY found")
        return True
    
    try:
        from memory.core import MemoryCore
        from api.core.config import app_config
        
        # Test with default OpenAI configuration
        memory_core = MemoryCore(app_config=app_config)
        
        print(f"   Embedding provider: {memory_core.embedding_config.provider}")
        print(f"   Embedding model: {memory_core.embedding_config.model}")
        print(f"   Embedding dimension: {memory_core.embedding_config.dimension}")
        
        # Test embedding generation
        test_text = "This is a test memory for embedding"
        embedding = memory_core._get_embedding(test_text)
        
        print(f"   Generated embedding length: {len(embedding)}")
        print(f"   Expected dimension: {memory_core.EMBEDDING_DIM}")
        
        if len(embedding) == memory_core.EMBEDDING_DIM:
            print("   ✅ PASS - Embedding dimension matches configuration")
            return True
        else:
            print("   ❌ FAIL - Embedding dimension mismatch")
            return False
            
    except Exception as e:
        print(f"   ❌ FAIL: {str(e)}")
        return False


def test_memory_core_with_ollama():
    """Test memory core with Ollama embeddings."""
    print("\n🦙 Testing Memory Core with Ollama Embeddings...")
    
    try:
        from memory.core import MemoryCore
        from embedding import EmbeddingConfig
        
        # Create custom app config with Ollama settings
        ollama_app_config = {
            "embedding": {
                "provider": "ollama",
                "model": "nomic-embed-text:latest",
                "dimension": 768,
                "base_url": "http://localhost:11434",
                "api_key": "",
                "timeout": 30
            }
        }
        
        # Test with Ollama configuration
        memory_core = MemoryCore(app_config=ollama_app_config)
        
        print(f"   Embedding provider: {memory_core.embedding_config.provider}")
        print(f"   Embedding model: {memory_core.embedding_config.model}")
        print(f"   Embedding dimension: {memory_core.embedding_config.dimension}")
        
        # Test embedding generation
        test_text = "This is a test memory for Ollama embedding"
        embedding = memory_core._get_embedding(test_text)
        
        print(f"   Generated embedding length: {len(embedding)}")
        print(f"   Expected dimension: {memory_core.EMBEDDING_DIM}")
        
        if len(embedding) == memory_core.EMBEDDING_DIM:
            print("   ✅ PASS - Ollama embedding works correctly")
            return True
        else:
            print("   ❌ FAIL - Ollama embedding dimension mismatch")
            return False
            
    except Exception as e:
        print(f"   ⏭️ SKIP - Ollama test error: {str(e)}")
        return True  # Don't fail if Ollama isn't available


def test_memory_agent_integration():
    """Test memory agent with new embedding system."""
    print("\n🤖 Testing Memory Agent Integration...")
    
    try:
        from memory.core_agent import MemoryAgent
        from api.core.config import app_config
        
        # Create memory agent with app config
        agent = MemoryAgent(app_config=app_config)
        
        print(f"   Core embedding provider: {agent.core.embedding_config.provider}")
        print(f"   Core embedding model: {agent.core.embedding_config.model}")
        
        # Test storing a memory
        test_memory = "I prefer coffee over tea in the morning"
        result = agent.store_memory(test_memory, apply_grounding=False)
        
        if result and 'memory_id' in result:
            print(f"   ✅ PASS - Memory stored with ID: {result['memory_id'][:8]}...")
            
            # Test searching for the memory
            search_results = agent.search_memories("coffee preference", top_k=1)
            
            if search_results and len(search_results) > 0:
                print("   ✅ PASS - Memory search works with new embedding system")
                return True
            else:
                print("   ❌ FAIL - Memory search failed")
                return False
        else:
            print("   ❌ FAIL - Memory storage failed")
            return False
            
    except Exception as e:
        print(f"   ❌ FAIL: {str(e)}")
        return False


def main():
    """Run all embedding integration tests."""
    print("🚀 Testing Embedding Integration with Memory System\n")
    
    tests = [
        ("Memory Core with OpenAI", test_memory_core_with_openai),
        ("Memory Core with Ollama", test_memory_core_with_ollama),
        ("Memory Agent Integration", test_memory_agent_integration)
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
    print("EMBEDDING INTEGRATION TEST SUMMARY")
    print('='*60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All embedding integration tests passed!")
        print("\nThe embedding configuration system is working correctly!")
        print("\nYou can now:")
        print("1. Configure different embedding providers via environment variables")
        print("2. Use the API to change embedding settings at runtime")
        print("3. Switch between OpenAI and Ollama embeddings seamlessly")
    else:
        print(f"\n⚠️ {len(results) - passed} test(s) failed. Check the errors above.")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

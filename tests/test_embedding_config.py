#!/usr/bin/env python3
"""
Test script for Embedding Configuration

This script tests the new embedding configuration system with both OpenAI and Ollama providers.
"""

import os
import sys
import json
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, '..')
sys.path.insert(0, '.')

def test_embedding_config_creation():
    """Test creating embedding configurations."""
    print("🧪 Testing Embedding Configuration Creation...")
    
    try:
        from embedding import EmbeddingConfig, create_embedding_client
        
        # Test OpenAI config
        openai_config = EmbeddingConfig(
            provider="openai",
            model="text-embedding-ada-002",
            dimension=1536,
            api_key="test-key"
        )
        print("   ✅ PASS - OpenAI config created")
        
        # Test Ollama config
        ollama_config = EmbeddingConfig(
            provider="ollama",
            model="nomic-embed-text",
            dimension=768,
            base_url="http://localhost:11434"
        )
        print("   ✅ PASS - Ollama config created")
        
        return True
    except Exception as e:
        print(f"   ❌ FAIL: {str(e)}")
        return False


def test_embedding_client_creation():
    """Test creating embedding clients."""
    print("\n🔧 Testing Embedding Client Creation...")
    
    try:
        from embedding import EmbeddingConfig, create_embedding_client
        
        # Test OpenAI client creation
        openai_config = EmbeddingConfig(
            provider="openai",
            model="text-embedding-ada-002",
            dimension=1536,
            api_key="test-key"
        )
        openai_client = create_embedding_client(openai_config)
        print("   ✅ PASS - OpenAI client created")
        
        # Test Ollama client creation
        ollama_config = EmbeddingConfig(
            provider="ollama",
            model="nomic-embed-text",
            dimension=768,
            base_url="http://localhost:11434"
        )
        ollama_client = create_embedding_client(ollama_config)
        print("   ✅ PASS - Ollama client created")
        
        # Test invalid provider
        try:
            invalid_config = EmbeddingConfig(
                provider="invalid",
                model="test",
                dimension=100
            )
            create_embedding_client(invalid_config)
            print("   ❌ FAIL - Should have rejected invalid provider")
            return False
        except ValueError:
            print("   ✅ PASS - Invalid provider rejected")
        
        return True
    except Exception as e:
        print(f"   ❌ FAIL: {str(e)}")
        return False


def test_app_config_integration():
    """Test integration with app config."""
    print("\n⚙️ Testing App Config Integration...")
    
    try:
        from embedding import get_embedding_config_from_app_config
        from api.core.config import app_config
        
        # Test extracting config from app_config
        embedding_config = get_embedding_config_from_app_config(app_config)
        print(f"   Provider: {embedding_config.provider}")
        print(f"   Model: {embedding_config.model}")
        print(f"   Dimension: {embedding_config.dimension}")
        print("   ✅ PASS - App config integration works")
        
        return True
    except Exception as e:
        print(f"   ❌ FAIL: {str(e)}")
        return False


def test_openai_connection():
    """Test OpenAI embedding connection (if API key available)."""
    print("\n📡 Testing OpenAI Connection...")
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("   ⏭️ SKIP - No OPENAI_API_KEY found")
        return True
    
    try:
        from embedding import EmbeddingConfig, create_embedding_client
        
        config = EmbeddingConfig(
            provider="openai",
            model="text-embedding-ada-002",
            dimension=1536,
            api_key=openai_api_key
        )
        
        client = create_embedding_client(config)
        connection_test = client.test_connection()
        
        if connection_test['success']:
            print("   ✅ PASS - OpenAI connection successful")
            print(f"   Model: {connection_test['model']}")
            print(f"   Dimension match: {connection_test['dimension_match']}")
            return True
        else:
            print(f"   ❌ FAIL - OpenAI connection failed: {connection_test.get('error')}")
            return False
            
    except Exception as e:
        print(f"   ❌ FAIL: {str(e)}")
        return False


def test_ollama_connection():
    """Test Ollama embedding connection (if server available)."""
    print("\n🦙 Testing Ollama Connection...")
    
    try:
        from embedding import EmbeddingConfig, create_embedding_client
        
        config = EmbeddingConfig(
            provider="ollama",
            model="nomic-embed-text",
            dimension=768,
            base_url="http://localhost:11434"
        )
        
        client = create_embedding_client(config)
        connection_test = client.test_connection()
        
        if connection_test['success']:
            print("   ✅ PASS - Ollama connection successful")
            print(f"   Model: {connection_test['model']}")
            print(f"   Available models: {connection_test.get('available_models', [])[:3]}...")
            print(f"   Dimension match: {connection_test['dimension_match']}")
            return True
        else:
            print(f"   ⏭️ SKIP - Ollama not available: {connection_test.get('error')}")
            return True  # Don't fail if Ollama isn't running
            
    except Exception as e:
        print(f"   ⏭️ SKIP - Ollama test error: {str(e)}")
        return True  # Don't fail if Ollama isn't available


def main():
    """Run all embedding configuration tests."""
    print("🚀 Testing Embedding Configuration System\n")
    
    tests = [
        ("Embedding Config Creation", test_embedding_config_creation),
        ("Embedding Client Creation", test_embedding_client_creation),
        ("App Config Integration", test_app_config_integration),
        ("OpenAI Connection", test_openai_connection),
        ("Ollama Connection", test_ollama_connection)
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
    print("EMBEDDING CONFIGURATION TEST SUMMARY")
    print('='*60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All embedding configuration tests passed!")
        print("\nNext steps:")
        print("1. Configure your preferred embedding provider in .env")
        print("2. Test the configuration: python tests/test_embedding_config.py")
        print("3. Start the web server and test via API: python web_app.py")
    else:
        print(f"\n⚠️ {len(results) - passed} test(s) failed. Check the errors above.")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

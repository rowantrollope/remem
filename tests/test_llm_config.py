#!/usr/bin/env python3
"""
Test script for LLM configuration system

This script tests the new two-tier LLM configuration system with both OpenAI and Ollama providers.
"""

import os
import json
import requests
from llm.llm_manager import LLMConfig, LLMManager, OpenAIClient, OllamaClient

def test_llm_clients():
    """Test LLM clients directly."""
    print("🧪 Testing LLM Clients Directly")
    print("=" * 50)
    
    # Test OpenAI client (if API key available)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        print("\n📡 Testing OpenAI Client...")
        try:
            openai_config = LLMConfig(
                provider="openai",
                model="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=50,
                api_key=openai_api_key
            )
            
            openai_client = OpenAIClient(openai_config)
            
            # Test connection
            connection_result = openai_client.test_connection()
            print(f"   Connection test: {'✅ PASS' if connection_result['success'] else '❌ FAIL'}")
            if not connection_result['success']:
                print(f"   Error: {connection_result.get('detail') or connection_result.get('error', 'Unknown error')}")
            
            # Test chat completion
            if connection_result['success']:
                response = openai_client.chat_completion([
                    {"role": "user", "content": "Say hello in one word"}
                ])
                print(f"   Chat completion: ✅ PASS")
                print(f"   Response: {response['content'][:50]}...")
                
        except Exception as e:
            print(f"   ❌ FAIL: {str(e)}")
    else:
        print("\n⚠️  Skipping OpenAI test - no API key found")
    
    # Test Ollama client (if server available)
    print("\n🦙 Testing Ollama Client...")
    try:
        ollama_config = LLMConfig(
            provider="ollama",
            model="llama2",  # Common model name
            temperature=0.7,
            max_tokens=50,
            base_url="http://localhost:11434"
        )
        
        ollama_client = OllamaClient(ollama_config)
        
        # Test connection
        connection_result = ollama_client.test_connection()
        print(f"   Connection test: {'✅ PASS' if connection_result['success'] else '❌ FAIL'}")
        if not connection_result['success']:
            print(f"   Error: {connection_result.get('detail') or connection_result.get('error', 'Unknown error')}")
            if 'available_models' in connection_result:
                print(f"   Available models: {connection_result['available_models']}")
        
        # Test chat completion (only if connection successful)
        if connection_result['success']:
            response = ollama_client.chat_completion([
                {"role": "user", "content": "Say hello in one word"}
            ])
            print(f"   Chat completion: ✅ PASS")
            print(f"   Response: {response['content'][:50]}...")
            
    except Exception as e:
        print(f"   ❌ FAIL: {str(e)}")


def test_llm_manager():
    """Test LLM Manager with two-tier system."""
    print("\n\n🎯 Testing LLM Manager (Two-Tier System)")
    print("=" * 50)
    
    try:
        # Create configurations for both tiers
        tier1_config = LLMConfig(
            provider="openai",
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=100,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        tier2_config = LLMConfig(
            provider="openai",  # Could be "ollama" if available
            model="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=50,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize manager
        manager = LLMManager(tier1_config, tier2_config)
        print("   Manager initialization: ✅ PASS")
        
        # Test connection for both tiers
        connection_results = manager.test_all_connections()
        print(f"   Tier 1 connection: {'✅ PASS' if connection_results['tier1']['success'] else '❌ FAIL'}")
        print(f"   Tier 2 connection: {'✅ PASS' if connection_results['tier2']['success'] else '❌ FAIL'}")
        
        # Test tier 1 (conversational)
        if connection_results['tier1']['success']:
            tier1_client = manager.get_tier1_client()
            response = tier1_client.chat_completion([
                {"role": "user", "content": "Explain AI in one sentence"}
            ])
            print(f"   Tier 1 chat: ✅ PASS")
            print(f"   Response: {response['content'][:80]}...")
        
        # Test tier 2 (utility)
        if connection_results['tier2']['success']:
            tier2_client = manager.get_tier2_client()
            response = tier2_client.chat_completion([
                {"role": "user", "content": "Extract key info: I like pizza"}
            ])
            print(f"   Tier 2 chat: ✅ PASS")
            print(f"   Response: {response['content'][:80]}...")
            
    except Exception as e:
        print(f"   ❌ FAIL: {str(e)}")


def test_api_endpoints():
    """Test the configuration API endpoints."""
    print("\n\n🌐 Testing Configuration API Endpoints")
    print("=" * 50)
    
    base_url = "http://localhost:5001"
    
    try:
        # Test GET /api/config
        print("\n📥 Testing GET /api/config...")
        response = requests.get(f"{base_url}/api/config", timeout=10)
        if response.status_code == 200:
            config = response.json()
            print("   ✅ PASS - Configuration retrieved")
            if 'config' in config and 'llm' in config['config']:
                print("   ✅ PASS - LLM configuration present")
                llm_config = config['config']['llm']
                print(f"   Tier 1 provider: {llm_config['tier1']['provider']}")
                print(f"   Tier 2 provider: {llm_config['tier2']['provider']}")
            else:
                print("   ❌ FAIL - LLM configuration missing")
        else:
            print(f"   ❌ FAIL - HTTP {response.status_code}")
    
    except Exception as e:
        print(f"   ❌ FAIL: {str(e)}")
    
    try:
        # Test POST /api/config/test with LLM configuration
        print("\n🧪 Testing POST /api/config/test...")
        test_config = {
            "llm": {
                "tier1": {
                    "provider": "openai",
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.7,
                    "max_tokens": 1000,
                    "api_key": os.getenv("OPENAI_API_KEY", "test-key")
                },
                "tier2": {
                    "provider": "ollama",
                    "model": "llama2",
                    "temperature": 0.1,
                    "max_tokens": 500,
                    "base_url": "http://localhost:11434"
                }
            }
        }
        
        response = requests.post(
            f"{base_url}/api/config/test",
            json=test_config,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("   ✅ PASS - Configuration test completed")
            if 'test_results' in result and 'tests' in result['test_results']:
                tests = result['test_results']['tests']
                if 'llm' in tests:
                    llm_tests = tests['llm']
                    print(f"   LLM tests valid: {'✅ PASS' if llm_tests['valid'] else '❌ FAIL'}")
                    for tier, tier_result in llm_tests.get('tier_tests', {}).items():
                        print(f"   {tier} valid: {'✅ PASS' if tier_result['valid'] else '❌ FAIL'}")
                        if 'connection_test' in tier_result:
                            conn_test = tier_result['connection_test']
                            print(f"   {tier} connection: {'✅ PASS' if conn_test['success'] else '❌ FAIL'}")
        else:
            print(f"   ❌ FAIL - HTTP {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
    
    except Exception as e:
        print(f"   ❌ FAIL: {str(e)}")


if __name__ == "__main__":
    print("🚀 LLM Configuration System Test Suite")
    print("=" * 60)
    
    # Test individual clients
    test_llm_clients()
    
    # Test manager
    test_llm_manager()
    
    # Test API endpoints (requires server to be running)
    print("\n⚠️  Note: API endpoint tests require the web server to be running")
    print("   Start with: python web_app.py")
    
    try:
        test_api_endpoints()
    except Exception as e:
        print(f"\n❌ API tests failed: {str(e)}")
        print("   Make sure the web server is running on localhost:5001")
    
    print("\n🏁 Test suite completed!")

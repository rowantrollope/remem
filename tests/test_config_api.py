#!/usr/bin/env python3
"""
Test script for Configuration Management API

This script demonstrates how to use the new configuration management endpoints
to get, update, test, and reload system configuration.
"""

import requests
import json
import time

# Configuration for testing
BASE_URL = "http://localhost:5001"
API_BASE = f"{BASE_URL}/api"

def test_api_endpoint(method, endpoint, data=None, description=""):
    """Test an API endpoint and print results."""
    url = f"{API_BASE}{endpoint}"
    
    print(f"\n{'='*60}")
    print(f"Testing: {method} {endpoint}")
    if description:
        print(f"Description: {description}")
    print(f"{'='*60}")
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        elif method == "PUT":
            response = requests.put(url, json=data, timeout=10)
        elif method == "DELETE":
            response = requests.delete(url, timeout=10)
        else:
            print(f"❌ Unsupported method: {method}")
            return
        
        print(f"Status Code: {response.status_code}")
        
        try:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
        except:
            print(f"Response (text): {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed. Make sure the web server is running on port 5001")
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Test the configuration management API."""
    print("🔧 Configuration Management API Test")
    print("=" * 60)
    print("Make sure the web server is running: python web_app.py")
    print("=" * 60)
    
    # Test 1: Get current configuration
    test_api_endpoint(
        "GET", 
        "/config",
        description="Get current system configuration"
    )
    
    # Test 2: Test configuration changes
    test_config = {
        "redis": {
            "host": "localhost",
            "port":6379,
            "db": 0
        },
        "openai": {
            "temperature": 0.2,
            "chat_model": "gpt-3.5-turbo"
        },
        "langgraph": {
            "model_name": "gpt-3.5-turbo",
            "temperature": 0.15
        },
        "memory_agent": {
            "default_top_k": 7,
            "apply_grounding_default": True
        }
    }
    
    test_api_endpoint(
        "POST",
        "/config/test",
        data=test_config,
        description="Test configuration changes without applying them"
    )
    
    # Test 3: Update configuration
    update_config = {
        "memory_agent": {
            "default_top_k": 8
        },
        "langgraph": {
            "temperature": 0.2
        }
    }
    
    test_api_endpoint(
        "PUT",
        "/config",
        data=update_config,
        description="Update memory agent and LangGraph configuration"
    )
    
    # Test 4: Get updated configuration
    test_api_endpoint(
        "GET",
        "/config",
        description="Get configuration after updates"
    )
    
    # Test 5: Reload configuration
    test_api_endpoint(
        "POST",
        "/config/reload",
        description="Reload configuration and restart memory agent"
    )
    
    # Test 6: Test invalid configuration
    invalid_config = {
        "redis": {
            "port": "invalid_port"
        },
        "openai": {
            "temperature": "not_a_number"
        }
    }
    
    test_api_endpoint(
        "POST",
        "/config/test",
        data=invalid_config,
        description="Test invalid configuration (should show errors)"
    )
    
    # Test 7: Test Redis connection change
    redis_test_config = {
        "redis": {
            "host": "localhost",
            "port":     ,
            "db": 1  # Different database
        }
    }
    
    test_api_endpoint(
        "POST",
        "/config/test",
        data=redis_test_config,
        description="Test Redis connection to different database"
    )
    
    print("\n" + "="*60)
    print("🎉 Configuration API Testing Complete!")
    print("="*60)
    print("\nConfiguration Management API Summary:")
    print("✅ GET /api/config - Get current configuration")
    print("✅ PUT /api/config - Update configuration")
    print("✅ POST /api/config/test - Test configuration without applying")
    print("✅ POST /api/config/reload - Reload configuration and restart agent")
    print("\nConfiguration Categories:")
    print("• redis: host, port, db, vectorset_key")
    print("• openai: api_key, organization, embedding_model, embedding_dimension, chat_model, temperature")
    print("• langgraph: model_name, temperature, system_prompt_enabled")
    print("• memory_agent: default_top_k, apply_grounding_default, validation_enabled")
    print("• web_server: host, port, debug, cors_enabled")

if __name__ == "__main__":
    main()

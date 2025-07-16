#!/usr/bin/env python3
"""
Simple Configuration API Test (no external dependencies)

This script demonstrates the configuration API using only Python standard library.
"""

import json
import urllib.request
import urllib.parse
import urllib.error

BASE_URL = "http://localhost:5001/api"

def make_request(method, endpoint, data=None):
    """Make HTTP request using urllib."""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if data:
            data = json.dumps(data).encode('utf-8')
        
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header('Content-Type', 'application/json')
        
        with urllib.request.urlopen(req, timeout=10) as response:
            result = json.loads(response.read().decode('utf-8'))
            return response.status, result
            
    except urllib.error.HTTPError as e:
        try:
            error_data = json.loads(e.read().decode('utf-8'))
            return e.code, error_data
        except:
            return e.code, {"detail": str(e)}
    except Exception as e:
        return None, {"detail": str(e)}

def test_endpoint(method, endpoint, data=None, description=""):
    """Test an endpoint and print results."""
    print(f"\n{'='*50}")
    print(f"Testing: {method} {endpoint}")
    if description:
        print(f"Description: {description}")
    print(f"{'='*50}")
    
    status, result = make_request(method, endpoint, data)
    
    if status:
        print(f"Status: {status}")
        print(f"Response: {json.dumps(result, indent=2)}")
    else:
        print(f"Error: {result.get('detail') or result.get('error', 'Unknown error')}")

def main():
    """Test configuration API with simple examples."""
    print("🔧 Simple Configuration API Test")
    print("Make sure web server is running: python web_app.py")
    
    # Test 1: Get current config
    test_endpoint("GET", "/config", description="Get current configuration")
    
    # Test 2: Update memory agent settings
    update_data = {
        "memory_agent": {
            "default_top_k": 7
        }
    }
    test_endpoint("PUT", "/config", update_data, "Update memory agent default_top_k")
    
    # Test 3: Test configuration
    test_data = {
        "redis": {
            "host": "localhost",
            "port": 6379
        }
    }
    test_endpoint("POST", "/config/test", test_data, "Test Redis configuration")
    
    # Test 4: Get config after update
    test_endpoint("GET", "/config", description="Get updated configuration")
    
    print(f"\n{'='*50}")
    print("✅ Simple Configuration Test Complete!")
    print("For full testing, use: python test_config_api.py")

if __name__ == "__main__":
    main()

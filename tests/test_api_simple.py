#!/usr/bin/env python3
"""
Simple API test for timestamp standardization.
"""

import requests
import json
from datetime import datetime

def test_memory_api():
    """Test the memory API directly."""
    print("🧪 Testing memory API...")
    
    url = "http://localhost:5000/api/memory"
    data = {
        "text": "API test memory for timestamp standardization",
        "apply_grounding": False
    }
    
    try:
        response = requests.post(url, json=data, timeout=10)
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            if 'created_at' in result:
                created_at = result['created_at']
                try:
                    parsed_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    print(f"✅ API returns ISO timestamp: {created_at}")
                    return True
                except ValueError as e:
                    print(f"❌ Invalid timestamp format: {e}")
                    return False
            else:
                print(f"❌ No created_at field in response")
                return False
        else:
            print(f"❌ API call failed")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_memory_api()

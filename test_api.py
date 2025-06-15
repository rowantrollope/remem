#!/usr/bin/env python3
"""
Test script for the /api/ask endpoint with validation
"""

import requests
import json

def test_ask_api():
    """Test the /api/ask endpoint with various inputs."""
    base_url = "http://localhost:5001"
    
    test_cases = [
        {
            "question": "What did I learn about Redis yesterday?",
            "expected_type": "search_result"
        },
        {
            "question": "hello",
            "expected_type": "help_message"
        },
        {
            "question": "Tell me about my meeting with Sarah",
            "expected_type": "search_result"
        },
        {
            "question": "how are you?",
            "expected_type": "help_message"
        },
        {
            "question": "asdfghjkl",
            "expected_type": "help_message"
        }
    ]
    
    print("🧪 Testing /api/ask endpoint...")
    print("="*60)
    
    for i, test_case in enumerate(test_cases, 1):
        question = test_case["question"]
        expected = test_case["expected_type"]
        
        print(f"\nTest {i}: '{question}'")
        print(f"Expected: {expected}")
        print("-" * 40)
        
        try:
            response = requests.post(
                f"{base_url}/api/ask",
                json={"question": question},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "")
                print(f"✅ Status: {response.status_code}")
                print(f"Answer: {answer}")
                
                # Check if it's a help message
                if "Ask me to remember anything" in answer:
                    print("Type: help_message")
                else:
                    print("Type: search_result")
            else:
                print(f"❌ Status: {response.status_code}")
                print(f"Error: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
    
    print("\n" + "="*60)
    print("✅ API testing completed!")

if __name__ == "__main__":
    test_api()

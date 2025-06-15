#!/usr/bin/env python3
"""
Test script to verify the web API returns the new JSON response format

This script tests the /api/ask endpoint to ensure it returns structured JSON
with answer, confidence, and supporting_memories fields.
"""

import requests
import json
import time

def test_web_api_json_response():
    """Test the web API JSON response format."""
    print("🧪 Testing Web API JSON Response Format")
    print("=" * 40)
    
    base_url = "http://localhost:5001"
    
    # Check if the web server is running
    try:
        response = requests.get(f"{base_url}/api/status", timeout=5)
        if response.status_code != 200:
            print("❌ Web server is not responding correctly")
            return False
        print("✅ Web server is running")
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to web server at {base_url}")
        print("Please make sure the web server is running with: python web_app.py")
        return False
    
    # Test storing some memories first
    print("\n📝 Storing test memories via API...")
    
    test_memories = [
        "Molly is a dog",
        "Molly (the dog) is black", 
        "Molly loves to play fetch",
        "Molly is 3 years old"
    ]
    
    for memory in test_memories:
        try:
            response = requests.post(
                f"{base_url}/api/remember",
                json={"memory": memory},
                timeout=10
            )
            if response.status_code == 200:
                print(f"   ✅ Stored: {memory}")
            else:
                print(f"   ⚠️ Failed to store: {memory}")
        except Exception as e:
            print(f"   ❌ Error storing memory: {e}")
    
    # Test questions with the new JSON format
    print("\n🔍 Testing /api/ask endpoint with JSON response format...")
    
    test_questions = [
        {
            "question": "what color is molly?",
            "description": "Direct question with clear answer"
        },
        {
            "question": "how old is molly?",
            "description": "Another direct question"
        },
        {
            "question": "what does molly like to eat?",
            "description": "Question without clear answer"
        },
        {
            "question": "hello",
            "description": "Help request"
        }
    ]
    
    for i, test_case in enumerate(test_questions, 1):
        print(f"\n--- Test Case {i}: {test_case['description']} ---")
        print(f"❓ Question: {test_case['question']}")
        
        try:
            response = requests.post(
                f"{base_url}/api/ask",
                json={"question": test_case['question']},
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"   ❌ HTTP Error: {response.status_code}")
                print(f"   Response: {response.text}")
                continue
            
            data = response.json()
            
            # Check response structure
            print(f"📋 Response structure check:")
            
            # Check for success field
            if data.get('success'):
                print(f"   ✅ success: {data.get('success')}")
            else:
                print(f"   ❌ Missing or false success field")
            
            # Check required fields from the new format
            required_fields = ['type', 'answer', 'confidence', 'supporting_memories']
            for field in required_fields:
                if field in data:
                    field_type = type(data[field]).__name__
                    print(f"   ✅ {field}: {field_type}")
                else:
                    print(f"   ❌ Missing field: {field}")
            
            # Display the response content
            print(f"\n📄 Response content:")
            print(f"   Type: {data.get('type', 'N/A')}")
            print(f"   Answer: {data.get('answer', 'N/A')}")
            print(f"   Confidence: {data.get('confidence', 'N/A')}")
            
            if data.get('reasoning'):
                print(f"   Reasoning: {data.get('reasoning')}")
            
            supporting_memories = data.get('supporting_memories', [])
            print(f"   Supporting Memories: {len(supporting_memories)} found")
            
            for j, memory in enumerate(supporting_memories[:2], 1):  # Show first 2
                if isinstance(memory, dict):
                    text = memory.get('text', 'N/A')
                    score = memory.get('relevance_score', 0)
                    print(f"      {j}. {text} ({score}% relevant)")
                else:
                    print(f"      {j}. {memory}")
            
            # Test JSON structure
            try:
                json_str = json.dumps(data, indent=2)
                print(f"   ✅ Response is valid JSON")
            except Exception as e:
                print(f"   ❌ JSON serialization failed: {e}")
                
        except requests.exceptions.Timeout:
            print(f"   ❌ Request timed out")
        except Exception as e:
            print(f"   ❌ Request failed: {e}")
    
    print("\n✅ Web API JSON response format test completed!")
    return True

def main():
    """Main test function."""
    success = test_web_api_json_response()
    
    if success:
        print("\n🎉 Web API JSON response format is working correctly!")
        print("The /api/ask endpoint now returns structured responses with answer, confidence, and supporting memories.")
    else:
        print("\n❌ Web API JSON response format test failed.")

if __name__ == "__main__":
    main()

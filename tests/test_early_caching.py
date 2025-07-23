#!/usr/bin/env python3
"""
Test script for early caching functionality in the /ask endpoint.

This test verifies that:
1. The first request to /ask processes normally and caches the response
2. The second identical request returns the cached response immediately
3. Cache metadata is properly included in responses
4. Performance improvement is measurable
"""

import requests
import json
import time
import os

BASE_URL = "http://localhost:5001"

def test_early_caching():
    """Test the early caching functionality for the /ask endpoint."""
    print("🧪 Testing Early Caching for /ask Endpoint")
    print("=" * 60)
    
    # Check if LangCache is configured
    langcache_configured = all([
        os.getenv("LANGCACHE_HOST"),
        os.getenv("LANGCACHE_API_KEY"), 
        os.getenv("LANGCACHE_CACHE_ID")
    ])
    
    if not langcache_configured:
        print("⚠️ LangCache not configured - early caching will be skipped")
        print("Set LANGCACHE_HOST, LANGCACHE_API_KEY, and LANGCACHE_CACHE_ID to test caching")
        return
    
    # Test question that should be cached
    test_question = "What do you know about me?"
    vectorstore_name = "test_memories"
    
    print(f"📝 Test Question: '{test_question}'")
    print(f"📦 Vectorstore: {vectorstore_name}")
    print()
    
    # Test 1: First request (should be cache miss)
    print("1. First Request (Cache Miss Expected)")
    print("-" * 40)
    
    start_time = time.time()
    try:
        response1 = requests.post(
            f"{BASE_URL}/api/memory/{vectorstore_name}/ask",
            json={
                "question": test_question,
                "top_k": 5,
                "min_similarity": 0.7
            },
            timeout=30
        )
        first_request_time = time.time() - start_time
        
        if response1.status_code == 200:
            data1 = response1.json()
            print(f"✅ Status: {response1.status_code}")
            print(f"⏱️ Response Time: {first_request_time:.3f} seconds")
            print(f"📊 Cache Hit: {data1.get('_cache_hit', False)}")
            print(f"💬 Answer: {data1.get('answer', 'No answer')[:100]}...")
            
            if data1.get('_cache_hit'):
                print("⚠️ Unexpected cache hit on first request")
            else:
                print("✅ Cache miss as expected")
                
        else:
            print(f"❌ Error: {response1.status_code} - {response1.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    
    print()
    
    # Test 2: Second identical request (should be cache hit)
    print("2. Second Identical Request (Cache Hit Expected)")
    print("-" * 40)
    
    # Wait a moment to ensure any async caching operations complete
    time.sleep(1)
    
    start_time = time.time()
    try:
        response2 = requests.post(
            f"{BASE_URL}/api/memory/{vectorstore_name}/ask",
            json={
                "question": test_question,
                "top_k": 5,
                "min_similarity": 0.7
            },
            timeout=30
        )
        second_request_time = time.time() - start_time
        
        if response2.status_code == 200:
            data2 = response2.json()
            print(f"✅ Status: {response2.status_code}")
            print(f"⏱️ Response Time: {second_request_time:.3f} seconds")
            print(f"📊 Cache Hit: {data2.get('_cache_hit', False)}")
            print(f"💬 Answer: {data2.get('answer', 'No answer')[:100]}...")
            
            if data2.get('_cache_hit'):
                print("✅ Cache hit as expected")
                print(f"📅 Cached At: {data2.get('_cached_at', 'Unknown')}")
                print(f"🎯 Similarity Score: {data2.get('_similarity_score', 'Unknown')}")
            else:
                print("⚠️ Expected cache hit but got cache miss")
                
        else:
            print(f"❌ Error: {response2.status_code} - {response2.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    
    print()
    
    # Test 3: Performance comparison
    print("3. Performance Analysis")
    print("-" * 40)
    
    if first_request_time > 0 and second_request_time > 0:
        if second_request_time < first_request_time:
            improvement = ((first_request_time - second_request_time) / first_request_time) * 100
            print(f"🚀 Performance Improvement: {improvement:.1f}%")
            print(f"⏱️ Time Saved: {first_request_time - second_request_time:.3f} seconds")
            
            if improvement > 10:
                print("✅ Significant performance improvement achieved")
            else:
                print("⚠️ Modest performance improvement")
        else:
            print("⚠️ Second request was not faster than first")
    
    print()
    
    # Test 4: Different question (should be cache miss)
    print("4. Different Question (Cache Miss Expected)")
    print("-" * 40)
    
    different_question = "What are my favorite foods?"
    
    start_time = time.time()
    try:
        response3 = requests.post(
            f"{BASE_URL}/api/memory/{vectorstore_name}/ask",
            json={
                "question": different_question,
                "top_k": 5,
                "min_similarity": 0.7
            },
            timeout=30
        )
        third_request_time = time.time() - start_time
        
        if response3.status_code == 200:
            data3 = response3.json()
            print(f"✅ Status: {response3.status_code}")
            print(f"⏱️ Response Time: {third_request_time:.3f} seconds")
            print(f"📊 Cache Hit: {data3.get('_cache_hit', False)}")
            print(f"💬 Answer: {data3.get('answer', 'No answer')[:100]}...")
            
            if not data3.get('_cache_hit'):
                print("✅ Cache miss as expected for different question")
            else:
                print("⚠️ Unexpected cache hit for different question")
                
        else:
            print(f"❌ Error: {response3.status_code} - {response3.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    
    print()
    print("=" * 60)
    print("✅ Early Caching Test Completed!")
    
    return True

def test_server_health():
    """Test if the server is running and healthy."""
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=5)
        return response.status_code == 200
    except:
        return False

if __name__ == "__main__":
    print("🔍 Checking server health...")
    if not test_server_health():
        print("❌ Server not healthy. Please start the web app first:")
        print("   python web_app.py")
        exit(1)
    
    print("✅ Server is healthy")
    print()
    
    test_early_caching()

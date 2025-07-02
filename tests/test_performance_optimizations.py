#!/usr/bin/env python3
"""
Performance Optimization Tests

This script tests the performance optimizations to validate that they
reduce LLM calls and improve response times while maintaining quality.
"""

import time
import requests
import json
from typing import Dict, Any, List
import statistics


class PerformanceTestSuite:
    """Test suite for performance optimizations."""
    
    def __init__(self, base_url: str = "http://localhost:5001"):
        """Initialize test suite with API base URL."""
        self.base_url = base_url
        self.results = {}
    
    def test_cache_effectiveness(self) -> Dict[str, Any]:
        """Test LLM response caching effectiveness."""
        print("🧪 Testing cache effectiveness...")

        # Test query that should be cached
        test_query = "What restaurants have I been to in Italy?"

        # First request (cache miss)
        start_time = time.time()
        response1 = requests.post(f"{self.base_url}/api/memory/search", json={
            "query": test_query,
            "top_k": 5
        })
        first_request_time = time.time() - start_time

        # Second request (should be cache hit for exact match)
        start_time = time.time()
        response2 = requests.post(f"{self.base_url}/api/memory/search", json={
            "query": test_query,
            "top_k": 5
        })
        second_request_time = time.time() - start_time

        # Third request with semantically similar query (should be semantic cache hit)
        similar_query = "Which Italian restaurants have I visited?"
        start_time = time.time()
        response3 = requests.post(f"{self.base_url}/api/memory/search", json={
            "query": similar_query,
            "top_k": 5
        })
        third_request_time = time.time() - start_time

        # Get cache metrics
        metrics_response = requests.get(f"{self.base_url}/api/performance/metrics")

        exact_cache_improvement = (first_request_time - second_request_time) / first_request_time * 100
        semantic_cache_improvement = (first_request_time - third_request_time) / first_request_time * 100

        return {
            "test": "cache_effectiveness",
            "first_request_time": first_request_time,
            "second_request_time": second_request_time,
            "third_request_time": third_request_time,
            "exact_cache_improvement_percent": exact_cache_improvement,
            "semantic_cache_improvement_percent": semantic_cache_improvement,
            "cache_metrics": metrics_response.json() if metrics_response.status_code == 200 else None,
            "passed": exact_cache_improvement > 10 or semantic_cache_improvement > 5  # Either type should work
        }
    
    def test_batch_processing(self) -> Dict[str, Any]:
        """Test batch processing optimizations."""
        print("🧪 Testing batch processing...")
        
        # Test multiple similar queries
        queries = [
            "What Italian restaurants do I like?",
            "What are my favorite pizza places?",
            "Where have I eaten good pasta?",
            "What restaurants in Rome have I visited?"
        ]
        
        # Individual requests
        start_time = time.time()
        individual_responses = []
        for query in queries:
            response = requests.post(f"{self.base_url}/api/memory/search", json={
                "query": query,
                "top_k": 3
            })
            individual_responses.append(response)
        individual_time = time.time() - start_time
        
        # Simulated batch processing (using K-line recall which uses optimized processing)
        start_time = time.time()
        batch_responses = []
        for query in queries:
            response = requests.post(f"{self.base_url}/api/klines/recall", json={
                "query": query,
                "top_k": 3,
                "use_llm_filtering": True
            })
            batch_responses.append(response)
        batch_time = time.time() - start_time
        
        improvement = (individual_time - batch_time) / individual_time * 100
        
        return {
            "test": "batch_processing",
            "individual_time": individual_time,
            "batch_time": batch_time,
            "improvement_percent": improvement,
            "query_count": len(queries),
            "passed": improvement > 5  # Expect at least 5% improvement
        }
    
    def test_memory_extraction_optimization(self) -> Dict[str, Any]:
        """Test optimized memory extraction."""
        print("🧪 Testing memory extraction optimization...")
        
        # Test conversation that should trigger memory extraction
        session_response = requests.post(f"{self.base_url}/api/agent/session", json={
            "system_prompt": "You are a helpful travel assistant.",
            "config": {"use_memory": True}
        })
        
        if session_response.status_code != 200:
            return {"test": "memory_extraction", "passed": False, "error": "Failed to create session"}
        
        session_id = session_response.json()["session_id"]
        
        # Send message that should extract memories
        test_message = "I really love Italian food, especially pizza from Naples. I'm also vegetarian and prefer restaurants with outdoor seating."
        
        start_time = time.time()
        message_response = requests.post(f"{self.base_url}/api/agent/session/{session_id}", json={
            "message": test_message,
            "store_memory": True
        })
        extraction_time = time.time() - start_time
        
        # Clean up
        requests.delete(f"{self.base_url}/api/agent/session/{session_id}")
        
        return {
            "test": "memory_extraction",
            "extraction_time": extraction_time,
            "response_status": message_response.status_code,
            "passed": message_response.status_code == 200 and extraction_time < 10  # Should complete in <10 seconds
        }
    
    def test_overall_performance(self) -> Dict[str, Any]:
        """Test overall API performance with optimizations."""
        print("🧪 Testing overall performance...")
        
        # Test multiple endpoints with timing
        endpoints_to_test = [
            {
                "name": "memory_search",
                "method": "POST",
                "url": "/api/memory/search",
                "data": {"query": "Italian restaurants", "top_k": 5}
            },
            {
                "name": "kline_recall",
                "method": "POST", 
                "url": "/api/klines/recall",
                "data": {"query": "What do I like to eat?", "top_k": 5}
            },
            {
                "name": "memory_info",
                "method": "GET",
                "url": "/api/memory",
                "data": None
            }
        ]
        
        results = {}
        
        for endpoint in endpoints_to_test:
            times = []
            
            # Run each test 3 times
            for _ in range(3):
                start_time = time.time()
                
                if endpoint["method"] == "POST":
                    response = requests.post(f"{self.base_url}{endpoint['url']}", json=endpoint["data"])
                else:
                    response = requests.get(f"{self.base_url}{endpoint['url']}")
                
                request_time = time.time() - start_time
                times.append(request_time)
            
            results[endpoint["name"]] = {
                "avg_time": statistics.mean(times),
                "min_time": min(times),
                "max_time": max(times),
                "times": times
            }
        
        return {
            "test": "overall_performance",
            "endpoint_results": results,
            "passed": all(result["avg_time"] < 5 for result in results.values())  # All should be <5 seconds
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all performance tests."""
        print("🚀 Starting performance optimization tests...")
        
        # Get initial metrics
        initial_metrics = requests.get(f"{self.base_url}/api/performance/metrics")
        
        tests = [
            self.test_cache_effectiveness,
            self.test_batch_processing,
            self.test_memory_extraction_optimization,
            self.test_overall_performance
        ]
        
        results = {}
        passed_tests = 0
        
        for test_func in tests:
            try:
                result = test_func()
                results[result["test"]] = result
                if result["passed"]:
                    passed_tests += 1
                    print(f"✅ {result['test']} - PASSED")
                else:
                    print(f"❌ {result['test']} - FAILED")
            except Exception as e:
                print(f"⚠️ {test_func.__name__} - ERROR: {e}")
                results[test_func.__name__] = {"test": test_func.__name__, "passed": False, "error": str(e)}
        
        # Get final metrics
        final_metrics = requests.get(f"{self.base_url}/api/performance/metrics")
        
        summary = {
            "total_tests": len(tests),
            "passed_tests": passed_tests,
            "success_rate": passed_tests / len(tests) * 100,
            "initial_metrics": initial_metrics.json() if initial_metrics.status_code == 200 else None,
            "final_metrics": final_metrics.json() if final_metrics.status_code == 200 else None,
            "test_results": results
        }
        
        print(f"\n📊 Test Summary: {passed_tests}/{len(tests)} tests passed ({summary['success_rate']:.1f}%)")
        
        return summary


def main():
    """Run performance tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test performance optimizations")
    parser.add_argument("--url", default="http://localhost:5001", help="API base URL")
    parser.add_argument("--output", help="Output file for results (JSON)")
    
    args = parser.parse_args()
    
    # Run tests
    test_suite = PerformanceTestSuite(args.url)
    results = test_suite.run_all_tests()
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📁 Results saved to {args.output}")
    
    # Print summary
    print("\n" + "="*50)
    print("PERFORMANCE TEST SUMMARY")
    print("="*50)
    
    for test_name, test_result in results["test_results"].items():
        status = "✅ PASSED" if test_result["passed"] else "❌ FAILED"
        print(f"{test_name}: {status}")
        
        if "improvement_percent" in test_result:
            print(f"  Improvement: {test_result['improvement_percent']:.1f}%")
        
        if "avg_time" in test_result:
            print(f"  Avg Time: {test_result['avg_time']:.2f}s")
    
    print(f"\nOverall Success Rate: {results['success_rate']:.1f}%")
    
    if results["final_metrics"]:
        cache_stats = results["final_metrics"]["performance_metrics"]["cache_stats"]
        print(f"Cache Hit Rate: {cache_stats['hit_rate_percent']:.1f}%")
        print(f"Total Cache Hits: {cache_stats['hits']}")


if __name__ == "__main__":
    main()

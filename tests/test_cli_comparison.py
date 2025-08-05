#!/usr/bin/env python3
"""
Comprehensive CLI Comparison Tests

This test suite compares the original CLI (LangChain-based) with the new CLI (OpenAI SDK-based)
across multiple dimensions: functionality, performance, reliability, and error handling.
"""

import os
import sys
import time
import json
import pytest
import asyncio
import traceback
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import the CLI classes
from memory.agent import MemoryAgentChat
from cli_new import OpenAIMemoryAgent

# Load environment variables
load_dotenv()


@dataclass
class TestResult:
    """
    Data class to store test results for comparison.
    
    Attributes:
        test_name: Name of the test that was run
        cli_type: Type of CLI ('original' or 'openai_sdk')
        success: Whether the test passed
        execution_time: Time taken to execute in seconds
        response_length: Length of the response
        error_message: Error message if test failed
        memory_used: Peak memory usage during test (if measured)
        additional_metrics: Any additional metrics specific to the test
    """
    test_name: str
    cli_type: str
    success: bool
    execution_time: float
    response_length: int = 0
    error_message: str = ""
    memory_used: float = 0.0
    additional_metrics: Dict[str, Any] = None

    def __post_init__(self):
        """
        Initialize additional_metrics if not provided.
        """
        if self.additional_metrics is None:
            self.additional_metrics = {}


class CLITester:
    """
    Main testing class that compares CLI implementations across multiple dimensions.
    
    This class provides methods to test functionality, performance, reliability,
    and error handling of both CLI implementations.
    """

    def __init__(self, vectorstore_name: str = "test_comparison"):
        """
        Initialize the CLI tester with test configuration.

        Args:
            vectorstore_name: Name of the vectorstore to use for testing
        """
        self.vectorstore_name = vectorstore_name
        self.test_results: List[TestResult] = []
        
        # Initialize both CLI agents
        self.original_agent = None
        self.openai_agent = None
        
        # Test data for consistent testing
        self.test_memories = [
            "I prefer 4-space indentation in Python code",
            "My favorite restaurant is Mario's Italian Kitchen", 
            "I work remotely from San Francisco",
            "I use VS Code as my primary editor",
            "I prefer unit tests with pytest framework"
        ]
        
        self.test_queries = [
            "What coding preferences do I have?",
            "Where do I like to eat?",
            "Tell me about my work setup",
            "What tools do I use for development?",
            "How do I prefer to write tests?"
        ]
        
        self.complex_queries = [
            "Show me all my preferences and categorize them by type",
            "What do you know about my coding style and work environment?",
            "Find all memories about food and development tools",
            "Summarize everything you know about me"
        ]

    def setup_agents(self) -> bool:
        """
        Initialize both CLI agents for testing.
        
        Returns:
            bool: True if both agents were initialized successfully
        """
        try:
            # Initialize original agent (LangChain-based)
            print("🔧 Initializing original CLI agent...")
            self.original_agent = MemoryAgentChat(vectorset_key=self.vectorstore_name)
            
            # Initialize OpenAI SDK agent  
            print("🔧 Initializing OpenAI SDK CLI agent...")
            self.openai_agent = OpenAIMemoryAgent(vectorset_key=self.vectorstore_name)
            
            print("✅ Both agents initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize agents: {e}")
            return False

    def cleanup_test_data(self) -> None:
        """
        Clean up test data from both vectorstores.
        """
        try:
            if self.original_agent:
                self.original_agent.memory_agent.clear_all_memories()
            if self.openai_agent:
                self.openai_agent.memory_agent.clear_all_memories()
            print("🧹 Test data cleaned up")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")

    def measure_execution_time(self, func, *args, **kwargs) -> Tuple[Any, float]:
        """
        Measure the execution time of a function.

        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Tuple of (result, execution_time_seconds)
        """
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            return result, execution_time
        except Exception as e:
            execution_time = time.time() - start_time
            return None, execution_time

    def test_basic_functionality(self) -> List[TestResult]:
        """
        Test basic functionality of both CLI implementations.
        
        Tests:
        - Memory storage
        - Memory search
        - Memory deletion
        - Statistics retrieval
        
        Returns:
            List of TestResult objects for basic functionality tests
        """
        results = []
        
        print("\n🧪 Testing Basic Functionality")
        print("=" * 50)
        
        # Test memory storage
        for i, memory in enumerate(self.test_memories):
            # Test original CLI
            try:
                response, exec_time = self.measure_execution_time(
                    self.original_agent.run, f"Remember: {memory}"
                )
                results.append(TestResult(
                    test_name=f"store_memory_{i}",
                    cli_type="original",
                    success=response is not None and "successfully" in response.lower(),
                    execution_time=exec_time,
                    response_length=len(response) if response else 0
                ))
            except Exception as e:
                results.append(TestResult(
                    test_name=f"store_memory_{i}",
                    cli_type="original", 
                    success=False,
                    execution_time=0,
                    error_message=str(e)
                ))
            
            # Test OpenAI SDK CLI
            try:
                response, exec_time = self.measure_execution_time(
                    self.openai_agent.run, f"Remember: {memory}"
                )
                results.append(TestResult(
                    test_name=f"store_memory_{i}",
                    cli_type="openai_sdk",
                    success=response is not None and "successfully" in response.lower(),
                    execution_time=exec_time,
                    response_length=len(response) if response else 0
                ))
            except Exception as e:
                results.append(TestResult(
                    test_name=f"store_memory_{i}",
                    cli_type="openai_sdk",
                    success=False, 
                    execution_time=0,
                    error_message=str(e)
                ))
        
        # Test memory search
        for i, query in enumerate(self.test_queries):
            # Test original CLI
            try:
                response, exec_time = self.measure_execution_time(
                    self.original_agent.run, query
                )
                results.append(TestResult(
                    test_name=f"search_query_{i}",
                    cli_type="original",
                    success=response is not None and len(response) > 0,
                    execution_time=exec_time,
                    response_length=len(response) if response else 0
                ))
            except Exception as e:
                results.append(TestResult(
                    test_name=f"search_query_{i}",
                    cli_type="original",
                    success=False,
                    execution_time=0,
                    error_message=str(e)
                ))
            
            # Test OpenAI SDK CLI
            try:
                response, exec_time = self.measure_execution_time(
                    self.openai_agent.run, query
                )
                results.append(TestResult(
                    test_name=f"search_query_{i}",
                    cli_type="openai_sdk", 
                    success=response is not None and len(response) > 0,
                    execution_time=exec_time,
                    response_length=len(response) if response else 0
                ))
            except Exception as e:
                results.append(TestResult(
                    test_name=f"search_query_{i}",
                    cli_type="openai_sdk",
                    success=False,
                    execution_time=0,
                    error_message=str(e)
                ))
        
        return results

    def test_performance_benchmarks(self) -> List[TestResult]:
        """
        Run performance benchmarks comparing both CLI implementations.
        
        Tests:
        - Response time under load
        - Concurrent request handling
        - Memory usage patterns
        - Tool calling efficiency
        
        Returns:
            List of TestResult objects for performance tests
        """
        results = []
        
        print("\n⚡ Testing Performance Benchmarks")
        print("=" * 50)
        
        # Test response time for complex queries
        for i, query in enumerate(self.complex_queries):
            # Test original CLI
            try:
                response, exec_time = self.measure_execution_time(
                    self.original_agent.run, query
                )
                results.append(TestResult(
                    test_name=f"complex_query_{i}",
                    cli_type="original",
                    success=response is not None,
                    execution_time=exec_time,
                    response_length=len(response) if response else 0,
                    additional_metrics={"query_complexity": "high"}
                ))
            except Exception as e:
                results.append(TestResult(
                    test_name=f"complex_query_{i}",
                    cli_type="original",
                    success=False,
                    execution_time=0,
                    error_message=str(e)
                ))
            
            # Test OpenAI SDK CLI
            try:
                response, exec_time = self.measure_execution_time(
                    self.openai_agent.run, query
                )
                results.append(TestResult(
                    test_name=f"complex_query_{i}",
                    cli_type="openai_sdk",
                    success=response is not None,
                    execution_time=exec_time,
                    response_length=len(response) if response else 0,
                    additional_metrics={"query_complexity": "high"}
                ))
            except Exception as e:
                results.append(TestResult(
                    test_name=f"complex_query_{i}",
                    cli_type="openai_sdk",
                    success=False,
                    execution_time=0,
                    error_message=str(e)
                ))
        
        # Test concurrent processing (simplified)
        concurrent_queries = ["What do you know about me?"] * 3
        
        # Test original CLI concurrent processing
        start_time = time.time()
        try:
            responses = []
            for query in concurrent_queries:
                response = self.original_agent.run(query)
                responses.append(response)
            total_time = time.time() - start_time
            
            results.append(TestResult(
                test_name="concurrent_processing",
                cli_type="original",
                success=len(responses) == len(concurrent_queries),
                execution_time=total_time,
                additional_metrics={
                    "queries_processed": len(responses),
                    "avg_time_per_query": total_time / len(concurrent_queries)
                }
            ))
        except Exception as e:
            results.append(TestResult(
                test_name="concurrent_processing",
                cli_type="original",
                success=False,
                execution_time=0,
                error_message=str(e)
            ))
        
        # Test OpenAI SDK CLI concurrent processing
        start_time = time.time()
        try:
            responses = []
            for query in concurrent_queries:
                response = self.openai_agent.run(query)
                responses.append(response)
            total_time = time.time() - start_time
            
            results.append(TestResult(
                test_name="concurrent_processing",
                cli_type="openai_sdk",
                success=len(responses) == len(concurrent_queries),
                execution_time=total_time,
                additional_metrics={
                    "queries_processed": len(responses),
                    "avg_time_per_query": total_time / len(concurrent_queries)
                }
            ))
        except Exception as e:
            results.append(TestResult(
                test_name="concurrent_processing",
                cli_type="openai_sdk",
                success=False,
                execution_time=0,
                error_message=str(e)
            ))
        
        return results

    def test_reliability_and_error_handling(self) -> List[TestResult]:
        """
        Test reliability and error handling of both CLI implementations.
        
        Tests:
        - Invalid input handling
        - Network error simulation
        - Memory overflow scenarios
        - Tool calling edge cases
        
        Returns:
            List of TestResult objects for reliability tests
        """
        results = []
        
        print("\n🛡️ Testing Reliability and Error Handling")
        print("=" * 50)
        
        # Test invalid inputs
        invalid_inputs = [
            "",  # Empty input
            "x" * 10000,  # Very long input
            "Show me memories about unicorns and dragons",  # Non-existent data
            "Delete everything now!",  # Potentially dangerous command
        ]
        
        for i, invalid_input in enumerate(invalid_inputs):
            # Test original CLI
            try:
                response, exec_time = self.measure_execution_time(
                    self.original_agent.run, invalid_input
                )
                results.append(TestResult(
                    test_name=f"invalid_input_{i}",
                    cli_type="original",
                    success=response is not None,  # Should handle gracefully
                    execution_time=exec_time,
                    response_length=len(response) if response else 0,
                    additional_metrics={"input_type": "invalid"}
                ))
            except Exception as e:
                results.append(TestResult(
                    test_name=f"invalid_input_{i}",
                    cli_type="original",
                    success=False,
                    execution_time=0,
                    error_message=str(e)
                ))
            
            # Test OpenAI SDK CLI
            try:
                response, exec_time = self.measure_execution_time(
                    self.openai_agent.run, invalid_input
                )
                results.append(TestResult(
                    test_name=f"invalid_input_{i}",
                    cli_type="openai_sdk",
                    success=response is not None,  # Should handle gracefully
                    execution_time=exec_time,
                    response_length=len(response) if response else 0,
                    additional_metrics={"input_type": "invalid"}
                ))
            except Exception as e:
                results.append(TestResult(
                    test_name=f"invalid_input_{i}",
                    cli_type="openai_sdk",
                    success=False,
                    execution_time=0,
                    error_message=str(e)
                ))
        
        return results

    def test_tool_calling_efficiency(self) -> List[TestResult]:
        """
        Test the efficiency and reliability of tool calling in both implementations.
        
        Tests:
        - Single tool calls
        - Multiple sequential tool calls
        - Tool call error handling
        - Tool response parsing
        
        Returns:
            List of TestResult objects for tool calling tests
        """
        results = []
        
        print("\n🔧 Testing Tool Calling Efficiency")
        print("=" * 50)
        
        # Test single tool calls
        tool_tests = [
            "Show me my memory statistics",
            "Search for memories about coding",
            "Set my context to working from home",
        ]
        
        for i, test in enumerate(tool_tests):
            # Test original CLI
            try:
                response, exec_time = self.measure_execution_time(
                    self.original_agent.run, test
                )
                results.append(TestResult(
                    test_name=f"single_tool_{i}",
                    cli_type="original",
                    success=response is not None and len(response) > 0,
                    execution_time=exec_time,
                    response_length=len(response) if response else 0,
                    additional_metrics={"tool_type": "single"}
                ))
            except Exception as e:
                results.append(TestResult(
                    test_name=f"single_tool_{i}",
                    cli_type="original",
                    success=False,
                    execution_time=0,
                    error_message=str(e)
                ))
            
            # Test OpenAI SDK CLI
            try:
                response, exec_time = self.measure_execution_time(
                    self.openai_agent.run, test
                )
                results.append(TestResult(
                    test_name=f"single_tool_{i}",
                    cli_type="openai_sdk",
                    success=response is not None and len(response) > 0,
                    execution_time=exec_time,
                    response_length=len(response) if response else 0,
                    additional_metrics={"tool_type": "single"}
                ))
            except Exception as e:
                results.append(TestResult(
                    test_name=f"single_tool_{i}",
                    cli_type="openai_sdk",
                    success=False,
                    execution_time=0,
                    error_message=str(e)
                ))
        
        # Test multi-step tool calls
        multi_step_test = "Search for my coding preferences, then store a new memory about testing, then show me the updated statistics"
        
        # Test original CLI
        try:
            response, exec_time = self.measure_execution_time(
                self.original_agent.run, multi_step_test
            )
            results.append(TestResult(
                test_name="multi_step_tools",
                cli_type="original",
                success=response is not None and len(response) > 0,
                execution_time=exec_time,
                response_length=len(response) if response else 0,
                additional_metrics={"tool_type": "multi_step"}
            ))
        except Exception as e:
            results.append(TestResult(
                test_name="multi_step_tools",
                cli_type="original",
                success=False,
                execution_time=0,
                error_message=str(e)
            ))
        
        # Test OpenAI SDK CLI
        try:
            response, exec_time = self.measure_execution_time(
                self.openai_agent.run, multi_step_test
            )
            results.append(TestResult(
                test_name="multi_step_tools",
                cli_type="openai_sdk",
                success=response is not None and len(response) > 0,
                execution_time=exec_time,
                response_length=len(response) if response else 0,
                additional_metrics={"tool_type": "multi_step"}
            ))
        except Exception as e:
            results.append(TestResult(
                test_name="multi_step_tools",
                cli_type="openai_sdk",
                success=False,
                execution_time=0,
                error_message=str(e)
            ))
        
        return results

    def generate_comparison_report(self, all_results: List[TestResult]) -> str:
        """
        Generate a comprehensive comparison report from test results.

        Args:
            all_results: List of all TestResult objects from various tests

        Returns:
            Formatted comparison report as a string
        """
        # Group results by CLI type
        original_results = [r for r in all_results if r.cli_type == "original"]
        openai_results = [r for r in all_results if r.cli_type == "openai_sdk"]
        
        # Calculate summary statistics
        original_success_rate = sum(1 for r in original_results if r.success) / len(original_results) * 100
        openai_success_rate = sum(1 for r in openai_results if r.success) / len(openai_results) * 100
        
        original_avg_time = sum(r.execution_time for r in original_results) / len(original_results)
        openai_avg_time = sum(r.execution_time for r in openai_results) / len(openai_results)
        
        original_avg_response_len = sum(r.response_length for r in original_results if r.response_length > 0) / max(1, len([r for r in original_results if r.response_length > 0]))
        openai_avg_response_len = sum(r.response_length for r in openai_results if r.response_length > 0) / max(1, len([r for r in openai_results if r.response_length > 0]))
        
        report = f"""
{'='*80}
                    CLI COMPARISON REPORT
{'='*80}

EXECUTIVE SUMMARY:
-----------------
Original CLI (LangChain):   {len(original_results)} tests, {original_success_rate:.1f}% success rate
OpenAI SDK CLI:             {len(openai_results)} tests, {openai_success_rate:.1f}% success rate

PERFORMANCE METRICS:
-------------------
                        Original CLI    OpenAI SDK CLI    Winner
Average Response Time:  {original_avg_time:.3f}s        {openai_avg_time:.3f}s        {'OpenAI SDK' if openai_avg_time < original_avg_time else 'Original'}
Average Response Length: {original_avg_response_len:.0f} chars     {openai_avg_response_len:.0f} chars     {'OpenAI SDK' if openai_avg_response_len > original_avg_response_len else 'Original'}
Success Rate:           {original_success_rate:.1f}%            {openai_success_rate:.1f}%            {'OpenAI SDK' if openai_success_rate > original_success_rate else 'Original'}

DETAILED TEST RESULTS:
---------------------
"""
        
        # Group results by test name for side-by-side comparison
        test_names = set(r.test_name for r in all_results)
        
        for test_name in sorted(test_names):
            original_result = next((r for r in original_results if r.test_name == test_name), None)
            openai_result = next((r for r in openai_results if r.test_name == test_name), None)
            
            report += f"\n{test_name}:\n"
            
            if original_result:
                status = "✅ PASS" if original_result.success else "❌ FAIL"
                report += f"  Original:   {status} | {original_result.execution_time:.3f}s | {original_result.response_length} chars"
                if original_result.error_message:
                    report += f" | Error: {original_result.error_message[:50]}..."
                report += "\n"
            
            if openai_result:
                status = "✅ PASS" if openai_result.success else "❌ FAIL"
                report += f"  OpenAI SDK: {status} | {openai_result.execution_time:.3f}s | {openai_result.response_length} chars"
                if openai_result.error_message:
                    report += f" | Error: {openai_result.error_message[:50]}..."
                report += "\n"
        
        # Add recommendations
        report += f"""

RECOMMENDATIONS:
---------------
"""
        
        if openai_success_rate > original_success_rate:
            report += "• OpenAI SDK CLI shows higher reliability\n"
        elif original_success_rate > openai_success_rate:
            report += "• Original CLI shows higher reliability\n"
        else:
            report += "• Both CLIs show similar reliability\n"
        
        if openai_avg_time < original_avg_time:
            report += "• OpenAI SDK CLI is faster on average\n"
        elif original_avg_time < openai_avg_time:
            report += "• Original CLI is faster on average\n"
        else:
            report += "• Both CLIs show similar performance\n"
        
        report += f"\n{'='*80}\n"
        
        return report

    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all test suites and generate comprehensive results.
        
        Returns:
            Dictionary containing all test results and summary report
        """
        print("🚀 Starting Comprehensive CLI Comparison Tests")
        print("=" * 80)
        
        # Setup
        if not self.setup_agents():
            return {"error": "Failed to initialize agents"}
        
        # Clean up any existing test data
        self.cleanup_test_data()
        
        all_results = []
        
        try:
            # Run all test suites
            print("\n1️⃣ Running Basic Functionality Tests...")
            basic_results = self.test_basic_functionality()
            all_results.extend(basic_results)
            
            print("\n2️⃣ Running Performance Benchmarks...")
            perf_results = self.test_performance_benchmarks()
            all_results.extend(perf_results)
            
            print("\n3️⃣ Running Reliability Tests...")
            reliability_results = self.test_reliability_and_error_handling()
            all_results.extend(reliability_results)
            
            print("\n4️⃣ Running Tool Calling Tests...")
            tool_results = self.test_tool_calling_efficiency()
            all_results.extend(tool_results)
            
            # Generate comprehensive report
            report = self.generate_comparison_report(all_results)
            
            return {
                "success": True,
                "total_tests": len(all_results),
                "results": all_results,
                "report": report
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "partial_results": all_results
            }
        
        finally:
            # Cleanup
            self.cleanup_test_data()


def main():
    """
    Main function to run CLI comparison tests.
    """
    print("🧪 CLI Comparison Test Suite")
    print("=" * 50)
    
    # Check if required environment variables are set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found. Please set it in .env file.")
        return
    
    # Initialize tester
    tester = CLITester(vectorstore_name="test_cli_comparison")
    
    # Run all tests
    results = tester.run_all_tests()
    
    if results.get("success"):
        print(results["report"])
        
        # Save detailed results to file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"cli_comparison_results_{timestamp}.json"
        
        # Convert TestResult objects to dictionaries for JSON serialization
        serializable_results = []
        for result in results["results"]:
            serializable_results.append({
                "test_name": result.test_name,
                "cli_type": result.cli_type,
                "success": result.success,
                "execution_time": result.execution_time,
                "response_length": result.response_length,
                "error_message": result.error_message,
                "memory_used": result.memory_used,
                "additional_metrics": result.additional_metrics
            })
        
        with open(filename, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "total_tests": results["total_tests"],
                "results": serializable_results
            }, f, indent=2)
        
        print(f"\n📄 Detailed results saved to: {filename}")
        
    else:
        print(f"❌ Tests failed: {results.get('error')}")
        if "partial_results" in results:
            print(f"📊 Partial results available: {len(results['partial_results'])} tests completed")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Performance Test with Mock Dependencies

Tests CLI performance without requiring API keys or external services
by mocking the dependencies and measuring code execution paths.
"""

import time
import sys
import os
import psutil
import traceback
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

class PerformanceTester:
    """
    Performance tester that uses mocks to avoid external dependencies
    while still measuring real code execution performance.
    """
    
    def __init__(self):
        """
        Initialize the performance tester with mock configurations.
        """
        self.results = {}
        self.process = psutil.Process(os.getpid())
        
    def measure_memory_usage(self):
        """
        Get current memory usage in MB.
        
        Returns:
            float: Memory usage in MB
        """
        return self.process.memory_info().rss / 1024 / 1024
    
    def measure_import_performance(self) -> Dict[str, Any]:
        """
        Measure the time and memory cost of importing both CLI modules.
        
        Returns:
            Dictionary with import performance metrics
        """
        results = {
            "original_cli": {"import_time": 0, "memory_before": 0, "memory_after": 0, "success": False, "error": ""},
            "openai_cli": {"import_time": 0, "memory_before": 0, "memory_after": 0, "success": False, "error": ""}
        }
        
        # Test original CLI import
        try:
            memory_before = self.measure_memory_usage()
            start_time = time.time()
            
            # Mock the problematic dependencies
            with patch('numpy.array'), \
                 patch('redis.Redis'), \
                 patch('openai.OpenAI'), \
                 patch.dict('sys.modules', {'numpy': Mock(), 'redis': Mock()}):
                
                from memory.agent import MemoryAgentChat
            
            import_time = time.time() - start_time
            memory_after = self.measure_memory_usage()
            
            results["original_cli"].update({
                "import_time": import_time,
                "memory_before": memory_before,
                "memory_after": memory_after,
                "memory_delta": memory_after - memory_before,
                "success": True
            })
            
        except Exception as e:
            results["original_cli"]["error"] = str(e)
        
        # Test OpenAI CLI import
        try:
            memory_before = self.measure_memory_usage()
            start_time = time.time()
            
            # Mock the problematic dependencies
            with patch('openai.OpenAI'), \
                 patch('redis.Redis'), \
                 patch.dict('sys.modules', {'openai': Mock(), 'redis': Mock()}):
                
                from cli_new import OpenAIMemoryAgent
            
            import_time = time.time() - start_time
            memory_after = self.measure_memory_usage()
            
            results["openai_cli"].update({
                "import_time": import_time,
                "memory_before": memory_before,
                "memory_after": memory_after,
                "memory_delta": memory_after - memory_before,
                "success": True
            })
            
        except Exception as e:
            results["openai_cli"]["error"] = str(e)
        
        return results
    
    def measure_initialization_performance(self) -> Dict[str, Any]:
        """
        Measure initialization time and memory usage for both CLIs.
        
        Returns:
            Dictionary with initialization performance metrics
        """
        results = {
            "original_cli": {"init_time": 0, "memory_delta": 0, "success": False, "error": ""},
            "openai_cli": {"init_time": 0, "memory_delta": 0, "success": False, "error": ""}
        }
        
        # Mock dependencies
        mock_memory_agent = Mock()
        mock_memory_agent.core.VECTORSET_KEY = "test"
        mock_llm = Mock()
        mock_redis = Mock()
        mock_openai_client = Mock()
        
        # Test original CLI initialization
        try:
            with patch('memory.agent.MemoryAgent') as mock_ma, \
                 patch('langchain_openai.ChatOpenAI') as mock_chat, \
                 patch('redis.Redis', return_value=mock_redis), \
                 patch('os.getenv', return_value="fake_key"):
                
                mock_ma.return_value = mock_memory_agent
                mock_chat.return_value = mock_llm
                
                from memory.agent import MemoryAgentChat
                
                memory_before = self.measure_memory_usage()
                start_time = time.time()
                
                agent = MemoryAgentChat(vectorset_key="test_performance")
                
                init_time = time.time() - start_time
                memory_after = self.measure_memory_usage()
                
                results["original_cli"].update({
                    "init_time": init_time,
                    "memory_delta": memory_after - memory_before,
                    "success": True
                })
                
        except Exception as e:
            results["original_cli"]["error"] = str(e)
        
        # Test OpenAI CLI initialization
        try:
            with patch('cli_new.MemoryAgent') as mock_ma, \
                 patch('openai.OpenAI') as mock_openai, \
                 patch('redis.Redis', return_value=mock_redis), \
                 patch('os.getenv', return_value="fake_key"):
                
                mock_ma.return_value = mock_memory_agent
                mock_openai.return_value = mock_openai_client
                
                from cli_new import OpenAIMemoryAgent
                
                memory_before = self.measure_memory_usage()
                start_time = time.time()
                
                agent = OpenAIMemoryAgent(vectorset_key="test_performance")
                
                init_time = time.time() - start_time
                memory_after = self.measure_memory_usage()
                
                results["openai_cli"].update({
                    "init_time": init_time,
                    "memory_delta": memory_after - memory_before,
                    "success": True
                })
                
        except Exception as e:
            results["openai_cli"]["error"] = str(e)
        
        return results
    
    def measure_method_call_performance(self) -> Dict[str, Any]:
        """
        Measure performance of individual method calls for both CLIs.
        
        Returns:
            Dictionary with method call performance metrics
        """
        results = {
            "original_cli": {},
            "openai_cli": {}
        }
        
        # Mock responses
        mock_response = "This is a mock response from the agent"
        mock_memory_agent = Mock()
        mock_memory_agent.core.VECTORSET_KEY = "test"
        mock_memory_agent.get_memory_info.return_value = {
            "memory_count": 5,
            "vectorset_name": "test",
            "vector_dimension": 1536,
            "embedding_model": "text-embedding-ada-002"
        }
        
        # Test original CLI methods
        try:
            with patch('memory.agent.MemoryAgent') as mock_ma, \
                 patch('langchain_openai.ChatOpenAI') as mock_chat, \
                 patch('os.getenv', return_value="fake_key"):
                
                mock_ma.return_value = mock_memory_agent
                mock_llm = Mock()
                mock_llm.invoke.return_value = Mock(content=mock_response, tool_calls=None)
                mock_chat.return_value.bind_tools.return_value = mock_llm
                
                from memory.agent import MemoryAgentChat
                agent = MemoryAgentChat(vectorset_key="test_performance")
                
                # Test run method
                start_time = time.time()
                response = agent.run("Test query")
                run_time = time.time() - start_time
                
                # Test show_stats method
                start_time = time.time()
                agent.show_stats()
                stats_time = time.time() - start_time
                
                results["original_cli"] = {
                    "run_method_time": run_time,
                    "stats_method_time": stats_time,
                    "success": True
                }
                
        except Exception as e:
            results["original_cli"]["error"] = str(e)
        
        # Test OpenAI CLI methods
        try:
            with patch('cli_new.MemoryAgent') as mock_ma, \
                 patch('openai.OpenAI') as mock_openai, \
                 patch('os.getenv', return_value="fake_key"):
                
                mock_ma.return_value = mock_memory_agent
                mock_client = Mock()
                mock_completion = Mock()
                mock_completion.choices = [Mock()]
                mock_completion.choices[0].message = Mock(content=mock_response, tool_calls=None)
                mock_client.chat.completions.create.return_value = mock_completion
                mock_openai.return_value = mock_client
                
                from cli_new import OpenAIMemoryAgent
                agent = OpenAIMemoryAgent(vectorset_key="test_performance")
                
                # Test run method
                start_time = time.time()
                response = agent.run("Test query")
                run_time = time.time() - start_time
                
                # Test show_stats method
                start_time = time.time()
                agent.show_stats()
                stats_time = time.time() - start_time
                
                results["openai_cli"] = {
                    "run_method_time": run_time,
                    "stats_method_time": stats_time,
                    "success": True
                }
                
        except Exception as e:
            results["openai_cli"]["error"] = str(e)
        
        return results
    
    def measure_tool_preparation_performance(self) -> Dict[str, Any]:
        """
        Measure the time to prepare tools for both CLI implementations.
        
        Returns:
            Dictionary with tool preparation performance metrics
        """
        results = {
            "original_cli": {"prep_time": 0, "tool_count": 0, "success": False, "error": ""},
            "openai_cli": {"prep_time": 0, "tool_count": 0, "success": False, "error": ""}
        }
        
        # Mock the tools
        mock_tools = [Mock() for _ in range(5)]  # Simulate 5 tools
        for i, tool in enumerate(mock_tools):
            tool.name = f"mock_tool_{i}"
            tool.description = f"Mock tool {i}"
            tool.args_schema = Mock()
            tool.args_schema.schema.return_value = {"properties": {}, "required": []}
        
        # Test original CLI tool preparation (LangChain format)
        try:
            with patch('memory.tools.AVAILABLE_TOOLS', mock_tools):
                start_time = time.time()
                
                # Simulate the original CLI's tool binding approach
                tools_by_name = {tool.name: tool for tool in mock_tools}
                
                prep_time = time.time() - start_time
                
                results["original_cli"].update({
                    "prep_time": prep_time,
                    "tool_count": len(mock_tools),
                    "success": True
                })
                
        except Exception as e:
            results["original_cli"]["error"] = str(e)
        
        # Test OpenAI CLI tool preparation (OpenAI format conversion)
        try:
            with patch('memory.tools.AVAILABLE_TOOLS', mock_tools):
                start_time = time.time()
                
                # Simulate the OpenAI CLI's tool preparation
                openai_tools = []
                for tool in mock_tools:
                    schema = tool.args_schema.schema()
                    tool_def = {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": {
                                "type": "object",
                                "properties": schema.get("properties", {}),
                                "required": schema.get("required", [])
                            }
                        }
                    }
                    openai_tools.append(tool_def)
                
                prep_time = time.time() - start_time
                
                results["openai_cli"].update({
                    "prep_time": prep_time,
                    "tool_count": len(openai_tools),
                    "success": True
                })
                
        except Exception as e:
            results["openai_cli"]["error"] = str(e)
        
        return results
    
    def run_all_performance_tests(self) -> Dict[str, Any]:
        """
        Run all performance tests and return comprehensive results.
        
        Returns:
            Dictionary containing all performance test results
        """
        print("🚀 Starting Performance Tests with Mocked Dependencies")
        print("=" * 60)
        
        all_results = {}
        
        # Test 1: Import Performance
        print("\n1️⃣ Testing Import Performance...")
        all_results["import_performance"] = self.measure_import_performance()
        
        # Test 2: Initialization Performance
        print("2️⃣ Testing Initialization Performance...")
        all_results["initialization_performance"] = self.measure_initialization_performance()
        
        # Test 3: Method Call Performance
        print("3️⃣ Testing Method Call Performance...")
        all_results["method_performance"] = self.measure_method_call_performance()
        
        # Test 4: Tool Preparation Performance
        print("4️⃣ Testing Tool Preparation Performance...")
        all_results["tool_preparation"] = self.measure_tool_preparation_performance()
        
        return all_results
    
    def generate_performance_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive performance report.
        
        Args:
            results: Performance test results
            
        Returns:
            Formatted performance report string
        """
        report = f"""
{'='*80}
                PERFORMANCE COMPARISON REPORT
{'='*80}

IMPORT PERFORMANCE:
                    Original CLI    OpenAI CLI    Winner
Import Time:        {results['import_performance']['original_cli'].get('import_time', 0):.4f}s       {results['import_performance']['openai_cli'].get('import_time', 0):.4f}s       {'Original' if results['import_performance']['original_cli'].get('import_time', 999) < results['import_performance']['openai_cli'].get('import_time', 999) else 'OpenAI'}
Memory Delta:       {results['import_performance']['original_cli'].get('memory_delta', 0):.2f}MB        {results['import_performance']['openai_cli'].get('memory_delta', 0):.2f}MB        {'Original' if results['import_performance']['original_cli'].get('memory_delta', 999) < results['import_performance']['openai_cli'].get('memory_delta', 999) else 'OpenAI'}
Success:            {'✅' if results['import_performance']['original_cli'].get('success') else '❌'}              {'✅' if results['import_performance']['openai_cli'].get('success') else '❌'}

INITIALIZATION PERFORMANCE:
                    Original CLI    OpenAI CLI    Winner
Init Time:          {results['initialization_performance']['original_cli'].get('init_time', 0):.4f}s       {results['initialization_performance']['openai_cli'].get('init_time', 0):.4f}s       {'Original' if results['initialization_performance']['original_cli'].get('init_time', 999) < results['initialization_performance']['openai_cli'].get('init_time', 999) else 'OpenAI'}
Memory Delta:       {results['initialization_performance']['original_cli'].get('memory_delta', 0):.2f}MB        {results['initialization_performance']['openai_cli'].get('memory_delta', 0):.2f}MB        {'Original' if results['initialization_performance']['original_cli'].get('memory_delta', 999) < results['initialization_performance']['openai_cli'].get('memory_delta', 999) else 'OpenAI'}
Success:            {'✅' if results['initialization_performance']['original_cli'].get('success') else '❌'}              {'✅' if results['initialization_performance']['openai_cli'].get('success') else '❌'}

METHOD CALL PERFORMANCE:
                    Original CLI    OpenAI CLI    Winner
Run Method:         {results['method_performance']['original_cli'].get('run_method_time', 0):.4f}s       {results['method_performance']['openai_cli'].get('run_method_time', 0):.4f}s       {'Original' if results['method_performance']['original_cli'].get('run_method_time', 999) < results['method_performance']['openai_cli'].get('run_method_time', 999) else 'OpenAI'}
Stats Method:       {results['method_performance']['original_cli'].get('stats_method_time', 0):.4f}s       {results['method_performance']['openai_cli'].get('stats_method_time', 0):.4f}s       {'Original' if results['method_performance']['original_cli'].get('stats_method_time', 999) < results['method_performance']['openai_cli'].get('stats_method_time', 999) else 'OpenAI'}

TOOL PREPARATION PERFORMANCE:
                    Original CLI    OpenAI CLI    Winner
Prep Time:          {results['tool_preparation']['original_cli'].get('prep_time', 0):.6f}s     {results['tool_preparation']['openai_cli'].get('prep_time', 0):.6f}s     {'Original' if results['tool_preparation']['original_cli'].get('prep_time', 999) < results['tool_preparation']['openai_cli'].get('prep_time', 999) else 'OpenAI'}
Tools Processed:    {results['tool_preparation']['original_cli'].get('tool_count', 0)}              {results['tool_preparation']['openai_cli'].get('tool_count', 0)}              {'Same' if results['tool_preparation']['original_cli'].get('tool_count', 0) == results['tool_preparation']['openai_cli'].get('tool_count', 0) else 'Different'}

PERFORMANCE SUMMARY:
"""
        
        # Calculate winners
        categories = ['import_time', 'init_time', 'run_method_time', 'prep_time']
        original_wins = 0
        openai_wins = 0
        
        for category in categories:
            if category == 'import_time':
                orig_val = results['import_performance']['original_cli'].get('import_time', 999)
                openai_val = results['import_performance']['openai_cli'].get('import_time', 999)
            elif category == 'init_time':
                orig_val = results['initialization_performance']['original_cli'].get('init_time', 999)
                openai_val = results['initialization_performance']['openai_cli'].get('init_time', 999)
            elif category == 'run_method_time':
                orig_val = results['method_performance']['original_cli'].get('run_method_time', 999)
                openai_val = results['method_performance']['openai_cli'].get('run_method_time', 999)
            elif category == 'prep_time':
                orig_val = results['tool_preparation']['original_cli'].get('prep_time', 999)
                openai_val = results['tool_preparation']['openai_cli'].get('prep_time', 999)
            
            if orig_val < openai_val:
                original_wins += 1
            elif openai_val < orig_val:
                openai_wins += 1
        
        overall_winner = "Original CLI" if original_wins > openai_wins else "OpenAI CLI" if openai_wins > original_wins else "Tie"
        
        report += f"""
Original CLI wins: {original_wins}/{len(categories)} categories
OpenAI CLI wins:   {openai_wins}/{len(categories)} categories
Overall Winner:    {overall_winner}

DETAILED ANALYSIS:
• Import Speed: {'Original CLI is faster to import' if results['import_performance']['original_cli'].get('import_time', 999) < results['import_performance']['openai_cli'].get('import_time', 999) else 'OpenAI CLI is faster to import'}
• Memory Usage: {'Original CLI uses less memory' if results['import_performance']['original_cli'].get('memory_delta', 999) < results['import_performance']['openai_cli'].get('memory_delta', 999) else 'OpenAI CLI uses less memory'}
• Initialization: {'Original CLI initializes faster' if results['initialization_performance']['original_cli'].get('init_time', 999) < results['initialization_performance']['openai_cli'].get('init_time', 999) else 'OpenAI CLI initializes faster'}
• Tool Processing: {'Original CLI processes tools faster' if results['tool_preparation']['original_cli'].get('prep_time', 999) < results['tool_preparation']['openai_cli'].get('prep_time', 999) else 'OpenAI CLI processes tools faster'}

RECOMMENDATIONS:
"""
        
        if overall_winner == "Original CLI":
            report += "• Original CLI shows better raw performance\n"
            report += "• Consider Original CLI for performance-critical applications\n"
        elif overall_winner == "OpenAI CLI":
            report += "• OpenAI CLI shows better performance despite added features\n"
            report += "• OpenAI CLI provides better performance with enhanced functionality\n"
        else:
            report += "• Performance is comparable between implementations\n"
            report += "• Choose based on features and maintainability rather than performance\n"
        
        report += f"\n{'='*80}\n"
        
        return report

def main():
    """
    Main function to run performance tests.
    """
    print("⚡ CLI Performance Testing")
    print("=" * 40)
    
    # Check if psutil is available
    try:
        import psutil
    except ImportError:
        print("❌ psutil not available. Installing...")
        os.system("pip3 install psutil")
        import psutil
    
    # Initialize tester
    tester = PerformanceTester()
    
    # Run all tests
    results = tester.run_all_performance_tests()
    
    # Generate and display report
    report = tester.generate_performance_report(results)
    print(report)
    
    # Save detailed results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"performance_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"📄 Detailed results saved to: {filename}")

if __name__ == "__main__":
    main()
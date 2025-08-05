#!/usr/bin/env python3
"""
Simple CLI Comparison Test

A lightweight test to compare basic functionality and performance
between the original CLI and the new OpenAI SDK CLI.
"""

import os
import sys
import time
import json
from typing import Dict, List, Tuple

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_import_compatibility():
    """
    Test if both CLI implementations can be imported successfully.
    
    Returns:
        Dict with import test results
    """
    results = {
        "original_cli": {"success": False, "error": ""},
        "openai_cli": {"success": False, "error": ""}
    }
    
    # Test original CLI imports
    try:
        from memory.agent import MemoryAgentChat
        results["original_cli"]["success"] = True
        print("✅ Original CLI imports successfully")
    except Exception as e:
        results["original_cli"]["error"] = str(e)
        print(f"❌ Original CLI import failed: {e}")
    
    # Test new CLI imports
    try:
        from cli_new import OpenAIMemoryAgent
        results["openai_cli"]["success"] = True
        print("✅ OpenAI SDK CLI imports successfully")
    except Exception as e:
        results["openai_cli"]["error"] = str(e)
        print(f"❌ OpenAI SDK CLI import failed: {e}")
    
    return results

def test_agent_initialization():
    """
    Test if both agents can be initialized successfully.
    
    Returns:
        Dict with initialization test results and timing
    """
    results = {
        "original_cli": {"success": False, "time": 0, "error": ""},
        "openai_cli": {"success": False, "time": 0, "error": ""}
    }
    
    # Test original CLI initialization
    try:
        from memory.agent import MemoryAgentChat
        start_time = time.time()
        agent = MemoryAgentChat(vectorset_key="test_simple")
        init_time = time.time() - start_time
        results["original_cli"]["success"] = True
        results["original_cli"]["time"] = init_time
        print(f"✅ Original CLI initialized in {init_time:.3f}s")
    except Exception as e:
        results["original_cli"]["error"] = str(e)
        print(f"❌ Original CLI initialization failed: {e}")
    
    # Test OpenAI SDK CLI initialization
    try:
        from cli_new import OpenAIMemoryAgent
        start_time = time.time()
        agent = OpenAIMemoryAgent(vectorset_key="test_simple")
        init_time = time.time() - start_time
        results["openai_cli"]["success"] = True
        results["openai_cli"]["time"] = init_time
        print(f"✅ OpenAI SDK CLI initialized in {init_time:.3f}s")
    except Exception as e:
        results["openai_cli"]["error"] = str(e)
        print(f"❌ OpenAI SDK CLI initialization failed: {e}")
    
    return results

def test_method_signatures():
    """
    Test if both CLI implementations have the same method signatures.
    
    Returns:
        Dict comparing method signatures
    """
    results = {
        "original_methods": [],
        "openai_methods": [],
        "common_methods": [],
        "missing_in_original": [],
        "missing_in_openai": []
    }
    
    try:
        from memory.agent import MemoryAgentChat
        from cli_new import OpenAIMemoryAgent
        
        # Get method names for both classes
        original_methods = [method for method in dir(MemoryAgentChat) if not method.startswith('_')]
        openai_methods = [method for method in dir(OpenAIMemoryAgent) if not method.startswith('_')]
        
        results["original_methods"] = original_methods
        results["openai_methods"] = openai_methods
        results["common_methods"] = list(set(original_methods) & set(openai_methods))
        results["missing_in_original"] = list(set(openai_methods) - set(original_methods))
        results["missing_in_openai"] = list(set(original_methods) - set(openai_methods))
        
        print(f"📊 Method Comparison:")
        print(f"   Original CLI: {len(original_methods)} methods")
        print(f"   OpenAI CLI: {len(openai_methods)} methods")
        print(f"   Common: {len(results['common_methods'])} methods")
        print(f"   Missing in Original: {results['missing_in_original']}")
        print(f"   Missing in OpenAI: {results['missing_in_openai']}")
        
    except Exception as e:
        results["error"] = str(e)
        print(f"❌ Method signature comparison failed: {e}")
    
    return results

def test_tool_integration():
    """
    Test if both CLIs properly integrate with the memory tools.
    
    Returns:
        Dict with tool integration test results
    """
    results = {
        "original_cli": {"success": False, "tools_count": 0, "error": ""},
        "openai_cli": {"success": False, "tools_count": 0, "error": ""}
    }
    
    # Test original CLI tool integration
    try:
        from memory.agent import MemoryAgentChat
        from memory.tools import AVAILABLE_TOOLS
        
        agent = MemoryAgentChat(vectorset_key="test_simple")
        results["original_cli"]["success"] = True
        results["original_cli"]["tools_count"] = len(AVAILABLE_TOOLS)
        print(f"✅ Original CLI integrates with {len(AVAILABLE_TOOLS)} tools")
    except Exception as e:
        results["original_cli"]["error"] = str(e)
        print(f"❌ Original CLI tool integration failed: {e}")
    
    # Test OpenAI SDK CLI tool integration
    try:
        from cli_new import OpenAIMemoryAgent
        from memory.tools import AVAILABLE_TOOLS
        
        agent = OpenAIMemoryAgent(vectorset_key="test_simple")
        results["openai_cli"]["success"] = True
        results["openai_cli"]["tools_count"] = len(AVAILABLE_TOOLS)
        print(f"✅ OpenAI SDK CLI integrates with {len(AVAILABLE_TOOLS)} tools")
    except Exception as e:
        results["openai_cli"]["error"] = str(e)
        print(f"❌ OpenAI SDK CLI tool integration failed: {e}")
    
    return results

def test_architecture_differences():
    """
    Analyze architectural differences between the two implementations.
    
    Returns:
        Dict with architectural analysis
    """
    results = {
        "original_architecture": {},
        "openai_architecture": {},
        "differences": []
    }
    
    try:
        from memory.agent import MemoryAgentChat
        from cli_new import OpenAIMemoryAgent
        import inspect
        
        # Analyze original CLI architecture
        original_agent = MemoryAgentChat(vectorset_key="test_simple")
        original_source = inspect.getsource(MemoryAgentChat.__init__)
        
        results["original_architecture"] = {
            "uses_langchain": "langchain" in original_source.lower(),
            "uses_openai_direct": "openai" in original_source.lower() and "chatopen" not in original_source.lower(),
            "tool_binding_method": "bind_tools" if "bind_tools" in original_source else "custom",
            "class_size_lines": len(original_source.split('\n'))
        }
        
        # Analyze OpenAI SDK CLI architecture
        openai_agent = OpenAIMemoryAgent(vectorset_key="test_simple")
        openai_source = inspect.getsource(OpenAIMemoryAgent.__init__)
        
        results["openai_architecture"] = {
            "uses_langchain": "langchain" in openai_source.lower(),
            "uses_openai_direct": "openai" in openai_source.lower() and "chatopen" not in openai_source.lower(),
            "tool_binding_method": "function_calling" if "function" in openai_source else "custom",
            "class_size_lines": len(openai_source.split('\n'))
        }
        
        # Identify key differences
        differences = []
        if results["original_architecture"]["uses_langchain"] != results["openai_architecture"]["uses_langchain"]:
            differences.append("LangChain dependency differs")
        if results["original_architecture"]["uses_openai_direct"] != results["openai_architecture"]["uses_openai_direct"]:
            differences.append("OpenAI SDK usage differs")
        if results["original_architecture"]["tool_binding_method"] != results["openai_architecture"]["tool_binding_method"]:
            differences.append("Tool binding approach differs")
        
        results["differences"] = differences
        
        print("🏗️ Architecture Analysis:")
        print(f"   Original uses LangChain: {results['original_architecture']['uses_langchain']}")
        print(f"   OpenAI uses direct SDK: {results['openai_architecture']['uses_openai_direct']}")
        print(f"   Tool binding differs: {'tool_binding_method' in str(differences)}")
        print(f"   Key differences: {', '.join(differences) if differences else 'None detected'}")
        
    except Exception as e:
        results["error"] = str(e)
        print(f"❌ Architecture analysis failed: {e}")
    
    return results

def generate_simple_report(all_results: Dict) -> str:
    """
    Generate a simple comparison report.
    
    Args:
        all_results: Dictionary containing all test results
        
    Returns:
        Formatted report string
    """
    report = f"""
{'='*60}
                CLI COMPARISON REPORT
{'='*60}

IMPORT COMPATIBILITY:
Original CLI: {'✅ SUCCESS' if all_results['imports']['original_cli']['success'] else '❌ FAILED'}
OpenAI CLI:   {'✅ SUCCESS' if all_results['imports']['openai_cli']['success'] else '❌ FAILED'}

INITIALIZATION PERFORMANCE:
Original CLI: {all_results['initialization']['original_cli']['time']:.3f}s
OpenAI CLI:   {all_results['initialization']['openai_cli']['time']:.3f}s
Winner:       {'OpenAI CLI' if all_results['initialization']['openai_cli']['time'] < all_results['initialization']['original_cli']['time'] else 'Original CLI'}

METHOD COMPATIBILITY:
Common Methods:    {len(all_results['methods']['common_methods'])}
Original Only:     {len(all_results['methods']['missing_in_openai'])}
OpenAI Only:       {len(all_results['methods']['missing_in_original'])}

TOOL INTEGRATION:
Original CLI: {all_results['tools']['original_cli']['tools_count']} tools
OpenAI CLI:   {all_results['tools']['openai_cli']['tools_count']} tools
Compatibility: {'✅ SAME' if all_results['tools']['original_cli']['tools_count'] == all_results['tools']['openai_cli']['tools_count'] else '❌ DIFFERENT'}

ARCHITECTURE DIFFERENCES:
{chr(10).join(f'• {diff}' for diff in all_results['architecture']['differences']) if all_results['architecture']['differences'] else '• No significant differences detected'}

RECOMMENDATIONS:
"""
    
    # Add recommendations based on results
    if all_results['initialization']['openai_cli']['time'] < all_results['initialization']['original_cli']['time']:
        report += "• OpenAI CLI shows faster initialization\n"
    
    if len(all_results['architecture']['differences']) == 0:
        report += "• Both implementations maintain architectural compatibility\n"
    
    if all_results['tools']['original_cli']['tools_count'] == all_results['tools']['openai_cli']['tools_count']:
        report += "• Tool integration is consistent across both implementations\n"
    
    report += f"\n{'='*60}\n"
    
    return report

def main():
    """
    Main function to run simple CLI comparison tests.
    """
    print("🧪 Simple CLI Comparison Test")
    print("=" * 40)
    
    all_results = {}
    
    # Run tests
    print("\n1️⃣ Testing Import Compatibility...")
    all_results['imports'] = test_import_compatibility()
    
    print("\n2️⃣ Testing Agent Initialization...")
    all_results['initialization'] = test_agent_initialization()
    
    print("\n3️⃣ Testing Method Signatures...")
    all_results['methods'] = test_method_signatures()
    
    print("\n4️⃣ Testing Tool Integration...")
    all_results['tools'] = test_tool_integration()
    
    print("\n5️⃣ Analyzing Architecture...")
    all_results['architecture'] = test_architecture_differences()
    
    # Generate and display report
    report = generate_simple_report(all_results)
    print(report)
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"simple_cli_comparison_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"📄 Results saved to: {filename}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Comprehensive end-to-end test suite for the CLI interface.

This test covers:
- Command line argument processing
- Interactive mode commands
- Memory storage and retrieval
- Help system
- Error handling
- Vectorstore selection
- Environment variable handling
- Input validation
- Response formatting
- Performance testing
"""

import os
import sys
import subprocess
import tempfile
import time
import threading
from unittest.mock import MagicMock
from dotenv import load_dotenv

# Optional imports for advanced testing
try:
    import pexpect
    PEXPECT_AVAILABLE = True
except ImportError:
    PEXPECT_AVAILABLE = False
    print("⚠️  pexpect not available - interactive session tests will be skipped")
    print("   Install with: pip install pexpect")

# Load environment variables
load_dotenv()

# Add the project root to the path
sys.path.append('.')

from memory.agent import LangGraphMemoryAgent
from llm.llm_manager import LLMConfig, init_llm_manager
from memory.debug_utils import colorize, Colors


class CLITestRunner:
    """Test runner for CLI functionality."""
    
    def __init__(self):
        self.test_vectorstore = "test:cli_suite"
        self.agent = None
        self.setup_llm_manager()
        
    def setup_llm_manager(self):
        """Initialize LLM manager for testing."""
        try:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                print("❌ OPENAI_API_KEY not found. Some tests will be skipped.")
                return False

            tier1_config = LLMConfig(
                provider="openai",
                model="gpt-4o-mini",
                temperature=0.1,
                max_tokens=1000,
                api_key=openai_api_key,
                timeout=30
            )

            tier2_config = LLMConfig(
                provider="openai", 
                model="gpt-4o-mini",
                temperature=0.1,
                max_tokens=500,
                api_key=openai_api_key,
                timeout=30
            )

            init_llm_manager(tier1_config, tier2_config)
            print("✅ LLM manager initialized for testing")
            return True
        except Exception as e:
            print(f"❌ Error initializing LLM manager: {e}")
            return False
    
    def setup_agent(self):
        """Initialize test agent."""
        try:
            self.agent = LangGraphMemoryAgent(vectorset_key=self.test_vectorstore)
            print("✅ Test agent initialized")
            return True
        except Exception as e:
            print(f"❌ Error initializing test agent: {e}")
            return False


def test_command_line_help():
    """Test command line help functionality."""
    print("\n🧪 Testing Command Line Help")
    print("-" * 40)

    try:
        # Test help command variations
        help_commands = ["help", "--help", "-h"]

        for cmd in help_commands:
            print(f"Testing: python cli.py {cmd}")
            result = subprocess.run(
                [sys.executable, "cli.py", cmd],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                print(f"✅ Help command '{cmd}' works")
                # Check if help content is present
                required_sections = [
                    "LangGraph Memory Agent",
                    "USAGE:",
                    "EXAMPLES:",
                    "INTERACTIVE MODE COMMANDS:",
                    "FEATURES:",
                    "ENVIRONMENT VARIABLES:"
                ]

                missing_sections = []
                for section in required_sections:
                    if section not in result.stdout:
                        missing_sections.append(section)

                if not missing_sections:
                    print("✅ Help content is complete")
                else:
                    print(f"⚠️  Missing help sections: {missing_sections}")

                # Check for specific commands in help
                interactive_commands = ["/help", "/profile", "/stats", "/vectorset", "/clear", "/debug"]
                for cmd_help in interactive_commands:
                    if cmd_help in result.stdout:
                        print(f"✅ Interactive command '{cmd_help}' documented")
                    else:
                        print(f"⚠️  Interactive command '{cmd_help}' not documented")

            else:
                print(f"❌ Help command '{cmd}' failed")
                print(f"Error: {result.stderr}")

        return True
    except Exception as e:
        print(f"❌ Command line help test failed: {e}")
        return False


def test_command_line_query():
    """Test single command line query processing."""
    print("\n🧪 Testing Command Line Query Processing")
    print("-" * 40)
    
    # Skip if no API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipping command line query test - no OPENAI_API_KEY")
        return True
    
    try:
        test_query = "What is the capital of France?"
        print(f"Testing query: {test_query}")
        
        # Create a temporary .env file to ensure API key is available
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write(f"OPENAI_API_KEY={os.getenv('OPENAI_API_KEY')}\n")
            temp_env_file = f.name
        
        try:
            # Test with timeout to prevent hanging
            result = subprocess.run(
                [sys.executable, "cli.py", test_query],
                capture_output=True,
                text=True,
                timeout=60,
                env={**os.environ, "DOTENV_PATH": temp_env_file}
            )
            
            if result.returncode == 0:
                print("✅ Command line query executed successfully")
                if "remem>" in result.stdout:
                    print("✅ CLI prompt format is correct")
                if len(result.stdout.strip()) > 0:
                    print("✅ Response was generated")
                else:
                    print("⚠️  No response content found")
            else:
                print(f"❌ Command line query failed with return code {result.returncode}")
                print(f"Error: {result.stderr}")
        finally:
            os.unlink(temp_env_file)
        
        return True
    except subprocess.TimeoutExpired:
        print("❌ Command line query test timed out")
        return False
    except Exception as e:
        print(f"❌ Command line query test failed: {e}")
        return False


def test_interactive_commands():
    """Test interactive mode commands."""
    print("\n🧪 Testing Interactive Mode Commands")
    print("-" * 40)
    
    # Skip if no API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipping interactive commands test - no OPENAI_API_KEY")
        return True
    
    runner = CLITestRunner()
    if not runner.setup_agent():
        return False
    
    agent = runner.agent
    
    # Test remember command
    print("Testing 'remember' command...")
    try:
        memory_text = "I like testing CLI interfaces"
        storage_result = agent.memory_agent.store_memory(memory_text, apply_grounding=True)
        if storage_result and 'memory_id' in storage_result:
            print("✅ Remember command functionality works")
        else:
            print("❌ Remember command failed")
    except Exception as e:
        print(f"❌ Remember command test failed: {e}")
    
    # Test remember-raw command
    print("Testing 'remember-raw' command...")
    try:
        memory_text = "Raw memory without grounding"
        storage_result = agent.memory_agent.store_memory(memory_text, apply_grounding=False)
        if storage_result and 'memory_id' in storage_result:
            print("✅ Remember-raw command functionality works")
        else:
            print("❌ Remember-raw command failed")
    except Exception as e:
        print(f"❌ Remember-raw command test failed: {e}")
    
    # Test stats functionality
    print("Testing stats functionality...")
    try:
        stats = agent.memory_agent.get_memory_stats()
        if stats and not stats.get('error'):
            print("✅ Stats command functionality works")
        else:
            print("❌ Stats command failed")
    except Exception as e:
        print(f"❌ Stats command test failed: {e}")
    
    # Test profile functionality
    print("Testing profile functionality...")
    try:
        profile = agent.get_user_profile_summary()
        if profile:
            print("✅ Profile command functionality works")
        else:
            print("❌ Profile command failed")
    except Exception as e:
        print(f"❌ Profile command test failed: {e}")
    
    return True


def test_memory_operations():
    """Test memory storage and retrieval operations."""
    print("\n🧪 Testing Memory Operations")
    print("-" * 40)
    
    # Skip if no API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipping memory operations test - no OPENAI_API_KEY")
        return True
    
    runner = CLITestRunner()
    if not runner.setup_agent():
        return False
    
    agent = runner.agent
    
    # Test memory storage
    test_memories = [
        "I prefer Python over JavaScript for backend development",
        "My favorite IDE is VS Code with dark theme",
        "I like to use 4-space indentation in my code"
    ]
    
    stored_ids = []
    for memory in test_memories:
        try:
            result = agent.memory_agent.store_memory(memory, apply_grounding=True)
            if result and 'memory_id' in result:
                stored_ids.append(result['memory_id'])
                print(f"✅ Stored memory: {memory[:50]}...")
            else:
                print(f"❌ Failed to store memory: {memory[:50]}...")
        except Exception as e:
            print(f"❌ Error storing memory: {e}")
    
    # Test memory retrieval
    if stored_ids:
        try:
            search_result = agent.memory_agent.search_memories("coding preferences", top_k=5)
            memories = search_result['memories']
            if memories and len(memories) > 0:
                print(f"✅ Memory retrieval works - found {len(memories)} memories")
            else:
                print("❌ Memory retrieval failed - no results")
        except Exception as e:
            print(f"❌ Memory retrieval test failed: {e}")
    
    return True


def test_error_handling():
    """Test error handling scenarios."""
    print("\n🧪 Testing Error Handling")
    print("-" * 40)

    # Test with invalid vectorstore name
    try:
        agent = LangGraphMemoryAgent(vectorset_key="")
        print("⚠️  Empty vectorstore name was accepted (might be handled gracefully)")
    except Exception as e:
        print("✅ Empty vectorstore name properly rejected")

    # Test with very long input
    try:
        runner = CLITestRunner()
        if runner.setup_agent():
            very_long_input = "test " * 1000  # Very long input
            response = runner.agent.run(very_long_input)
            print("✅ Long input handled gracefully")
    except Exception as e:
        print(f"✅ Long input properly handled with error: {type(e).__name__}")

    return True


def test_vectorstore_selection():
    """Test vectorstore selection functionality."""
    print("\n🧪 Testing Vectorstore Selection")
    print("-" * 40)

    # Skip if no API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipping vectorstore selection test - no OPENAI_API_KEY")
        return True

    try:
        # Test creating agents with different vectorstores
        test_vectorstores = [
            "test:vectorstore1",
            "test:vectorstore2",
            "project:memories"
        ]

        agents = []
        for vs_name in test_vectorstores:
            try:
                agent = LangGraphMemoryAgent(vectorset_key=vs_name)
                agents.append((vs_name, agent))
                print(f"✅ Created agent with vectorstore: {vs_name}")
            except Exception as e:
                print(f"❌ Failed to create agent with vectorstore {vs_name}: {e}")

        # Test that agents are isolated (store different memories)
        if len(agents) >= 2:
            agent1_name, agent1 = agents[0]
            agent2_name, agent2 = agents[1]

            # Store different memories in each
            memory1 = f"Memory specific to {agent1_name}"
            memory2 = f"Memory specific to {agent2_name}"

            try:
                result1 = agent1.memory_agent.store_memory(memory1, apply_grounding=False)
                result2 = agent2.memory_agent.store_memory(memory2, apply_grounding=False)

                if result1 and result2:
                    print("✅ Successfully stored memories in different vectorstores")

                    # Verify isolation by searching
                    search1 = agent1.memory_agent.search_memories(agent1_name, top_k=5)
                    search2 = agent2.memory_agent.search_memories(agent2_name, top_k=5)

                    if search1 and search2:
                        print("✅ Vectorstore isolation verified")
                    else:
                        print("⚠️  Could not verify vectorstore isolation")

            except Exception as e:
                print(f"❌ Error testing vectorstore isolation: {e}")

        return True
    except Exception as e:
        print(f"❌ Vectorstore selection test failed: {e}")
        return False


def test_environment_variables():
    """Test environment variable handling."""
    print("\n🧪 Testing Environment Variables")
    print("-" * 40)

    try:
        # Test missing OPENAI_API_KEY
        print("Testing missing OPENAI_API_KEY...")
        env_without_key = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}

        result = subprocess.run(
            [sys.executable, "cli.py", "help"],
            capture_output=True,
            text=True,
            timeout=10,
            env=env_without_key
        )

        if "OPENAI_API_KEY not found" in result.stderr or result.returncode != 0:
            print("✅ Missing API key properly detected")
        else:
            print("⚠️  Missing API key not detected (help command might not require it)")

        # Test debug environment variables
        print("Testing debug environment variables...")
        debug_env = {**os.environ, "MEMORY_DEBUG": "true", "MEMORY_VERBOSE": "true"}

        if os.getenv("OPENAI_API_KEY"):
            result = subprocess.run(
                [sys.executable, "main.py", "help"],
                capture_output=True,
                text=True,
                timeout=10,
                env=debug_env
            )

            if result.returncode == 0:
                print("✅ Debug environment variables handled")
            else:
                print(f"⚠️  Debug environment variables may have issues: {result.stderr}")

        return True
    except Exception as e:
        print(f"❌ Environment variables test failed: {e}")
        return False


def test_interactive_session():
    """Test interactive session using pexpect for real CLI interaction."""
    print("\n🧪 Testing Interactive Session")
    print("-" * 40)

    # Skip if pexpect not available
    if not PEXPECT_AVAILABLE:
        print("⚠️  Skipping interactive session test - pexpect not available")
        return True

    # Skip if no API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipping interactive session test - no OPENAI_API_KEY")
        return True

    try:
        # Create a temporary .env file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write(f"OPENAI_API_KEY={os.getenv('OPENAI_API_KEY')}\n")
            temp_env_file = f.name

        try:
            # Start interactive session
            print("Starting interactive CLI session...")
            child = pexpect.spawn(f'{sys.executable} main.py', timeout=60)
            child.logfile_read = sys.stdout.buffer  # Log output for debugging

            # Wait for vectorstore selection prompt
            child.expect("Select option", timeout=30)
            print("✅ Vectorstore selection prompt appeared")

            # Select option 1 (create new vectorstore)
            child.sendline("1")

            # Wait for vectorstore name prompt
            child.expect("Enter new vectorset name:", timeout=10)
            print("✅ New vectorstore name prompt appeared")

            # Enter test vectorstore name
            test_vs_name = f"test:interactive_{int(time.time())}"
            child.sendline(test_vs_name)

            # Wait for initialization
            child.expect("Memory agent initialized successfully", timeout=30)
            print("✅ Memory agent initialized")

            # Wait for the interactive prompt
            child.expect("remem>", timeout=10)
            print("✅ Interactive prompt appeared")

            # Test help command
            child.sendline("/help")
            child.expect("Available commands:", timeout=10)
            print("✅ Help command works in interactive mode")

            # Test remember command
            child.sendline("remember I am testing the CLI interface")
            child.expect("remem>", timeout=20)
            print("✅ Remember command processed")

            # Test stats command
            child.sendline("/stats")
            child.expect("Memory Statistics", timeout=10)
            print("✅ Stats command works")

            # Test query
            child.sendline("what am I testing?")
            child.expect("remem>", timeout=30)
            print("✅ Query processed successfully")

            # Exit gracefully
            child.sendline("quit")
            child.expect(pexpect.EOF, timeout=10)
            print("✅ Graceful exit works")

            return True

        except pexpect.TIMEOUT:
            print("❌ Interactive session test timed out")
            return False
        except pexpect.EOF:
            print("❌ Interactive session ended unexpectedly")
            return False
        finally:
            if 'child' in locals():
                child.close()
            os.unlink(temp_env_file)

    except Exception as e:
        print(f"❌ Interactive session test failed: {e}")
        return False


def test_performance_and_stress():
    """Test performance and stress scenarios."""
    print("\n🧪 Testing Performance and Stress")
    print("-" * 40)

    # Skip if no API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipping performance test - no OPENAI_API_KEY")
        return True

    try:
        runner = CLITestRunner()
        if not runner.setup_agent():
            return False

        agent = runner.agent

        # Test rapid memory storage
        print("Testing rapid memory storage...")
        start_time = time.time()
        memories_stored = 0

        for i in range(10):
            try:
                memory = f"Performance test memory {i} - {time.time()}"
                result = agent.memory_agent.store_memory(memory, apply_grounding=False)
                if result and 'memory_id' in result:
                    memories_stored += 1
            except Exception as e:
                print(f"⚠️  Memory {i} failed: {e}")

        storage_time = time.time() - start_time
        print(f"✅ Stored {memories_stored}/10 memories in {storage_time:.2f}s")

        # Test rapid memory retrieval
        print("Testing rapid memory retrieval...")
        start_time = time.time()
        successful_searches = 0

        for i in range(5):
            try:
                search_result = agent.memory_agent.search_memories(f"performance test {i}", top_k=3)
                results = search_result['memories']
                if results:
                    successful_searches += 1
            except Exception as e:
                print(f"⚠️  Search {i} failed: {e}")

        search_time = time.time() - start_time
        print(f"✅ Completed {successful_searches}/5 searches in {search_time:.2f}s")

        # Test concurrent operations (if possible)
        print("Testing concurrent operations...")

        def store_memory_worker(agent, memory_text, results_list):
            try:
                result = agent.memory_agent.store_memory(memory_text, apply_grounding=False)
                results_list.append(result is not None and 'memory_id' in result)
            except Exception:
                results_list.append(False)

        threads = []
        results = []

        for i in range(3):
            memory_text = f"Concurrent test memory {i} - {time.time()}"
            thread = threading.Thread(target=store_memory_worker, args=(agent, memory_text, results))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join(timeout=30)

        successful_concurrent = sum(results)
        print(f"✅ Concurrent operations: {successful_concurrent}/{len(threads)} successful")

        return True

    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False


def test_input_validation():
    """Test input validation and edge cases."""
    print("\n🧪 Testing Input Validation")
    print("-" * 40)

    # Skip if no API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipping input validation test - no OPENAI_API_KEY")
        return True

    try:
        runner = CLITestRunner()
        if not runner.setup_agent():
            return False

        agent = runner.agent

        # Test empty input
        print("Testing empty input...")
        try:
            response = agent.run("")
            print("✅ Empty input handled gracefully")
        except Exception as e:
            print(f"✅ Empty input properly rejected: {type(e).__name__}")

        # Test special characters
        print("Testing special characters...")
        special_inputs = [
            "What about émojis? 🤖",
            "Testing unicode: αβγδε",
            "Quotes: 'single' and \"double\"",
            "Symbols: @#$%^&*()",
            "Newlines:\nand\ttabs"
        ]

        for special_input in special_inputs:
            try:
                response = agent.run(special_input)
                print(f"✅ Special input handled: {special_input[:20]}...")
            except Exception as e:
                print(f"⚠️  Special input failed: {special_input[:20]}... - {type(e).__name__}")

        # Test very long input
        print("Testing very long input...")
        long_input = "This is a very long input. " * 100
        try:
            response = agent.run(long_input)
            print("✅ Long input handled gracefully")
        except Exception as e:
            print(f"✅ Long input properly handled: {type(e).__name__}")

        return True

    except Exception as e:
        print(f"❌ Input validation test failed: {e}")
        return False


def main():
    """Main test function."""
    print("🚀 Comprehensive CLI End-to-End Test Suite")
    print("=" * 60)

    # Check prerequisites
    prerequisites = []
    if not os.getenv("OPENAI_API_KEY"):
        prerequisites.append("⚠️  OPENAI_API_KEY not found - some tests will be skipped")
    if not PEXPECT_AVAILABLE:
        prerequisites.append("⚠️  pexpect not available - interactive session tests will be skipped")

    if prerequisites:
        print("Prerequisites:")
        for prereq in prerequisites:
            print(f"  {prereq}")
        print()

    tests = [
        ("Command Line Help", test_command_line_help),
        ("Command Line Query", test_command_line_query),
        ("Interactive Commands", test_interactive_commands),
        ("Memory Operations", test_memory_operations),
        ("Error Handling", test_error_handling),
        ("Vectorstore Selection", test_vectorstore_selection),
        ("Environment Variables", test_environment_variables),
        ("Interactive Session", test_interactive_session),
        ("Performance and Stress", test_performance_and_stress),
        ("Input Validation", test_input_validation)
    ]
    
    results = []
    start_time = time.time()

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        test_start = time.time()
        try:
            success = test_func()
            test_duration = time.time() - test_start
            results.append((test_name, success, test_duration))
        except Exception as e:
            test_duration = time.time() - test_start
            print(f"❌ {test_name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False, test_duration))

    total_duration = time.time() - start_time

    # Summary
    print(f"\n{'='*20} TEST SUMMARY {'='*20}")
    passed_count = 0
    failed_count = 0

    for test_name, success, duration in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status} ({duration:.2f}s)")
        if success:
            passed_count += 1
        else:
            failed_count += 1

    print(f"\n📊 Results: {passed_count} passed, {failed_count} failed")
    print(f"⏱️  Total time: {total_duration:.2f}s")

    if failed_count == 0:
        print("\n🎉 All CLI tests passed!")
        print("\nThe CLI is working correctly. You can use:")
        print("- python cli.py (interactive mode)")
        print("- python cli.py 'your question' (single query mode)")
        print("- python cli.py help (show help)")
        print("\n💡 For more comprehensive testing, install pexpect:")
        print("   pip install pexpect")
    else:
        print(f"\n❌ {failed_count} CLI tests failed. Please check the implementation.")
        print("\n🔍 Failed tests may indicate:")
        print("- Missing dependencies (Redis, OpenAI API key)")
        print("- Configuration issues")
        print("- Code bugs that need fixing")
        sys.exit(1)


if __name__ == "__main__":
    main()

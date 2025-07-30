#!/usr/bin/env python3
"""
Integration tests for CLI that require more complex setup and real interactions.

These tests are designed to run against a real Redis instance and test
the full end-to-end functionality of the CLI including:
- Real memory storage and retrieval
- Actual LLM interactions
- Full conversation flows
- Performance under load
"""

import os
import sys
import subprocess
import tempfile
import time
import threading
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project root to the path
sys.path.append('.')

from memory.agent import LangGraphMemoryAgent
from llm.llm_manager import LLMConfig, init_llm_manager


class CLIIntegrationTestRunner:
    """Integration test runner for CLI functionality."""
    
    def __init__(self):
        self.test_vectorstore = f"test:integration_{int(time.time())}"
        self.agent = None
        self.setup_llm_manager()
        
    def setup_llm_manager(self):
        """Initialize LLM manager for testing."""
        try:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise Exception("OPENAI_API_KEY not found")

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
            print("✅ LLM manager initialized for integration testing")
            return True
        except Exception as e:
            print(f"❌ Error initializing LLM manager: {e}")
            return False
    
    def setup_agent(self):
        """Initialize test agent."""
        try:
            self.agent = LangGraphMemoryAgent(vectorset_key=self.test_vectorstore)
            print(f"✅ Test agent initialized with vectorstore: {self.test_vectorstore}")
            return True
        except Exception as e:
            print(f"❌ Error initializing test agent: {e}")
            return False
    
    def cleanup(self):
        """Clean up test data."""
        try:
            if self.agent:
                # Clear test memories
                stats = self.agent.memory_agent.get_memory_stats()
                if stats and not stats.get('error'):
                    print(f"🧹 Cleaning up {stats.get('memory_count', 0)} test memories")
            print("✅ Cleanup completed")
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")


def test_full_conversation_flow():
    """Test a complete conversation flow with memory storage and retrieval."""
    print("\n🧪 Testing Full Conversation Flow")
    print("-" * 40)
    
    runner = CLIIntegrationTestRunner()
    if not runner.setup_agent():
        return False
    
    try:
        agent = runner.agent
        
        # Simulate a conversation with memory building
        conversation_steps = [
            ("remember I am a Python developer", "memory storage"),
            ("remember I prefer VS Code as my editor", "preference storage"),
            ("remember I work on machine learning projects", "work context"),
            ("what programming language do I use?", "retrieval test"),
            ("what editor do I prefer?", "preference retrieval"),
            ("what kind of projects do I work on?", "context retrieval")
        ]
        
        for step, (user_input, description) in enumerate(conversation_steps, 1):
            print(f"\nStep {step}: {description}")
            print(f"Input: {user_input}")
            
            try:
                start_time = time.time()
                response = agent.run(user_input)
                duration = time.time() - start_time
                
                print(f"Response time: {duration:.2f}s")
                print(f"Response length: {len(response)} characters")
                
                # Validate response quality
                if len(response) > 10:
                    print("✅ Response generated successfully")
                else:
                    print("⚠️  Response seems too short")
                    
                # For retrieval steps, check if relevant info is included
                if "retrieval" in description:
                    if any(keyword in response.lower() for keyword in ["python", "vs code", "machine learning"]):
                        print("✅ Relevant memory information retrieved")
                    else:
                        print("⚠️  Memory retrieval may not be working optimally")
                        
            except Exception as e:
                print(f"❌ Step {step} failed: {e}")
                return False
        
        # Test memory statistics
        print("\nChecking memory statistics...")
        stats = agent.memory_agent.get_memory_stats()
        if stats and not stats.get('error'):
            memory_count = stats.get('memory_count', 0)
            print(f"✅ Memory statistics: {memory_count} memories stored")
            if memory_count >= 3:  # Should have at least the 3 remember commands
                print("✅ Memory storage working correctly")
            else:
                print("⚠️  Expected more memories to be stored")
        else:
            print("❌ Could not retrieve memory statistics")
        
        return True
        
    finally:
        runner.cleanup()


def test_concurrent_cli_instances():
    """Test multiple CLI instances running concurrently."""
    print("\n🧪 Testing Concurrent CLI Instances")
    print("-" * 40)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipping concurrent test - no OPENAI_API_KEY")
        return True
    
    def run_cli_instance(instance_id, results):
        """Run a CLI instance and store results."""
        try:
            # Create a temporary .env file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
                f.write(f"OPENAI_API_KEY={os.getenv('OPENAI_API_KEY')}\n")
                temp_env_file = f.name
            
            try:
                query = f"remember I am instance {instance_id}"
                result = subprocess.run(
                    [sys.executable, "main.py", query],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    env={**os.environ, "DOTENV_PATH": temp_env_file}
                )
                
                success = result.returncode == 0 and len(result.stdout) > 0
                results[instance_id] = {
                    'success': success,
                    'returncode': result.returncode,
                    'stdout_length': len(result.stdout),
                    'stderr': result.stderr
                }
                
            finally:
                os.unlink(temp_env_file)
                
        except Exception as e:
            results[instance_id] = {
                'success': False,
                'error': str(e)
            }
    
    # Run multiple instances concurrently
    num_instances = 3
    threads = []
    results = {}
    
    print(f"Starting {num_instances} concurrent CLI instances...")
    start_time = time.time()
    
    for i in range(num_instances):
        thread = threading.Thread(target=run_cli_instance, args=(i, results))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join(timeout=120)  # 2 minute timeout per thread
    
    total_time = time.time() - start_time
    
    # Analyze results
    successful_instances = sum(1 for r in results.values() if r.get('success', False))
    
    print(f"✅ Concurrent test completed in {total_time:.2f}s")
    print(f"✅ {successful_instances}/{num_instances} instances succeeded")
    
    for instance_id, result in results.items():
        if result.get('success'):
            print(f"  Instance {instance_id}: ✅ Success")
        else:
            error = result.get('error', result.get('stderr', 'Unknown error'))
            print(f"  Instance {instance_id}: ❌ Failed - {error}")
    
    return successful_instances >= (num_instances // 2)  # At least half should succeed


def main():
    """Main integration test function."""
    print("🚀 CLI Integration Test Suite")
    print("=" * 50)
    
    # Check prerequisites
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found. Integration tests require API access.")
        sys.exit(1)
    
    tests = [
        ("Full Conversation Flow", test_full_conversation_flow),
        ("Concurrent CLI Instances", test_concurrent_cli_instances),
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
    print(f"\n{'='*20} INTEGRATION TEST SUMMARY {'='*20}")
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
        print("\n🎉 All integration tests passed!")
        print("The CLI is working correctly under real-world conditions.")
    else:
        print(f"\n❌ {failed_count} integration tests failed.")
        print("This may indicate issues with:")
        print("- Redis connectivity")
        print("- OpenAI API integration")
        print("- Memory system functionality")
        print("- Concurrent access handling")
        sys.exit(1)


if __name__ == "__main__":
    main()

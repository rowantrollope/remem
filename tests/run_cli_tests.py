#!/usr/bin/env python3
"""
CLI Test Runner

This script runs the comprehensive CLI test suite including:
1. Basic functionality tests (test_cli.py)
2. Integration tests (test_cli_integration.py)

Usage:
    python run_cli_tests.py                    # Run all tests
    python run_cli_tests.py --basic            # Run only basic tests
    python run_cli_tests.py --integration      # Run only integration tests
    python run_cli_tests.py --help             # Show help
"""

import os
import sys
import subprocess
import argparse
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def check_prerequisites():
    """Check if all prerequisites are met for testing."""
    issues = []
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        issues.append("❌ OPENAI_API_KEY not found in environment")
    
    # Check for Redis (try to import redis and connect)
    try:
        import redis
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", 6379))
        redis_db = int(os.getenv("REDIS_DB", 0))
        
        client = redis.Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)
        client.ping()
        print("✅ Redis connection successful")
    except ImportError:
        issues.append("❌ Redis package not installed (pip install redis)")
    except Exception as e:
        issues.append(f"❌ Redis connection failed: {e}")
    
    # Check for required Python packages
    required_packages = [
        ("dotenv", "python-dotenv"),
        ("memory.agent", "local memory module"),
    ]
    
    for package, install_name in required_packages:
        try:
            __import__(package)
        except ImportError:
            issues.append(f"❌ {package} not available ({install_name})")
    
    # Check for optional packages
    optional_packages = [
        ("pexpect", "pexpect - for interactive session tests"),
    ]
    
    for package, description in optional_packages:
        try:
            __import__(package)
            print(f"✅ {description} available")
        except ImportError:
            print(f"⚠️  {description} not available (some tests will be skipped)")
    
    return issues


def run_test_suite(test_file, description):
    """Run a specific test suite."""
    print(f"\n{'='*60}")
    print(f"🧪 Running {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, test_file],
            cwd=os.getcwd(),
            timeout=600  # 10 minute timeout
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n✅ {description} completed successfully in {duration:.2f}s")
            return True
        else:
            print(f"\n❌ {description} failed with return code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\n❌ {description} timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"\n❌ {description} failed with exception: {e}")
        return False


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="Run CLI test suites",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_cli_tests.py                    # Run all tests
  python run_cli_tests.py --basic            # Run only basic tests
  python run_cli_tests.py --integration      # Run only integration tests
  
Prerequisites:
  - OPENAI_API_KEY environment variable
  - Redis server running
  - Required Python packages installed
        """
    )
    
    parser.add_argument(
        "--basic",
        action="store_true",
        help="Run only basic functionality tests"
    )
    
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run only integration tests"
    )
    
    parser.add_argument(
        "--skip-prereq-check",
        action="store_true",
        help="Skip prerequisite checks"
    )
    
    args = parser.parse_args()
    
    print("🚀 CLI Test Runner")
    print("=" * 50)
    
    # Check prerequisites unless skipped
    if not args.skip_prereq_check:
        print("\n🔍 Checking Prerequisites...")
        issues = check_prerequisites()
        
        if issues:
            print("\n❌ Prerequisites not met:")
            for issue in issues:
                print(f"  {issue}")
            print("\nPlease resolve these issues before running tests.")
            print("Use --skip-prereq-check to bypass this check.")
            sys.exit(1)
        else:
            print("✅ All prerequisites met")
    
    # Determine which tests to run
    run_basic = args.basic or not (args.basic or args.integration)
    run_integration = args.integration or not (args.basic or args.integration)
    
    results = []
    start_time = time.time()
    
    # Run basic tests
    if run_basic:
        success = run_test_suite("tests/test_cli.py", "Basic CLI Tests")
        results.append(("Basic Tests", success))
    
    # Run integration tests
    if run_integration:
        success = run_test_suite("tests/test_cli_integration.py", "Integration Tests")
        results.append(("Integration Tests", success))
    
    total_duration = time.time() - start_time
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST RUNNER SUMMARY")
    print(f"{'='*60}")
    
    passed_count = 0
    failed_count = 0
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
        if success:
            passed_count += 1
        else:
            failed_count += 1
    
    print(f"\nResults: {passed_count} passed, {failed_count} failed")
    print(f"Total time: {total_duration:.2f}s")
    
    if failed_count == 0:
        print("\n🎉 All test suites passed!")
        print("\nThe CLI is working correctly and ready for use.")
        print("\nNext steps:")
        print("- Try the CLI: python main.py")
        print("- Read the help: python main.py help")
        print("- Start a conversation: python main.py 'Hello!'")
    else:
        print(f"\n❌ {failed_count} test suite(s) failed.")
        print("\nPlease check the test output above for details.")
        print("Common issues:")
        print("- Redis not running or not accessible")
        print("- OpenAI API key invalid or missing")
        print("- Missing Python dependencies")
        print("- Network connectivity issues")
        sys.exit(1)


if __name__ == "__main__":
    main()

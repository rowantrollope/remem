"""
Main entry point for the LangGraph Memory Agent CLI.
"""

import os
import sys
from dotenv import load_dotenv
from memory.agent import LangGraphMemoryAgent
from memory.debug_utils import (
    section_header, format_user_response, success_print, error_print,
    info_print, colorize, Colors
)

# Import LLM manager
from llm.llm_manager import init_llm_manager, LLMConfig


def initialize_llm_manager():
    """Initialize the LLM manager with default configuration for CLI usage."""
    try:
        # Get OpenAI API key from environment
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            error_print("OPENAI_API_KEY not found in environment variables.")
            return False

        # Create default configuration using OpenAI for both tiers
        tier1_config = LLMConfig(
            provider="openai",
            model="gpt-4o-mini",  # Use cost-effective model for CLI
            temperature=0.7,
            max_tokens=2000,
            api_key=openai_api_key,
            timeout=30
        )

        tier2_config = LLMConfig(
            provider="openai",
            model="gpt-4o-mini",  # Use same model for consistency
            temperature=0.1,      # Lower temperature for internal operations
            max_tokens=1000,
            api_key=openai_api_key,
            timeout=30
        )

        # Initialize the global LLM manager
        init_llm_manager(tier1_config, tier2_config)
        success_print("LLM manager initialized successfully")
        return True

    except Exception as e:
        error_print(f"Failed to initialize LLM manager: {e}")
        return False

def get_available_vectorstores():
    """Get list of available vectorstores by checking Redis directly."""
    try:
        from memory.core import MemoryCore

        # Get all Redis keys to find vectorstores
        import redis
        import os

        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", 6379))
        redis_db = int(os.getenv("REDIS_DB", 0))

        redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)

        # Get all keys and filter for vectorstore patterns
        all_keys = redis_client.keys("*")
        vectorstore_names = set()

        # Look for keys that might be vectorstores
        for key in all_keys:
            # Skip cache and other system keys
            if any(skip in key for skip in ['embeddingCache', 'background_processor', ':log']):
                continue

            # Check if this key has vector data by trying to get info about it
            try:
                # Try to get memory count for this potential vectorstore
                core = MemoryCore(vectorset_key=key)
                info = core.get_memory_info()
                if info and not info.get('error') and info.get('memory_count', 0) > 0:
                    vectorstore_names.add((key, info.get('memory_count', 0)))
            except:
                continue

        return sorted(list(vectorstore_names), key=lambda x: x[1], reverse=True)  # Sort by memory count

    except Exception as e:
        error_print(f"Error getting vectorstores: {e}")
        return []

def get_vectorstore_name():
    """Interactive vectorset selection."""
    section_header("🗄️  Vectorset Selection")

    # Get available vectorsets
    existing_vectorstores = get_available_vectorstores()

    if existing_vectorstores:
        print("Available vectorsets with memories:")
        for i, (name, count) in enumerate(existing_vectorstores, 1):
            print(f"  {i}. {name} ({count} memories)")
        print(f"  {len(existing_vectorstores) + 1}. Create new vectorset")
        print(f"  {len(existing_vectorstores) + 2}. Enter custom name")

        while True:
            try:
                choice = input(f"\nSelect option (1-{len(existing_vectorstores) + 2}): ").strip()
                choice_num = int(choice)

                if 1 <= choice_num <= len(existing_vectorstores):
                    return existing_vectorstores[choice_num - 1][0]
                elif choice_num == len(existing_vectorstores) + 1:
                    # Create new vectorstore
                    return get_new_vectorstore_name()
                elif choice_num == len(existing_vectorstores) + 2:
                    # Enter custom name
                    return get_custom_vectorstore_name()
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
    else:
        print("No existing vectorsets found with memories.")
        print("Options:")
        print("  1. Create new vectorset")
        print("  2. Enter custom name")

        while True:
            choice = input("\nSelect option (1-2): ").strip()
            if choice == "1":
                return get_new_vectorstore_name()
            elif choice == "2":
                return get_custom_vectorstore_name()
            else:
                print("Invalid choice. Please enter 1 or 2.")

def get_new_vectorstore_name():
    """Get name for a new vectorset."""
    print("\n📝 Create New Vectorset")
    print("Examples: 'myproject:memories', 'personal:notes', 'work:research'")

    while True:
        name = input("Enter new vectorset name: ").strip()
        if name:
            if ':' not in name:
                print("💡 Consider using format 'project:type' (e.g., 'myproject:memories')")
                confirm = input(f"Use '{name}' anyway? (y/N): ").strip().lower()
                if confirm != 'y':
                    continue
            return name
        else:
            print("Please enter a valid name.")

def get_custom_vectorstore_name():
    """Get custom vectorstore name."""
    print("\n✏️  Enter Custom Vectorstore Name")

    while True:
        name = input("Vectorstore name: ").strip()
        if name:
            return name
        else:
            print("Please enter a valid name.")

def show_cli_help():
    """Show command-line help information."""
    help_text = f"""
{colorize('🧠 LangGraph Memory Agent - Command Line Help', Colors.BRIGHT_CYAN)}
{colorize('=' * 60, Colors.GRAY)}

{colorize('USAGE:', Colors.BRIGHT_YELLOW)}
  python main.py [question]           Ask a question directly
  python main.py                      Start interactive chat mode
  python main.py help                 Show this help message

{colorize('EXAMPLES:', Colors.BRIGHT_YELLOW)}
  python main.py "what coding style do I prefer?"
  python main.py "remember that I like 4-space indentation"
  python main.py "what are my travel preferences?"

{colorize('ENVIRONMENT VARIABLES:', Colors.BRIGHT_YELLOW)}
  {colorize('MEMORY_DEBUG=true', Colors.WHITE)}    Enable detailed debug output
  {colorize('MEMORY_VERBOSE=true', Colors.WHITE)}  Enable verbose logging

{colorize('DEBUG EXAMPLES:', Colors.BRIGHT_YELLOW)}
  MEMORY_DEBUG=true python main.py "your question"
  MEMORY_VERBOSE=true python main.py

{colorize('INTERACTIVE MODE COMMANDS:', Colors.BRIGHT_YELLOW)}
  {colorize('/help', Colors.WHITE)}       Show interactive help
  {colorize('/profile', Colors.WHITE)}    Show your user profile summary
  {colorize('/stats', Colors.WHITE)}      Show memory system statistics
  {colorize('/vectorset', Colors.WHITE)} Switch to a different vectorstore
  {colorize('/clear', Colors.WHITE)}      Clear conversation history
  {colorize('/debug', Colors.WHITE)}      Toggle debug mode
  {colorize('quit', Colors.WHITE)}        Exit the program

{colorize('FEATURES:', Colors.BRIGHT_YELLOW)}
  • Automatic memory extraction from conversations
  • Intelligent question answering with confidence scoring
  • Multiple vectorstore support for different projects
  • Contextual grounding for time/location-dependent memories
  • Duplicate prevention and smart filtering

{colorize('CONFIGURATION:', Colors.BRIGHT_YELLOW)}
  • Requires OPENAI_API_KEY in environment or .env file
  • Redis server must be running (default: localhost:6379)
  • Optional LangCache for response caching

{colorize('=' * 60, Colors.GRAY)}
"""
    print(help_text)

def main():
    """Main function to run the memory agent."""
    # Load environment variables
    load_dotenv()

    # Check for help command first
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
        if user_input.lower() in ['help', '--help', '-h']:
            show_cli_help()
            return

    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        error_print("OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file with your OpenAI API key.")
        print("You can copy .env.example to .env and fill in your API key.")
        sys.exit(1)

    # Initialize LLM manager for memory operations
    if not initialize_llm_manager():
        print("Memory extraction and advanced features will be limited.")
        print("Continuing with basic functionality...")

    # Get vectorstore name from user
    vectorstore_name = get_vectorstore_name()

    section_header("🧠 Initializing LangGraph Memory Agent")
    info_print(f"Vectorset: {colorize(vectorstore_name, Colors.BRIGHT_BLUE)}")
    try:
        # Create the memory agent with specified vectorstore
        agent = LangGraphMemoryAgent(vectorset_key=vectorstore_name)
        success_print("Memory agent initialized successfully")
    except Exception as e:
        error_print(f"Failed to initialize memory agent: {e}")
        print("Make sure Redis is running and all dependencies are installed.")
        sys.exit(1)

    try:
        # Check if we have command line arguments
        if len(sys.argv) > 1:
            # Run with command line input (help already handled above)
            user_input = " ".join(sys.argv[1:])
            user_prompt = colorize(f"remem> {user_input}", Colors.BRIGHT_CYAN)
            print(f"\n{user_prompt}")

            response = agent.run(user_input)
            formatted_response = format_user_response(response)
            print(formatted_response)
        else:
            # Start interactive chat
            section_header(f"🧠 Memory Agent - Vectorset: {vectorstore_name}")
            print("You can store and retrieve memories!")
            print("\nExamples:")
            print("- 'Remember that I like pizza'")
            print("- 'What do I like to eat?'")
            print("- 'Store this: I met John at the coffee shop'")
            print("- 'Set context: I'm at home working on my laptop'")
            print(f"\nType {colorize('/help', Colors.BRIGHT_YELLOW)} for available commands")
            agent.chat()

    except Exception as e:
        print(f"Error running memory agent: {e}")
        print("Make sure you have installed all dependencies with: pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()

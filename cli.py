#!/usr/bin/env python3
"""
CLI entry point for the Memory Agent using OpenAI Agents SDK approach.

This is a clean implementation using direct ChatOpenAI tool calling
without framework abstractions for maximum reliability and performance.
"""

import os
import sys
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from memory.core_agent import MemoryAgent
from memory.tools import AVAILABLE_TOOLS, set_memory_agent
from memory.debug_utils import (
    section_header, format_user_response, success_print, error_print,
    info_print, colorize, Colors
)

# Import LLM manager
from llm.llm_manager import init_llm_manager, LLMConfig


class OpenAIMemoryAgent:
    """
    Memory agent using direct OpenAI Agents SDK approach.
    
    This implementation uses the OpenAI Python SDK directly with function calling
    for maximum control, reliability, and performance.
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.1,
                 vectorset_key: str = None):
        """
        Initialize the OpenAI memory agent with direct SDK integration.

        Args:
            model_name: OpenAI model to use for conversations
            temperature: Temperature setting for response generation
            vectorset_key: Name of the vectorset to use for memory storage
        """
        # Initialize the underlying memory agent
        self.memory_agent = MemoryAgent(vectorset_key=vectorset_key)
        
        # Set the global memory agent for tools
        set_memory_agent(self.memory_agent)
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name
        self.temperature = temperature
        
        # Create tool mapping for execution
        self.tools_by_name = {tool.name: tool for tool in AVAILABLE_TOOLS}
        
        # Prepare tools for OpenAI function calling format
        self.openai_tools = self._prepare_openai_tools()
        
        # Conversation history for context
        self.conversation_history = []
        self.max_history_length = 20  # Keep last 20 messages
    
    def _prepare_openai_tools(self) -> List[Dict[str, Any]]:
        """
        Convert memory tools to OpenAI function calling format.
        
        Returns:
            List of tool definitions in OpenAI format
        """
        openai_tools = []
        
        for tool in AVAILABLE_TOOLS:
            # Extract schema from the tool
            schema = tool.args_schema.schema() if tool.args_schema else {}
            
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
        
        return openai_tools
    
    def _get_system_prompt(self) -> str:
        """
        Get a comprehensive system prompt for the memory agent.
        
        Returns:
            System prompt string with clear instructions and capabilities
        """
        return """You are a helpful memory assistant with access to powerful memory management tools.

Available tools and their purposes:
- search_memories: Find relevant memories using semantic similarity search
- store_memory: Store new memories with optional contextual grounding
- delete_memory: Delete memories by ID or search description
- set_context: Set current context for memory grounding (location, activity, people)
- get_memory_stats: Get memory system statistics and information

Core Guidelines:
- Always search memories first to understand what you know about the user
- Use tools intelligently to help the user manage their memories
- Be natural and conversational in your responses
- When showing memories to users, include memory IDs for reference
- Apply contextual grounding when storing memories to improve future retrieval
- Provide confidence scores and reasoning when answering questions

Advanced Capabilities:
- Multi-step reasoning using multiple tool calls in sequence
- Intelligent duplicate detection and memory cleanup
- Context-aware memory storage with grounding
- Sophisticated search with relevance scoring
- Memory extraction from conversational data

You can handle complex requests by using multiple tools in sequence to provide
comprehensive and accurate responses."""
    
    def run(self, user_input: str, max_iterations: int = 5) -> str:
        """
        Process user input using OpenAI function calling with memory tools.

        Args:
            user_input: The user's message or question
            max_iterations: Maximum number of tool calling iterations

        Returns:
            The agent's final response
        """
        # Create messages list with conversation history
        messages = []
        
        # Add system message
        messages.append({
            "role": "system", 
            "content": self._get_system_prompt()
        })
        
        # Include recent conversation history
        recent_history = self.conversation_history[-self.max_history_length:]
        for msg in recent_history:
            messages.append(msg)
        
        # Add current user message
        user_message = {"role": "user", "content": user_input}
        messages.append(user_message)
        
        # Process with function calling iterations
        for iteration in range(max_iterations):
            try:
                # Call OpenAI with tools
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=self.openai_tools,
                    tool_choice="auto",
                    temperature=self.temperature
                )
                
                assistant_message = response.choices[0].message
                messages.append({
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": assistant_message.tool_calls
                })
                
                # Check if there are tool calls
                if assistant_message.tool_calls:
                    # Execute tools and add tool messages
                    for tool_call in assistant_message.tool_calls:
                        tool_result = self._execute_tool(tool_call)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result
                        })
                    # Continue to next iteration
                else:
                    # No tool calls, we have our final response
                    final_response = assistant_message.content
                    break
                    
            except Exception as e:
                error_response = f"Error during processing: {str(e)}"
                final_response = error_response
                break
        else:
            # Hit max iterations
            final_response = "I apologize, but I reached the maximum number of processing iterations. Please try a simpler request."
        
        # Update conversation history (only user message and final AI response)
        self.conversation_history.append(user_message)
        if final_response:
            self.conversation_history.append({
                "role": "assistant", 
                "content": final_response
            })
        
        # Trim conversation history
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
        
        return final_response or "I apologize, but I couldn't generate a response."
    
    def _execute_tool(self, tool_call) -> str:
        """
        Execute a single tool call and return the result.

        Args:
            tool_call: OpenAI tool call object

        Returns:
            String result of the tool execution
        """
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        
        if tool_name in self.tools_by_name:
            try:
                tool = self.tools_by_name[tool_name]
                result = tool._run(**tool_args)
                return str(result)
            except Exception as e:
                return f"Error executing {tool_name}: {str(e)}"
        else:
            return f"Unknown tool: {tool_name}"
    
    def show_help(self):
        """
        Display comprehensive help information for the memory agent.
        """
        help_text = f"""
{colorize('🧠 OpenAI Memory Agent - Help', Colors.BRIGHT_CYAN)}
{colorize('=' * 60, Colors.GRAY)}

{colorize('BASIC USAGE:', Colors.BRIGHT_YELLOW)}
  • Ask questions about your stored information and preferences
  • Store information by asking me to remember something
  • Delete memories by description or ID
  • Set context for better memory organization

{colorize('EXAMPLE CONVERSATIONS:', Colors.BRIGHT_YELLOW)}
  {colorize('remem>', Colors.CYAN)} "Remember that I prefer 4-space indentation in Python"
  {colorize('remem>', Colors.CYAN)} "What coding preferences do I have?"
  {colorize('remem>', Colors.CYAN)} "Show me all my memories about travel"
  {colorize('remem>', Colors.CYAN)} "Delete memories about old projects"
  {colorize('remem>', Colors.CYAN)} "Set context: I'm working from home on the Redis project"

{colorize('SPECIAL COMMANDS:', Colors.BRIGHT_YELLOW)}
  {colorize('/help', Colors.WHITE)}       - Show this help message
  {colorize('/profile', Colors.WHITE)}    - Show your complete user profile summary
  {colorize('/stats', Colors.WHITE)}      - Show memory system statistics
  {colorize('/vectorset', Colors.WHITE)} - Switch to a different vectorstore
  {colorize('/clear', Colors.WHITE)}      - Clear conversation history
  {colorize('quit', Colors.WHITE)}        - Exit the program (or Ctrl+C)

{colorize('ADVANCED FEATURES:', Colors.BRIGHT_YELLOW)}
  • {colorize('OpenAI Function Calling:', Colors.GREEN)} Direct tool integration with reliable execution
  • {colorize('Contextual Grounding:', Colors.GREEN)} Memories include context for better retrieval
  • {colorize('Smart Search:', Colors.GREEN)} Semantic similarity with relevance scoring
  • {colorize('Multi-step Reasoning:', Colors.GREEN)} Complex queries handled automatically
  • {colorize('Memory Management:', Colors.GREEN)} Intelligent storage and duplicate detection

{colorize('ARCHITECTURE:', Colors.BRIGHT_YELLOW)}
  • Built on OpenAI Agents SDK for maximum reliability
  • Direct function calling without framework abstractions
  • Redis VectorSet for high-performance semantic search
  • Configurable embedding providers (OpenAI/Ollama)

{colorize('=' * 60, Colors.GRAY)}
"""
        print(help_text)
    
    def show_stats(self):
        """
        Show memory system statistics with enhanced formatting.
        """
        try:
            info = self.memory_agent.get_memory_info()

            section_header("Memory Statistics")

            if 'error' in info:
                error_print(f"Error getting stats: {info['error']}")
                return

            print(f"Vectorstore: {colorize(info['vectorset_name'], Colors.BRIGHT_CYAN)}")
            print(f"Total Memories: {colorize(str(info['memory_count']), Colors.BRIGHT_GREEN)}")
            print(f"Vector Dimension: {colorize(str(info['vector_dimension']), Colors.BRIGHT_BLUE)}")
            print(f"Embedding Model: {colorize(info['embedding_model'], Colors.WHITE)}")
            redis_info = f"{info['redis_host']}:{info['redis_port']}"
            print(f"Redis: {colorize(redis_info, Colors.GRAY)}")
            print(f"Last Updated: {colorize(info['timestamp'][:19], Colors.GRAY)}")
            print(f"OpenAI Model: {colorize(self.model_name, Colors.BRIGHT_BLUE)}")
            print(f"Temperature: {colorize(str(self.temperature), Colors.BRIGHT_BLUE)}")

            if 'note' in info:
                print(f"ℹ️  Note: {info['note']}")

        except Exception as e:
            error_print(f"Failed to get memory statistics: {e}")
    
    def get_user_profile_summary(self) -> str:
        """
        Get a comprehensive summary of what the agent knows about the user.
        
        Returns:
            Formatted user profile summary
        """
        # Delegate to the underlying memory agent
        if hasattr(self.memory_agent, 'get_user_profile_summary'):
            return self.memory_agent.get_user_profile_summary()
        else:
            return "Profile summary not available"
    
    def clear_conversation_history(self):
        """
        Clear the conversation history to start fresh.
        """
        self.conversation_history = []
        info_print("Conversation history cleared")
    
    def switch_vectorstore(self):
        """
        Switch to a different vectorstore with user selection.
        """
        try:
            # Import the vectorstore selection function
            from cli import get_vectorstore_name

            print(f"\n{colorize('Current vectorstore:', Colors.BRIGHT_CYAN)} {self.memory_agent.core.VECTORSET_KEY}")
            print("Select a new vectorstore:")

            # Get new vectorstore name
            new_vectorstore = get_vectorstore_name()

            if new_vectorstore == self.memory_agent.core.VECTORSET_KEY:
                info_print("Already using that vectorstore - no change needed")
                return

            # Create new memory agent with the selected vectorstore
            old_vectorstore = self.memory_agent.core.VECTORSET_KEY
            self.memory_agent = MemoryAgent(vectorset_key=new_vectorstore)

            # Update the global memory agent for tools
            set_memory_agent(self.memory_agent)

            # Clear conversation history since we're switching context
            self.clear_conversation_history()

            success_print(f"Switched from '{old_vectorstore}' to '{new_vectorstore}'")
            info_print("Conversation history cleared for new context")

        except Exception as e:
            error_print(f"Failed to switch vectorstore: {e}")
            print("Continuing with current vectorstore.")
    
    def chat(self):
        """
        Start an interactive chat session with the OpenAI memory agent.
        """
        section_header("OpenAI Memory Agent Chat")
        print("Powered by OpenAI Agents SDK with direct function calling!")
        print("\nExamples:")
        print("- 'Remember that I prefer detailed comments above functions'")
        print("- 'What do I know about my coding preferences?'")
        print("- 'Show me all memories about this project and summarize them'")
        print(f"\nType {colorize('/help', Colors.BRIGHT_YELLOW)} for available commands")

        try:
            first_prompt = True
            while True:
                try:
                    # Create prompt with vectorstore name
                    vectorstore_name = self.memory_agent.core.VECTORSET_KEY
                    prompt = f"{colorize(f'({vectorstore_name})', Colors.GRAY)} {colorize('remem>', Colors.CYAN)} "

                    # Add newline before prompt ONLY for first time
                    if first_prompt:
                        print()
                    first_prompt = False

                    user_input = input(prompt).strip()

                    if user_input.lower() in ['quit', 'exit', 'bye']:
                        print(f"\n{colorize('Goodbye! 👋', Colors.BRIGHT_GREEN)}")
                        break

                    if not user_input:
                        continue

                    # Handle special commands
                    if user_input.lower() in ['/help', 'help']:
                        self.show_help()
                        continue
                    elif user_input.lower() in ['/profile', 'profile']:
                        profile = self.get_user_profile_summary()
                        print(f"\n{profile}")
                        continue
                    elif user_input.lower() in ['/stats', 'stats']:
                        self.show_stats()
                        continue
                    elif user_input.lower() in ['/vectorset', 'vectorstore']:
                        self.switch_vectorstore()
                        continue
                    elif user_input.lower() in ['/clear', 'clear']:
                        self.clear_conversation_history()
                        continue

                    # Regular conversation - use OpenAI function calling
                    response = self.run(user_input)
                    formatted_response = format_user_response(response)
                    print(formatted_response)

                except KeyboardInterrupt:
                    print(f"\n\n{colorize('Goodbye! 👋', Colors.BRIGHT_GREEN)}")
                    break
                except Exception as e:
                    error_print(f"Error: {e}")
                    print("Please try again.")
        finally:
            pass


def initialize_llm_manager():
    """
    Initialize the LLM manager with default configuration for CLI usage.
    
    Returns:
        bool: True if initialization was successful, False otherwise
    """
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
    """
    Get list of available vectorstores by checking Redis directly.
    
    Returns:
        List of tuples (vectorstore_name, memory_count) sorted by memory count
    """
    try:
        from memory.core import MemoryCore

        # Get all Redis keys to find vectorstores
        import redis

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
    """
    Interactive vectorset selection with enhanced UI.
    
    Returns:
        str: Selected vectorstore name
    """
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
    """
    Get name for a new vectorset with validation.
    
    Returns:
        str: New vectorstore name
    """
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
    """
    Get custom vectorstore name.
    
    Returns:
        str: Custom vectorstore name
    """
    print("\n✏️  Enter Custom Vectorstore Name")

    while True:
        name = input("Vectorstore name: ").strip()
        if name:
            return name
        else:
            print("Please enter a valid name.")


def show_cli_help():
    """
    Show comprehensive command-line help information.
    """
    help_text = f"""
{colorize('🧠 OpenAI Memory Agent - Command Line Help', Colors.BRIGHT_CYAN)}
{colorize('=' * 70, Colors.GRAY)}

{colorize('USAGE:', Colors.BRIGHT_YELLOW)}
  python cli.py [question]           Ask a question directly
  python cli.py                      Start interactive chat mode
  python cli.py help                 Show this help message

{colorize('EXAMPLES:', Colors.BRIGHT_YELLOW)}
  python cli.py "what coding style do I prefer?"
  python cli.py "remember that I like detailed function comments"
  python cli.py "what are my travel preferences?"

{colorize('ENVIRONMENT VARIABLES:', Colors.BRIGHT_YELLOW)}
  {colorize('MEMORY_DEBUG=true', Colors.WHITE)}    Enable detailed debug output
  {colorize('MEMORY_VERBOSE=true', Colors.WHITE)}  Enable verbose logging

{colorize('DEBUG EXAMPLES:', Colors.BRIGHT_YELLOW)}
  MEMORY_DEBUG=true python cli.py "your question"
  MEMORY_VERBOSE=true python cli.py

{colorize('INTERACTIVE MODE COMMANDS:', Colors.BRIGHT_YELLOW)}
  {colorize('/help', Colors.WHITE)}       Show interactive help
  {colorize('/profile', Colors.WHITE)}    Show your user profile summary
  {colorize('/stats', Colors.WHITE)}      Show memory system statistics
  {colorize('/vectorset', Colors.WHITE)} Switch to a different vectorstore
  {colorize('/clear', Colors.WHITE)}      Clear conversation history
  {colorize('quit', Colors.WHITE)}        Exit the program

{colorize('FEATURES:', Colors.BRIGHT_YELLOW)}
  • OpenAI Agents SDK with direct function calling
  • Intelligent memory operations with contextual grounding
  • Question answering with confidence scoring
  • Multiple vectorstore support for different projects
  • Smart memory deletion by ID or description
  • Natural conversation with reliable tool execution

{colorize('ARCHITECTURE:', Colors.BRIGHT_YELLOW)}
  • Built on OpenAI Agents SDK for maximum reliability
  • Direct function calling without framework abstractions
  • Redis VectorSet for high-performance semantic search
  • Configurable embedding providers (OpenAI/Ollama)

{colorize('CONFIGURATION:', Colors.BRIGHT_YELLOW)}
  • Requires OPENAI_API_KEY in environment or .env file
  • Redis server must be running (default: localhost:6379)
  • Optional LangCache for response caching

{colorize('=' * 70, Colors.GRAY)}
"""
    print(help_text)


def main():
    """
    Main function to run the OpenAI memory agent CLI.
    """
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

    section_header("🧠 Initializing OpenAI Memory Agent")
    info_print(f"Vectorset: {colorize(vectorstore_name, Colors.BRIGHT_BLUE)}")
    info_print(f"Architecture: {colorize('OpenAI Agents SDK', Colors.BRIGHT_GREEN)}")
    
    try:
        # Create the OpenAI memory agent with specified vectorstore
        agent = OpenAIMemoryAgent(vectorset_key=vectorstore_name)
        success_print("OpenAI memory agent initialized successfully")
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
            agent.chat()

    except Exception as e:
        print(f"Error running memory agent: {e}")
        print("Make sure you have installed all dependencies with: pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
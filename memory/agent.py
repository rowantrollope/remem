#!/usr/bin/env python3
"""
Memory Agent

Clean, reliable memory agent using direct ChatOpenAI with tools.
"""

import os
from typing import List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage

from .core_agent import MemoryAgent
from .tools import AVAILABLE_TOOLS, set_memory_agent
from .debug_utils import (
    format_user_response, section_header, colorize, Colors, error_print, success_print, info_print
)

# Load environment variables
load_dotenv()


class MemoryAgentChat:
    """Memory agent using direct ChatOpenAI with tools."""
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.1,
                 vectorset_key: str = None):
        """Initialize the memory agent.

        Args:
            model_name: OpenAI model to use
            temperature: Temperature for the model
            vectorset_key: Name of the vectorset to use (defaults to "memories")
        """
        # Initialize the underlying memory agent
        self.memory_agent = MemoryAgent(vectorset_key=vectorset_key)
        
        # Set the global memory agent for tools
        set_memory_agent(self.memory_agent)
        
        # Initialize ChatOpenAI with tools
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        ).bind_tools(AVAILABLE_TOOLS)
        
        # Create tool mapping for execution
        self.tools_by_name = {tool.name: tool for tool in AVAILABLE_TOOLS}
        
        # Conversation history for context
        self.conversation_history = []
        self.max_history_length = 20  # Keep last 20 messages
    
    def _get_system_prompt(self) -> str:
        """Get a clean, minimal system prompt."""
        return """You are a helpful memory assistant with access to tools for managing memories.

Available tools:
- search_memories: Find relevant memories using similarity search
- store_memory: Store new memories (only when explicitly requested)
- delete_memory: Delete memories by ID or search description
- set_context: Set current context for memory grounding
- get_memory_stats: Get memory system statistics

Guidelines:
- Search memories first to understand what you know about the user
- Use tools as needed to help the user
- Be natural and conversational
- When showing memories to users, include memory IDs for reference

You can handle complex requests by using multiple tools in sequence."""
    
    def run(self, user_input: str, max_iterations: int = 5) -> str:
        """Run the agent with user input.

        Args:
            user_input: The user's message
            max_iterations: Maximum number of tool calling iterations

        Returns:
            The agent's response
        """
        # Create user message
        user_message = HumanMessage(content=user_input)
        
        # Build messages list with conversation history
        messages = []
        
        # Add system message
        system_msg = SystemMessage(content=self._get_system_prompt())
        messages.append(system_msg)
        
        # Include recent conversation history
        recent_history = self.conversation_history[-self.max_history_length:]
        messages.extend(recent_history)
        
        # Add current user message
        messages.append(user_message)
        
        # Iterate with tool calling
        for iteration in range(max_iterations):
            try:
                # Call the model
                response = self.llm.invoke(messages)
                messages.append(response)
                
                # Check if there are tool calls
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    # Execute tools
                    tool_messages = self._execute_tools(response.tool_calls)
                    messages.extend(tool_messages)
                    # Continue to next iteration
                else:
                    # No tool calls, we're done
                    final_response = response.content
                    break
            except Exception as e:
                error_response = f"Error: {str(e)}"
                final_response = error_response
                break
        else:
            # Hit max iterations
            final_response = "I apologize, but I reached the maximum number of tool iterations. Please try a simpler request."
        
        # Update conversation history (only user message and final AI response)
        self.conversation_history.append(user_message)
        if final_response:
            self.conversation_history.append(AIMessage(content=final_response))
        
        # Trim conversation history
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
        
        return final_response or "I apologize, but I couldn't generate a response."
    
    def _execute_tools(self, tool_calls) -> List[ToolMessage]:
        """Execute tool calls and return tool messages."""
        tool_messages = []
        
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            tool_call_id = tool_call.get("id", tool_name)
            
            if tool_name in self.tools_by_name:
                try:
                    tool = self.tools_by_name[tool_name]
                    result = tool._run(**tool_args)
                    
                    tool_message = ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call_id
                    )
                    tool_messages.append(tool_message)
                    
                except Exception as e:
                    error_message = ToolMessage(
                        content=f"Error executing {tool_name}: {str(e)}",
                        tool_call_id=tool_call_id
                    )
                    tool_messages.append(error_message)
            else:
                error_message = ToolMessage(
                    content=f"Unknown tool: {tool_name}",
                    tool_call_id=tool_call_id
                )
                tool_messages.append(error_message)
        
        return tool_messages
    
    def show_help(self):
        """Display help information for the memory agent."""
        help_text = f"""
{colorize('🧠 Memory Agent - Help', Colors.BRIGHT_CYAN)}
{colorize('=' * 50, Colors.GRAY)}

{colorize('BASIC USAGE:', Colors.BRIGHT_YELLOW)}
  • Ask questions about your preferences and stored information
  • Store information by asking me to remember something
  • Delete memories by description or ID

{colorize('EXAMPLE CONVERSATIONS:', Colors.BRIGHT_YELLOW)}
  {colorize('remem>', Colors.CYAN)} "Remember that I prefer 4-space indentation"
  {colorize('remem>', Colors.CYAN)} "What coding style do I prefer?"
  {colorize('remem>', Colors.CYAN)} "Show me all my memories"
  {colorize('remem>', Colors.CYAN)} "Delete memories about code style"

{colorize('SPECIAL COMMANDS:', Colors.BRIGHT_YELLOW)}
  {colorize('/help', Colors.WHITE)}       - Show this help message
  {colorize('/profile', Colors.WHITE)}    - Show your complete user profile summary
  {colorize('/stats', Colors.WHITE)}      - Show memory system statistics
  {colorize('/vectorset', Colors.WHITE)} - Switch to a different vectorstore
  {colorize('/clear', Colors.WHITE)}      - Clear conversation history
  {colorize('quit', Colors.WHITE)}        - Exit the program (or Ctrl+C)

{colorize('MEMORY FEATURES:', Colors.BRIGHT_YELLOW)}
  • {colorize('Intelligent Tool Use:', Colors.GREEN)} Reliable tool calling with proper iteration limits
  • {colorize('Contextual Search:', Colors.GREEN)} Finds relevant memories for your questions
  • {colorize('Smart Deletion:', Colors.GREEN)} Can delete by description or ID
  • {colorize('Natural Conversation:', Colors.GREEN)} Direct, clean implementation

{colorize('=' * 50, Colors.GRAY)}
"""
        print(help_text)
    
    def show_stats(self):
        """Show memory system statistics."""
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

            if 'note' in info:
                print(f"ℹ️  Note: {info['note']}")

        except Exception as e:
            error_print(f"Failed to get memory statistics: {e}")
    
    def get_user_profile_summary(self) -> str:
        """Get a comprehensive summary of what the agent knows about the user."""
        # Delegate to the underlying memory agent
        return self.memory_agent.get_user_profile_summary() if hasattr(self.memory_agent, 'get_user_profile_summary') else "Profile summary not available"
    
    def clear_conversation_history(self):
        """Clear the conversation history to start fresh."""
        self.conversation_history = []
        info_print("Conversation history cleared")
    
    def switch_vectorstore(self):
        """Switch to a different vectorstore."""
        try:
            # Import the vectorstore selection function
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(__file__)))
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
        """Start an interactive chat session."""
        section_header("Memory Agent Chat")
        print("Intelligent tool execution with natural conversation!")
        print("\nExamples:")
        print("- 'Remember that I like pizza'")
        print("- 'What do I like to eat?'")
        print("- 'Show me all my memories and delete any that aren't real preferences'")
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

                    # Regular conversation - let the agent handle everything
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


if __name__ == "__main__":
    # Create and run the memory agent
    agent = MemoryAgentChat()
    agent.chat()
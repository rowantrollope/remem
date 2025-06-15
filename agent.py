"""
LangGraph agent implementation using OpenAI.
"""

import os
from typing import Dict, Any, List
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain.tools import BaseTool

from tools import AVAILABLE_TOOLS

# Load environment variables
load_dotenv()


class AgentState:
    """State object for the agent."""
    
    def __init__(self):
        self.messages: List[Any] = []
        self.next_action: str = ""
        self.tool_calls: List[Dict] = []


class LangGraphAgent:
    """A basic LangGraph agent using OpenAI."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.1):
        """Initialize the agent.
        
        Args:
            model_name: OpenAI model to use
            temperature: Temperature for the model
        """
        # Initialize the OpenAI model
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Bind tools to the model
        self.llm_with_tools = self.llm.bind_tools(AVAILABLE_TOOLS)

        # Create a tool mapping for execution
        self.tools_by_name = {tool.name: tool for tool in AVAILABLE_TOOLS}
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        
        # Define the graph
        workflow = StateGraph(dict)
        
        # Add nodes
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", self._call_tools)
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END
            }
        )
        
        # Add edge from tools back to agent
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()
    
    def _call_model(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Call the language model."""
        messages = state.get("messages", [])
        
        # Add system message if this is the first call
        if not messages:
            system_msg = SystemMessage(content="""You are a helpful AI assistant. You have access to several tools:
            - get_weather: Get weather information for a city
            - calculate: Perform mathematical calculations
            - search_web: Search the web for information
            
            Use these tools when appropriate to help answer user questions.""")
            messages = [system_msg] + messages
        
        response = self.llm_with_tools.invoke(messages)
        
        # Update state
        state["messages"] = messages + [response]
        
        return state
    
    def _call_tools(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tools based on the model's response."""
        messages = state["messages"]
        last_message = messages[-1]

        # Execute tool calls
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_call_id = tool_call["id"]

                # Execute the tool
                if tool_name in self.tools_by_name:
                    try:
                        tool = self.tools_by_name[tool_name]
                        result = tool.invoke(tool_args)

                        # Create tool message
                        tool_message = ToolMessage(
                            content=str(result),
                            tool_call_id=tool_call_id
                        )
                        messages.append(tool_message)
                    except Exception as e:
                        # Handle tool execution errors
                        error_message = ToolMessage(
                            content=f"Error executing {tool_name}: {str(e)}",
                            tool_call_id=tool_call_id
                        )
                        messages.append(error_message)
                else:
                    # Handle unknown tool
                    error_message = ToolMessage(
                        content=f"Unknown tool: {tool_name}",
                        tool_call_id=tool_call_id
                    )
                    messages.append(error_message)

        state["messages"] = messages
        return state
    
    def _should_continue(self, state: Dict[str, Any]) -> str:
        """Determine whether to continue or end the conversation."""
        messages = state["messages"]
        last_message = messages[-1]
        
        # If the last message has tool calls, continue to execute tools
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "continue"
        else:
            return "end"
    
    def run(self, user_input: str) -> str:
        """Run the agent with user input.
        
        Args:
            user_input: The user's message
            
        Returns:
            The agent's response
        """
        # Initialize state with user message
        initial_state = {
            "messages": [HumanMessage(content=user_input)]
        }
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        # Extract the final response
        final_message = result["messages"][-1]
        if hasattr(final_message, 'content'):
            return final_message.content
        else:
            return str(final_message)
    
    def chat(self):
        """Start an interactive chat session."""
        print("LangGraph Agent Chat (type 'quit' to exit)")
        print("-" * 40)
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            try:
                response = self.run(user_input)
                print(f"\nAgent: {response}")
            except Exception as e:
                print(f"\nError: {e}")
                print("Please try again.")


if __name__ == "__main__":
    # Create and run the agent
    agent = LangGraphAgent()
    agent.chat()

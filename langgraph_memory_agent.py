#!/usr/bin/env python3
"""
LangGraph Memory Agent

A sophisticated memory agent that uses LangGraph for workflow orchestration
and intelligent tool selection for memory operations.
"""

import os
import json
from typing import Dict, Any, List
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from memory_agent import MemoryAgent
from tools import AVAILABLE_TOOLS, set_memory_agent

# Load environment variables
load_dotenv()


class LangGraphMemoryAgent:
    """A sophisticated memory agent using LangGraph for workflow orchestration."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.1):
        """Initialize the LangGraph memory agent.
        
        Args:
            model_name: OpenAI model to use
            temperature: Temperature for the model
        """
        # Initialize the underlying memory agent
        self.memory_agent = MemoryAgent()
        
        # Set the global memory agent for tools
        set_memory_agent(self.memory_agent)
        
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
        """Build the LangGraph workflow for memory operations."""
        
        # Define the graph
        workflow = StateGraph(dict)
        
        # Add nodes
        workflow.add_node("analyzer", self._analyze_question)
        workflow.add_node("memory_agent", self._call_memory_model)
        workflow.add_node("tools", self._call_tools)
        workflow.add_node("synthesizer", self._synthesize_response)
        
        # Set entry point
        workflow.set_entry_point("analyzer")
        
        # Add edges
        workflow.add_edge("analyzer", "memory_agent")
        
        # Add conditional edges from memory_agent
        workflow.add_conditional_edges(
            "memory_agent",
            self._should_use_tools,
            {
                "use_tools": "tools",
                "synthesize": "synthesizer"
            }
        )
        
        # Add edge from tools back to memory_agent for potential iteration
        workflow.add_edge("tools", "memory_agent")
        
        # End at synthesizer
        workflow.add_edge("synthesizer", END)
        
        return workflow.compile()
    
    def _analyze_question(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the incoming question to understand intent and complexity."""
        messages = state.get("messages", [])
        if not messages:
            return state
            
        user_message = messages[-1].content if messages else ""
        
        # Store analysis in state for later use
        state["question_analysis"] = {
            "original_question": user_message,
            "analysis_complete": True
        }
        
        return state
    
    def _call_memory_model(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Call the language model with memory-specific system prompt."""
        messages = state.get("messages", [])
        
        # Add system message if this is the first call or if we need to refresh context
        if not any(isinstance(msg, SystemMessage) for msg in messages):
            system_msg = SystemMessage(content="""You are an intelligent memory assistant with access to sophisticated memory tools. Your goal is to help users store, retrieve, and analyze their personal memories effectively.

Available memory tools:
- store_memory: Store new memories with contextual grounding
- search_memories: Find relevant memories using vector similarity search
- answer_with_confidence: Answer questions with sophisticated confidence analysis and structured JSON responses
- format_memory_results: Format memory search results for display
- set_context: Set current context (location, activity, people) for better memory grounding
- get_memory_stats: Get statistics about stored memories
- analyze_question_type: Analyze what type of question the user is asking

For memory-related questions, you have two approaches:

SIMPLE QUESTIONS: For straightforward memory questions, use answer_with_confidence tool which provides:
- Sophisticated confidence analysis (high/medium/low)
- Structured JSON responses with supporting memories
- Proper relevance scoring and citations

COMPLEX QUESTIONS: For multi-step or complex questions, use the individual tools:
1. Use analyze_question_type to understand complexity
2. Use search_memories to find relevant information
3. Use format_memory_results to present findings nicely
4. Synthesize your own response

For storage requests, always use store_memory.

Always be helpful and provide context about the memories you find. When using answer_with_confidence, present the structured response in a user-friendly way.""")
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
    
    def _should_use_tools(self, state: Dict[str, Any]) -> str:
        """Determine whether to use tools or synthesize response."""
        messages = state["messages"]
        last_message = messages[-1]
        
        # If the last message has tool calls, use tools
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "use_tools"
        else:
            return "synthesize"
    
    def _synthesize_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize the final response based on tool results."""
        # The response is already in the last message, just mark as complete
        state["response_complete"] = True
        return state
    
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
    
    def answer_question(self, question: str, top_k: int = 5, filterBy: str = None) -> Dict[str, Any]:
        """Answer a question using the LangGraph workflow with proper confidence analysis.

        Uses LangGraph workflow for tool orchestration,
        but delegates to memory agent for the final analysis and confidence scoring.

        Args:
            question: The question to answer
            top_k: Number of memories to retrieve (passed to tools)
            filterBy: Optional filter expression

        Returns:
            Dictionary with structured response for API compatibility
        """
        print(f"🤔 LangGraph Agent processing: {question}")

        # For questions that need memory analysis, use memory agent's
        # sophisticated confidence analysis instead of the LangGraph workflow
        # This preserves the high-quality confidence scoring and structured responses

        # First, check if this is a memory-related question
        validation_result = self.memory_agent._validate_and_preprocess_question(question)

        if validation_result["type"] == "help":
            # For non-memory questions, we could use the LangGraph workflow
            # But for now, return the help message
            return {
                "type": "help",
                "answer": validation_result["content"],
                "confidence": "n/a",
                "supporting_memories": []
            }

        # For memory questions, use the original memory agent's sophisticated analysis
        # This preserves the confidence scoring and structured JSON responses
        print("🧠 Using original memory agent for sophisticated confidence analysis...")
        return self.memory_agent.answer_question(question, top_k=top_k, filterBy=filterBy)
    
    def chat(self):
        """Start an interactive chat session."""
        print("LangGraph Memory Agent Chat (type 'quit' to exit)")
        print("-" * 50)
        
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
    agent = LangGraphMemoryAgent()
    agent.chat()

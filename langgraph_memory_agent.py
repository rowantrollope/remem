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

        # Conversation buffer for memory extraction
        self.conversation_buffer = []
        self.extraction_threshold = 2  # Extract after 3 meaningful exchanges
        self.extraction_context = "I am a personal assistant. Extract user preferences, constraints, and important personal information."

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
            system_msg = SystemMessage(content="""You are an intelligent memory assistant with access to sophisticated memory tools. Your primary goals are:

1. **PROACTIVE MEMORY EXTRACTION**: Automatically identify and extract valuable information from conversations
2. **HELPFUL RESPONSES**: Provide immediate assistance using existing memories
3. **CONTINUOUS LEARNING**: Build long-term understanding of user preferences and context

Available memory tools:
- store_memory: Store individual memories with contextual grounding
- search_memories: Find relevant memories using vector similarity search
- answer_with_confidence: Answer questions with sophisticated confidence analysis
- extract_and_store_memories: **NEW** - Intelligently extract multiple memories from conversational data
- format_memory_results: Format memory search results for display
- set_context: Set current context (location, activity, people) for better memory grounding
- get_memory_stats: Get statistics about stored memories
- analyze_question_type: Analyze what type of question the user is asking

**MEMORY EXTRACTION STRATEGY**:
When users share personal information, preferences, constraints, or important details, you should:

1. **IMMEDIATE RESPONSE**: First, provide a helpful response to their message
2. **SMART EXTRACTION**: Then use extract_and_store_memories to capture valuable information

**Extract memories when users mention**:
- Preferences: "I prefer...", "I like...", "I hate...", "I always..."
- Constraints: "My budget is...", "I need...", "I can't..."
- Personal details: Family info, dietary restrictions, accessibility needs
- Important facts: Names, dates, locations, relationships

**Example flow**:
User: "I prefer window seats when flying and my wife is vegetarian"
1. Respond: "Got it! I'll remember your seating preference and your wife's dietary needs."
2. Extract: Use extract_and_store_memories with context like "I am a travel agent app"

**For questions**: Use answer_with_confidence for sophisticated analysis with confidence scoring.

**Context prompts for extraction**:
- Travel: "I am a travel agent app. Extract user preferences, constraints, and personal details."
- General: "I am a personal assistant. Extract user preferences and important information."

Always be proactive about learning while being helpful and conversational.""")
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
        """Run the agent with user input and automatic memory extraction.

        Args:
            user_input: The user's message

        Returns:
            The agent's response
        """
        # Processing user input silently

        # Add to conversation buffer
        self.conversation_buffer.append({
            "sender": "user",
            "text": user_input,
            "timestamp": self._get_timestamp()
        })

        # Initialize state with user message
        initial_state = {
            "messages": [HumanMessage(content=user_input)]
        }

        # Run the graph
        result = self.graph.invoke(initial_state)

        # Extract the final response
        final_message = result["messages"][-1]
        response = ""
        if hasattr(final_message, 'content'):
            response = final_message.content
        else:
            response = str(final_message)

        # Add assistant response to buffer
        self.conversation_buffer.append({
            "sender": "assistant",
            "text": response,
            "timestamp": self._get_timestamp()
        })

        # Check if we should extract memories from recent conversation
        self._check_and_extract_memories()

        return response

    def _get_timestamp(self) -> str:
        """Get current timestamp as ISO string."""
        from datetime import datetime
        return datetime.now().isoformat()

    def _check_and_extract_memories(self):
        """Check if conversation buffer should trigger memory extraction."""
        # Only extract if we have enough conversation
        if len(self.conversation_buffer) < self.extraction_threshold * 2:  # *2 for user+assistant pairs
            return

        # Check if recent messages contain extractable information
        recent_messages = self.conversation_buffer[-6:]  # Last 3 exchanges
        if not self._contains_extractable_info(recent_messages):
            return

        # Extract memories from recent conversation
        try:
            conversation_text = self._format_conversation_for_extraction(recent_messages)

            # Extract directly using the memory agent
            result = self.memory_agent.extract_and_store_memories(
                raw_input=conversation_text,
                context_prompt=self.extraction_context,
                store_raw=False,
                apply_grounding=True
            )

            if result["total_extracted"] > 0:
                print(f"🧠 MEMORY: Auto-extracted {result['total_extracted']} memories from conversation")

            # Clear processed messages from buffer (keep last 2 for context)
            self.conversation_buffer = self.conversation_buffer[-2:]

        except Exception as e:
            print(f"⚠️ MEMORY: Auto-extraction failed: {e}")

    def _contains_extractable_info(self, messages) -> bool:
        """Check if messages contain information worth extracting using LLM evaluation."""
        # Combine messages into conversation text
        conversation_text = ' '.join([msg['text'] for msg in messages])

        # Skip very short messages
        if len(conversation_text.strip()) < 10:
            return False

        # Use LLM to evaluate if the text contains extractable information
        try:
            evaluation_prompt = f"""You are an intelligent memory evaluation system. Your task is to determine if the following conversational text contains information that would be valuable to remember for future interactions.

VALUABLE INFORMATION INCLUDES:
- Personal preferences (likes, dislikes, habits)
- Constraints and requirements (budget, time, accessibility needs)
- Personal details (family, dietary restrictions, important dates)
- Factual information about people, places, or things
- Goals and intentions
- Important contextual details

STRICTLY IGNORE:
- Temporary information (current weather, today's schedule, immediate tasks)
- Conversational filler or pleasantries ("Hi there", "How are you?")
- General questions without personal context ("What's the best way to...")
- Information requests that don't reveal user preferences
- Time-sensitive information that won't be relevant later
- Assistant responses or suggestions

CONVERSATIONAL TEXT TO EVALUATE:
"{conversation_text}"

Respond with ONLY "YES" if the text contains valuable information worth remembering, or "NO" if it doesn't. Do not include any explanation."""

            response = self.llm.invoke([{"role": "user", "content": evaluation_prompt}])
            result = response.content.strip().upper()
            return result == "YES"

        except Exception as e:
            print(f"⚠️ LLM evaluation failed, falling back to keyword approach: {e}")
            return self._contains_extractable_info_fallback(messages)

    def _contains_extractable_info_fallback(self, messages) -> bool:
        """Fallback keyword-based approach for checking extractable information."""
        extractable_keywords = [
            'prefer', 'like', 'love', 'hate', 'dislike', 'always', 'never', 'usually',
            'budget', 'family', 'wife', 'husband', 'kids', 'children', 'allergic', 'allergy',
            'need', 'want', 'can\'t', 'cannot', 'must', 'have to', 'dietary', 'vegetarian',
            'vegan', 'gluten', 'accessibility', 'wheelchair', 'mobility', 'window seat',
            'aisle seat', 'michelin', 'restaurant', 'hotel', 'flight', 'travel'
        ]

        conversation_text = ' '.join([msg['text'] for msg in messages]).lower()
        return any(keyword in conversation_text for keyword in extractable_keywords)

    def _format_conversation_for_extraction(self, messages) -> str:
        """Format conversation messages for memory extraction."""
        return '\n'.join([f"{msg['sender'].title()}: {msg['text']}" for msg in messages])

    def set_extraction_context(self, context_prompt: str):
        """Set the context prompt for automatic memory extraction."""
        self.extraction_context = context_prompt

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
        # Processing question silently

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

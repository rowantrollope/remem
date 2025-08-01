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
import openai

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable

from .core_agent import MemoryAgent
from .tools import AVAILABLE_TOOLS, set_memory_agent
from .debug_utils import (
    format_user_response, section_header, colorize, Colors, error_print, success_print, info_print
)

# Load environment variables
load_dotenv()


class DirectOpenAIRunnable(Runnable):
    """Custom Runnable that uses direct OpenAI API calls instead of LangChain ChatOpenAI.

    This implements the lean-core pattern: keep LangChain-Core interfaces but use
    direct SDK calls for better performance and fewer dependencies.
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.1, **kwargs):
        self.model = model
        self.temperature = temperature
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.functions = kwargs.get('functions', [])

    def invoke(self, input_data, config=None):
        """Invoke the OpenAI API directly with LangChain message format."""
        if isinstance(input_data, list):
            messages = input_data
        else:
            messages = input_data.get("messages", [])

        # Convert LangChain messages to OpenAI format
        openai_messages = []
        for msg in messages:
            if hasattr(msg, 'content'):
                role = "user"
                if hasattr(msg, 'type'):
                    if msg.type == "system":
                        role = "system"
                    elif msg.type == "ai":
                        role = "assistant"
                    elif msg.type == "human":
                        role = "user"

                openai_messages.append({
                    "role": role,
                    "content": msg.content
                })

        # Call OpenAI API directly
        try:
            kwargs = {
                "model": self.model,
                "messages": openai_messages,
                "temperature": self.temperature
            }

            # Add function calling if functions are provided
            if self.functions:
                kwargs["functions"] = self.functions
                kwargs["function_call"] = "auto"

            response = self.client.chat.completions.create(**kwargs)

            # Convert back to LangChain format
            content = response.choices[0].message.content or ""
            ai_message = AIMessage(content=content)

            # Handle function calls
            if response.choices[0].message.function_call:
                ai_message.additional_kwargs = {
                    "function_call": {
                        "name": response.choices[0].message.function_call.name,
                        "arguments": response.choices[0].message.function_call.arguments
                    }
                }

            return ai_message

        except Exception as e:
            error_print(f"OpenAI API error: {e}")
            return AIMessage(content=f"Error: {str(e)}")

    def bind_tools(self, tools):
        """Bind tools/functions to this runnable."""
        functions = []
        for tool in tools:
            function_def = {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }

            # Add parameters if the tool has them
            if hasattr(tool, 'args_schema') and tool.args_schema:
                try:
                    schema = tool.args_schema.model_json_schema()
                except AttributeError:
                    schema = tool.args_schema.schema()

                if 'properties' in schema:
                    function_def["parameters"]["properties"] = schema['properties']
                if 'required' in schema:
                    function_def["parameters"]["required"] = schema['required']

            functions.append(function_def)

        return DirectOpenAIRunnable(
            model=self.model,
            temperature=self.temperature,
            functions=functions
        )



class LangGraphMemoryAgent:
    """A sophisticated memory agent using LangGraph for workflow orchestration."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.1,
                 vectorset_key: str = None):
        """Initialize the LangGraph memory agent.

        Args:
            model_name: OpenAI model to use
            temperature: Temperature for the model
            vectorset_key: Name of the vectorset to use (defaults to "memories")
        """
        # Initialize the underlying memory agent
        self.memory_agent = MemoryAgent(vectorset_key=vectorset_key)
        
        # Set the global memory agent for tools
        set_memory_agent(self.memory_agent)
        
        # Initialize the direct OpenAI runnable (lean-core pattern)
        self.llm = DirectOpenAIRunnable(
            model=model_name,
            temperature=temperature
        )

        # Bind tools to the model
        self.llm_with_tools = self.llm.bind_tools(AVAILABLE_TOOLS)

        # Create a tool mapping for execution
        self.tools_by_name = {tool.name: tool for tool in AVAILABLE_TOOLS}

        # Note: Automatic memory extraction removed - memories should only be stored when explicitly requested

        # Conversation history for LangGraph context
        self.conversation_history = []  # Store LangGraph messages for context
        self.max_history_length = 20  # Keep last 20 messages (10 exchanges)

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
        
        # Add conditional edges from memory agent
        workflow.add_conditional_edges(
            "memory_agent",
            self._should_use_tools,
            {
                "use_tools": "tools",
                "synthesize": "synthesizer"
            }
        )
        
        # Add edge from tools back to memory agent for potential iteration
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
        custom_system_prompt = state.get("custom_system_prompt")

        # Add system message if this is the first call or if we need to refresh context
        if not any(isinstance(msg, SystemMessage) for msg in messages):
            if custom_system_prompt:
                # Use custom system prompt with focused memory tools context
                system_content = f"""{custom_system_prompt}

**Memory Tools Available**:
- search_memories: Find relevant existing memories using vector similarity search
- extract_and_store_memories: Extract only NEW significant information not already stored
- answer_with_confidence: Answer questions with confidence analysis

**Memory Strategy**:
1. **SEARCH MEMORIES FIRST**: Always use search_memories to find relevant information about the user before responding
2. **USE CONTEXT**: Include relevant memories in your response to provide personalized assistance
3. **SELECTIVE EXTRACTION**: Only extract significant new information that would be valuable for future assistance
4. **AVOID OVER-EXTRACTION**: Don't store minor details, temporary information, or duplicates

**IMPORTANT - Memory Display Format**:
When displaying memory search results to users, ALWAYS include the memory ID for each memory. Format like this:
1. [Memory text]
   ID: [memory_id]

This allows users to reference specific memories for deletion or other operations.

**What to Extract**:
- Important preferences (dietary restrictions, travel preferences, seating preferences, food likes/dislikes)
- Significant constraints (budget limits, accessibility needs, time constraints)
- Key personal details (family composition, location, occupation)
- Major plans or goals (upcoming trips, important events, life changes)
- Explicit details that might be useful based on the conversation and the type of assistant you are. For example a travel agent might find it useful to remember that the user has been to Italy before.

**What NOT to Extract**:
- Minor conversational details
- Temporary information
- Information already stored
- Assistant responses

**CRITICAL WORKFLOW - FOLLOW THIS FOR EVERY REQUEST**:
For every user request:
1. **ALWAYS START** with search_memories to find relevant information about the user
2. **USE THE CONTEXT** from retrieved memories in your response
3. **PROVIDE PERSONALIZED HELP** based on what you know about them
4. **OPTIONALLY** extract new information if significant

**Example**:
User: "give me some recommendations for things we can do"
1. **MUST DO**: search_memories("user travel plans family preferences")
2. **USE RESULTS**: Found memories like "user has 213 kids", "all kids are vegetarians", "planning trip to Paris"
3. **PERSONALIZED RESPONSE**: "Based on your trip to Paris with your 213 vegetarian kids, here are some recommendations..."

**NEVER** respond without first searching for relevant user context!

Be helpful and conversational while being selective about what information is worth remembering."""
            else:
                # Use focused default memory-focused system prompt
                system_content = """You are a helpful assistant with memory capabilities. Your goal is to provide excellent assistance while selectively remembering important information for future interactions.

**Memory Tools Available**:
- search_memories: Find relevant existing memories
- extract_and_store_memories: Store only significant new information
- answer_with_confidence: Answer questions with confidence analysis

**Memory Approach**:
1. **SEARCH FIRST**: Always use search_memories to find relevant information about the user before responding
2. **USE CONTEXT**: Include relevant memories in your response to provide personalized assistance
3. **SELECTIVE MEMORY**: Only remember significant information that would improve future assistance
4. **AVOID DUPLICATES**: Don't store information that's already captured

**IMPORTANT - Memory Display Format**:
When displaying memory search results to users, ALWAYS include the memory ID for each memory. Format like this:
1. [Memory text]
   ID: [memory_id]

This allows users to reference specific memories for deletion or other operations.

**Remember These Types of Information**:
- Important preferences (dietary restrictions, travel preferences, seating preferences, food likes/dislikes)
- Significant constraints (budget limits, accessibility needs, time constraints)
- Key personal details (family composition, location, occupation)
- Major plans or goals (upcoming trips, important events, life changes)

**Don't Remember**:
- Minor conversational details
- Temporary information
- Information already stored
- Your own responses

**CRITICAL WORKFLOW - FOLLOW THIS FOR EVERY REQUEST**:
For every user request:
1. **ALWAYS START** with search_memories to find relevant information about the user
2. **USE THE CONTEXT** from retrieved memories in your response
3. **PROVIDE PERSONALIZED HELP** based on what you know about them
4. **OPTIONALLY** extract new information if significant

**NEVER** respond without first searching for relevant user context!"""

            system_msg = SystemMessage(content=system_content)
            messages = [system_msg] + messages
        
        response = self.llm_with_tools.invoke(messages)
        
        # Update state
        state["messages"] = messages + [response]
        
        return state
    
    def _call_tools(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tools based on the model's response."""
        messages = state["messages"]
        last_message = messages[-1]

        # Execute function calls (from our DirectOpenAIRunnable)
        if hasattr(last_message, 'additional_kwargs') and 'function_call' in last_message.additional_kwargs:
            function_call = last_message.additional_kwargs['function_call']
            tool_name = function_call["name"]

            try:
                tool_args = json.loads(function_call["arguments"])
            except json.JSONDecodeError:
                tool_args = {}

            # Execute the tool
            if tool_name in self.tools_by_name:
                try:
                    tool = self.tools_by_name[tool_name]
                    result = tool.invoke(tool_args)

                    # Create tool message (using LangChain-Core ToolMessage)
                    tool_message = ToolMessage(
                        content=str(result),
                        tool_call_id=tool_name  # Use tool name as ID
                    )
                    messages.append(tool_message)
                except Exception as e:
                    # Handle tool execution errors
                    error_message = ToolMessage(
                        content=f"Error executing {tool_name}: {str(e)}",
                        tool_call_id=tool_name
                    )
                    messages.append(error_message)
            else:
                # Handle unknown tool
                error_message = ToolMessage(
                    content=f"Unknown tool: {tool_name}",
                    tool_call_id=tool_name
                )
                messages.append(error_message)

        state["messages"] = messages
        return state
    
    def _should_use_tools(self, state: Dict[str, Any]) -> str:
        """Determine whether to use tools or synthesize response."""
        messages = state["messages"]
        last_message = messages[-1]

        # Check for function calls from our DirectOpenAIRunnable
        if hasattr(last_message, 'additional_kwargs') and 'function_call' in last_message.additional_kwargs:
            return "use_tools"
        # Legacy check for tool_calls (in case we switch back)
        elif hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "use_tools"
        else:
            return "synthesize"
    
    def _synthesize_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize the final response based on tool results."""
        # The response is already in the last message, just mark as complete
        state["response_complete"] = True
        return state
    
    def run(self, user_input: str, system_prompt: str = None) -> str:
        """Run the agent with user input and automatic memory extraction.

        Args:
            user_input: The user's message
            system_prompt: Optional custom system prompt to override default

        Returns:
            The agent's response
        """
        # Processing user input silently

        # Note: Conversation buffer removed - no automatic memory extraction

        # Add current user message to conversation history
        current_user_message = HumanMessage(content=user_input)

        # Build messages list with conversation history + current message
        messages_for_graph = []

        # Include recent conversation history (limit to avoid token limits)
        recent_history = self.conversation_history[-self.max_history_length:]
        messages_for_graph.extend(recent_history)

        # Add current user message
        messages_for_graph.append(current_user_message)

        # Initialize state with conversation history and optional custom system prompt
        initial_state = {
            "messages": messages_for_graph,
            "custom_system_prompt": system_prompt
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
        # Note: Assistant response buffer removed - no automatic memory extraction

        # Add both user message and assistant response to conversation history
        self.conversation_history.append(current_user_message)
        self.conversation_history.append(final_message)

        # Trim conversation history to prevent memory bloat
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]

        # Note: Automatic memory extraction removed - memories should only be stored when explicitly requested

        return response

    def _get_timestamp(self) -> str:
        """Get current timestamp as ISO string in UTC."""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()

    # Note: _check_and_extract_memories method removed - no automatic memory extraction



    # Note: Administrative command filtering and conversation formatting methods removed - no automatic extraction

    def _would_create_duplicates(self, conversation_text: str) -> bool:
        """Check if the conversation text would likely create duplicate memories."""
        import hashlib

        # Create a hash of the conversation content
        content_hash = hashlib.md5(conversation_text.encode()).hexdigest()

        # Check if we've processed very similar content recently
        if content_hash in self.recent_extraction_hashes:
            return True

        # Also check for semantic similarity with recent extractions
        # Search for similar content in recent memories
        try:
            # Extract key phrases from the conversation for similarity check
            key_phrases = self._extract_key_phrases(conversation_text)
            if not key_phrases:
                return False

            # Search for each key phrase in recent memories
            for phrase in key_phrases:
                search_result = self.memory_agent.search_memories(phrase, top_k=3, min_similarity=0.85)
                recent_memories = search_result['memories']
                if recent_memories:
                    # Check if any recent memory is very similar and recent (within last hour)
                    from datetime import datetime, timedelta, timezone
                    one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)

                    for memory in recent_memories:
                        if not isinstance(memory, dict):
                            continue

                        memory_score = memory.get('score', 0)
                        memory_time_str = memory.get('created_at', '')

                        if memory_time_str and memory_score > 0.9:
                            try:
                                memory_time = datetime.fromisoformat(str(memory_time_str).replace('Z', '+00:00'))
                                if memory_time > one_hour_ago:
                                    print(f"🔄 MEMORY: Found very similar recent memory: {memory.get('text', '')[:50]}...")
                                    return True
                            except Exception as e:
                                print(f"⚠️ MEMORY: Error parsing timestamp: {e}")
                                continue

        except Exception as e:
            print(f"⚠️ MEMORY: Duplicate check failed: {e}")

        return False

    def _extract_key_phrases(self, text: str) -> list:
        """Extract key phrases from text for duplicate detection."""
        # Simple keyword extraction - could be enhanced with NLP
        import re

        # Remove common words and extract meaningful phrases
        text_lower = text.lower()

        # Look for specific patterns that indicate factual information
        patterns = [
            r'planning (?:a )?trip to (\w+)',
            r'family of (\d+)',
            r'(?:wife|husband|spouse) is (\w+)',
            r'prefer (\w+(?:\s+\w+)*)',
            r'like (\w+(?:\s+\w+)*)',
            r'love (\w+(?:\s+\w+)*)',
            r'hate (\w+(?:\s+\w+)*)',
            r'budget (?:is|of) ([\d,]+)',
            r'allergic to (\w+)',
            r'live in (\w+)',
            r'work (?:as|in) (\w+)',
            r'window (\w+)',
            r'aisle (\w+)',
            r'flying (\w+)',
            r'(\w+) seat',
        ]

        key_phrases = []
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            key_phrases.extend(matches)

        # Also extract noun phrases (simple approach)
        words = re.findall(r'\b\w+\b', text_lower)
        # Don't filter out preference words - they're important for identifying user preferences!
        important_words = [word for word in words if len(word) > 3 and word not in [
            'that', 'this', 'with', 'have', 'will', 'would', 'could', 'should',
            'going', 'want', 'need'  # Removed 'like', 'love', 'hate', 'prefer' from exclusion list
        ]]

        # Take combinations of important words
        for i in range(len(important_words) - 1):
            phrase = f"{important_words[i]} {important_words[i+1]}"
            if len(phrase) > 6:  # Only meaningful phrases
                key_phrases.append(phrase)

        return key_phrases[:5]  # Return top 5 key phrases

    def _store_extraction_hash(self, conversation_text: str):
        """Store hash of extracted conversation to prevent future duplicates."""
        import hashlib

        content_hash = hashlib.md5(conversation_text.encode()).hexdigest()
        self.recent_extraction_hashes.append(content_hash)

        # Keep only recent hashes
        if len(self.recent_extraction_hashes) > self.max_hash_history:
            self.recent_extraction_hashes = self.recent_extraction_hashes[-self.max_hash_history:]

    def set_extraction_context(self, context_prompt: str):
        """Set the context prompt for automatic memory extraction."""
        self.extraction_context = context_prompt

    def clear_conversation_history(self):
        """Clear the conversation history to start fresh."""
        self.conversation_history = []
        self.conversation_buffer = []
        print("🔄 MEMORY: Conversation history cleared")

    def show_conversation_history(self):
        """Show the current conversation history for debugging."""
        print(f"💬 CONVERSATION HISTORY ({len(self.conversation_history)} messages):")
        for i, msg in enumerate(self.conversation_history, 1):
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            print(f"   {i}. {role}: {content}")
        if not self.conversation_history:
            print("   (No conversation history)")
        print()

    def get_user_profile_summary(self) -> str:
        """Get a comprehensive summary of what the agent knows about the user.

        This method searches for various types of user information and provides
        a structured summary of the user's profile based on stored memories.

        Returns:
            Formatted string with user profile summary
        """
        try:
            # Search for different categories of user information
            profile_queries = [
                "family members spouse children kids",
                "preferences likes dislikes favorite",
                "budget constraints limitations requirements",
                "dietary restrictions allergies food preferences",
                "travel experiences places visited hotels restaurants",
                "location home address where lives",
                "work job career occupation",
                "future plans goals upcoming trips events"
            ]

            all_memories = []
            for query in profile_queries:
                memories = self.memory_agent.search_memories(query, top_k=5, min_similarity=0.6)
                all_memories.extend(memories)

            # Remove duplicates based on memory ID
            unique_memories = {}
            for memory in all_memories:
                memory_id = memory.get('id', memory.get('memory_id'))
                if memory_id not in unique_memories:
                    unique_memories[memory_id] = memory

            if not unique_memories:
                return "No user profile information has been stored yet. Start a conversation to build the user profile!"

            # Format the profile summary
            profile_summary = "🧠 USER PROFILE SUMMARY\n"
            profile_summary += "=" * 50 + "\n\n"

            # Group memories by category
            categories = {
                "Family & Relationships": [],
                "Preferences & Tastes": [],
                "Constraints & Requirements": [],
                "Experiences & History": [],
                "Future Plans & Goals": [],
                "Personal Context": []
            }

            for memory in unique_memories.values():
                text = memory.get('text', memory.get('final_text', ''))
                score = memory.get('score', 0) * 100
                time = memory.get('created_at', 'Unknown time')

                # Simple categorization based on keywords
                text_lower = text.lower()
                if any(word in text_lower for word in ['family', 'wife', 'husband', 'child', 'kid', 'son', 'daughter', 'parent']):
                    categories["Family & Relationships"].append(f"• {text} ({score:.1f}% relevance, {time})")
                elif any(word in text_lower for word in ['prefer', 'like', 'love', 'hate', 'favorite', 'enjoy']):
                    categories["Preferences & Tastes"].append(f"• {text} ({score:.1f}% relevance, {time})")
                elif any(word in text_lower for word in ['budget', 'limit', 'constraint', 'need', 'require', 'allergy', 'dietary']):
                    categories["Constraints & Requirements"].append(f"• {text} ({score:.1f}% relevance, {time})")
                elif any(word in text_lower for word in ['visited', 'been to', 'tried', 'experienced', 'stayed at']):
                    categories["Experiences & History"].append(f"• {text} ({score:.1f}% relevance, {time})")
                elif any(word in text_lower for word in ['planning', 'going to', 'will', 'next', 'upcoming', 'goal']):
                    categories["Future Plans & Goals"].append(f"• {text} ({score:.1f}% relevance, {time})")
                else:
                    categories["Personal Context"].append(f"• {text} ({score:.1f}% relevance, {time})")

            # Add non-empty categories to summary
            for category, items in categories.items():
                if items:
                    profile_summary += f"{category}:\n"
                    for item in items[:5]:  # Limit to top 5 per category
                        profile_summary += f"  {item}\n"
                    profile_summary += "\n"

            profile_summary += f"Total memories: {len(unique_memories)}\n"
            profile_summary += "=" * 50

            return profile_summary

        except Exception as e:
            return f"Error generating user profile summary: {str(e)}"

    def find_duplicate_memories(self, similarity_threshold: float = 0.9) -> Dict[str, Any]:
        """Find potential duplicate memories in the system.

        Args:
            similarity_threshold: Similarity threshold for duplicate detection (default: 0.9)

        Returns:
            Dictionary with duplicate groups and statistics
        """
        try:
            # Get all memories
            all_memories = self.memory_agent.search_memories("", top_k=1000, min_similarity=0.0)

            if len(all_memories) < 2:
                return {
                    "duplicate_groups": [],
                    "total_memories": len(all_memories),
                    "potential_duplicates": 0,
                    "message": "Not enough memories to check for duplicates"
                }

            duplicate_groups = []
            processed_ids = set()

            for i, memory in enumerate(all_memories):
                if memory.get('id') in processed_ids:
                    continue

                # Search for similar memories
                similar_memories = self.memory_agent.search_memories(
                    memory['text'],
                    top_k=10,
                    min_similarity=similarity_threshold
                )

                # Filter out the memory itself and find actual duplicates
                duplicates = []
                for similar in similar_memories:
                    if (similar.get('id') != memory.get('id') and
                        similar.get('id') not in processed_ids and
                        similar['score'] >= similarity_threshold):
                        duplicates.append(similar)
                        processed_ids.add(similar.get('id'))

                if duplicates:
                    group = {
                        "original": memory,
                        "duplicates": duplicates,
                        "group_size": len(duplicates) + 1
                    }
                    duplicate_groups.append(group)
                    processed_ids.add(memory.get('id'))

            total_duplicates = sum(group["group_size"] - 1 for group in duplicate_groups)

            return {
                "duplicate_groups": duplicate_groups,
                "total_memories": len(all_memories),
                "potential_duplicates": total_duplicates,
                "duplicate_groups_count": len(duplicate_groups),
                "message": f"Found {len(duplicate_groups)} groups with {total_duplicates} potential duplicates"
            }

        except Exception as e:
            return {
                "error": f"Error finding duplicates: {str(e)}",
                "duplicate_groups": [],
                "total_memories": 0,
                "potential_duplicates": 0
            }

    def answer_question(self, question: str, top_k: int = 5, filterBy: str = None, vectorset_key: str = None) -> Dict[str, Any]:
        """Answer a question using the LangGraph workflow with proper confidence analysis.

        Uses LangGraph workflow for tool orchestration,
        but delegates to memory agent for the final analysis and confidence scoring.

        Args:
            question: The question to answer
            top_k: Number of memories to retrieve (passed to tools)
            filterBy: Optional filter expression
            vectorset_key: Optional vectorset key to use instead of the instance default

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
        return self.memory_agent.answer_question(question, top_k=top_k, filterBy=filterBy, vectorset_key=vectorset_key)
    
    def show_help(self):
        """Display help information for the memory agent."""
        help_text = f"""
{colorize('🧠 LangGraph Remem CLI - Help', Colors.BRIGHT_CYAN)}
{colorize('=' * 50, Colors.GRAY)}

{colorize('BASIC USAGE:', Colors.BRIGHT_YELLOW)}
  • Ask questions about your preferences and stored information
  • Store new information by mentioning it in conversation
  • The agent automatically extracts and remembers important details

{colorize('EXAMPLE CONVERSATIONS:', Colors.BRIGHT_YELLOW)}
  {colorize('remem>', Colors.CYAN)} "Remember that I prefer 4-space indentation"
  {colorize('remem>', Colors.CYAN)} "What coding style do I prefer?"
  {colorize('remem>', Colors.CYAN)} "I like window seats when flying"
  {colorize('remem>', Colors.CYAN)} "What are my travel preferences?"

{colorize('DIRECT MEMORY COMMANDS:', Colors.BRIGHT_YELLOW)}
  {colorize('remember <text>', Colors.WHITE)}     - Store memory directly with grounding
  {colorize('remember-raw <text>', Colors.WHITE)} - Store memory directly without grounding

{colorize('SPECIAL COMMANDS:', Colors.BRIGHT_YELLOW)}
  {colorize('/help', Colors.WHITE)}       - Show this help message
  {colorize('/profile', Colors.WHITE)}    - Show your complete user profile summary
  {colorize('/stats', Colors.WHITE)}      - Show memory system statistics
  {colorize('/vectorset', Colors.WHITE)} - Switch to a different vectorstore
  {colorize('/clear', Colors.WHITE)}      - Clear conversation history
  {colorize('/debug', Colors.WHITE)}      - Toggle debug mode on/off
  {colorize('quit', Colors.WHITE)}        - Exit the program (or Ctrl+C)

{colorize('MEMORY FEATURES:', Colors.BRIGHT_YELLOW)}
  • {colorize('Smart Extraction:', Colors.GREEN)} Automatically identifies important information
  • {colorize('Contextual Search:', Colors.GREEN)} Finds relevant memories for your questions
  • {colorize('Duplicate Prevention:', Colors.GREEN)} Avoids storing redundant information
  • {colorize('Confidence Scoring:', Colors.GREEN)} Provides confidence levels for answers

{colorize('ENVIRONMENT VARIABLES:', Colors.BRIGHT_YELLOW)}
  {colorize('MEMORY_DEBUG=true', Colors.WHITE)}   - Enable detailed debug output
  {colorize('MEMORY_VERBOSE=true', Colors.WHITE)} - Enable verbose logging

{colorize('TIPS:', Colors.BRIGHT_YELLOW)}
  • Be specific when storing preferences (e.g., "I prefer 4-space tabs" vs "I like tabs")
  • Ask follow-up questions to get more detailed information
  • Use natural language - the agent understands conversational input
  • The agent learns from your conversations and improves over time

{colorize('=' * 50, Colors.GRAY)}
"""
        print(help_text)

    def show_stats(self):
        """Show memory system statistics."""
        try:
            info = self.memory_agent.get_memory_info()

            section_header("Remem Statistics")

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

    def toggle_debug(self):
        """Toggle debug mode on/off."""
        import os
        current_debug = os.getenv("MEMORY_DEBUG", "false").lower() == "true"
        new_debug = not current_debug

        # Set environment variable for current session
        os.environ["MEMORY_DEBUG"] = "true" if new_debug else "false"

        if new_debug:
            success_print("Debug mode enabled - you'll see detailed operation logs")
        else:
            success_print("Debug mode disabled - output will be clean and minimal")

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
        section_header("LangGraph Memory Agent Chat")
        print("Type '/help' for available commands or 'quit' to exit")

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

                    # Handle special commands with / prefix (also accept without / for backward compatibility)
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
                        success_print("Conversation history cleared")
                        continue
                    elif user_input.lower() in ['/debug', 'debug']:
                        self.toggle_debug()
                        continue

                    # Handle direct memory storage commands
                    elif user_input.lower().startswith('remember '):
                        memory_text = user_input[9:].strip()
                        if memory_text:
                            try:
                                storage_result = self.memory_agent.store_memory(memory_text, apply_grounding=True)
                                success_print(f"Memory stored with ID: {storage_result['memory_id']}")

                                # Show grounding information if applied
                                if storage_result.get('grounding_applied'):
                                    original = storage_result.get('original_text', '')
                                    final = storage_result.get('final_text', '')
                                    if original != final:
                                        print(f"\n🌍 Contextual grounding applied:")
                                        print(f"  Original: {original}")
                                        print(f"  Grounded: {final}")
                            except Exception as e:
                                error_print(f"Failed to store memory: {e}")
                        else:
                            error_print("Please provide memory text after 'remember'")
                        continue

                    elif user_input.lower().startswith('remember-raw '):
                        memory_text = user_input[12:].strip()
                        if memory_text:
                            try:
                                storage_result = self.memory_agent.store_memory(memory_text, apply_grounding=False)
                                success_print(f"Memory stored (no grounding) with ID: {storage_result['memory_id']}")
                            except Exception as e:
                                error_print(f"Failed to store memory: {e}")
                        else:
                            error_print("Please provide memory text after 'remember-raw'")
                        continue

                    # Regular conversation
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
    # Create and run the agent
    agent = LangGraphMemoryAgent()
    agent.chat()

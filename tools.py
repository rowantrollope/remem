"""
Memory tools for the LangGraph Memory Agent.
"""

from langchain.tools import tool
from typing import Dict, Any, List, Optional
import json


# Memory Tools - Global memory agent instance will be set by the LangGraph agent
_memory_agent = None

def set_memory_agent(agent):
    """Set the global memory agent instance for tools to use."""
    global _memory_agent
    _memory_agent = agent

@tool
def store_memory(memory_text: str, apply_grounding: bool = True) -> str:
    """Store a new memory with optional contextual grounding.

    Args:
        memory_text: The memory text to store
        apply_grounding: Whether to apply contextual grounding to resolve context-dependent references

    Returns:
        Success message with memory ID
    """
    if not _memory_agent:
        return "Error: Memory agent not initialized"

    try:
        memory_id = _memory_agent.store_memory(memory_text, apply_grounding=apply_grounding)
        return f"Successfully stored memory with ID: {memory_id}"
    except Exception as e:
        return f"Error storing memory: {str(e)}"

@tool
def search_memories(query: str, top_k: int = 3, filter_expr: str = None) -> str:
    """Search for relevant memories using vector similarity.

    Args:
        query: Search query text
        top_k: Number of top results to return (default: 3)
        filter_expr: Optional filter expression for Redis VSIM command

    Returns:
        JSON string containing list of relevant memories with metadata
    """
    if not _memory_agent:
        return "Error: Memory agent not initialized"

    try:
        memories = _memory_agent.search_memories(query, top_k, filter_expr)

        # Format memories for LLM consumption
        formatted_memories = []
        for memory in memories:
            formatted_memories.append({
                "id": memory["id"],
                "text": memory["text"],
                "relevance_score": round(memory["score"] * 100, 1),
                "timestamp": memory["formatted_time"],
                "tags": memory["tags"],
                "grounding_applied": memory.get("grounding_applied", False)
            })

        return json.dumps({
            "memories": formatted_memories,
            "count": len(formatted_memories),
            "query": query
        }, indent=2)
    except Exception as e:
        return f"Error searching memories: {str(e)}"

@tool
def set_context(location: str = None, activity: str = None, people: str = None) -> str:
    """Set current context for memory grounding.

    Args:
        location: Current location (e.g., "Jakarta, Indonesia", "Home office")
        activity: Current activity (e.g., "working", "traveling", "meeting")
        people: Comma-separated list of people present (e.g., "John,Sarah")

    Returns:
        Success message with updated context
    """
    if not _memory_agent:
        return "Error: Memory agent not initialized"

    try:
        people_list = [p.strip() for p in people.split(",")] if people else None
        _memory_agent.set_context(
            location=location,
            activity=activity,
            people_present=people_list
        )
        return f"Context updated - Location: {location}, Activity: {activity}, People: {people_list}"
    except Exception as e:
        return f"Error setting context: {str(e)}"

@tool
def get_memory_stats() -> str:
    """Get statistics about stored memories.

    Returns:
        JSON string with memory statistics and system information
    """
    if not _memory_agent:
        return "Error: Memory agent not initialized"

    try:
        stats = _memory_agent.get_memory_info()
        return json.dumps({
            "memory_count": stats.get("memory_count", 0),
            "vector_dimension": stats.get("vector_dimension", 0),
            "embedding_model": stats.get("embedding_model", ""),
            "vectorset_name": stats.get("vectorset_name", ""),
            "timestamp": stats.get("timestamp", "")
        }, indent=2)
    except Exception as e:
        return f"Error getting memory stats: {str(e)}"

@tool
def analyze_question_type(question: str) -> str:
    """Analyze what type of question this is to determine the best approach.

    Args:
        question: The user's question

    Returns:
        JSON string with question analysis
    """
    # Simple heuristic analysis - could be enhanced with LLM
    question_lower = question.lower()

    analysis = {
        "question": question,
        "type": "unknown",
        "complexity": "simple",
        "suggested_approach": "direct_search"
    }

    # Determine question type
    if any(word in question_lower for word in ["what", "tell me about", "describe"]):
        analysis["type"] = "factual"
    elif any(word in question_lower for word in ["when", "what time", "what date"]):
        analysis["type"] = "temporal"
    elif any(word in question_lower for word in ["where", "which place", "location"]):
        analysis["type"] = "spatial"
    elif any(word in question_lower for word in ["who", "which person"]):
        analysis["type"] = "personal"
    elif any(word in question_lower for word in ["how", "why", "explain"]):
        analysis["type"] = "analytical"
        analysis["complexity"] = "complex"
        analysis["suggested_approach"] = "multi_step"

    return json.dumps(analysis, indent=2)

@tool
def answer_with_confidence(question: str, top_k: int = 5, filter_expr: str = None) -> str:
    """Answer a question with sophisticated confidence analysis and structured response.

    This tool uses the original memory agent's sophisticated confidence scoring
    and returns properly formatted JSON with supporting memories.

    Args:
        question: The question to answer
        top_k: Number of memories to retrieve for context
        filter_expr: Optional filter expression for Redis VSIM command

    Returns:
        JSON string with structured answer including confidence and supporting memories
    """
    if not _memory_agent:
        return json.dumps({"error": "Memory agent not initialized"})

    try:
        # Use the memory agent's sophisticated answer_question method
        response = _memory_agent.answer_question(question, top_k=top_k, filterBy=filter_expr)

        # Return as JSON string for the LLM to process
        return json.dumps(response, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Error answering question: {str(e)}"})

@tool
def format_memory_results(memories_json: str) -> str:
    """Format memory search results for display.

    Args:
        memories_json: JSON string containing memory search results

    Returns:
        Formatted string of memories for display
    """
    if not _memory_agent:
        return "Error: Memory agent not initialized"

    try:
        memories_data = json.loads(memories_json)
        memories = memories_data.get("memories", [])

        # Convert to the format expected by format_memory_results
        formatted_memories = []
        for memory in memories:
            formatted_memories.append({
                "text": memory["text"],
                "score": memory["relevance_score"] / 100.0,  # Convert back to 0-1 scale
                "formatted_time": memory["timestamp"],
                "tags": memory["tags"]
            })

        return _memory_agent.format_memory_results(formatted_memories)
    except Exception as e:
        return f"Error formatting memories: {str(e)}"

# List of available memory tools
AVAILABLE_TOOLS = [
    store_memory,
    search_memories,
    set_context,
    get_memory_stats,
    analyze_question_type,
    answer_with_confidence,
    format_memory_results
]

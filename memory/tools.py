"""
Memory tools for the LangGraph Memory Agent.
Standard LangChain tools for use with LangGraph's ToolNode.
"""

from typing import Optional
import json
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool


# Memory Tools - Global memory agent instance will be set by the LangGraph agent
_memory_agent = None

def set_memory_agent(agent):
    """Set the global memory agent instance for tools to use."""
    global _memory_agent
    _memory_agent = agent

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
        storage_result = _memory_agent.store_memory(memory_text, apply_grounding=apply_grounding)
        memory_id = storage_result['memory_id']

        # Create a detailed response message
        message = f"Successfully stored memory with ID: {memory_id}"

        # Add enhanced grounding display if grounding was applied
        if storage_result['grounding_applied']:
            from .debug_utils import format_grounding_display
            grounding_display = format_grounding_display(storage_result)
            if grounding_display:
                message += f"\n\n{grounding_display}"

        return message
    except Exception as e:
        return f"Error storing memory: {str(e)}"


def search_memories(query: str, top_k: int = 10, filter_expr: str = None) -> str:
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
        # Use embedding optimization for better vector similarity search
        validation_result = _memory_agent.processing.validate_and_preprocess_question(query)

        if validation_result["type"] == "search":
            # Use the embedding-optimized query for vector search
            search_query = validation_result.get("embedding_query") or validation_result["content"]
            search_result = _memory_agent.search_memories(search_query, top_k, filter_expr)
        else:
            # For help queries, still search but with original query
            search_result = _memory_agent.search_memories(query, top_k, filter_expr)

        memories = search_result['memories']
        filtering_info = search_result['filtering_info']

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
            "query": query,
            "filtering_info": filtering_info
        }, indent=2)
    except Exception as e:
        return f"Error searching memories: {str(e)}"


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
            # Handle both possible score formats
            score = memory.get("relevance_score", memory.get("score", 0))
            if score > 1:  # If score is in 0-100 range, convert to 0-1
                score = score / 100.0

            formatted_memories.append({
                "id": memory.get("id", "unknown"),  # Preserve memory ID
                "text": memory["text"],
                "score": score,
                "formatted_time": memory.get("timestamp", memory.get("formatted_time", "")),
                "tags": memory.get("tags", [])
            })

        return _memory_agent.format_memory_results(formatted_memories)
    except Exception as e:
        return f"Error formatting memories: {str(e)}"


def extract_and_store_memories(raw_input: str, context_prompt: str, existing_memories_json: str = None) -> str:
    """Extract and store valuable memories from conversational data using intelligent LLM analysis.

    This tool analyzes conversational input to identify and store only NEW information that is not
    already captured in existing memories, avoiding duplicates through context-aware extraction.

    Args:
        raw_input: The conversational data to analyze (e.g., recent chat messages)
        context_prompt: Application context to guide extraction (e.g., "I am a travel agent app")
        existing_memories_json: Optional JSON string of existing memories to avoid duplicates

    Returns:
        JSON string with extraction results including extracted memories and summary
    """
    if not _memory_agent:
        return "Error: Memory agent not initialized"

    try:
        # Parse existing memories if provided
        existing_memories = None
        if existing_memories_json:
            try:
                existing_memories = json.loads(existing_memories_json)
            except json.JSONDecodeError:
                print("⚠️ Warning: Could not parse existing_memories_json, proceeding without context")

        result = _memory_agent.extract_and_store_memories(
            raw_input=raw_input,
            context_prompt=context_prompt,
            apply_grounding=True,
            existing_memories=existing_memories
        )

        # Format for LLM consumption
        return json.dumps({
            "success": True,
            "total_extracted": result["total_extracted"],
            "total_filtered": result["total_filtered"],
            "duplicates_skipped": result.get("duplicates_skipped", 0),
            "extraction_summary": result["extraction_summary"],
            "extracted_memories": [
                {
                    "text": memory["extracted_text"],
                    "confidence": memory["confidence"],
                    "category": memory["category"],
                    "memory_id": memory["memory_id"]
                }
                for memory in result["extracted_memories"]
            ]
        }, indent=2)

    except Exception as e:
        return f"Error extracting memories: {str(e)}"


def find_duplicate_memories(similarity_threshold: float = 0.9) -> str:
    """Find potential duplicate memories in the system.

    This tool helps identify and manage duplicate memories that may have been
    stored multiple times, allowing for cleanup and optimization.

    Args:
        similarity_threshold: Similarity threshold for duplicate detection (0.0-1.0, default: 0.9)

    Returns:
        JSON string with duplicate groups and statistics
    """
    if not _memory_agent:
        return "Error: Memory agent not initialized"

    try:
        # Use the LangGraph agent's find_duplicate_memories method if available
        if hasattr(_memory_agent, 'find_duplicate_memories'):
            result = _memory_agent.find_duplicate_memories(similarity_threshold)
        else:
            # Fallback to basic duplicate detection
            all_memories = _memory_agent.search_memories("", top_k=100, min_similarity=0.0)
            result = {
                "message": "Basic duplicate detection not available",
                "total_memories": len(all_memories),
                "duplicate_groups": [],
                "potential_duplicates": 0
            }

        return json.dumps(result, indent=2)

    except Exception as e:
        return f"Error finding duplicates: {str(e)}"


def delete_memory(memory_id: str) -> str:
    """Delete a specific memory by its ID or by searching for matching memories.

    Args:
        memory_id: The UUID of the memory to delete, or a search query to find memories to delete

    Returns:
        Success message or error message
    """
    if not _memory_agent:
        return "Error: Memory agent not initialized"

    try:
        # First, try to delete by exact ID if it looks like a UUID
        if len(memory_id) == 36 and memory_id.count('-') == 4:
            success = _memory_agent.delete_memory(memory_id)
            if success:
                return f"Successfully deleted memory with ID: {memory_id}"
            else:
                return f"Memory with ID {memory_id} not found"
        
        # If not a UUID, treat as a search query to find memories to delete
        # Search for memories matching the query
        search_result = _memory_agent.search_memories(memory_id, top_k=10, min_similarity=0.5)
        memories = search_result['memories']
        
        if not memories:
            return f"No memories found matching '{memory_id}'"
        
        # If there's only one memory, delete it
        if len(memories) == 1:
            memory_to_delete = memories[0]
            success = _memory_agent.delete_memory(memory_to_delete['id'])
            if success:
                return f"Successfully deleted memory: {memory_to_delete['text'][:100]}..."
            else:
                return f"Failed to delete memory with ID: {memory_to_delete['id']}"
        
        # If multiple memories found, return them for user to choose
        response = f"Found {len(memories)} memories matching '{memory_id}':\n\n"
        for i, memory in enumerate(memories[:5], 1):  # Show top 5
            response += f"{i}. {memory['text'][:100]}...\n"
            response += f"   ID: {memory['id']}\n"
            response += f"   Relevance: {memory['score']*100:.1f}%\n\n"
        
        response += "To delete a specific memory, use its ID like: delete_memory('2fe50a39-ccb9-4a21-af2c-09e9b0d91472')"
        return response
        
    except Exception as e:
        return f"Error deleting memory: {str(e)}"


def clear_all_memories() -> str:
    """Clear all stored memories from the current vectorstore.

    This will permanently delete ALL memories. Use with caution.

    Returns:
        Success message with deletion count or error message
    """
    if not _memory_agent:
        return "Error: Memory agent not initialized"

    try:
        result = _memory_agent.clear_all_memories()

        if result.get('success'):
            memories_deleted = result.get('memories_deleted', 0)
            return f"Successfully cleared all memories. Deleted {memories_deleted} memories."
        else:
            error_msg = result.get('error', 'Unknown error')
            return f"Failed to clear memories: {error_msg}"
    except Exception as e:
        return f"Error clearing memories: {str(e)}"

# Pydantic schemas for tool arguments
class StoreMemoryArgs(BaseModel):
    memory_text: str = Field(description="The memory text to store")
    apply_grounding: bool = Field(default=True, description="Whether to apply contextual grounding")

class DeleteMemoryArgs(BaseModel): 
    memory_id: str = Field(description="Memory ID (UUID) or search query to find memories to delete")
    
class SearchMemoriesArgs(BaseModel):
    query: str = Field(description="Search query text")
    top_k: int = Field(default=10, description="Number of top results to return")
    filter_expr: Optional[str] = Field(default=None, description="Optional filter expression")

class SetContextArgs(BaseModel):
    location: Optional[str] = Field(default=None, description="Current location")
    activity: Optional[str] = Field(default=None, description="Current activity")
    people: Optional[str] = Field(default=None, description="Comma-separated list of people present")

class GetMemoryStatsArgs(BaseModel):
    pass  # No arguments needed

class StoreMemoryTool(BaseTool):
    name: str = "store_memory"
    description: str = "Store a new memory with optional contextual grounding."
    args_schema: type = StoreMemoryArgs

    def _run(self, memory_text: str, apply_grounding: bool = True) -> str:
        return store_memory(memory_text, apply_grounding)

class DeleteMemoryTool(BaseTool):
    name: str = "delete_memory"
    description: str = "Delete a memory by ID or search for memories to delete by description."
    args_schema: type = DeleteMemoryArgs

    def _run(self, memory_id: str) -> str:
        return delete_memory(memory_id)

class SearchMemoriesTool(BaseTool):
    name: str = "search_memories"
    description: str = "Search for relevant memories using vector similarity."
    args_schema: type = SearchMemoriesArgs

    def _run(self, query: str, top_k: int = 10, filter_expr: Optional[str] = None) -> str:
        return search_memories(query, top_k, filter_expr)

class SetContextTool(BaseTool):
    name: str = "set_context"
    description: str = "Set current context for memory grounding."
    args_schema: type = SetContextArgs

    def _run(self, location: Optional[str] = None, activity: Optional[str] = None, 
             people: Optional[str] = None) -> str:
        return set_context(location, activity, people)

class GetMemoryStatsTool(BaseTool):
    name: str = "get_memory_stats"
    description: str = "Get statistics about stored memories."
    args_schema: type = GetMemoryStatsArgs

    def _run(self) -> str:
        return get_memory_stats()

# List of available memory tools (proper LangChain tools)
AVAILABLE_TOOLS = [
    StoreMemoryTool(),
    SearchMemoriesTool(),
    DeleteMemoryTool(),
    SetContextTool(),
    GetMemoryStatsTool(),
]

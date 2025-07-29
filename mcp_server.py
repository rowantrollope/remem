#!/usr/bin/env python3
"""
Model Context Protocol (MCP) Server for Remem Memory Agent

This MCP server exposes the remem memory system capabilities as standardized tools
that can be used by any MCP-compatible client (like Claude Desktop, IDEs, etc.).

The server provides tools for:
- Storing memories (Memories)
- Searching memories
- Getting memory statistics
- Managing context
- Advanced memory operations with k-lines
- LangCache integration for caching tool responses
"""

import os
import sys
import json
import hashlib
from typing import Optional
from functools import wraps
from dotenv import load_dotenv

# Import MCP SDK
from mcp.server.fastmcp import FastMCP

# Import remem memory system
from memory.core_agent import MemoryAgent

# Import LangCache client
from clients.langcache_client import LangCacheClient, is_cache_enabled_for_operation

# Load environment variables
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("remem-memory")

# Global instances
memory_agent: Optional[MemoryAgent] = None
langcache_client: Optional[LangCacheClient] = None

def init_langcache_client() -> bool:
    """Initialize the LangCache client with proper error handling."""
    global langcache_client
    try:
        # Only initialize if environment variables are set
        if all([os.getenv("LANGCACHE_HOST"), os.getenv("LANGCACHE_API_KEY"), os.getenv("LANGCACHE_CACHE_ID")]):
            langcache_client = LangCacheClient()
            print("✅ LangCache client initialized", file=sys.stderr)
            return True
        else:
            print("⚠️ LangCache environment variables not set, caching disabled", file=sys.stderr)
            return False
    except Exception as e:
        print(f"⚠️ Error initializing LangCache client: {e}", file=sys.stderr)
        return False

def init_memory_agent(vectorstore_name: str = "augment:remem") -> bool:
    """Initialize the memory agent with proper error handling."""
    global memory_agent
    try:
        memory_agent = MemoryAgent(vectorset_key=vectorstore_name)
        return True
    except Exception as e:
        print(f"Error initializing memory agent: {e}", file=sys.stderr)
        return False

def create_cache_key(func_name: str, **kwargs) -> str:
    """Create a cache key from function name and arguments."""
    # Create a deterministic key from function name and sorted arguments
    args_str = json.dumps(kwargs, sort_keys=True, default=str)
    key_content = f"{func_name}:{args_str}"
    return hashlib.md5(key_content.encode()).hexdigest()

def cached_tool(operation_type: str = "mcp_tool"):
    """Decorator to add LangCache caching to MCP tool functions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Check if caching is enabled for this operation type
            if not langcache_client or not is_cache_enabled_for_operation(operation_type):
                return await func(*args, **kwargs)

            try:
                # Create cache key
                cache_key = create_cache_key(func.__name__, **kwargs)

                # Create a simple message format for LangCache
                messages = [{"role": "user", "content": f"{func.__name__}: {json.dumps(kwargs, default=str)}"}]

                # Try to get from cache
                cached_response = langcache_client.search_cache(messages)
                if cached_response:
                    print(f"🎯 CACHE HIT for {func.__name__}", file=sys.stderr)
                    return cached_response['content']

                # Call the original function
                result = await func(*args, **kwargs)

                # Store in cache if we got a result
                if result and isinstance(result, str):
                    langcache_client.store_cache(messages, result)
                    print(f"💾 CACHED result for {func.__name__}", file=sys.stderr)

                return result

            except Exception as e:
                print(f"⚠️ Cache error for {func.__name__}: {e}", file=sys.stderr)
                # Fall back to calling the original function
                return await func(*args, **kwargs)

        return wrapper
    return decorator

@mcp.tool()
async def store_memory(
    text: str,
    memory_type: str = "neme",
    vectorstore_name: str = "augment:remem",
    apply_grounding: bool = True
) -> str:
    """Store a new memory (neme) in the memory system.
    
    Args:
        text: The memory content to store
        memory_type: Type of memory (default: "neme")
        vectorstore_name: Name of the vectorstore to use
        apply_grounding: Whether to apply contextual grounding
    """
    if not memory_agent:
        return "Error: Memory agent not initialized"
    
    try:
        # Store the memory using the core agent
        result = memory_agent.store_memory(
            memory_text=text,
            apply_grounding=apply_grounding,
            vectorset_key=vectorstore_name
        )
        
        if result.get('success'):
            memory_id = result.get('memory_id', 'unknown')
            return f"Memory stored successfully with ID: {memory_id}"
        else:
            return f"Failed to store memory: {result.get('message', 'Unknown error')}"
            
    except Exception as e:
        return f"Error storing memory: {str(e)}"

@mcp.tool()
@cached_tool(operation_type="memory_search")
async def search_memories(
    query: str,
    top_k: int = 5,
    vectorstore_name: str = "augment:remem",
    memory_type: str = "neme",
    min_similarity: float = 0.7
) -> str:
    """Search for relevant memories using vector similarity.
    
    Args:
        query: Search query text
        top_k: Number of results to return
        vectorstore_name: Name of the vectorstore to search
        memory_type: Type of memories to search for
        min_similarity: Minimum similarity threshold
    """
    if not memory_agent:
        return "Error: Memory agent not initialized"
    
    try:
        # Search memories using the core agent
        result = memory_agent.search_memories(
            query=query,
            top_k=top_k,
            filterBy=f'type == "{memory_type}"',
            min_similarity=min_similarity,
            vectorset_key=vectorstore_name
        )
        
        if result.get('success'):
            memories = result.get('memories', [])
            if not memories:
                return f"No memories found for query: '{query}'"
            
            # Format the results
            formatted_results = []
            for i, memory in enumerate(memories, 1):
                text = memory.get('text', memory.get('final_text', 'No content'))
                score = memory.get('score', 0)
                timestamp = memory.get('timestamp', 'Unknown')
                formatted_results.append(
                    f"{i}. [{score:.3f}] {text}\n   Stored: {timestamp}"
                )
            
            return f"Found {len(memories)} memories:\n\n" + "\n\n".join(formatted_results)
        else:
            return f"Search failed: {result.get('message', 'Unknown error')}"
            
    except Exception as e:
        return f"Error searching memories: {str(e)}"

@mcp.tool()
async def get_memory_stats(vectorstore_name: str = "augment:remem") -> str:
    """Get statistics about the memory system.
    
    Args:
        vectorstore_name: Name of the vectorstore to get stats for
    """
    if not memory_agent:
        return "Error: Memory agent not initialized"
    
    try:
        # Get memory info using the core agent
        result = memory_agent.get_memory_info()
        
        if result.get('success'):
            info = result.get('info', {})
            stats = [
                f"Memory Count: {info.get('memory_count', 0)}",
                f"Vector Dimension: {info.get('vector_dimension', 'Unknown')}",
                f"Embedding Model: {info.get('embedding_model', 'Unknown')}",
                f"Vectorstore: {vectorstore_name}",
            ]
            
            # Add memory type breakdown if available
            type_breakdown = info.get('memory_types', {})
            if type_breakdown:
                stats.append("\nMemory Types:")
                for mem_type, count in type_breakdown.items():
                    stats.append(f"  {mem_type}: {count}")
            
            return "\n".join(stats)
        else:
            return f"Failed to get memory stats: {result.get('message', 'Unknown error')}"
            
    except Exception as e:
        return f"Error getting memory stats: {str(e)}"

@mcp.tool()
@cached_tool(operation_type="question_answering")
async def answer_question(
    question: str,
    top_k: int = 5,
    vectorstore_name: str = "augment:remem",
    confidence_threshold: float = 0.7
) -> str:
    """Answer a question using the memory system with confidence scoring.
    
    Args:
        question: The question to answer
        top_k: Number of memories to consider
        vectorstore_name: Name of the vectorstore to search
        confidence_threshold: Minimum confidence threshold for answers
    """
    if not memory_agent:
        return "Error: Memory agent not initialized"
    
    try:
        # Use the memory agent's sophisticated question answering
        result = memory_agent.answer_question(
            question=question,
            top_k=top_k
        )
        
        if result.get('success'):
            answer = result.get('answer', 'No answer generated')
            confidence = result.get('confidence', 0)
            memories_used = result.get('memories_used', 0)
            
            if confidence < confidence_threshold:
                return f"Low confidence answer (confidence: {confidence:.2f}):\n{answer}\n\nBased on {memories_used} memories. Consider storing more relevant information."
            else:
                return f"Answer (confidence: {confidence:.2f}):\n{answer}\n\nBased on {memories_used} relevant memories."
        else:
            return f"Failed to answer question: {result.get('message', 'Unknown error')}"
            
    except Exception as e:
        return f"Error answering question: {str(e)}"

@mcp.tool()
async def extract_and_store_memories(
    conversation_text: str,
    vectorstore_name: str = "augment:remem",
    apply_grounding: bool = True
) -> str:
    """Extract and store memories from a conversation or text.
    
    Args:
        conversation_text: The text to extract memories from
        vectorstore_name: Name of the vectorstore to store in
        apply_grounding: Whether to apply contextual grounding
    """
    if not memory_agent:
        return "Error: Memory agent not initialized"
    
    try:
        # Extract and store memories using the core agent
        result = memory_agent.extract_and_store_memories(
            raw_input=conversation_text,
            context_prompt="Extract important information from this conversation",
            apply_grounding=apply_grounding
        )
        
        if result.get('success'):
            extracted_count = result.get('extracted_count', 0)
            stored_count = result.get('stored_count', 0)
            
            if stored_count > 0:
                return f"Successfully extracted and stored {stored_count} memories from {extracted_count} candidates."
            else:
                return "No new memories were extracted from the provided text."
        else:
            return f"Failed to extract memories: {result.get('message', 'Unknown error')}"
            
    except Exception as e:
        return f"Error extracting memories: {str(e)}"

@mcp.tool()
async def set_context(
    location: str = "",
    activity: str = "",
    people_present: str = "",
    environment: str = "",
    vectorstore_name: str = "augment:remem"
) -> str:
    """Set contextual information for memory grounding.

    Args:
        location: Current location
        activity: Current activity
        people_present: People present (comma-separated)
        environment: Environmental context
        vectorstore_name: Name of the vectorstore
    """
    if not memory_agent:
        return "Error: Memory agent not initialized"

    try:
        # Set context using the core agent (returns None)
        memory_agent.set_context(
            location=location,
            activity=activity,
            people_present=people_present.split(',') if people_present else [],
            **{k: v for k, v in {"environment_context": environment}.items() if v}
        )

        # Get the updated context to confirm
        context = memory_agent.core._get_current_context()
        context_info = []
        if context.get('spatial', {}).get('location'):
            context_info.append(f"Location: {context['spatial']['location']}")
        if context.get('spatial', {}).get('activity'):
            context_info.append(f"Activity: {context['spatial']['activity']}")
        if context.get('social', {}).get('people_present'):
            context_info.append(f"People: {', '.join(context['social']['people_present'])}")
        if context.get('environmental'):
            env_items = [f"{k}: {v}" for k, v in context['environmental'].items()]
            if env_items:
                context_info.append(f"Environment: {', '.join(env_items)}")

        return f"Context updated successfully:\n" + "\n".join(context_info) if context_info else "Context cleared."

    except Exception as e:
        return f"Error setting context: {str(e)}"

@mcp.tool()
async def get_context(vectorstore_name: str = "augment:remem") -> str:
    """Get current contextual information.

    Args:
        vectorstore_name: Name of the vectorstore
    """
    if not memory_agent:
        return "Error: Memory agent not initialized"

    try:
        # Get context using the core agent's private method
        context = memory_agent.core._get_current_context()

        context_info = []
        if context.get('spatial', {}).get('location'):
            context_info.append(f"Location: {context['spatial']['location']}")
        if context.get('spatial', {}).get('activity'):
            context_info.append(f"Activity: {context['spatial']['activity']}")
        if context.get('social', {}).get('people_present'):
            context_info.append(f"People: {', '.join(context['social']['people_present'])}")
        if context.get('environmental'):
            env_items = [f"{k}: {v}" for k, v in context['environmental'].items()]
            if env_items:
                context_info.append(f"Environment: {', '.join(env_items)}")
        if context.get('temporal'):
            context_info.append(f"Date: {context['temporal']['date']}")
            context_info.append(f"Time: {context['temporal']['time']}")

        if context_info:
            return "Current context:\n" + "\n".join(context_info)
        else:
            return "No context information currently set."

    except Exception as e:
        return f"Error getting context: {str(e)}"

@mcp.tool()
async def recall_with_klines(
    query: str,
    top_k: int = 5,
    vectorstore_name: str = "augment:remem",
    use_advanced_filtering: bool = True
) -> str:
    """Advanced memory recall using k-lines for sophisticated reasoning.

    Args:
        query: Query for memory recall
        top_k: Number of memories to recall
        vectorstore_name: Name of the vectorstore
        use_advanced_filtering: Whether to use LLM-based filtering
    """
    if not memory_agent:
        return "Error: Memory agent not initialized"

    try:
        # Use recall_memories for advanced memory retrieval
        result = memory_agent.recall_memories(
            query=query,
            top_k=top_k,
            min_similarity=0.7
        )

        # result is already a formatted string from recall_memories
        if result and result.strip():
            return result
        else:
            return f"No relevant memories found for: '{query}'"

    except Exception as e:
        return f"Error in k-line recall: {str(e)}"

@mcp.tool()
async def get_cache_stats(
) -> str:
    """Get LangCache statistics and health information.

    Returns:
        Cache statistics including hit rate, total requests, and health status
    """
    if not langcache_client:
        return "LangCache not initialized. Set LANGCACHE_HOST, LANGCACHE_API_KEY, and LANGCACHE_CACHE_ID environment variables."

    try:
        # Get cache statistics
        stats = langcache_client.get_stats()

        # Get health check
        health = langcache_client.health_check()

        result = {
            "cache_stats": stats,
            "health": health,
            "cache_enabled": True
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return f"Error getting cache stats: {str(e)}"

@mcp.tool()
async def clear_cache_stats(
) -> str:
    """Clear LangCache statistics counters.

    Returns:
        Confirmation message
    """
    if not langcache_client:
        return "LangCache not initialized."

    try:
        langcache_client.clear_stats()
        return "Cache statistics cleared successfully."

    except Exception as e:
        return f"Error clearing cache stats: {str(e)}"

@mcp.tool()
async def delete_memory(
    memory_id: str,
    vectorstore_name: str = "augment:remem"
) -> str:
    """Delete a specific memory by its ID.

    Args:
        memory_id: The UUID of the memory to delete
        vectorstore_name: Name of the vectorstore to delete from

    Returns:
        Confirmation message or error
    """
    if not memory_agent:
        return "Memory agent not initialized."

    try:
        success = memory_agent.delete_memory(memory_id, vectorstore_key=vectorstore_name)

        if success:
            return f"Successfully deleted memory with ID: {memory_id}"
        else:
            return f"Memory with ID {memory_id} not found in vectorstore '{vectorstore_name}'"

    except Exception as e:
        return f"Error deleting memory: {str(e)}"

@mcp.tool()
async def clear_all_memories(
    vectorstore_name: str = "augment:remem",
    confirm: bool = False
) -> str:
    """Clear all memories from a vectorstore.

    Args:
        vectorstore_name: Name of the vectorstore to clear
        confirm: Must be set to True to actually perform the deletion (safety measure)

    Returns:
        Confirmation message or error
    """
    if not memory_agent:
        return "Memory agent not initialized."

    if not confirm:
        return "⚠️ This will delete ALL memories from the vectorstore. To confirm, call this tool with confirm=True"

    try:
        result = memory_agent.clear_all_memories(vectorstore_key=vectorstore_name)

        if result.get('success'):
            memories_deleted = result.get('memories_deleted', 0)
            return f"✅ Successfully cleared all memories from '{vectorstore_name}'. Deleted {memories_deleted} memories."
        else:
            error_msg = result.get('error', 'Unknown error')
            return f"❌ Failed to clear memories: {error_msg}"

    except Exception as e:
        return f"Error clearing memories: {str(e)}"

if __name__ == "__main__":
    # Initialize the memory agent
    if not init_memory_agent():
        print("Failed to initialize memory agent", file=sys.stderr)
        sys.exit(1)

    # Initialize LangCache client (optional)
    init_langcache_client()

    print("Remem MCP Server initialized successfully", file=sys.stderr)
    print("Available tools:", file=sys.stderr)
    print("  - store_memory: Store new memories with contextual grounding", file=sys.stderr)
    print("  - search_memories: Search memories using vector similarity (cached)", file=sys.stderr)
    print("  - get_memory_stats: Get memory system statistics", file=sys.stderr)
    print("  - answer_question: Answer questions using memory with confidence scoring (cached)", file=sys.stderr)
    print("  - extract_and_store_memories: Extract memories from conversations", file=sys.stderr)
    print("  - set_context: Set contextual information for grounding", file=sys.stderr)
    print("  - get_context: Get current contextual information", file=sys.stderr)
    print("  - recall_with_klines: Advanced memory recall with k-line reasoning", file=sys.stderr)
    print("  - get_cache_stats: Get LangCache statistics and health information", file=sys.stderr)
    print("  - clear_cache_stats: Clear LangCache statistics counters", file=sys.stderr)

    # Run the MCP server
    mcp.run(transport='stdio')

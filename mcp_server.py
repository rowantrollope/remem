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

# Import LLM manager
from llm.llm_manager import LLMConfig, init_llm_manager as initialize_llm_manager

# Import LangCache client
from clients.langcache_client import LangCacheClient, is_cache_enabled_for_operation

# Load environment variables
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("remem-memory")

# Global instances
memory_agent: Optional[MemoryAgent] = None
langcache_client: Optional[LangCacheClient] = None

def init_llm_manager() -> bool:
    """Initialize the LLM manager for context analysis and grounding."""
    try:
        # Create default LLM configurations for MCP server
        tier1_config = LLMConfig(
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.1,
            max_tokens=1000,
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=30
        )

        tier2_config = LLMConfig(
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.1,
            max_tokens=500,
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=30
        )

        initialize_llm_manager(tier1_config, tier2_config)
        print("✅ LLM manager initialized for MCP server", file=sys.stderr)
        return True
    except Exception as e:
        print(f"⚠️ Error initializing LLM manager: {e}", file=sys.stderr)
        return False

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

def init_memory_agent(vectorstore_name: str) -> bool:
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
    vectorstore_name: str,
    apply_grounding: bool = True
) -> str:
    """Store a new memory in the memory system.
    
    Args:
        text: The memory content to store
        vectorstore_name: Name of the vectorstore to use
        apply_grounding: Whether to apply contextual grounding
    """
    if not memory_agent:
        return "Error: Memory agent not initialized"

    print(f"🔍 MCP DEBUG: Storing memory with grounding={apply_grounding}", file=sys.stderr)

    try:
        # Store the memory using the core agent
        result = memory_agent.store_memory(
            memory_text=text,
            apply_grounding=apply_grounding,
            vectorset_key=vectorstore_name
        )

        print(f"🔍 MCP DEBUG: Memory storage result: {result}", file=sys.stderr)

        # The core store_memory method returns the memory data directly if successful
        if result and result.get('memory_id'):
            memory_id = result.get('memory_id')
            grounding_applied = result.get('grounding_applied', False)

            response = f"Memory stored successfully with ID: {memory_id}"
            if grounding_applied:
                original = result.get('original_text', '')
                final = result.get('final_text', '')
                if original != final:
                    response += f"\n\n🌍 Contextual grounding applied:\n  Original: {original}\n  Grounded: {final}"

            return response
        else:
            return f"Failed to store memory: Invalid response from memory system"

    except Exception as e:
        return f"Error storing memory: {str(e)}"

@mcp.tool()
@cached_tool(operation_type="memory_search")
async def search_memories(
    query: str,
    vectorstore_name: str,
    top_k: int = 5,
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
        # Search memories using the memory agent - now returns dict with memories and filtering info
        search_result = memory_agent.search_memories(
            query=query,
            top_k=top_k,
            min_similarity=min_similarity,
            vectorset_key=vectorstore_name
        )

        memories = search_result['memories']
        filtering_info = search_result['filtering_info']

        if not memories:
            excluded_count = filtering_info['excluded_count']
            if excluded_count > 0:
                return f"No memories found for query: '{query}' (found {excluded_count} memories below similarity threshold {min_similarity})"
            else:
                return f"No memories found for query: '{query}'"

        # Format the results
        formatted_results = []
        for i, memory in enumerate(memories, 1):
            text = memory.get('text', memory.get('final_text', 'No content'))
            id = memory.get('id', f'unknown-{i}')
            # Use relevance score if available, fallback to similarity score
            relevance_score = memory.get('relevance_score', memory.get('score', 0))
            similarity_score = memory.get('score', 0)
            timestamp = memory.get('timestamp', 'Unknown')
            access_count = memory.get('access_count', 0)

            # Show both relevance and similarity scores for transparency
            score_info = f"[{relevance_score:.3f}]"
            if relevance_score != similarity_score:
                score_info += f" (sim: {similarity_score:.3f}, refs: {access_count})"

            formatted_results.append(
                f"{i}. {score_info} {text}\n   ID: {id}\n   Stored: {timestamp}"
            )

        # Include filtering information in the response
        result = f"Found {len(memories)} memories"
        if filtering_info['excluded_count'] > 0:
            result += f" ({filtering_info['excluded_count']} excluded below similarity threshold {min_similarity})"
        result += f":\n\n" + "\n\n".join(formatted_results)

        return result
            
    except Exception as e:
        return f"Error searching memories: {str(e)}"

@mcp.tool()
async def get_memory_stats(vectorstore_name: str) -> str:
    """Get statistics about the memory system.
    
    Args:
        vectorstore_name: Name of the vectorstore to get stats for
    """
    if not memory_agent:
        return "Error: Memory agent not initialized"
    
    try:
        # Get memory info using the core agent
        info = memory_agent.get_memory_info()

        # Check if there's an error in the result
        if info.get('error'):
            return f"Failed to get memory stats: {info.get('error')}"

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

        # Add note if present
        if info.get('note'):
            stats.append(f"\nNote: {info.get('note')}")

        return "\n".join(stats)
            
    except Exception as e:
        return f"Error getting memory stats: {str(e)}"

@mcp.tool()
@cached_tool(operation_type="question_answering")
async def answer_question(
    question: str,
    vectorstore_name: str,
    top_k: int = 5,
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
    vectorstore_name: str,
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
    vectorstore_name: str,
    location: str = "",
    activity: str = "",
    people_present: str = "",
    environment: str = ""
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
async def get_context(vectorstore_name: str) -> str:
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
    vectorstore_name: str,
    top_k: int = 5,
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
async def get_relevance_config(vectorstore_name: str) -> str:
    """Get current relevance scoring configuration.

    Args:
        vectorstore_name: Name of the vectorstore to get config for
    """
    if not memory_agent:
        return "Error: Memory agent not initialized"

    try:
        # Initialize memory agent for the specific vectorstore if needed
        if not init_memory_agent(vectorstore_name):
            return f"Error: Could not initialize memory agent for vectorstore '{vectorstore_name}'"

        config = memory_agent.get_relevance_config()

        result = "Current Relevance Scoring Configuration:\n\n"
        result += f"Vector Weight: {config['vector_weight']:.2f} (similarity score importance)\n"
        result += f"Temporal Weight: {config['temporal_weight']:.2f} (recency importance)\n"
        result += f"Usage Weight: {config['usage_weight']:.2f} (access frequency importance)\n"
        result += f"Recency Decay Days: {config['recency_decay_days']:.1f} (creation date decay)\n"
        result += f"Access Decay Days: {config['access_decay_days']:.1f} (last access decay)\n"
        result += f"Usage Boost Factor: {config['usage_boost_factor']:.2f} (access count multiplier)\n"
        result += f"Max Temporal Boost: {config['max_temporal_boost']:.2f} (temporal cap)\n"
        result += f"Max Usage Boost: {config['max_usage_boost']:.2f} (usage cap)\n"

        return result

    except Exception as e:
        return f"Error getting relevance config: {str(e)}"

@mcp.tool()
async def update_relevance_config(
    vectorstore_name: str,
    vector_weight: float = None,
    temporal_weight: float = None,
    usage_weight: float = None,
    recency_decay_days: float = None,
    access_decay_days: float = None,
    usage_boost_factor: float = None,
    max_temporal_boost: float = None,
    max_usage_boost: float = None
) -> str:
    """Update relevance scoring configuration parameters.

    Args:
        vectorstore_name: Name of the vectorstore to update config for
        vector_weight: Weight for vector similarity score (0.0-1.0)
        temporal_weight: Weight for temporal recency component (0.0-1.0)
        usage_weight: Weight for usage frequency component (0.0-1.0)
        recency_decay_days: Days for creation recency to decay to ~37%
        access_decay_days: Days for last access recency to decay to ~37%
        usage_boost_factor: Multiplier for access count boost
        max_temporal_boost: Maximum boost from temporal factors
        max_usage_boost: Maximum boost from usage factors
    """
    if not memory_agent:
        return "Error: Memory agent not initialized"

    try:
        # Initialize memory agent for the specific vectorstore if needed
        if not init_memory_agent(vectorstore_name):
            return f"Error: Could not initialize memory agent for vectorstore '{vectorstore_name}'"

        # Build update parameters (only include non-None values)
        update_params = {}
        if vector_weight is not None:
            update_params['vector_weight'] = vector_weight
        if temporal_weight is not None:
            update_params['temporal_weight'] = temporal_weight
        if usage_weight is not None:
            update_params['usage_weight'] = usage_weight
        if recency_decay_days is not None:
            update_params['recency_decay_days'] = recency_decay_days
        if access_decay_days is not None:
            update_params['access_decay_days'] = access_decay_days
        if usage_boost_factor is not None:
            update_params['usage_boost_factor'] = usage_boost_factor
        if max_temporal_boost is not None:
            update_params['max_temporal_boost'] = max_temporal_boost
        if max_usage_boost is not None:
            update_params['max_usage_boost'] = max_usage_boost

        if not update_params:
            return "No parameters provided to update"

        # Update the configuration
        updated_config = memory_agent.update_relevance_config(**update_params)

        result = "Relevance configuration updated successfully!\n\n"
        result += "New Configuration:\n"
        result += f"Vector Weight: {updated_config['vector_weight']:.2f}\n"
        result += f"Temporal Weight: {updated_config['temporal_weight']:.2f}\n"
        result += f"Usage Weight: {updated_config['usage_weight']:.2f}\n"
        result += f"Recency Decay Days: {updated_config['recency_decay_days']:.1f}\n"
        result += f"Access Decay Days: {updated_config['access_decay_days']:.1f}\n"
        result += f"Usage Boost Factor: {updated_config['usage_boost_factor']:.2f}\n"
        result += f"Max Temporal Boost: {updated_config['max_temporal_boost']:.2f}\n"
        result += f"Max Usage Boost: {updated_config['max_usage_boost']:.2f}\n"

        return result

    except Exception as e:
        return f"Error updating relevance config: {str(e)}"

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
    vectorstore_name: str
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
    vectorstore_name: str,
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
    # Initialize the LLM manager first (required for grounding)
    if not init_llm_manager():
        print("Failed to initialize LLM manager", file=sys.stderr)
        sys.exit(1)

    # Initialize the memory agent
    if not init_memory_agent("augment:global"):
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

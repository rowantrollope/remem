#!/usr/bin/env python3
"""
Model Context Protocol (MCP) Server for Remem Memory Agent

This MCP server exposes the remem memory system capabilities as standardized tools
that can be used by any MCP-compatible client (like Claude Desktop, IDEs, etc.).

The server provides tools for:
- Storing memories with contextual grounding
- Searching memories with vector similarity
- Deleting memories by ID or clearing all
- Getting memory statistics
- Managing context for grounding
- Relevance scoring configuration
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

# Global instances - use dictionary to cache memory agents per vectorstore
# This provides significant performance improvements by avoiding re-initialization
# for repeated operations on the same vectorstore while still supporting
# multiple vectorstores dynamically as needed by different Claude Code sessions
memory_agents: dict[str, MemoryAgent] = {}
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

def get_memory_agent(vectorstore_name: str) -> Optional[MemoryAgent]:
    """Get or create a memory agent for the specified vectorstore.
    
    This function implements intelligent caching of memory agent instances to optimize
    performance. Each vectorstore gets its own memory agent instance that is reused
    across multiple tool calls. This is particularly important for Claude Code which
    may make many sequential calls to the same vectorstore during a coding session.
    
    Args:
        vectorstore_name: The name of the vectorstore to get/create an agent for
        
    Returns:
        MemoryAgent instance or None if initialization fails
    """
    global memory_agents
    
    # Return cached instance if it exists - this avoids expensive re-initialization
    # and maintains any state/context that may have been set on the agent
    if vectorstore_name in memory_agents:
        return memory_agents[vectorstore_name]
    
    try:
        # Create new memory agent instance for this vectorstore
        # The MemoryAgent constructor will handle Redis connection validation
        # and embedding model initialization automatically
        agent = MemoryAgent(vectorset_key=vectorstore_name)
        memory_agents[vectorstore_name] = agent
        print(f"✅ Initialized memory agent for vectorstore: {vectorstore_name}", file=sys.stderr)
        return agent
    except Exception as e:
        print(f"❌ Error initializing memory agent for '{vectorstore_name}': {e}", file=sys.stderr)
        return None

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
    content: str,
    vectorstore_name: str,
    tags: str = "",
    apply_grounding: bool = True
) -> str:
    """Store a new memory in the memory system with optional tagging.
    
    This tool stores memories with contextual grounding applied by default to improve
    memory relevance and searchability. The grounding process enriches the memory
    content with current context information to make future retrieval more accurate.
    
    Args:
        content: The memory content to store - this is the main information to remember
        vectorstore_name: Name of the vectorstore to use (e.g., 'claude:global', 'claude:remem')
        tags: Optional comma-separated tags to categorize the memory (e.g., 'preference,coding,style')
        apply_grounding: Whether to apply contextual grounding to improve memory quality
    
    Returns:
        Success message with memory ID or error description
    """
    memory_agent = get_memory_agent(vectorstore_name)
    if not memory_agent:
        return f"Error: Could not initialize memory agent for vectorstore '{vectorstore_name}'"

    print(f"🔍 MCP DEBUG: Storing memory with grounding={apply_grounding}", file=sys.stderr)

    try:
        # Parse tags if provided and append them to the memory text
        # The system will automatically extract these as tags during parsing
        memory_text_with_tags = content
        if tags:
            tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
            if tag_list:
                # Append tags in quotes so they get extracted as tags by _parse_memory_text
                quoted_tags = ' '.join(f'"{tag}"' for tag in tag_list)
                memory_text_with_tags = f"{content} {quoted_tags}"
        
        # Store the memory using the core agent with enhanced error handling
        # The memory agent will handle embedding generation, grounding application,
        # and storage in the Redis VectorSet automatically
        result = memory_agent.store_memory(
            memory_text=memory_text_with_tags,
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
    """Search for relevant memories using advanced vector similarity with relevance scoring.
    
    This tool performs sophisticated memory search using vector embeddings combined with
    relevance scoring that considers similarity, recency, and usage patterns. The search
    results are ranked by a composite score that balances semantic similarity with
    temporal and usage-based relevance factors.
    
    Args:
        query: Search query text - describe what you're looking for
        vectorstore_name: Name of the vectorstore to search (e.g., 'claude:global', 'claude:remem')
        top_k: Maximum number of results to return (1-20)
        min_similarity: Minimum similarity threshold (0.0-1.0, higher = more strict)
    
    Returns:
        Formatted search results with relevance scores and metadata
    """
    memory_agent = get_memory_agent(vectorstore_name)
    if not memory_agent:
        return f"Error: Could not initialize memory agent for vectorstore '{vectorstore_name}'"
    
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
    """Get comprehensive statistics about the memory system for a specific vectorstore.
    
    This tool provides detailed information about the memory system including memory
    count, vector dimensions, embedding model configuration, and memory type breakdown.
    This is particularly useful for understanding the current state of your memory
    system and monitoring storage usage.
    
    Args:
        vectorstore_name: Name of the vectorstore to get stats for (e.g., 'claude:global', 'claude:remem')
    
    Returns:
        Detailed statistics including memory count, vector dimensions, and configuration info
    """
    memory_agent = get_memory_agent(vectorstore_name)
    if not memory_agent:
        return f"Error: Could not initialize memory agent for vectorstore '{vectorstore_name}'"
    
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
    memory_agent = get_memory_agent(vectorstore_name)
    if not memory_agent:
        return f"Error: Could not initialize memory agent for vectorstore '{vectorstore_name}'"

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
    memory_agent = get_memory_agent(vectorstore_name)
    if not memory_agent:
        return f"Error: Could not initialize memory agent for vectorstore '{vectorstore_name}'"

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
async def get_relevance_config(vectorstore_name: str) -> str:
    """Get current relevance scoring configuration.

    Args:
        vectorstore_name: Name of the vectorstore to get config for
    """
    memory_agent = get_memory_agent(vectorstore_name)
    if not memory_agent:
        return f"Error: Could not initialize memory agent for vectorstore '{vectorstore_name}'"

    try:
        # Use the already initialized memory agent from get_memory_agent above
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
    memory_agent = get_memory_agent(vectorstore_name)
    if not memory_agent:
        return f"Error: Could not initialize memory agent for vectorstore '{vectorstore_name}'"

    try:
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
    memory_agent = get_memory_agent(vectorstore_name)
    if not memory_agent:
        return f"Error: Could not initialize memory agent for vectorstore '{vectorstore_name}'"

    try:
        success = memory_agent.delete_memory(memory_id, vectorset_key=vectorstore_name)

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
    memory_agent = get_memory_agent(vectorstore_name)
    if not memory_agent:
        return f"Error: Could not initialize memory agent for vectorstore '{vectorstore_name}'"

    if not confirm:
        return "⚠️ This will delete ALL memories from the vectorstore. To confirm, call this tool with confirm=True"

    try:
        result = memory_agent.clear_all_memories(vectorset_key=vectorstore_name)

        if result.get('success'):
            memories_deleted = result.get('memories_deleted', 0)
            return f"✅ Successfully cleared all memories from '{vectorstore_name}'. Deleted {memories_deleted} memories."
        else:
            error_msg = result.get('error', 'Unknown error')
            return f"❌ Failed to clear memories: {error_msg}"

    except Exception as e:
        return f"Error clearing memories: {str(e)}"

def main() -> None:
    """Main entry point for the Remem MCP Server.
    
    This function initializes all required components for the MCP server including
    the LLM manager for contextual grounding, optional LangCache client for performance
    optimization, and starts the MCP server with STDIO transport for Claude Desktop
    integration.
    
    The server uses dynamic memory agent initialization, meaning memory agents are
    created on-demand for each vectorstore as tools are called, rather than 
    pre-initializing with a hardcoded vectorstore name.
    """
    # Initialize the LLM manager first - this is required for contextual grounding
    # functionality which enhances memory storage quality by adding relevant context
    if not init_llm_manager():
        print("❌ Failed to initialize LLM manager - contextual grounding will be disabled", file=sys.stderr)
        sys.exit(1)

    # Initialize LangCache client (optional) - this provides caching for expensive
    # operations like memory search and question answering to improve performance
    init_langcache_client()

    print("✅ Remem MCP Server initialized successfully", file=sys.stderr)
    print("🔧 Available tools:", file=sys.stderr)
    print("  - store_memory: Store new memories with contextual grounding and tagging", file=sys.stderr)
    print("  - search_memories: Search memories using advanced vector similarity (cached)", file=sys.stderr)
    print("  - get_memory_stats: Get comprehensive memory system statistics", file=sys.stderr)
    print("  - set_context: Set contextual information for memory grounding", file=sys.stderr)
    print("  - get_context: Get current contextual information", file=sys.stderr)
    print("  - delete_memory: Delete a specific memory by ID", file=sys.stderr)
    print("  - clear_all_memories: Clear all memories from a vectorstore (with confirmation)", file=sys.stderr)
    print("  - get_relevance_config: Get current relevance scoring configuration", file=sys.stderr)
    print("  - update_relevance_config: Update relevance scoring parameters", file=sys.stderr)
    print(f"📊 Memory agents will be initialized dynamically per vectorstore", file=sys.stderr)
    print(f"🚀 Starting MCP server with STDIO transport...", file=sys.stderr)

    # Run the MCP server with STDIO transport - this is the standard method
    # for integration with Claude Desktop and other MCP-compatible clients
    mcp.run(transport='stdio')


if __name__ == "__main__":
    main()

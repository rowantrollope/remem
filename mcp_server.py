#!/usr/bin/env python3
"""
Model Context Protocol (MCP) Server for Remem Memory Agent

This MCP server exposes the remem memory system capabilities as standardized tools
that can be used by any MCP-compatible client (like Claude Desktop, IDEs, etc.).

The server provides tools for:
- Storing memories (nemes)
- Searching memories
- Getting memory statistics
- Managing context
- Advanced memory operations with k-lines
"""

import os
import sys
import json
import asyncio
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# Import MCP SDK
from mcp.server.fastmcp import FastMCP

# Import remem memory system
from memory.core_agent import MemoryAgent
from memory.core import RelevanceConfig

# Load environment variables
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("remem-memory")

# Global memory agent instance
memory_agent: Optional[MemoryAgent] = None

def init_memory_agent(vectorstore_name: str = "memories") -> bool:
    """Initialize the memory agent with proper error handling."""
    global memory_agent
    try:
        memory_agent = MemoryAgent(vectorset_key=vectorstore_name)
        return True
    except Exception as e:
        print(f"Error initializing memory agent: {e}", file=sys.stderr)
        return False

@mcp.tool()
async def store_memory(
    text: str,
    memory_type: str = "neme",
    vectorstore_name: str = "memories",
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
async def search_memories(
    query: str,
    top_k: int = 5,
    vectorstore_name: str = "memories",
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
async def get_memory_stats(vectorstore_name: str = "memories") -> str:
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
async def answer_question(
    question: str,
    top_k: int = 5,
    vectorstore_name: str = "memories",
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
    vectorstore_name: str = "memories",
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
    vectorstore_name: str = "memories"
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
async def get_context(vectorstore_name: str = "memories") -> str:
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
    vectorstore_name: str = "memories",
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

if __name__ == "__main__":
    # Initialize the memory agent
    if not init_memory_agent():
        print("Failed to initialize memory agent", file=sys.stderr)
        sys.exit(1)

    print("Remem MCP Server initialized successfully", file=sys.stderr)
    print("Available tools:", file=sys.stderr)
    print("  - store_memory: Store new memories with contextual grounding", file=sys.stderr)
    print("  - search_memories: Search memories using vector similarity", file=sys.stderr)
    print("  - get_memory_stats: Get memory system statistics", file=sys.stderr)
    print("  - answer_question: Answer questions using memory with confidence scoring", file=sys.stderr)
    print("  - extract_and_store_memories: Extract memories from conversations", file=sys.stderr)
    print("  - set_context: Set contextual information for grounding", file=sys.stderr)
    print("  - get_context: Get current contextual information", file=sys.stderr)
    print("  - recall_with_klines: Advanced memory recall with k-line reasoning", file=sys.stderr)

    # Run the MCP server
    mcp.run(transport='stdio')

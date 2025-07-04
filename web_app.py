#!/usr/bin/env python3
"""
REST API for Memory Agent

"""

import os
import json
import uuid
import requests
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, Query, Path, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from memory.agent import LangGraphMemoryAgent
from memory.core_agent import MemoryAgent
from llm_manager import LLMManager, LLMConfig, init_llm_manager as initialize_llm_manager, get_llm_manager
from optimizations.performance_optimizer import init_performance_optimizer, get_performance_optimizer, optimize_memory_agent

app = FastAPI(
    title="Memory Agent API",
    description="REST API for Memory Agent with LangGraph integration",
    version="1.0.0"
)

# Enable CORS for all routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the memory agent and OpenAI client
memory_agent = None
openai_client = None

# Initialize OpenAI client
try:
    from openai import OpenAI
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except ImportError:
    print("⚠️ OpenAI package not installed. Chat sessions will not work.")
except Exception as e:
    print(f"⚠️ Failed to initialize OpenAI client: {e}")

# Global configuration store
app_config = {
    "redis": {
        "host": os.getenv("REDIS_HOST", "localhost"),
        "port": int(os.getenv("REDIS_PORT", "6381")),
        "db": int(os.getenv("REDIS_DB", "0")),
        "vectorset_key": "memories"
    },
    "llm": {
        "tier1": {
            "provider": "openai",  # "openai" or "ollama"
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 2000,
            "base_url": None,  # For Ollama: "http://localhost:11434"
            "api_key": os.getenv("OPENAI_API_KEY", ""),
            "timeout": 30
        },
        "tier2": {
            "provider": "ollama",  # "openai" or "ollama"
            "model": "deepseek-r1:7b",
            "temperature": 0.1,
            "max_tokens": 1000,
            "base_url": "http://localhost:11434",  # For Ollama: "http://localhost:11434"
            "api_key": os.getenv("OPENAI_API_KEY", ""),
            "timeout": 30
        }
    },
    "openai": {
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "organization": os.getenv("OPENAI_ORG_ID", ""),
        "embedding_model": "text-embedding-ada-002",
        "embedding_dimension": 1536,
        "chat_model": "gpt-3.5-turbo",
        "temperature": 0.1
    },
    "langgraph": {
        "model_name": "gpt-3.5-turbo",
        "temperature": 0.1,
        "system_prompt_enabled": True
    },
    "memory_agent": {
        "default_top_k": 5,
        "apply_grounding_default": True,
        "validation_enabled": True
    },
    "web_server": {
        "host": "0.0.0.0",
        "port": 5001,
        "debug": True,
        "cors_enabled": True
    },
    "performance": {
        "cache_enabled": True,
        "use_semantic_cache": True,
        "cache_default_ttl": 3600,
        "optimization_enabled": True,
        "batch_processing_enabled": True,
        "semantic_similarity_threshold": 0.85,
        "cache_ttl_settings": {
            "query_optimization": 7200,
            "memory_relevance": 1800,
            "context_analysis": 3600,
            "memory_grounding": 1800,
            "extraction_evaluation": 900,
            "memory_extraction_evaluation": 300,  # Short TTL for memory evaluation
            "conversation": 300,
            "answer_generation": 1800
        },
        "semantic_similarity_thresholds": {
            "query_optimization": 0.90,
            "memory_relevance": 0.85,
            "context_analysis": 0.88,
            "memory_grounding": 0.82,
            "extraction_evaluation": 0.80,
            "memory_extraction_evaluation": 0.70,  # Lower threshold for memory evaluation to reduce false cache hits
            "conversation": 0.95,
            "answer_generation": 0.87
        }
    }
}

# =============================================================================
# PYDANTIC MODELS - Request/Response validation models
# =============================================================================

class MemoryStoreRequest(BaseModel):
    text: str = Field(..., description="Memory text to store")
    apply_grounding: bool = Field(True, description="Whether to apply contextual grounding")

class MemorySearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    top_k: int = Field(5, ge=1, description="Number of results to return")
    filter: Optional[str] = Field(None, description="Filter expression for Redis VSIM command")
    optimize_query: bool = Field(False, description="Whether to optimize query for embedding search")
    min_similarity: float = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity score threshold")

class MemoryDeleteRequest(BaseModel):
    vectorstore_name: Optional[str] = Field(None, description="Name of the vectorstore to delete from")

class ContextSetRequest(BaseModel):
    location: Optional[str] = Field(None, description="Current location")
    activity: Optional[str] = Field(None, description="Current activity")
    people_present: Optional[List[str]] = Field(None, description="List of people present")



class KLineAnswerRequest(BaseModel):
    question: str = Field(..., description="Question to answer")
    top_k: int = Field(5, ge=1, description="Number of memories to retrieve for context")
    filter: Optional[str] = Field(None, description="Filter expression for Redis VSIM command")
    min_similarity: float = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity score threshold")

class AgentChatRequest(BaseModel):
    message: str = Field(..., description="User message/question")
    system_prompt: Optional[str] = Field(None, description="Custom system prompt to override default behavior")

class AgentSessionCreateRequest(BaseModel):
    system_prompt: str = Field(..., description="Custom system prompt for the agent session")
    session_id: Optional[str] = Field(None, description="Custom session ID, auto-generated if not provided")
    config: Optional[Dict[str, Any]] = Field(None, description="Additional configuration options")

class AgentSessionMessageRequest(BaseModel):
    message: str = Field(..., description="The user's message")
    stream: bool = Field(False, description="Whether to stream the response")
    store_memory: bool = Field(True, description="Whether to extract and store memories from this conversation")
    top_k: int = Field(10, ge=1, description="Number of memories to search and return")
    min_similarity: float = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity score threshold")

class LLMConfigTier(BaseModel):
    provider: str = Field(..., description="LLM provider: 'openai' or 'ollama'")
    model: str = Field(..., description="Model name")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Response creativity")
    max_tokens: int = Field(1000, ge=1, description="Maximum response length")
    base_url: Optional[str] = Field(None, description="Base URL for Ollama")
    api_key: Optional[str] = Field(None, description="API key")
    timeout: int = Field(30, ge=1, description="Request timeout in seconds")

class LLMConfigUpdate(BaseModel):
    tier1: Optional[LLMConfigTier] = None
    tier2: Optional[LLMConfigTier] = None

class CacheClearRequest(BaseModel):
    pattern: Optional[str] = Field(None, description="Cache pattern to clear")
    operation_type: Optional[str] = Field(None, description="Operation type to clear")

class ConfigUpdateRequest(BaseModel):
    redis: Optional[Dict[str, Any]] = Field(None, description="Redis configuration")
    llm: Optional[Dict[str, Any]] = Field(None, description="LLM configuration")
    openai: Optional[Dict[str, Any]] = Field(None, description="OpenAI configuration")
    langgraph: Optional[Dict[str, Any]] = Field(None, description="LangGraph configuration")
    memory_agent: Optional[Dict[str, Any]] = Field(None, description="Memory agent configuration")
    web_server: Optional[Dict[str, Any]] = Field(None, description="Web server configuration")
    performance: Optional[Dict[str, Any]] = Field(None, description="Performance configuration")

def init_llm_manager():
    """Initialize the LLM manager with current configuration."""
    try:
        # Create LLM configurations for both tiers
        tier1_config = LLMConfig(
            provider=app_config["llm"]["tier1"]["provider"],
            model=app_config["llm"]["tier1"]["model"],
            temperature=app_config["llm"]["tier1"]["temperature"],
            max_tokens=app_config["llm"]["tier1"]["max_tokens"],
            base_url=app_config["llm"]["tier1"]["base_url"],
            api_key=app_config["llm"]["tier1"]["api_key"],
            timeout=app_config["llm"]["tier1"]["timeout"]
        )

        tier2_config = LLMConfig(
            provider=app_config["llm"]["tier2"]["provider"],
            model=app_config["llm"]["tier2"]["model"],
            temperature=app_config["llm"]["tier2"]["temperature"],
            max_tokens=app_config["llm"]["tier2"]["max_tokens"],
            base_url=app_config["llm"]["tier2"]["base_url"],
            api_key=app_config["llm"]["tier2"]["api_key"],
            timeout=app_config["llm"]["tier2"]["timeout"]
        )

        # Initialize the global LLM manager
        initialize_llm_manager(tier1_config, tier2_config)
        return True
    except Exception as e:
        print(f"Failed to initialize LLM manager: {e}")
        return False

def reinitialize_llm_manager_with_optimizations():
    """Reinitialize LLM manager and reapply performance optimizations if enabled."""
    try:
        # Reinitialize the LLM manager
        if not init_llm_manager():
            return False, "Failed to reinitialize LLM manager"

        # Reapply performance optimizations if enabled
        if app_config["performance"]["optimization_enabled"] and app_config["performance"]["cache_enabled"]:
            try:
                optimizer = get_performance_optimizer()
                if optimizer:
                    llm_manager = get_llm_manager()
                    cached_llm_manager = optimizer.optimize_llm_manager(llm_manager)
                    
                    # Replace the global LLM manager with the cached version
                    import llm_manager as llm_manager_module
                    llm_manager_module.llm_manager = cached_llm_manager
                    
                    cache_type = "semantic vectorset" if app_config["performance"]["use_semantic_cache"] else "hash-based"
                    print(f"✅ LLM manager reinitialized and wrapped with {cache_type} caching")
                else:
                    print("✅ LLM manager reinitialized (no performance optimizer available)")
            except Exception as e:
                print(f"⚠️ LLM manager reinitialized but performance optimization failed: {e}")
                # Don't fail completely, the LLM manager is still functional
        else:
            print("✅ LLM manager reinitialized (performance optimizations disabled)")

        return True, "LLM manager reinitialized successfully"
    except Exception as e:
        error_msg = f"Failed to reinitialize LLM manager: {e}"
        print(f"❌ {error_msg}")
        return False, error_msg

def init_memory_agent():
    """Initialize the LangGraph memory agent with current configuration."""
    global memory_agent
    try:
        # Initialize performance optimizer if enabled
        optimizer = None
        if app_config["performance"]["optimization_enabled"]:
            optimizer = init_performance_optimizer(
                redis_host=app_config["redis"]["host"],
                redis_port=app_config["redis"]["port"],
                redis_db=app_config["redis"]["db"],
                cache_enabled=app_config["performance"]["cache_enabled"],
                use_semantic_cache=app_config["performance"]["use_semantic_cache"]
            )
            cache_type = "semantic vectorset" if app_config["performance"]["use_semantic_cache"] else "hash-based"
            print(f"✅ Performance optimizer initialized ({cache_type} caching)")

            # CRITICAL: Wrap the LLM manager with caching
            if app_config["performance"]["cache_enabled"]:
                llm_manager = get_llm_manager()
                cached_llm_manager = optimizer.optimize_llm_manager(llm_manager)
                
                # Replace the global LLM manager with the cached version
                import llm_manager as llm_manager_module
                llm_manager_module.llm_manager = cached_llm_manager
                print(f"✅ LLM manager wrapped with {cache_type} caching")

        # Create memory agent with current Redis configuration
        base_memory_agent = MemoryAgent(
            redis_host=app_config["redis"]["host"],
            redis_port=app_config["redis"]["port"],
            redis_db=app_config["redis"]["db"],
            vectorset_key=app_config["redis"]["vectorset_key"]
        )

        # Apply performance optimizations if enabled
        if app_config["performance"]["optimization_enabled"]:
            base_memory_agent = optimize_memory_agent(
                base_memory_agent,
                cache_enabled=app_config["performance"]["cache_enabled"]
            )
            print("✅ Memory agent optimizations applied")

        # Create LangGraph agent with current OpenAI configuration
        memory_agent = LangGraphMemoryAgent(
            model_name=app_config["langgraph"]["model_name"],
            temperature=app_config["langgraph"]["temperature"],
            vectorset_key=app_config["redis"]["vectorset_key"]
        )

        # Replace the underlying memory agent with our configured one
        memory_agent.memory_agent = base_memory_agent

        return True
    except Exception as e:
        print(f"Failed to initialize LangGraph memory agent: {e}")
        return False

# =============================================================================
# API UTILITIES AND VALIDATION
# =============================================================================

# Reserved vectorstore names that cannot be used by users
RESERVED_VECTORSTORE_NAMES = ["all", "info", "search", "context", "health", "metrics", "config"]

def validate_vectorstore_name(name: str) -> None:
    """Validate that vectorstore name is not reserved.
    
    Args:
        name: Vectorstore name to validate
        
    Raises:
        HTTPException: If name is reserved
    """
    if name.lower() in RESERVED_VECTORSTORE_NAMES:
        raise HTTPException(
            status_code=400, 
            detail=f"'{name}' is a reserved name. Reserved names: {', '.join(RESERVED_VECTORSTORE_NAMES)}"
        )

def get_vectorstore_name(vectorstore_param: Optional[str] = None) -> str:
    """Get vectorstore name with fallback to default.
    
    Args:
        vectorstore_param: Optional vectorstore name from URL parameter
        
    Returns:
        Vectorstore name (explicit or default)
    """
    return vectorstore_param or app_config["redis"]["vectorset_key"]

# =============================================================================
# NEME API - Fundamental Memory Operations (Inspired by Minsky's "Nemes")
# =============================================================================
#
# Nemes are the fundamental units of memory in Minsky's Society of Mind theory.
# These APIs handle atomic memory storage, retrieval, and basic operations.
# Think of these as the building blocks of knowledge that can be combined
# by higher-level cognitive processes.
#
# Core operations:
# - Store atomic memories with contextual grounding
# - Vector similarity search across stored memories
# - Memory lifecycle management (delete, clear)
# - Context management for grounding operations
# =============================================================================

@app.post('/api/memory')
async def api_store_neme_default(request: MemoryStoreRequest):
    """Store a new atomic memory (Neme) in the default vectorstore.

    In Minsky's framework, a Neme is a fundamental unit of memory - an atomic
    piece of knowledge that can be contextually grounded and later recalled
    by higher-level cognitive processes.

    Returns:
        JSON with success status, memory_id, message, and grounding information
    """
    return await _store_memory_impl(request, vectorstore_name=None)

@app.post('/api/memory/{vectorstore_name}')
async def api_store_neme_vectorstore(vectorstore_name: str, request: MemoryStoreRequest):
    """Store a new atomic memory (Neme) in a specific vectorstore.

    Args:
        vectorstore_name: Name of the vectorstore to store in

    Returns:
        JSON with success status, memory_id, message, and grounding information
    """
    validate_vectorstore_name(vectorstore_name)
    return await _store_memory_impl(request, vectorstore_name=vectorstore_name)

async def _store_memory_impl(request: MemoryStoreRequest, vectorstore_name: Optional[str] = None):
    """Implementation for storing memories with vectorstore support.
    
    Returns:
        JSON with success status, memory_id, message, and grounding information:
        - success (bool): Whether the operation succeeded
        - memory_id (str): UUID of the stored memory
        - message (str): Success message
        - original_text (str): Original memory text (after parsing)
        - final_text (str): Final stored text (grounded or original)
        - grounding_applied (bool): Whether grounding was applied
        - tags (list): Extracted tags from the memory
        - created_at (str): ISO 8601 UTC timestamp of memory creation
        - grounding_info (dict, optional): Details about grounding changes if applied
        - context_snapshot (dict, optional): Context used for grounding if applied
    """
    try:
        memory_text = request.text.strip()
        apply_grounding = request.apply_grounding
        
        # Use vectorstore from URL path, fallback to default
        final_vectorstore_name = get_vectorstore_name(vectorstore_name)

        if not memory_text:
            raise HTTPException(status_code=400, detail='Memory text is required')

        print(f"💾 NEME API: Storing atomic memory - '{memory_text[:60]}{'...' if len(memory_text) > 60 else ''}'")
        print(f"📦 Vectorstore: {final_vectorstore_name}")

        # Use existing memory agent with vectorstore parameter
        if not memory_agent:
            raise HTTPException(status_code=500, detail='Memory agent not initialized')

        storage_result = memory_agent.memory_agent.store_memory(
            memory_text,
            apply_grounding=apply_grounding,
            vectorset_key=final_vectorstore_name
        )

        # Prepare response with grounding information
        response_data = {
            'success': True,
            'memory_id': storage_result['memory_id'],
            'message': 'Memory stored successfully',
            'original_text': storage_result['original_text'],
            'final_text': storage_result['final_text'],
            'grounding_applied': storage_result['grounding_applied'],
            'tags': storage_result['tags'],
            'created_at': storage_result['created_at'],
            'vectorstore_name': final_vectorstore_name
        }

        # Include grounding information if available
        if 'grounding_info' in storage_result:
            response_data['grounding_info'] = storage_result['grounding_info']

        # Include context snapshot if available
        if 'context_snapshot' in storage_result:
            response_data['context_snapshot'] = storage_result['context_snapshot']

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/api/memory/search')
async def api_search_nemes_default(request: MemorySearchRequest):
    """Search atomic memories (Nemes) in the default vectorstore using vector similarity.

    This performs direct vector similarity search across stored Nemes,
    returning the most relevant atomic memories for a given query.

    Returns:
        JSON with success status, memories array, count, and memory breakdown by type
    """
    return await _search_memories_impl(request, vectorstore_name=None)

@app.post('/api/memory/{vectorstore_name}/search')
async def api_search_nemes_vectorstore(vectorstore_name: str, request: MemorySearchRequest):
    """Search atomic memories (Nemes) in a specific vectorstore using vector similarity.

    Args:
        vectorstore_name: Name of the vectorstore to search in

    Returns:
        JSON with success status, memories array, count, and memory breakdown by type
    """
    validate_vectorstore_name(vectorstore_name)
    return await _search_memories_impl(request, vectorstore_name=vectorstore_name)

async def _search_memories_impl(request: MemorySearchRequest, vectorstore_name: Optional[str] = None):
    """Implementation for searching memories with vectorstore support.
    
    Returns:
        JSON with success status, memories array, count, and memory breakdown by type
    """
    try:
        query = request.query.strip()
        top_k = request.top_k
        filter_expr = request.filter
        optimize_query = request.optimize_query
        min_similarity = request.min_similarity
        
        # Use vectorstore from URL path, fallback to default
        final_vectorstore_name = get_vectorstore_name(vectorstore_name)

        if not query:
            raise HTTPException(status_code=400, detail='Query is required')

        print(f"🔍 NEME API: Searching memories: {query} (top_k: {top_k}, min_similarity: {min_similarity})")
        print(f"📦 Vectorstore: {final_vectorstore_name}")
        if filter_expr:
            print(f"🔍 Filter: {filter_expr}")
        if optimize_query:
            print(f"🔍 Query optimization: enabled")

        # Use existing memory agent with vectorstore parameter
        if not memory_agent:
            raise HTTPException(status_code=500, detail='Memory agent not initialized')

        # Use the memory agent for search operations with optional optimization
        if optimize_query:
            validation_result = memory_agent.memory_agent.processing.validate_and_preprocess_question(query)
            if validation_result["type"] == "search":
                search_query = validation_result.get("embedding_query") or validation_result["content"]
                print(f"🔍 Using optimized search query: '{search_query}'")
                search_result = memory_agent.memory_agent.search_memories_with_filtering_info(
                    search_query, top_k, filter_expr, min_similarity, vectorset_key=final_vectorstore_name
                )
            else:
                search_result = memory_agent.memory_agent.search_memories_with_filtering_info(
                    query, top_k, filter_expr, min_similarity, vectorset_key=final_vectorstore_name
                )
        else:
            search_result = memory_agent.memory_agent.search_memories_with_filtering_info(
                query, top_k, filter_expr, min_similarity, vectorset_key=final_vectorstore_name
            )

        memories = search_result['memories']
        filtering_info = search_result['filtering_info']
        print(f"🔍 NEME API: Search result type: {type(search_result)}")
        print(f"🔍 NEME API: Filtering info: {filtering_info}")

        return {
            'success': True,
            'query': query,
            'memories': memories,
            'count': len(memories),
            'filtering_info': filtering_info,
            'vectorstore_name': final_vectorstore_name
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/memory')
async def api_get_neme_info_default():
    """Get atomic memory (Neme) statistics and system information for the default vectorstore.

    Returns:
        JSON with memory count, vector dimension, embedding model, and system info
    """
    return await _get_memory_info_impl(vectorstore_name=None)

@app.get('/api/memory/{vectorstore_name}')
async def api_get_neme_info_vectorstore(vectorstore_name: str):
    """Get atomic memory (Neme) statistics and system information for a specific vectorstore.

    Args:
        vectorstore_name: Name of the vectorstore to get info for

    Returns:
        JSON with memory count, vector dimension, embedding model, and system info
    """
    validate_vectorstore_name(vectorstore_name)
    return await _get_memory_info_impl(vectorstore_name=vectorstore_name)

async def _get_memory_info_impl(vectorstore_name: Optional[str] = None):
    """Implementation for getting memory info with vectorstore support.
    
    Returns:
        JSON with memory count, vector dimension, embedding model, and system info
    """
    try:
        if not memory_agent:
            raise HTTPException(status_code=500, detail='Memory agent not initialized')

        # Use vectorstore from URL path, fallback to default
        final_vectorstore_name = get_vectorstore_name(vectorstore_name)

        # Get comprehensive memory information from underlying agent
        memory_info = memory_agent.memory_agent.get_memory_info()

        if 'error' in memory_info:
            raise HTTPException(status_code=500, detail=memory_info['error'])

        # Add vectorstore name to response
        memory_info['vectorstore_name'] = final_vectorstore_name

        return {
            'success': True,
            **memory_info
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete('/api/memory/{memory_id}')
async def api_delete_neme_default(memory_id: str = Path(..., description="UUID of the memory to delete")):
    """Delete a specific atomic memory (Neme) by ID from the default vectorstore.

    Args:
        memory_id: UUID of the memory to delete

    Returns:
        JSON with success status and deletion details
    """
    return await _delete_memory_impl(memory_id, vectorstore_name=None)

@app.delete('/api/memory/{vectorstore_name}/{memory_id}')
async def api_delete_neme_vectorstore(vectorstore_name: str, memory_id: str):
    """Delete a specific atomic memory (Neme) by ID from a specific vectorstore.

    Args:
        vectorstore_name: Name of the vectorstore to delete from
        memory_id: UUID of the memory to delete

    Returns:
        JSON with success status and deletion details
    """
    validate_vectorstore_name(vectorstore_name)
    return await _delete_memory_impl(memory_id, vectorstore_name=vectorstore_name)

async def _delete_memory_impl(memory_id: str, vectorstore_name: Optional[str] = None):
    """Implementation for deleting a memory with vectorstore support.
    
    Returns:
        JSON with success status and deletion details
    """
    try:
        if not memory_id or not memory_id.strip():
            raise HTTPException(status_code=400, detail='Memory ID is required')

        # Use vectorstore from URL path, fallback to default
        final_vectorstore_name = get_vectorstore_name(vectorstore_name)

        print(f"🗑️ NEME API: Deleting atomic memory: {memory_id}")
        print(f"📦 Vectorstore: {final_vectorstore_name}")

        # Use existing memory agent with vectorstore parameter
        if not memory_agent:
            raise HTTPException(status_code=500, detail='Memory agent not initialized')

        success = memory_agent.memory_agent.delete_memory(
            memory_id.strip(),
            vectorset_key=final_vectorstore_name
        )

        if success:
            return {
                'success': True,
                'message': f'Neme {memory_id} deleted successfully',
                'memory_id': memory_id,
                'vectorstore_name': final_vectorstore_name
            }
        else:
            raise HTTPException(
                status_code=404, 
                detail=f'Neme {memory_id} not found or could not be deleted'
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete('/api/memory/all')
async def api_delete_all_nemes_default():
    """Clear all atomic memories (Nemes) from the default vectorstore.

    Returns:
        JSON with success status, deletion count, and operation details
    """
    return await _delete_all_memories_impl(vectorstore_name=None)

@app.delete('/api/memory/{vectorstore_name}/all')
async def api_delete_all_nemes_vectorstore(vectorstore_name: str):
    """Clear all atomic memories (Nemes) from a specific vectorstore.

    Args:
        vectorstore_name: Name of the vectorstore to clear

    Returns:
        JSON with success status, deletion count, and operation details
    """
    validate_vectorstore_name(vectorstore_name)
    return await _delete_all_memories_impl(vectorstore_name=vectorstore_name)

async def _delete_all_memories_impl(vectorstore_name: Optional[str] = None):
    """Implementation for deleting all memories with vectorstore support.
    
    Returns:
        JSON with success status, deletion count, and operation details
    """
    try:
        # Use vectorstore from URL path, fallback to default
        final_vectorstore_name = get_vectorstore_name(vectorstore_name)

        print("🗑️ NEME API: Clearing all atomic memories...")
        print(f"📦 Vectorstore: {final_vectorstore_name}")

        # Use existing memory agent with vectorstore parameter
        if not memory_agent:
            raise HTTPException(status_code=500, detail='Memory agent not initialized')

        result = memory_agent.memory_agent.clear_all_memories(vectorset_key=final_vectorstore_name)

        if result['success']:
            return {
                'success': True,
                'message': result['message'],
                'memories_deleted': result['memories_deleted'],
                'vectorset_existed': result['vectorset_existed'],
                'vectorstore_name': final_vectorstore_name
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=result['error']
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/api/memory/context')
async def api_set_neme_context_default(request: ContextSetRequest, additional_context: Dict[str, Any] = Body({})):
    """Set current context for memory grounding in the default vectorstore.

    Args:
        request: Context parameters (location, activity, people_present)
        additional_context: Additional fields will be stored as environment context

    Returns:
        JSON with success status and updated context
    """
    return await _set_context_impl(request, additional_context, vectorstore_name=None)

@app.post('/api/memory/{vectorstore_name}/context')
async def api_set_neme_context_vectorstore(vectorstore_name: str, request: ContextSetRequest, additional_context: Dict[str, Any] = Body({})):
    """Set current context for memory grounding in a specific vectorstore.

    Args:
        vectorstore_name: Name of the vectorstore to set context for
        request: Context parameters (location, activity, people_present)
        additional_context: Additional fields will be stored as environment context

    Returns:
        JSON with success status and updated context
    """
    validate_vectorstore_name(vectorstore_name)
    return await _set_context_impl(request, additional_context, vectorstore_name=vectorstore_name)

async def _set_context_impl(request: ContextSetRequest, additional_context: Dict[str, Any], vectorstore_name: Optional[str] = None):
    """Implementation for setting context with vectorstore support.
    
    Returns:
        JSON with success status and updated context
    """
    try:
        if not memory_agent:
            raise HTTPException(status_code=500, detail='Memory agent not initialized')

        # Use vectorstore from URL path, fallback to default
        final_vectorstore_name = get_vectorstore_name(vectorstore_name)

        # Extract context parameters
        location = request.location
        activity = request.activity
        people_present = request.people_present or []

        # Use additional_context for environment context
        environment_context = additional_context

        print(f"🌍 NEME API: Setting context - Location: {location}, Activity: {activity}, People: {people_present}")
        print(f"📦 Vectorstore: {final_vectorstore_name}")

        # Set context on underlying memory agent
        memory_agent.memory_agent.set_context(
            location=location,
            activity=activity,
            people_present=people_present if people_present else None,
            **environment_context
        )

        return {
            'success': True,
            'message': 'Context updated successfully',
            'context': {
                'location': location,
                'activity': activity,
                'people_present': people_present,
                'environment': environment_context
            },
            'vectorstore_name': final_vectorstore_name
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/memory/context')
async def api_get_neme_context_default():
    """Get current context information for memory grounding from the default vectorstore.

    Returns:
        JSON with success status and current context (temporal, spatial, social, environmental)
    """
    return await _get_context_impl(vectorstore_name=None)

@app.get('/api/memory/{vectorstore_name}/context')
async def api_get_neme_context_vectorstore(vectorstore_name: str):
    """Get current context information for memory grounding from a specific vectorstore.

    Args:
        vectorstore_name: Name of the vectorstore to get context for

    Returns:
        JSON with success status and current context (temporal, spatial, social, environmental)
    """
    validate_vectorstore_name(vectorstore_name)
    return await _get_context_impl(vectorstore_name=vectorstore_name)

async def _get_context_impl(vectorstore_name: Optional[str] = None):
    """Implementation for getting context with vectorstore support.
    
    Returns:
        JSON with success status and current context (temporal, spatial, social, environmental)
    """
    try:
        if not memory_agent:
            raise HTTPException(status_code=500, detail='Memory agent not initialized')

        # Use vectorstore from URL path, fallback to default
        final_vectorstore_name = get_vectorstore_name(vectorstore_name)

        current_context = memory_agent.memory_agent.core._get_current_context()

        return {
            'success': True,
            'context': current_context,
            'vectorstore_name': final_vectorstore_name
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# BACKWARD COMPATIBILITY - DEPRECATED ENDPOINTS
# =============================================================================
# 
# These endpoints maintain backward compatibility with the old API design.
# They are deprecated and will be removed in a future version.
# Client applications should migrate to the new vectorstore-aware endpoints.

# DEPRECATED: Use DELETE /api/memory/all or DELETE /api/memory/{vectorstore_name}/all
@app.delete('/api/memory')  
async def api_delete_all_nemes_legacy(request: Optional[MemoryDeleteRequest] = None):
    """DEPRECATED: Use DELETE /api/memory/all or DELETE /api/memory/{vectorstore_name}/all
    
    Clear all atomic memories (Nemes) from the system.
    
    Args:
        request: Optional request body with vectorstore_name
        
    Returns:
        JSON with success status, deletion count, and operation details
    """
    # Get vectorstore_name from request body if provided
    vectorstore_name = request.vectorstore_name if request else None
    return await _delete_all_memories_impl(vectorstore_name=vectorstore_name)

# =============================================================================
# K-LINE API - Reflective Operations (Inspired by Minsky's "K-lines")
# =============================================================================
#
# K-lines (Knowledge lines) represent temporary mental states that activate
# and connect relevant Nemes for specific cognitive tasks. In Minsky's theory,
# K-lines are the mechanism by which the mind constructs coherent mental states
# from distributed memory fragments.
#
# These APIs handle:
# - Constructing mental states by recalling and filtering relevant memories
# - Question answering with confidence scoring and reasoning
# - Extracting valuable memories from conversational data
# - Advanced cognitive operations that combine multiple Nemes
# =============================================================================



@app.post('/api/memory/ask')
async def api_kline_answer(request: KLineAnswerRequest):
    """Answer a question using K-line construction and reasoning.

    This operation constructs a mental state from relevant Nemes and applies
    sophisticated reasoning to answer questions with confidence scoring.
    It represents the full cognitive process of memory recall + reasoning.

    K-lines are constructed but NOT stored - they exist only as temporary mental states.

    Returns:
        JSON with structured response including answer, confidence, reasoning, supporting memories,
        and the constructed mental state (K-line)
    """
    try:
        if not memory_agent:
            raise HTTPException(status_code=500, detail='Memory agent not initialized')

        question = request.question.strip()
        top_k = request.top_k
        filter_expr = request.filter
        min_similarity = request.min_similarity

        if not question:
            raise HTTPException(status_code=400, detail='Question is required')

        print(f"🤔 K-LINE API: Answering question via mental state construction: {question} (top_k: {top_k})")
        if filter_expr:
            print(f"🔍 Filter: {filter_expr}")

        # Use the memory agent's sophisticated answer_question method
        # This constructs a K-line (mental state) and applies reasoning
        answer_response = memory_agent.memory_agent.answer_question(question, top_k=top_k, filterBy=filter_expr, min_similarity=min_similarity)

        # Construct K-line (mental state) from the supporting memories
        supporting_memories = answer_response.get('supporting_memories', [])
        if supporting_memories:
            kline_result = memory_agent.memory_agent.construct_kline(
                query=question,
                memories=supporting_memories,
                answer=answer_response.get('answer'),
                confidence=answer_response.get('confidence'),
                reasoning=answer_response.get('reasoning')
            )
            print(f"🧠 Constructed K-line with coherence score: {kline_result.get('coherence_score', 0):.3f}")
        else:
            kline_result = {
                'mental_state': 'No relevant memories found to construct mental state.',
                'coherence_score': 0.0,
                'summary': 'Empty mental state'
            }

        # Prepare the response
        response_data = {
            'success': True,
            'question': question,
            **answer_response,  # Spread the structured response (answer, confidence, supporting_memories, etc.)
            'kline': kline_result  # Include the constructed mental state
        }

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# =============================================================================
# AGENT API - High-Level Orchestration (Full Cognitive Architecture)
# =============================================================================
#
# The Agent API represents the highest level of cognitive architecture,
# orchestrating both Nemes (atomic memories) and K-lines (mental states)
# to provide sophisticated conversational and reasoning capabilities.
#
# These APIs handle:
# - Full conversational agents with memory integration
# - Session management with persistent context
# - Complex multi-step reasoning workflows
# - Integration of memory operations with language generation
# =============================================================================

@app.post('/api/agent/chat')
async def api_agent_chat(request: AgentChatRequest):
    """Full conversational agent with integrated memory architecture.

    This endpoint orchestrates the complete cognitive architecture:
    - Searches relevant Nemes (atomic memories)
    - Constructs K-lines (mental states) for context
    - Applies sophisticated reasoning and language generation
    - Optionally extracts new memories from the conversation

    Returns:
        JSON with success status, original message, and agent response
    """
    try:
        if not memory_agent:
            raise HTTPException(status_code=500, detail='Memory agent not initialized')

        message = request.message.strip()
        system_prompt = request.system_prompt.strip() if request.system_prompt else None

        if not message:
            raise HTTPException(status_code=400, detail='Message is required')

        print(f"💬 AGENT API: Processing chat message - '{message}'")
        if system_prompt:
            print(f"🎯 AGENT API: Using custom system prompt - '{system_prompt[:60]}{'...' if len(system_prompt) > 60 else ''}'")

        # Use the LangGraph agent's run method with optional custom system prompt
        response = memory_agent.run(message, system_prompt=system_prompt if system_prompt else None)

        return {
            'success': True,
            'message': message,
            'response': response,
            'system_prompt_used': bool(system_prompt)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





# =============================================================================
# CHAT APIs - General purpose chat interface
# =============================================================================

@app.post('/api/agent/session')
async def api_create_agent_session(request: AgentSessionCreateRequest):
    """Create a new agent session with integrated memory capabilities.

    Returns:
        JSON with session_id and confirmation
    """
    try:
        system_prompt = request.system_prompt.strip()
        if not system_prompt:
            raise HTTPException(status_code=400, detail='system_prompt is required')

        session_id = request.session_id or str(uuid.uuid4())
        config = request.config or {}

        # Extract memory configuration with default to true for agent sessions
        use_memory = config.get('use_memory', True)

        # Store session in memory (in production, use Redis or database)
        if not hasattr(app, 'chat_sessions'):
            app.chat_sessions = {}

        session_data = {
            'system_prompt': system_prompt,
            'messages': [],
            'config': config,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'last_activity': datetime.now(timezone.utc).isoformat(),
            'use_memory': use_memory
        }
        print(f"")
        # Add memory-specific fields if memory is enabled
        if use_memory:
            session_data.update({
                'conversation_buffer': [],  # Buffer for memory extraction
                'extraction_threshold': 2,  # Number of messages before extraction
                'last_extraction': None,    # Timestamp of last memory extraction
                'memory_context': f"Chat session for: {system_prompt[:100]}..."  # Context for memory extraction
            })

        app.chat_sessions[session_id] = session_data

        memory_status = "enabled" if use_memory else "disabled"
        print(f"🆕 AGENT API: Created agent session {session_id} (memory: {memory_status})")

        return {
            'success': True,
            'session_id': session_id,
            'system_prompt': system_prompt,
            'use_memory': use_memory,
            'created_at': session_data['created_at']
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/api/agent/session/{session_id}')
async def api_agent_session_message(session_id: str = Path(..., description="The agent session ID"), request: AgentSessionMessageRequest = Body(...)):
    """Send a message to an agent session with full cognitive architecture.

    This endpoint provides the complete agent experience:
    - Searches relevant Nemes for context
    - Constructs K-lines (mental states)
    - Generates contextually-aware responses
    - Automatically extracts new memories from conversations

    Args:
        session_id: The agent session ID
        request: Request body with message and options

    Returns:
        JSON with the assistant's response and conversation context.
        For memory-enabled sessions, includes 'memory_context' with relevant memories used in the response.
    """
    try:
        if not hasattr(app, 'chat_sessions') or session_id not in app.chat_sessions:
            raise HTTPException(status_code=404, detail='Agent session not found')

        message = request.message.strip()
        if not message:
            raise HTTPException(status_code=400, detail='message is required')

        stream = request.stream
        store_memory = request.store_memory
        top_k = request.top_k
        min_similarity = request.min_similarity

        # Validate top_k parameter
        if not isinstance(top_k, int) or top_k < 1:
            raise HTTPException(status_code=400, detail='top_k must be a positive integer')

        # Validate min_similarity parameter
        if not isinstance(min_similarity, (int, float)) or min_similarity < 0.0 or min_similarity > 1.0:
            raise HTTPException(status_code=400, detail='min_similarity must be a number between 0.0 and 1.0')

        session = app.chat_sessions[session_id]
        use_memory = session.get('use_memory', False)

        if use_memory:
            print(f"1) User said: '{message[:80]}{'...' if len(message) > 80 else ''}'")
        else:
            print(f"💬 [{session_id}] AGENT API: Agent session (memory: disabled): User: {message}")

        # Handle memory-enabled sessions
        if use_memory:
            return _handle_memory_enabled_message(session_id, session, message, stream, store_memory, top_k, min_similarity)
        else:
            # Add user message to session for non-memory sessions
            user_message = {
                'role': 'user',
                'content': message,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            session['messages'].append(user_message)
            return _handle_standard_message(session_id, session, stream)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _handle_standard_message(session_id, session, stream):
    """Handle message processing for sessions without memory."""
    # Prepare messages for LLM (include system prompt)
    llm_messages = [
        {'role': 'system', 'content': session['system_prompt']}
    ]

    # Add conversation history
    for msg in session['messages']:
        llm_messages.append({
            'role': msg['role'],
            'content': msg['content']
        })

    # Get response from Tier 1 LLM (primary conversational)
    try:
        llm_manager = get_llm_manager()
        tier1_client = llm_manager.get_tier1_client()

        response = tier1_client.chat_completion(
            messages=llm_messages,
            temperature=session['config'].get('temperature', 0.7),
            max_tokens=session['config'].get('max_tokens', 1000)
        )

        if stream:
            # TODO: Implement streaming response
            raise HTTPException(status_code=501, detail='Streaming not yet implemented')

        assistant_response = response['content']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'LLM error: {str(e)}')

    # Add assistant response to session
    assistant_message = {
        'role': 'assistant',
        'content': assistant_response,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
    session['messages'].append(assistant_message)
    session['last_activity'] = datetime.now(timezone.utc).isoformat()

    return {
        'success': True,
        'session_id': session_id,
        'message': assistant_response,
        'conversation_length': len(session['messages']),
        'timestamp': assistant_message['timestamp']
    }


def _handle_memory_enabled_message(session_id, session, user_message, stream, store_memory=True, top_k=10, min_similarity=0.9):
    """Handle message processing for sessions with memory enabled."""
    if not memory_agent:
        raise HTTPException(status_code=500, detail='Memory agent not initialized but session requires memory')
    
    # Temporarily disable caching for agent sessions to ensure fresh memory retrieval
    # Store original processing module and replace with non-cached version during session
    original_processing = None
    if hasattr(memory_agent.memory_agent, 'processing'):
        original_processing = memory_agent.memory_agent.processing
        # Import the non-optimized version
        from memory.processing import MemoryProcessing
        memory_agent.memory_agent.processing = MemoryProcessing()
        print("🚫 Temporarily disabled memory processing cache for agent session")
    
    try:
        # Create user message object
        user_message_obj = {
            'role': 'user',
            'content': user_message,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Add user message to session messages for conversation history
        session['messages'].append(user_message_obj)

        # Add to conversation buffer for memory extraction
        if 'conversation_buffer' not in session:
            session['conversation_buffer'] = []

        session['conversation_buffer'].append(user_message_obj)

        # Retrieve relevant memories for context using advanced filtering with embedding optimization
        relevant_memories = []
        try:
            print(f"2) Searching memories for: '{user_message[:60]}{'...' if len(user_message) > 60 else ''}' (top_k: {top_k})")

            # Use embedding optimization for better vector similarity search
            validation_result = memory_agent.memory_agent.processing.validate_and_preprocess_question(user_message)

            if validation_result["type"] == "search":
                # Use the embedding-optimized query for vector search
                search_query = validation_result.get("embedding_query") or validation_result["content"]
                print(f"2b) Using optimized search query: '{search_query}'")

                search_result = memory_agent.memory_agent.search_memories_with_filtering_info(
                    query=search_query,
                    top_k=top_k,
                    min_similarity=min_similarity
                )
                memory_results = search_result['memories']
                filtering_info = search_result['filtering_info']
            else:
                # For help queries, still search but with original message
                search_result = memory_agent.memory_agent.search_memories_with_filtering_info(
                    query=user_message,
                    top_k=top_k,
                    min_similarity=min_similarity
                )
                memory_results = search_result['memories']
                filtering_info = search_result['filtering_info']

            if memory_results:
                print(f"3) Found {len(memory_results)} memories, sending directly to LLM (no pre-filtering)")
                relevant_memories = memory_results
            else:
                print(f"3) No relevant memories found")
                relevant_memories = []
        except Exception as e:
            print(f"⚠️ Memory search failed: {e}")

        # Prepare enhanced system prompt with memory context
        enhanced_system_prompt = session['system_prompt']
        if relevant_memories:
            memory_context = "\n\n==== MEMORY CONTEXT ====\n"
            memory_context += "Here are potentially relevant memories from previous interactions:\n\n"
            for i, memory in enumerate(relevant_memories, 1):
                # Include similarity score and timestamp for context
                score_percent = memory.get('score', 0) * 100
                timestamp = memory.get('formatted_time', 'Unknown time')
                memory_context += f"Memory {i} (Similarity: {score_percent:.1f}%, {timestamp}):\n{memory['text']}\n\n"

            memory_context += "INSTRUCTIONS FOR USING MEMORIES:\n"
            memory_context += "- Only use memories that are directly relevant to the current request\n"
            memory_context += "- Ignore memories that don't relate to the user's current question or need\n"
            memory_context += "- Use relevant memories to provide personalized, contextual responses\n"
            memory_context += "- Consider the user's preferences, constraints, and past experiences when applicable\n"
            memory_context += "- If no memories are relevant, respond based on the current conversation only\n"
            memory_context += "==== END MEMORY CONTEXT ====\n"
            enhanced_system_prompt += memory_context

        # Prepare messages for LLM
        llm_messages = [
            {'role': 'system', 'content': enhanced_system_prompt}
        ]

        # Add recent conversation history (limit to last 10 messages to avoid token limits)
        recent_messages = session['messages'][-10:]
        for msg in recent_messages:
            llm_messages.append({
                'role': msg['role'],
                'content': msg['content']
            })

        # Get response from Tier 1 LLM (primary conversational)
        try:
            llm_manager = get_llm_manager()
            tier1_client = llm_manager.get_tier1_client()
            print(f"4) Sending message to Tier 1 LLM: {llm_messages}")
            response = tier1_client.chat_completion(
                messages=llm_messages,
                temperature=session['config'].get('temperature', 0.7),
                max_tokens=session['config'].get('max_tokens', 1000),
                bypass_cache=True  # Disable cache for agent sessions to ensure fresh memory retrieval
            )

            if stream:
                # TODO: Implement streaming response
                raise HTTPException(status_code=501, detail='Streaming not yet implemented')

            assistant_response = response['content']
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'LLM error: {str(e)}')

        # Add assistant response to session and buffer
        assistant_message = {
            'role': 'assistant',
            'content': assistant_response,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        session['messages'].append(assistant_message)
        session['conversation_buffer'].append(assistant_message)
        session['last_activity'] = datetime.now(timezone.utc).isoformat()

        # Check if we should extract memories (only if store_memory is True)
        if store_memory:
            _check_and_extract_memories(session_id, session)
        else:
            print(f"🧠 [{session_id}] MEMORY: Skipping memory extraction (store_memory=False)")

        response_data = {
            'success': True,
            'session_id': session_id,
            'message': assistant_response,
            'conversation_length': len(session['messages']),
            'timestamp': assistant_message['timestamp']
        }

        # Add memory info if memories were used
        if relevant_memories:
            response_data['memory_context'] = {
                'memories_used': len(relevant_memories),
                'memories': relevant_memories,  # Include all memory details for rendering
                'filtering_info': filtering_info if 'filtering_info' in locals() else None
            }
        print(f"7) memories_used: {len(relevant_memories)}")
        return response_data
    
    finally:
        # Restore original processing module if it was replaced
        if original_processing is not None:
            memory_agent.memory_agent.processing = original_processing
            print("✅ Restored original memory processing module")


def _check_and_extract_memories(session_id, session):
    """Extract memories from the most recent user message."""
    buffer = session.get('conversation_buffer', [])

    # Get the most recent user message
    recent_user_messages = [msg for msg in buffer if msg['role'] == 'user']
    if not recent_user_messages:
        print(f"🧠 [{session_id}] MEMORY: No user messages to extract from")
        return

    # Get the latest user message
    latest_user_message = recent_user_messages[-1]

    print(f"4) Extracting memories from: '{latest_user_message['content'][:60]}{'...' if len(latest_user_message['content']) > 60 else ''}'")

    try:
        # Format the latest user message for extraction
        # We only want to extract user preferences, facts, and constraints, not LLM suggestions
        conversation_text = f"User: {latest_user_message['content']}\n"

        # STEP 1: Search for existing relevant memories first (context-aware approach)
        print(f"1) Searching for existing memories related to: '{latest_user_message['content'][:50]}...'")
        existing_memories = memory_agent.memory_agent.search_memories(
            latest_user_message['content'],
            top_k=10,
            min_similarity=0.7
        )

        if existing_memories:
            print(f"2) Found {len(existing_memories)} existing relevant memories")
        else:
            print(f"2) No existing relevant memories found")

        # STEP 2: Extract memories using context-aware approach
        result = memory_agent.memory_agent.extract_and_store_memories(
            raw_input=conversation_text,
            context_prompt=session.get('memory_context', 'Extract ONLY user preferences, constraints, facts, and important personal details from the user messages. Do NOT extract assistant suggestions or recommendations.'),
            apply_grounding=True,
            existing_memories=existing_memories  # Pass existing memories for context-aware extraction
        )

        if result["total_extracted"] > 0:
            extracted_memories = result.get("extracted_memories", [])
            memory_texts = [mem.get("final_text", mem.get("raw_text", "Unknown")) for mem in extracted_memories]
            print(f"5) Identified {result['total_extracted']} memories: {', '.join([f'"{text[:40]}{"..." if len(text) > 40 else ""}"' for text in memory_texts])}")
            print(f"6) Saved {result['total_extracted']} memories to database")
            session['last_extraction'] = datetime.now(timezone.utc).isoformat()
        else:
            print(f"5) No memories identified for extraction")

        # Since we process each message individually, we can keep a smaller buffer
        # Keep only the last 4 messages (2 exchanges) for context
        session['conversation_buffer'] = buffer[-4:]

    except Exception as e:
        print(f"⚠️ Memory extraction failed: {e}")




@app.get('/api/agent/session/{session_id}')
async def api_get_agent_session(session_id: str = Path(..., description="The agent session ID")):
    """Get agent session information and conversation history.

    Args:
        session_id: The agent session ID

    Returns:
        JSON with session details and message history
    """
    try:
        if not hasattr(app, 'chat_sessions') or session_id not in app.chat_sessions:
            raise HTTPException(status_code=404, detail='Agent session not found')

        session = app.chat_sessions[session_id]

        response_data = {
            'success': True,
            'session_id': session_id,
            'system_prompt': session['system_prompt'],
            'messages': session['messages'],
            'config': session['config'],
            'created_at': session['created_at'],
            'last_activity': session['last_activity'],
            'message_count': len(session['messages']),
            'use_memory': session.get('use_memory', False)
        }

        # Add memory-specific information if memory is enabled
        if session.get('use_memory', False):
            response_data.update({
                'memory_info': {
                    'extraction_threshold': session.get('extraction_threshold', 2),
                    'last_extraction': session.get('last_extraction'),
                    'buffer_size': len(session.get('conversation_buffer', []))
                }
            })

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete('/api/agent/session/{session_id}')
async def api_delete_agent_session(session_id: str = Path(..., description="The agent session ID")):
    """Delete an agent session.

    Args:
        session_id: The agent session ID

    Returns:
        JSON confirmation of deletion
    """
    try:
        if not hasattr(app, 'chat_sessions') or session_id not in app.chat_sessions:
            raise HTTPException(status_code=404, detail='Agent session not found')

        del app.chat_sessions[session_id]

        print(f"🗑️ AGENT API: Deleted agent session {session_id}")

        return {
            'success': True,
            'message': f'Agent session {session_id} deleted'
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/agent/sessions')
async def api_list_agent_sessions():
    """List all active agent sessions.

    Returns:
        JSON with list of active sessions
    """
    try:
        if not hasattr(app, 'chat_sessions'):
            app.chat_sessions = {}

        sessions = []
        for session_id, session in app.chat_sessions.items():
            sessions.append({
                'session_id': session_id,
                'created_at': session['created_at'],
                'last_activity': session['last_activity'],
                'message_count': len(session['messages']),
                'system_prompt_preview': session['system_prompt'][:100] + '...' if len(session['system_prompt']) > 100 else session['system_prompt']
            })

        return {
            'success': True,
            'sessions': sessions,
            'total_sessions': len(sessions)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# SYSTEM APIs - Health checks and status
# =============================================================================

@app.get('/api/health')
async def api_health():
    """System health check."""
    return {
        'status': 'healthy' if memory_agent else 'unhealthy',
        'service': 'LangGraph Memory Agent API',
        'timestamp': datetime.now(timezone.utc).isoformat()
    }

# =============================================================================
# LLM APIs - LLM provider management and model information
# =============================================================================

# Simple in-memory cache for Ollama models
ollama_models_cache = {
    'data': None,
    'timestamp': None,
    'ttl': 45  # Cache for 45 seconds
}

@app.get('/api/llm/config')
async def api_get_llm_config():
    """Get current LLM configuration.

    Returns:
        JSON with LLM configuration including tier1 and tier2 settings
    """
    try:
        # Create a safe copy of LLM config without sensitive data
        safe_llm_config = json.loads(json.dumps(app_config["llm"]))

        # Mask API keys
        if safe_llm_config["tier1"]["api_key"]:
            safe_llm_config["tier1"]["api_key"] = safe_llm_config["tier1"]["api_key"][:8] + "..." + safe_llm_config["tier1"]["api_key"][-4:]
        if safe_llm_config["tier2"]["api_key"]:
            safe_llm_config["tier2"]["api_key"] = safe_llm_config["tier2"]["api_key"][:8] + "..." + safe_llm_config["tier2"]["api_key"][-4:]

        # Add runtime information
        runtime_info = {
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Add LLM manager status
        try:
            llm_mgr = get_llm_manager()
            runtime_info["llm_manager_initialized"] = True
            runtime_info["tier1_provider"] = llm_mgr.tier1_config.provider
            runtime_info["tier2_provider"] = llm_mgr.tier2_config.provider
            runtime_info["tier1_model"] = llm_mgr.tier1_config.model
            runtime_info["tier2_model"] = llm_mgr.tier2_config.model
        except Exception:
            runtime_info["llm_manager_initialized"] = False

        return {
            'success': True,
            'llm_config': safe_llm_config,
            'runtime': runtime_info
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put('/api/llm/config')
async def api_update_llm_config(request: LLMConfigUpdate):
    """Update LLM configuration.

    Returns:
        JSON with success status, updated configuration, and any warnings
    """
    try:
        data = request.dict(exclude_unset=True)

        if not data:
            raise HTTPException(status_code=400, detail='LLM configuration data is required')

        warnings = []
        changes_made = []
        requires_restart = False

        # Update LLM configuration
        for tier in ['tier1', 'tier2']:
            if tier in data:
                tier_config = data[tier]

                # Handle string fields
                for key in ['provider', 'model', 'base_url']:
                    if key in tier_config:
                        old_value = app_config['llm'][tier].get(key)
                        new_value = tier_config[key]

                        if old_value != new_value:
                            app_config['llm'][tier][key] = new_value
                            changes_made.append(f"llm.{tier}.{key}: {old_value} → {new_value}")
                            requires_restart = True

                # Handle API key with masking
                if 'api_key' in tier_config:
                    old_value = app_config['llm'][tier].get('api_key')
                    new_value = tier_config['api_key']

                    if old_value != new_value:
                        app_config['llm'][tier]['api_key'] = new_value
                        masked_old = old_value[:8] + "..." + old_value[-4:] if old_value else "None"
                        masked_new = new_value[:8] + "..." + new_value[-4:] if new_value else "None"
                        changes_made.append(f"llm.{tier}.api_key: {masked_old} → {masked_new}")
                        requires_restart = True

                # Handle numeric fields
                for key in ['temperature', 'max_tokens', 'timeout']:
                    if key in tier_config:
                        try:
                            old_value = app_config['llm'][tier].get(key)
                            if key == 'temperature':
                                new_value = float(tier_config[key])
                            else:
                                new_value = int(tier_config[key])

                            if old_value != new_value:
                                app_config['llm'][tier][key] = new_value
                                changes_made.append(f"llm.{tier}.{key}: {old_value} → {new_value}")
                                requires_restart = True
                        except (ValueError, TypeError):
                            raise HTTPException(status_code=400, detail=f'LLM {tier}.{key} must be a number')

                # Validate provider
                if 'provider' in tier_config:
                    provider = tier_config['provider'].lower()
                    if provider not in ['openai', 'ollama']:
                        raise HTTPException(status_code=400, detail=f'LLM provider must be "openai" or "ollama", got "{provider}"')

        # Reinitialize LLM manager if changes were made
        llm_reinitialized = False
        llm_reinit_error = None
        
        if changes_made:
            print(f"🔄 LLM configuration changed, reinitializing LLM manager...")
            success, message = reinitialize_llm_manager_with_optimizations()
            llm_reinitialized = success
            if not success:
                llm_reinit_error = message
                warnings.append(f"LLM reinitialization failed: {message}")

        # Prepare response
        response_data = {
            'success': True,
            'changes_made': changes_made,
            'llm_reinitialized': llm_reinitialized,
            'warnings': warnings
        }

        if llm_reinit_error:
            response_data['llm_reinit_error'] = llm_reinit_error

        if changes_made and llm_reinitialized:
            response_data['message'] = 'LLM configuration updated and applied successfully.'
        elif changes_made and not llm_reinitialized:
            response_data['message'] = 'LLM configuration updated but failed to apply. Manual restart may be required.'
        elif changes_made:
            response_data['message'] = 'LLM configuration updated successfully.'
        else:
            response_data['message'] = 'No changes were made to the LLM configuration.'

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/llm/ollama/models')
async def api_get_ollama_models(base_url: str = Query('http://localhost:11434', description="Ollama server base URL")):
    """Get available models from Ollama instance.

    Args:
        base_url: Ollama server base URL (default: http://localhost:11434)

    Returns:
        JSON with success status and models array containing model information:
        - success (bool): Whether the operation succeeded
        - models (list): Array of model objects with name, size, digest, modified_at, details
        - count (int): Number of models available
        - base_url (str): The Ollama server URL used
        - cached (bool): Whether the response was served from cache
    """
    try:
        base_url = base_url.rstrip('/')
        cache_key = base_url  # Use base_url as cache key for different instances
        
        # Check cache first
        current_time = datetime.now(timezone.utc)
        cache_data = ollama_models_cache.get('data')
        cache_timestamp = ollama_models_cache.get('timestamp')
        cache_base_url = ollama_models_cache.get('base_url')
        
        if (cache_data and cache_timestamp and cache_base_url == base_url and
            (current_time - cache_timestamp).total_seconds() < ollama_models_cache['ttl']):
            print(f"🎯 LLM API: Serving Ollama models from cache (age: {(current_time - cache_timestamp).total_seconds():.1f}s)")
            return {
                'success': True,
                'models': cache_data,
                'count': len(cache_data),
                'base_url': base_url,
                'cached': True,
                'cache_age_seconds': (current_time - cache_timestamp).total_seconds()
            }

        print(f"🦙 LLM API: Fetching available models from Ollama at {base_url}")
        
        # Make request to Ollama /api/tags endpoint
        ollama_url = f"{base_url}/api/tags"
        
        try:
            response = requests.get(ollama_url, timeout=10)
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise HTTPException(
                status_code=503,
                detail={
                    'success': False,
                    'error': f'Unable to connect to Ollama server at {base_url}',
                    'message': 'Please ensure Ollama is running and accessible',
                    'base_url': base_url
                }
            )
        except requests.exceptions.Timeout:
            raise HTTPException(
                status_code=504,
                detail={
                    'success': False,
                    'error': f'Timeout connecting to Ollama server at {base_url}',
                    'message': 'Ollama server did not respond within 10 seconds',
                    'base_url': base_url
                }
            )
        except requests.exceptions.HTTPError as e:
            raise HTTPException(
                status_code=502,
                detail={
                    'success': False,
                    'error': f'HTTP error from Ollama server: {e.response.status_code}',
                    'message': f'Ollama server returned an error: {e.response.text}',
                    'base_url': base_url
                }
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={
                    'success': False,
                    'error': f'Unexpected error connecting to Ollama: {str(e)}',
                    'base_url': base_url
                }
            )

        # Parse Ollama response
        try:
            ollama_data = response.json()
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=502,
                detail={
                    'success': False,
                    'error': 'Invalid JSON response from Ollama server',
                    'message': 'Ollama server returned malformed JSON',
                    'base_url': base_url
                }
            )

        # Transform Ollama response to expected format
        models = []
        ollama_models = ollama_data.get('models', [])
        
        for model in ollama_models:
            # Extract model information and transform to frontend format
            transformed_model = {
                'name': model.get('name', 'unknown'),
                'size': model.get('size', 0),
                'digest': model.get('digest', ''),
                'modified_at': model.get('modified_at', ''),
                'details': model.get('details', {})
            }
            
            # Ensure details is a dictionary and has expected fields
            if not isinstance(transformed_model['details'], dict):
                transformed_model['details'] = {}
            
            # Add additional computed fields for frontend convenience
            transformed_model['size_gb'] = round(transformed_model['size'] / (1024**3), 2) if transformed_model['size'] > 0 else 0
            transformed_model['family'] = transformed_model['details'].get('family', 'unknown')
            transformed_model['parameter_size'] = transformed_model['details'].get('parameter_size', 'unknown')
            
            models.append(transformed_model)

        # Update cache
        ollama_models_cache.update({
            'data': models,
            'timestamp': current_time,
            'base_url': base_url
        })

        print(f"🦙 LLM API: Found {len(models)} models, cached for {ollama_models_cache['ttl']} seconds")

        return {
            'success': True,
            'models': models,
            'count': len(models),
            'base_url': base_url,
            'cached': False,
            'cache_ttl_seconds': ollama_models_cache['ttl']
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                'success': False,
                'error': str(e),
                'message': 'Internal server error while fetching Ollama models'
            }
        )

# =============================================================================
# PERFORMANCE APIs - Performance monitoring and optimization
# =============================================================================

@app.get('/api/performance/metrics')
async def api_get_performance_metrics():
    """Get current performance metrics and cache statistics.

    Returns:
        JSON with performance metrics, cache statistics, and optimization status
    """
    try:
        optimizer = get_performance_optimizer()

        if not optimizer:
            raise HTTPException(
                status_code=500,
                detail='Performance optimizer not initialized'
            )

        metrics = optimizer.get_performance_metrics()

        return {
            'success': True,
            'performance_metrics': metrics,
            'configuration': {
                'optimization_enabled': app_config["performance"]["optimization_enabled"],
                'cache_enabled': app_config["performance"]["cache_enabled"],
                'batch_processing_enabled': app_config["performance"]["batch_processing_enabled"],
                'cache_default_ttl': app_config["performance"]["cache_default_ttl"]
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/api/performance/cache/clear')
async def api_clear_performance_cache(request: Optional[CacheClearRequest] = None):
    """Clear performance cache entries.

    Args:
        request: Optional request body with pattern and operation_type
            - pattern (str): Cache pattern to clear (for hash-based cache)
            - operation_type (str): Operation type to clear (for semantic cache)
                                   Options: query_optimization, memory_relevance, context_analysis,
                                           memory_grounding, extraction_evaluation, conversation, answer_generation
            If neither provided, clears all cache entries

    Returns:
        JSON with clearing results
    """
    try:
        optimizer = get_performance_optimizer()

        if not optimizer:
            raise HTTPException(
                status_code=500,
                detail='Performance optimizer not initialized'
            )

        pattern = request.pattern if request else None
        operation_type = request.operation_type if request else None

        result = optimizer.clear_cache(pattern=pattern, operation_type=operation_type)

        return {
            'success': True,
            **result,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/performance/cache/analyze')
async def api_analyze_cache_effectiveness():
    """Analyze cache effectiveness and provide optimization recommendations.

    Returns:
        JSON with cache analysis and recommendations
    """
    try:
        optimizer = get_performance_optimizer()

        if not optimizer:
            raise HTTPException(
                status_code=500,
                detail='Performance optimizer not initialized'
            )

        analysis = optimizer.analyze_cache_effectiveness()

        return {
            'success': True,
            'cache_analysis': analysis,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete('/api/performance/cache/memory-evaluation')
async def api_clear_memory_evaluation_cache():
    """Clear semantic cache entries for memory evaluation to resolve false cache hits.

    Returns:
        JSON with clearing results
    """
    try:
        optimizer = get_performance_optimizer()

        if not optimizer:
            raise HTTPException(
                status_code=500,
                detail='Performance optimizer not initialized'
            )

        result = optimizer.clear_cache(operation_type='memory_extraction_evaluation')

        return {
            'success': True,
            'message': 'Memory evaluation cache cleared',
            **result,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# CONFIGURATION MANAGEMENT APIs - Runtime configuration management
# =============================================================================

@app.get('/api/config')
async def api_get_config():
    """Get current system configuration.

    Returns:
        JSON with complete system configuration including Redis, OpenAI, LangGraph,
        memory agent, and web server settings
    """
    try:
        # Create a safe copy of config without sensitive data
        safe_config = json.loads(json.dumps(app_config))

        # Mask sensitive information
        if safe_config["openai"]["api_key"]:
            safe_config["openai"]["api_key"] = safe_config["openai"]["api_key"][:8] + "..." + safe_config["openai"]["api_key"][-4:]

        # Mask LLM API keys
        if safe_config["llm"]["tier1"]["api_key"]:
            safe_config["llm"]["tier1"]["api_key"] = safe_config["llm"]["tier1"]["api_key"][:8] + "..." + safe_config["llm"]["tier1"]["api_key"][-4:]
        if safe_config["llm"]["tier2"]["api_key"]:
            safe_config["llm"]["tier2"]["api_key"] = safe_config["llm"]["tier2"]["api_key"][:8] + "..." + safe_config["llm"]["tier2"]["api_key"][-4:]

        # Add runtime information
        runtime_info = {
            "memory_agent_initialized": memory_agent is not None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Add LLM manager status
        try:
            llm_mgr = get_llm_manager()
            runtime_info["llm_manager_initialized"] = True
            runtime_info["llm_tier1_provider"] = llm_mgr.tier1_config.provider
            runtime_info["llm_tier2_provider"] = llm_mgr.tier2_config.provider
        except Exception:
            runtime_info["llm_manager_initialized"] = False

        if memory_agent:
            try:
                memory_info = memory_agent.memory_agent.get_memory_info()
                runtime_info["memory_count"] = memory_info.get("memory_count", 0)
                runtime_info["redis_connected"] = True
                runtime_info["actual_redis_host"] = memory_info.get("redis_host", "unknown")
                runtime_info["actual_redis_port"] = memory_info.get("redis_port", "unknown")
            except Exception as e:
                runtime_info["redis_connected"] = False
                runtime_info["redis_error"] = str(e)

        return {
            'success': True,
            'config': safe_config,
            'runtime': runtime_info
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail={'error': str(e)})

@app.put('/api/config')
async def api_update_config(request: ConfigUpdateRequest):
    """Update system configuration.

    Args:
        request: Configuration object with any subset of configuration categories:
            - redis: {host, port, db, vectorset_key}
            - llm: {tier1: {provider, model, temperature, max_tokens, base_url, api_key, timeout}, tier2: {...}}
            - openai: {api_key, organization, embedding_model, embedding_dimension, chat_model, temperature}
            - langgraph: {model_name, temperature, system_prompt_enabled}
            - memory_agent: {default_top_k, apply_grounding_default, validation_enabled}
            - web_server: {host, port, debug, cors_enabled}
            - performance: {cache_enabled, optimization_enabled, etc.}

    Returns:
        JSON with success status, updated configuration, and any warnings
    """
    try:
        data = request.dict(exclude_unset=True)

        if not data:
            raise HTTPException(status_code=400, detail='Configuration data is required')

        warnings = []
        changes_made = []
        requires_restart = False

        # Update Redis configuration
        if 'redis' in data:
            redis_config = data['redis']
            for key in ['host', 'port', 'db', 'vectorset_key']:
                if key in redis_config:
                    old_value = app_config['redis'].get(key)
                    new_value = redis_config[key]

                    # Validate port and db are integers
                    if key in ['port', 'db']:
                        try:
                            new_value = int(new_value)
                        except (ValueError, TypeError):
                            raise HTTPException(status_code=400, detail=f'Redis {key} must be an integer')

                    if old_value != new_value:
                        app_config['redis'][key] = new_value
                        changes_made.append(f"redis.{key}: {old_value} → {new_value}")
                        requires_restart = True

        # Update LLM configuration
        if 'llm' in data:
            llm_config = data['llm']
            for tier in ['tier1', 'tier2']:
                if tier in llm_config:
                    tier_config = llm_config[tier]

                    # Handle string fields
                    for key in ['provider', 'model', 'base_url']:
                        if key in tier_config:
                            old_value = app_config['llm'][tier].get(key)
                            new_value = tier_config[key]

                            if old_value != new_value:
                                app_config['llm'][tier][key] = new_value
                                changes_made.append(f"llm.{tier}.{key}: {old_value} → {new_value}")
                                requires_restart = True

                    # Handle API key with masking
                    if 'api_key' in tier_config:
                        old_value = app_config['llm'][tier].get('api_key')
                        new_value = tier_config['api_key']

                        if old_value != new_value:
                            app_config['llm'][tier]['api_key'] = new_value
                            masked_old = old_value[:8] + "..." + old_value[-4:] if old_value else "None"
                            masked_new = new_value[:8] + "..." + new_value[-4:] if new_value else "None"
                            changes_made.append(f"llm.{tier}.api_key: {masked_old} → {masked_new}")
                            requires_restart = True

                    # Handle numeric fields
                    for key in ['temperature', 'max_tokens', 'timeout']:
                        if key in tier_config:
                            try:
                                old_value = app_config['llm'][tier].get(key)
                                if key == 'temperature':
                                    new_value = float(tier_config[key])
                                else:
                                    new_value = int(tier_config[key])

                                if old_value != new_value:
                                    app_config['llm'][tier][key] = new_value
                                    changes_made.append(f"llm.{tier}.{key}: {old_value} → {new_value}")
                                    requires_restart = True
                            except (ValueError, TypeError):
                                raise HTTPException(status_code=400, detail=f'LLM {tier}.{key} must be a number')

                    # Validate provider
                    if 'provider' in tier_config:
                        provider = tier_config['provider'].lower()
                        if provider not in ['openai', 'ollama']:
                            raise HTTPException(status_code=400, detail=f'LLM provider must be "openai" or "ollama", got "{provider}"')

        # Update OpenAI configuration
        if 'openai' in data:
            openai_config = data['openai']
            for key in ['api_key', 'organization', 'embedding_model', 'chat_model']:
                if key in openai_config:
                    old_value = app_config['openai'].get(key)
                    new_value = openai_config[key]

                    if old_value != new_value:
                        app_config['openai'][key] = new_value
                        if key == 'api_key':
                            # Mask API key in logs
                            masked_old = old_value[:8] + "..." + old_value[-4:] if old_value else "None"
                            masked_new = new_value[:8] + "..." + new_value[-4:] if new_value else "None"
                            changes_made.append(f"openai.{key}: {masked_old} → {masked_new}")
                        else:
                            changes_made.append(f"openai.{key}: {old_value} → {new_value}")
                        requires_restart = True

            # Handle numeric fields
            for key in ['embedding_dimension', 'temperature']:
                if key in openai_config:
                    try:
                        old_value = app_config['openai'].get(key)
                        new_value = float(openai_config[key]) if key == 'temperature' else int(openai_config[key])

                        if old_value != new_value:
                            app_config['openai'][key] = new_value
                            changes_made.append(f"openai.{key}: {old_value} → {new_value}")
                            requires_restart = True
                    except (ValueError, TypeError):
                        raise HTTPException(status_code=400, detail=f'OpenAI {key} must be a number')

        # Update LangGraph configuration
        if 'langgraph' in data:
            langgraph_config = data['langgraph']
            for key in ['model_name']:
                if key in langgraph_config:
                    old_value = app_config['langgraph'].get(key)
                    new_value = langgraph_config[key]

                    if old_value != new_value:
                        app_config['langgraph'][key] = new_value
                        changes_made.append(f"langgraph.{key}: {old_value} → {new_value}")
                        requires_restart = True

            # Handle numeric and boolean fields
            if 'temperature' in langgraph_config:
                try:
                    old_value = app_config['langgraph'].get('temperature')
                    new_value = float(langgraph_config['temperature'])

                    if old_value != new_value:
                        app_config['langgraph']['temperature'] = new_value
                        changes_made.append(f"langgraph.temperature: {old_value} → {new_value}")
                        requires_restart = True
                except (ValueError, TypeError):
                    raise HTTPException(status_code=400, detail='LangGraph temperature must be a number')

            if 'system_prompt_enabled' in langgraph_config:
                old_value = app_config['langgraph'].get('system_prompt_enabled')
                new_value = bool(langgraph_config['system_prompt_enabled'])

                if old_value != new_value:
                    app_config['langgraph']['system_prompt_enabled'] = new_value
                    changes_made.append(f"langgraph.system_prompt_enabled: {old_value} → {new_value}")
                    # This doesn't require restart, just affects future conversations

        # Update Memory Agent configuration
        if 'memory_agent' in data:
            memory_config = data['memory_agent']

            if 'default_top_k' in memory_config:
                try:
                    old_value = app_config['memory_agent'].get('default_top_k')
                    new_value = int(memory_config['default_top_k'])

                    if old_value != new_value:
                        app_config['memory_agent']['default_top_k'] = new_value
                        changes_made.append(f"memory_agent.default_top_k: {old_value} → {new_value}")
                except (ValueError, TypeError):
                    raise HTTPException(status_code=400, detail='Memory agent default_top_k must be an integer')

            for key in ['apply_grounding_default', 'validation_enabled']:
                if key in memory_config:
                    old_value = app_config['memory_agent'].get(key)
                    new_value = bool(memory_config[key])

                    if old_value != new_value:
                        app_config['memory_agent'][key] = new_value
                        changes_made.append(f"memory_agent.{key}: {old_value} → {new_value}")

        # Update Web Server configuration
        if 'web_server' in data:
            web_config = data['web_server']

            for key in ['host']:
                if key in web_config:
                    old_value = app_config['web_server'].get(key)
                    new_value = web_config[key]

                    if old_value != new_value:
                        app_config['web_server'][key] = new_value
                        changes_made.append(f"web_server.{key}: {old_value} → {new_value}")
                        warnings.append(f"Web server {key} change requires application restart to take effect")

            if 'port' in web_config:
                try:
                    old_value = app_config['web_server'].get('port')
                    new_value = int(web_config['port'])

                    if old_value != new_value:
                        app_config['web_server']['port'] = new_value
                        changes_made.append(f"web_server.port: {old_value} → {new_value}")
                        warnings.append("Web server port change requires application restart to take effect")
                except (ValueError, TypeError):
                    raise HTTPException(status_code=400, detail='Web server port must be an integer')

            for key in ['debug', 'cors_enabled']:
                if key in web_config:
                    old_value = app_config['web_server'].get(key)
                    new_value = bool(web_config[key])

                    if old_value != new_value:
                        app_config['web_server'][key] = new_value
                        changes_made.append(f"web_server.{key}: {old_value} → {new_value}")
                        warnings.append(f"Web server {key} change requires application restart to take effect")

        # Update Performance configuration
        if 'performance' in data:
            perf_config = data['performance']

            for key in ['cache_enabled', 'optimization_enabled', 'batch_processing_enabled', 'use_semantic_cache']:
                if key in perf_config:
                    old_value = app_config['performance'].get(key)
                    new_value = bool(perf_config[key])

                    if old_value != new_value:
                        app_config['performance'][key] = new_value
                        changes_made.append(f"performance.{key}: {old_value} → {new_value}")
                        if key in ['optimization_enabled', 'cache_enabled', 'use_semantic_cache']:
                            requires_restart = True

            for key in ['cache_default_ttl']:
                if key in perf_config:
                    try:
                        old_value = app_config['performance'].get(key)
                        new_value = int(perf_config[key])

                        if old_value != new_value:
                            app_config['performance'][key] = new_value
                            changes_made.append(f"performance.{key}: {old_value} → {new_value}")
                    except (ValueError, TypeError):
                        raise HTTPException(status_code=400, detail=f'Performance {key} must be an integer')

            if 'semantic_similarity_threshold' in perf_config:
                try:
                    old_value = app_config['performance'].get('semantic_similarity_threshold')
                    new_value = float(perf_config['semantic_similarity_threshold'])

                    if old_value != new_value:
                        app_config['performance']['semantic_similarity_threshold'] = new_value
                        changes_made.append(f"performance.semantic_similarity_threshold: {old_value} → {new_value}")
                except (ValueError, TypeError):
                    raise HTTPException(status_code=400, detail='Performance semantic_similarity_threshold must be a float')

            if 'cache_ttl_settings' in perf_config:
                ttl_settings = perf_config['cache_ttl_settings']
                if isinstance(ttl_settings, dict):
                    for ttl_key, ttl_value in ttl_settings.items():
                        try:
                            old_value = app_config['performance']['cache_ttl_settings'].get(ttl_key)
                            new_value = int(ttl_value)

                            if old_value != new_value:
                                app_config['performance']['cache_ttl_settings'][ttl_key] = new_value
                                changes_made.append(f"performance.cache_ttl_settings.{ttl_key}: {old_value} → {new_value}")
                        except (ValueError, TypeError):
                            raise HTTPException(status_code=400, detail=f'Performance cache TTL {ttl_key} must be an integer')

        # Check if LLM configuration was changed and reinitialize if needed
        llm_reinitialized = False
        llm_reinit_error = None
        llm_config_changed = any('llm.' in change for change in changes_made)
        
        if llm_config_changed:
            print(f"🔄 LLM configuration changed, reinitializing LLM manager...")
            success, message = reinitialize_llm_manager_with_optimizations()
            llm_reinitialized = success
            if not success:
                llm_reinit_error = message
                warnings.append(f"LLM reinitialization failed: {message}")
            else:
                # If LLM reinitialization was successful, we don't need a full restart for LLM changes
                requires_restart = any(change for change in changes_made if not change.startswith('llm.'))

        # Prepare response
        response_data = {
            'success': True,
            'changes_made': changes_made,
            'requires_restart': requires_restart,
            'warnings': warnings
        }

        # Add LLM reinitialization info if LLM config was changed
        if llm_config_changed:
            response_data['llm_reinitialized'] = llm_reinitialized
            if llm_reinit_error:
                response_data['llm_reinit_error'] = llm_reinit_error

        if requires_restart:
            response_data['message'] = 'Configuration updated. Memory agent restart required for changes to take effect.'
        elif changes_made and llm_config_changed and llm_reinitialized:
            response_data['message'] = 'Configuration updated and LLM changes applied successfully.'
        elif changes_made and llm_config_changed and not llm_reinitialized:
            response_data['message'] = 'Configuration updated but LLM changes failed to apply. Manual restart may be required.'
        elif changes_made:
            response_data['message'] = 'Configuration updated successfully.'
        else:
            response_data['message'] = 'No changes were made to the configuration.'

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={'error': str(e)})

@app.post('/api/config/reload')
async def api_reload_config():
    """Reload configuration and restart the memory agent.

    This endpoint reinitializes the memory agent with the current configuration.
    Useful after making configuration changes that require a restart.

    Returns:
        JSON with success status and reload details
    """
    try:
        global memory_agent

        print("🔄 Reloading configuration and restarting memory agent...")

        # Store old agent state for comparison
        old_agent_initialized = memory_agent is not None
        old_memory_count = 0

        if memory_agent:
            try:
                memory_info = memory_agent.memory_agent.get_memory_info()
                old_memory_count = memory_info.get("memory_count", 0)
            except:
                pass

        # Reinitialize the LLM manager with current configuration
        llm_success = init_llm_manager()

        # Reinitialize the memory agent with current configuration
        success = init_memory_agent() and llm_success

        if success:
            # Get new agent state
            new_memory_count = 0
            redis_connected = False

            try:
                memory_info = memory_agent.memory_agent.get_memory_info()
                new_memory_count = memory_info.get("memory_count", 0)
                redis_connected = True
            except Exception as e:
                print(f"Warning: Could not get memory info after reload: {e}")

            return {
                'success': True,
                'message': 'Configuration reloaded and memory agent restarted successfully',
                'reload_details': {
                    'agent_was_initialized': old_agent_initialized,
                    'agent_now_initialized': True,
                    'redis_connected': redis_connected,
                    'memory_count_before': old_memory_count,
                    'memory_count_after': new_memory_count,
                    'config_applied': {
                        'redis_host': app_config['redis']['host'],
                        'redis_port': app_config['redis']['port'],
                        'redis_db': app_config['redis']['db'],
                        'langgraph_model': app_config['langgraph']['model_name'],
                        'langgraph_temperature': app_config['langgraph']['temperature']
                    }
                },
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail={
                    'success': False,
                    'error': 'Failed to reinitialize memory agent with new configuration',
                    'message': 'Check Redis connection and OpenAI API key',
                    'reload_details': {
                        'agent_was_initialized': old_agent_initialized,
                        'agent_now_initialized': False,
                        'config_attempted': {
                            'redis_host': app_config['redis']['host'],
                            'redis_port': app_config['redis']['port'],
                            'redis_db': app_config['redis']['db']
                        }
                    }
                }
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                'success': False,
                'error': str(e),
                'message': 'Error occurred during configuration reload'
            }
        )

@app.post('/api/config/test')
async def api_test_config(request: ConfigUpdateRequest):
    """Test configuration without applying it.

    Args:
        request: Configuration object to test (same format as PUT /api/config)

    Returns:
        JSON with test results for each configuration component
    """
    try:
        data = request.dict(exclude_unset=True)

        if not data:
            raise HTTPException(status_code=400, detail='Configuration data is required for testing')

        test_results = {
            'overall_valid': True,
            'tests': {}
        }

        # Test Redis configuration
        if 'redis' in data:
            redis_config = data['redis']
            redis_test = {
                'valid': True,
                'errors': [],
                'warnings': []
            }

            # Validate Redis parameters
            test_host = redis_config.get('host', app_config['redis']['host'])
            test_port = redis_config.get('port', app_config['redis']['port'])
            test_db = redis_config.get('db', app_config['redis']['db'])

            try:
                test_port = int(test_port)
                test_db = int(test_db)
            except (ValueError, TypeError):
                redis_test['valid'] = False
                redis_test['errors'].append('Port and DB must be integers')

            if redis_test['valid']:
                # Test Redis connection
                try:
                    import redis
                    test_client = redis.Redis(host=test_host, port=test_port, db=test_db, socket_timeout=5)
                    test_client.ping()
                    redis_test['connection_successful'] = True
                except Exception as e:
                    redis_test['valid'] = False
                    redis_test['connection_successful'] = False
                    redis_test['errors'].append(f'Redis connection failed: {str(e)}')

            test_results['tests']['redis'] = redis_test
            if not redis_test['valid']:
                test_results['overall_valid'] = False

        # Test OpenAI configuration
        if 'openai' in data:
            openai_config = data['openai']
            openai_test = {
                'valid': True,
                'errors': [],
                'warnings': []
            }

            # Validate API key format
            test_api_key = openai_config.get('api_key', app_config['openai']['api_key'])
            if test_api_key and not test_api_key.startswith('sk-'):
                openai_test['warnings'].append('API key does not start with "sk-" - may be invalid')

            # Validate numeric parameters
            if 'embedding_dimension' in openai_config:
                try:
                    dim = int(openai_config['embedding_dimension'])
                    if dim <= 0:
                        openai_test['errors'].append('Embedding dimension must be positive')
                        openai_test['valid'] = False
                except (ValueError, TypeError):
                    openai_test['errors'].append('Embedding dimension must be an integer')
                    openai_test['valid'] = False

            if 'temperature' in openai_config:
                try:
                    temp = float(openai_config['temperature'])
                    if temp < 0 or temp > 2:
                        openai_test['warnings'].append('Temperature should be between 0 and 2')
                except (ValueError, TypeError):
                    openai_test['errors'].append('Temperature must be a number')
                    openai_test['valid'] = False

            # Test OpenAI API connection (if API key provided)
            if test_api_key:
                try:
                    import openai
                    test_client = openai.OpenAI(api_key=test_api_key)
                    # Try a simple API call with timeout
                    models = test_client.models.list()
                    openai_test['api_connection_successful'] = True
                except Exception as e:
                    openai_test['api_connection_successful'] = False
                    openai_test['warnings'].append(f'OpenAI API test failed: {str(e)}')

            test_results['tests']['openai'] = openai_test
            if not openai_test['valid']:
                test_results['overall_valid'] = False

        # Test LLM configuration
        if 'llm' in data:
            llm_config = data['llm']
            llm_test = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'tier_tests': {}
            }

            for tier in ['tier1', 'tier2']:
                if tier in llm_config:
                    tier_config = llm_config[tier]
                    tier_test = {
                        'valid': True,
                        'errors': [],
                        'warnings': []
                    }

                    # Validate provider
                    provider = tier_config.get('provider', '').lower()
                    if provider and provider not in ['openai', 'ollama']:
                        tier_test['errors'].append(f'Provider must be "openai" or "ollama", got "{provider}"')
                        tier_test['valid'] = False

                    # Validate numeric fields
                    if 'temperature' in tier_config:
                        try:
                            temp = float(tier_config['temperature'])
                            if temp < 0 or temp > 2:
                                tier_test['warnings'].append('Temperature should be between 0 and 2')
                        except (ValueError, TypeError):
                            tier_test['errors'].append('Temperature must be a number')
                            tier_test['valid'] = False

                    if 'max_tokens' in tier_config:
                        try:
                            tokens = int(tier_config['max_tokens'])
                            if tokens <= 0:
                                tier_test['errors'].append('Max tokens must be positive')
                                tier_test['valid'] = False
                        except (ValueError, TypeError):
                            tier_test['errors'].append('Max tokens must be an integer')
                            tier_test['valid'] = False

                    if 'timeout' in tier_config:
                        try:
                            timeout = int(tier_config['timeout'])
                            if timeout <= 0:
                                tier_test['errors'].append('Timeout must be positive')
                                tier_test['valid'] = False
                        except (ValueError, TypeError):
                            tier_test['errors'].append('Timeout must be an integer')
                            tier_test['valid'] = False

                    # Test connection if enough info provided
                    if provider and tier_config.get('model'):
                        try:
                            test_config = LLMConfig(
                                provider=provider,
                                model=tier_config['model'],
                                temperature=tier_config.get('temperature', 0.7),
                                max_tokens=tier_config.get('max_tokens', 1000),
                                base_url=tier_config.get('base_url'),
                                api_key=tier_config.get('api_key'),
                                timeout=tier_config.get('timeout', 30)
                            )

                            if provider == 'openai':
                                from llm_manager import OpenAIClient
                                test_client = OpenAIClient(test_config)
                            else:  # ollama
                                from llm_manager import OllamaClient
                                test_client = OllamaClient(test_config)

                            connection_result = test_client.test_connection()
                            tier_test['connection_test'] = connection_result

                            if not connection_result['success']:
                                tier_test['warnings'].append(f'Connection test failed: {connection_result.get("error", "Unknown error")}')
                        except Exception as e:
                            tier_test['warnings'].append(f'Could not test connection: {str(e)}')

                    llm_test['tier_tests'][tier] = tier_test
                    if not tier_test['valid']:
                        llm_test['valid'] = False

            test_results['tests']['llm'] = llm_test
            if not llm_test['valid']:
                test_results['overall_valid'] = False

        # Test LangGraph configuration
        if 'langgraph' in data:
            langgraph_config = data['langgraph']
            langgraph_test = {
                'valid': True,
                'errors': [],
                'warnings': []
            }

            # Validate temperature
            if 'temperature' in langgraph_config:
                try:
                    temp = float(langgraph_config['temperature'])
                    if temp < 0 or temp > 2:
                        langgraph_test['warnings'].append('Temperature should be between 0 and 2')
                except (ValueError, TypeError):
                    langgraph_test['errors'].append('Temperature must be a number')
                    langgraph_test['valid'] = False

            # Validate model name
            if 'model_name' in langgraph_config:
                model_name = langgraph_config['model_name']
                if not model_name or not isinstance(model_name, str):
                    langgraph_test['errors'].append('Model name must be a non-empty string')
                    langgraph_test['valid'] = False

            test_results['tests']['langgraph'] = langgraph_test
            if not langgraph_test['valid']:
                test_results['overall_valid'] = False

        # Test Memory Agent configuration
        if 'memory_agent' in data:
            memory_config = data['memory_agent']
            memory_test = {
                'valid': True,
                'errors': [],
                'warnings': []
            }

            # Validate default_top_k
            if 'default_top_k' in memory_config:
                try:
                    top_k = int(memory_config['default_top_k'])
                    if top_k <= 0:
                        memory_test['errors'].append('default_top_k must be positive')
                        memory_test['valid'] = False
                    elif top_k > 100:
                        memory_test['warnings'].append('default_top_k > 100 may impact performance')
                except (ValueError, TypeError):
                    memory_test['errors'].append('default_top_k must be an integer')
                    memory_test['valid'] = False

            test_results['tests']['memory_agent'] = memory_test
            if not memory_test['valid']:
                test_results['overall_valid'] = False

        # Test Web Server configuration
        if 'web_server' in data:
            web_config = data['web_server']
            web_test = {
                'valid': True,
                'errors': [],
                'warnings': []
            }

            # Validate port
            if 'port' in web_config:
                try:
                    port = int(web_config['port'])
                    if port < 1 or port > 65535:
                        web_test['errors'].append('Port must be between 1 and 65535')
                        web_test['valid'] = False
                    elif port < 1024:
                        web_test['warnings'].append('Ports below 1024 may require administrator privileges')
                except (ValueError, TypeError):
                    web_test['errors'].append('Port must be an integer')
                    web_test['valid'] = False

            test_results['tests']['web_server'] = web_test
            if not web_test['valid']:
                test_results['overall_valid'] = False

        return {
            'success': True,
            'test_results': test_results,
            'message': 'Configuration test completed',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={'error': str(e)})

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    print("🚀 Starting Memory Agent Web Server...")

    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment variables.")
        print("Please set your OpenAI API key in the .env file or environment.")
        raise RuntimeError("OPENAI_API_KEY not found")

    # Initialize LLM manager
    if init_llm_manager():
        print("✅ LLM manager ready")
    else:
        print("❌ Failed to initialize LLM manager")
        raise RuntimeError("Failed to initialize LLM manager")

    # Initialize LangGraph memory agent
    if init_memory_agent():
        print("✅ Memory agent ready")
        print("🌐 Server running at http://localhost:5001")
        print("📖 API docs available at http://localhost:5001/docs")
        print()
    else:
        print("❌ Failed to initialize memory agent")
        raise RuntimeError("Failed to initialize memory agent")



if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        "web_app:app",
        host="0.0.0.0",
        port=5001,
        log_level="info",
        reload=False
    )

#!/usr/bin/env python3
"""
REST API for Memory Agent

"""

import os
import json
import uuid
import urllib.request
import urllib.error
import redis
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, Query, Path, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from memory.agent import LangGraphMemoryAgent
from memory.core_agent import MemoryAgent
from llm_manager import LLMManager, LLMConfig, init_llm_manager as initialize_llm_manager, get_llm_manager


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
        "port": int(os.getenv("REDIS_PORT", "6379")),
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
            "provider": "openai",  # "openai" or "ollama"
            "model": "gpt-3.5-turbo",
            "temperature": 0.1,
            "max_tokens": 1000,
            "base_url": None,  # For Ollama: "http://localhost:11434"
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
    "langcache": {
        "enabled": True,  # Master switch for all caching
        "cache_types": {
            "memory_extraction": True,      # Cache memory extraction from conversations
            "query_optimization": True,     # Cache query validation and preprocessing
            "embedding_optimization": True, # Cache query optimization for vector search
            "context_analysis": True,       # Cache memory context analysis
            "memory_grounding": True        # Cache memory grounding operations
        }
    },

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



class ConfigUpdateRequest(BaseModel):
    redis: Optional[Dict[str, Any]] = Field(None, description="Redis configuration")
    llm: Optional[Dict[str, Any]] = Field(None, description="LLM configuration")
    openai: Optional[Dict[str, Any]] = Field(None, description="OpenAI configuration")
    langgraph: Optional[Dict[str, Any]] = Field(None, description="LangGraph configuration")
    memory_agent: Optional[Dict[str, Any]] = Field(None, description="Memory agent configuration")
    web_server: Optional[Dict[str, Any]] = Field(None, description="Web server configuration")
    langcache: Optional[Dict[str, Any]] = Field(None, description="LangCache configuration")


# =============================================================================
# ASYNC MEMORY PROCESSING MODELS
# =============================================================================

class RawMemoryStoreRequest(BaseModel):
    session_data: str = Field(..., description="Complete chat session text or conversation history")
    session_id: Optional[str] = Field(None, description="Optional session ID for tracking")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata (user_id, session_type, etc.)")

class RawMemoryStoreResponse(BaseModel):
    success: bool
    raw_memory_id: str
    queued_at: str
    estimated_processing_time: str
    queue_position: Optional[int] = None

class ProcessedMemoryResponse(BaseModel):
    success: bool
    session_id: str
    discrete_memories: List[Dict[str, Any]]
    session_summary: Dict[str, Any]
    processing_stats: Dict[str, Any]

class MemoryHierarchyRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Filter by session ID")
    memory_type: Optional[str] = Field(None, description="Filter by memory type: 'discrete', 'summary', 'raw'")
    start_date: Optional[str] = Field(None, description="Start date filter (ISO format)")
    end_date: Optional[str] = Field(None, description="End date filter (ISO format)")
    limit: int = Field(50, ge=1, le=1000, description="Maximum number of results")

class BackgroundProcessorStatus(BaseModel):
    success: bool
    processor_running: bool
    queue_size: int
    processed_today: int
    last_processed_at: Optional[str]
    processing_interval_seconds: int
    retention_policy: Dict[str, Any]

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

def reinitialize_llm_manager():
    """Reinitialize LLM manager."""
    try:
        # Reinitialize the LLM manager
        if not init_llm_manager():
            return False, "Failed to reinitialize LLM manager"

        print("✅ LLM manager reinitialized successfully")
        return True, "LLM manager reinitialized successfully"
    except Exception as e:
        error_msg = f"Failed to reinitialize LLM manager: {e}"
        print(f"❌ {error_msg}")
        return False, error_msg

def init_memory_agent():
    """Initialize the LangGraph memory agent with current configuration."""
    global memory_agent

    # Check if already initialized to prevent double initialization
    if memory_agent is not None:
        print("ℹ️ Memory agent already initialized, skipping re-initialization")
        return True

    try:
        # Create memory agent with current Redis configuration
        base_memory_agent = MemoryAgent(
            redis_host=app_config["redis"]["host"],
            redis_port=app_config["redis"]["port"],
            redis_db=app_config["redis"]["db"],
            vectorset_key=app_config["redis"]["vectorset_key"]
        )



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

# Note: Default vectorstore support has been removed. All endpoints now require explicit vectorstore names.

# =============================================================================
# NEME API - Fundamental Memory Operations (Inspired by Minsky's "Memories")
# =============================================================================
#
# Memories are the fundamental units of memory in Minsky's Society of Mind theory.
# These APIs handle atomic memory storage, retrieval, and basic operations.
# Think of these as the building blocks of knowledge that can be combined
# by higher-level cognitive processes.
#
# All endpoints require explicit vectorstore names for proper isolation and clarity.
# No default vectorstore support - this ensures explicit, predictable behavior.
#
# Core operations:
# - Store atomic memories with contextual grounding
# - Vector similarity search across stored memories
# - Memory lifecycle management (delete, clear)
# - Context management for grounding operations
#
# API Structure: /api/memory/{vectorstore_name}/{operation}
# =============================================================================

@app.post('/api/memory/{vectorstore_name}/search')
async def api_search_Memories_vectorstore(vectorstore_name: str, request: MemorySearchRequest):
    """Search atomic memories (Memories) in a specific vectorstore using vector similarity.

    Args:
        vectorstore_name: Name of the vectorstore to search in

    Returns:
        JSON with success status, memories array, count, and memory breakdown by type
    """
    validate_vectorstore_name(vectorstore_name)
    return await _search_memories_impl(request, vectorstore_name=vectorstore_name)

@app.post('/api/memory/{vectorstore_name}/context')
async def api_set_neme_context_vectorstore(vectorstore_name: str, request: ContextSetRequest):
    """Set current context for memory grounding in a specific vectorstore.

    Args:
        vectorstore_name: Name of the vectorstore to set context for
        request: Context parameters (location, activity, people_present)

    Returns:
        JSON with success status and updated context
    """
    validate_vectorstore_name(vectorstore_name)
    return await _set_context_impl(request, {}, vectorstore_name=vectorstore_name)


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

async def _store_memory_impl(request: MemoryStoreRequest, vectorstore_name: str):
    """Implementation for storing memories with explicit vectorstore support.

    Args:
        vectorstore_name: Explicit vectorstore name (required)

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

        if not memory_text:
            raise HTTPException(status_code=400, detail='Memory text is required')

        print(f"💾 NEME API: Storing atomic memory - '{memory_text[:60]}{'...' if len(memory_text) > 60 else ''}'")
        print(f"📦 Vectorstore: {vectorstore_name}")

        # Use existing memory agent with vectorstore parameter
        if not memory_agent:
            raise HTTPException(status_code=500, detail='Memory agent not initialized')

        storage_result = memory_agent.memory_agent.store_memory(
            memory_text,
            apply_grounding=apply_grounding,
            vectorset_key=vectorstore_name
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
            'vectorstore_name': vectorstore_name
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

async def _search_memories_impl(request: MemorySearchRequest, vectorstore_name: str):
    """Implementation for searching memories with explicit vectorstore support.

    Args:
        vectorstore_name: Explicit vectorstore name (required)

    Returns:
        JSON with success status, memories array, count, and memory breakdown by type
    """
    try:
        query = request.query.strip()
        top_k = request.top_k
        filter_expr = request.filter
        optimize_query = request.optimize_query
        min_similarity = request.min_similarity

        if not query:
            raise HTTPException(status_code=400, detail='Query is required')

        print(f"🔍 NEME API: Searching memories: {query} (top_k: {top_k}, min_similarity: {min_similarity})")
        print(f"📦 Vectorstore: {vectorstore_name}")
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
                    search_query, top_k, filter_expr, min_similarity, vectorset_key=vectorstore_name
                )
            else:
                search_result = memory_agent.memory_agent.search_memories_with_filtering_info(
                    query, top_k, filter_expr, min_similarity, vectorset_key=vectorstore_name
                )
        else:
            search_result = memory_agent.memory_agent.search_memories_with_filtering_info(
                query, top_k, filter_expr, min_similarity, vectorset_key=vectorstore_name
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
            'vectorstore_name': vectorstore_name
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

async def _get_memory_info_impl(vectorstore_name: str):
    """Implementation for getting memory info with explicit vectorstore support.

    Args:
        vectorstore_name: Explicit vectorstore name (required)

    Returns:
        JSON with memory count, vector dimension, embedding model, and system info
    """
    try:
        if not memory_agent:
            raise HTTPException(status_code=500, detail='Memory agent not initialized')

        redis_client = memory_agent.memory_agent.core.redis_client

        try:
            # Get memory count using VCARD for the specific vectorstore
            memory_count = redis_client.execute_command("VCARD", vectorstore_name)
            memory_count = int(memory_count) if memory_count else 0

            # Get vector dimension using VDIM for the specific vectorstore
            dimension = redis_client.execute_command("VDIM", vectorstore_name)
            dimension = int(dimension) if dimension else 0

            # Get detailed vector set info using VINFO for the specific vectorstore
            vinfo_result = redis_client.execute_command("VINFO", vectorstore_name)

            # Parse VINFO result (returns key-value pairs)
            vinfo_dict = {}
            if vinfo_result:
                for i in range(0, len(vinfo_result), 2):
                    if i + 1 < len(vinfo_result):
                        key = vinfo_result[i].decode('utf-8') if isinstance(vinfo_result[i], bytes) else vinfo_result[i]
                        value = vinfo_result[i + 1]
                        # Try to decode bytes, but keep original type for numbers
                        if isinstance(value, bytes):
                            try:
                                value = value.decode('utf-8')
                            except:
                                value = str(value)
                        vinfo_dict[key] = value

            memory_info = {
                'memory_count': memory_count,
                'vector_dimension': dimension,
                'vectorset_name': vectorstore_name,
                'vectorset_info': vinfo_dict,
                'embedding_model': 'text-embedding-ada-002',
                'redis_host': redis_client.connection_pool.connection_kwargs.get('host', 'unknown'),
                'redis_port': redis_client.connection_pool.connection_kwargs.get('port', 'unknown'),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        except redis.ResponseError:
            # VectorSet doesn't exist yet - this is normal when no memories have been stored
            memory_info = {
                'memory_count': 0,
                'vector_dimension': 0,
                'vectorset_name': vectorstore_name,
                'vectorset_info': {},
                'embedding_model': 'text-embedding-ada-002',
                'redis_host': redis_client.connection_pool.connection_kwargs.get('host', 'unknown'),
                'redis_port': redis_client.connection_pool.connection_kwargs.get('port', 'unknown'),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'note': 'No memories stored yet - VectorSet will be created when first memory is added'
            }

        return {
            'success': True,
            **memory_info
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete('/api/memory/{vectorstore_name}/all')
async def api_delete_all_Memories_vectorstore(vectorstore_name: str):
    """Clear all atomic memories (Memories) from a specific vectorstore.

    Args:
        vectorstore_name: Name of the vectorstore to clear

    Returns:
        JSON with success status, deletion count, and operation details
    """
    validate_vectorstore_name(vectorstore_name)
    return await _delete_all_memories_impl(vectorstore_name=vectorstore_name)

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

async def _delete_memory_impl(memory_id: str, vectorstore_name: str):
    """Implementation for deleting a memory with explicit vectorstore support.

    Args:
        vectorstore_name: Explicit vectorstore name (required)

    Returns:
        JSON with success status and deletion details
    """
    try:
        if not memory_id or not memory_id.strip():
            raise HTTPException(status_code=400, detail='Memory ID is required')

        print(f"🗑️ NEME API: Deleting atomic memory: {memory_id}")
        print(f"📦 Vectorstore: {vectorstore_name}")

        # Use existing memory agent with vectorstore parameter
        if not memory_agent:
            raise HTTPException(status_code=500, detail='Memory agent not initialized')

        success = memory_agent.memory_agent.delete_memory(
            memory_id.strip(),
            vectorset_key=vectorstore_name
        )

        if success:
            return {
                'success': True,
                'message': f'Neme {memory_id} deleted successfully',
                'memory_id': memory_id,
                'vectorstore_name': vectorstore_name
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

async def _delete_all_memories_impl(vectorstore_name: str):
    """Implementation for deleting all memories with explicit vectorstore support.

    Args:
        vectorstore_name: Explicit vectorstore name (required)

    Returns:
        JSON with success status, deletion count, and operation details
    """
    try:
        print("🗑️ NEME API: Clearing all atomic memories...")
        print(f"📦 Vectorstore: {vectorstore_name}")

        # Use existing memory agent with vectorstore parameter
        if not memory_agent:
            raise HTTPException(status_code=500, detail='Memory agent not initialized')

        result = memory_agent.memory_agent.clear_all_memories(vectorset_key=vectorstore_name)

        if result['success']:
            return {
                'success': True,
                'message': result['message'],
                'memories_deleted': result['memories_deleted'],
                'vectorset_existed': result['vectorset_existed'],
                'vectorstore_name': vectorstore_name
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

async def _set_context_impl(request: ContextSetRequest, additional_context: Dict[str, Any], vectorstore_name: str):
    """Implementation for setting context with explicit vectorstore support.

    Args:
        vectorstore_name: Explicit vectorstore name (required)

    Returns:
        JSON with success status and updated context
    """
    try:
        if not memory_agent:
            raise HTTPException(status_code=500, detail='Memory agent not initialized')

        # Extract context parameters
        location = request.location
        activity = request.activity
        people_present = request.people_present or []

        # Use additional_context for environment context
        environment_context = additional_context

        print(f"🌍 NEME API: Setting context - Location: {location}, Activity: {activity}, People: {people_present}")
        print(f"📦 Vectorstore: {vectorstore_name}")

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
            'vectorstore_name': vectorstore_name
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

async def _get_context_impl(vectorstore_name: str):
    """Implementation for getting context with explicit vectorstore support.

    Args:
        vectorstore_name: Explicit vectorstore name (required)

    Returns:
        JSON with success status and current context (temporal, spatial, social, environmental)
    """
    try:
        if not memory_agent:
            raise HTTPException(status_code=500, detail='Memory agent not initialized')

        current_context = memory_agent.memory_agent.core._get_current_context()

        return {
            'success': True,
            'context': current_context,
            'vectorstore_name': vectorstore_name
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# =============================================================================
# K-LINE API - Reflective Operations (Inspired by Minsky's "K-lines")
# =============================================================================
#
# K-lines (Knowledge lines) represent temporary mental states that activate
# and connect relevant Memories for specific cognitive tasks. In Minsky's theory,
# K-lines are the mechanism by which the mind constructs coherent mental states
# from distributed memory fragments.
#
# All endpoints require explicit vectorstore names for proper isolation.
#
# These APIs handle:
# - Constructing mental states by recalling and filtering relevant memories
# - Question answering with confidence scoring and reasoning
# - Extracting valuable memories from conversational data
# - Advanced cognitive operations that combine multiple Memories
#
# API Structure: /api/memory/{vectorstore_name}/{operation}
# =============================================================================



@app.post('/api/memory/{vectorstore_name}/ask')
async def api_kline_answer(vectorstore_name: str, request: KLineAnswerRequest):
    """Answer a question using K-line construction and reasoning.

    Args:
        vectorstore_name: Name of the vectorstore to search in

    This operation constructs a mental state from relevant Memories and applies
    sophisticated reasoning to answer questions with confidence scoring.
    It represents the full cognitive process of memory recall + reasoning.

    K-lines are constructed but NOT stored - they exist only as temporary mental states.

    Returns:
        JSON with structured response including answer, confidence, reasoning, supporting memories,
        and the constructed mental state (K-line)
    """
    try:
        validate_vectorstore_name(vectorstore_name)

        if not memory_agent:
            raise HTTPException(status_code=500, detail='Memory agent not initialized')

        question = request.question.strip()
        top_k = request.top_k
        filter_expr = request.filter
        min_similarity = request.min_similarity

        if not question:
            raise HTTPException(status_code=400, detail='Question is required')

        print(f"🤔 K-LINE API: Answering question via mental state construction: {question} (top_k: {top_k})")
        print(f"📦 Vectorstore: {vectorstore_name}")
        if filter_expr:
            print(f"🔍 Filter: {filter_expr}")

        # Use the memory agent's sophisticated answer_question method
        # This constructs a K-line (mental state) and applies reasoning
        answer_response = memory_agent.memory_agent.answer_question(question, top_k=top_k, filterBy=filter_expr, min_similarity=min_similarity, vectorset_key=vectorstore_name)

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
            'vectorstore_name': vectorstore_name,
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
# orchestrating both Memories (atomic memories) and K-lines (mental states)
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
    - Searches relevant Memories (atomic memories)
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
# ASYNC MEMORY PROCESSING APIs - Raw memory storage and background processing
# =============================================================================

@app.post('/api/memory/{vectorstore_name}/store_raw')
async def api_store_raw_memory(vectorstore_name: str, request: RawMemoryStoreRequest):
    """Store raw chat session data for asynchronous memory processing.

    This endpoint accepts complete chat session histories and queues them for
    background processing. The background processor will extract discrete memories,
    generate session summaries, and create a hierarchical memory structure.

    Args:
        vectorstore_name: Name of the vectorstore to store processed memories in
        request: Raw memory data with session content and metadata

    Returns:
        JSON with success status, raw_memory_id, and queue information
    """
    validate_vectorstore_name(vectorstore_name)

    try:
        session_data = request.session_data.strip()
        if not session_data:
            raise HTTPException(status_code=400, detail='session_data is required')

        session_id = request.session_id or str(uuid.uuid4())
        metadata = request.metadata or {}

        # Generate unique ID for this raw memory
        raw_memory_id = str(uuid.uuid4())
        current_time = datetime.now(timezone.utc)

        # Create raw memory record
        raw_memory_record = {
            "raw_memory_id": raw_memory_id,
            "session_id": session_id,
            "session_data": session_data,
            "metadata": metadata,
            "vectorstore_name": vectorstore_name,
            "created_at": current_time.isoformat(),
            "status": "queued",
            "processing_attempts": 0
        }

        # Get Redis client from memory agent
        if not memory_agent:
            raise HTTPException(status_code=500, detail='Memory agent not initialized')

        redis_client = memory_agent.memory_agent.core.redis_client

        # Store raw memory in Redis
        raw_memory_key = f"{vectorstore_name}:raw_memory:{raw_memory_id}"
        redis_client.set(raw_memory_key, json.dumps(raw_memory_record))

        # Add to processing queue (sorted set with timestamp as score)
        queue_key = "RAW_MEMORY_QUEUE"
        timestamp_score = current_time.timestamp()
        redis_client.zadd(queue_key, {raw_memory_key: timestamp_score})

        # Get current queue position
        queue_position = redis_client.zrank(queue_key, raw_memory_key)
        queue_size = redis_client.zcard(queue_key)

        print(f"📥 ASYNC MEMORY: Queued raw memory {raw_memory_id} for processing")
        print(f"📦 Vectorstore: {vectorstore_name}")
        print(f"📊 Queue position: {queue_position + 1}/{queue_size}")

        # Estimate processing time based on queue position
        estimated_time = "1-2 minutes" if queue_position < 5 else f"{queue_position * 2}-{queue_position * 3} minutes"

        return RawMemoryStoreResponse(
            success=True,
            raw_memory_id=raw_memory_id,
            queued_at=current_time.isoformat(),
            estimated_processing_time=estimated_time,
            queue_position=queue_position + 1
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/memory/{vectorstore_name}/processing_status')
async def api_get_processing_status(vectorstore_name: str):
    """Get status of the background memory processing system.

    Args:
        vectorstore_name: Name of the vectorstore to check status for

    Returns:
        JSON with processor status, queue size, and processing statistics
    """
    validate_vectorstore_name(vectorstore_name)

    try:
        if not memory_agent:
            raise HTTPException(status_code=500, detail='Memory agent not initialized')

        redis_client = memory_agent.memory_agent.core.redis_client

        # Get queue information
        queue_key = "RAW_MEMORY_QUEUE"
        queue_size = redis_client.zcard(queue_key)

        # Get processing statistics for today
        today = datetime.now(timezone.utc).date()
        stats_key = f"processing_stats:{today.isoformat()}"
        daily_stats = redis_client.hgetall(stats_key)

        processed_today = int(daily_stats.get(b'processed_count', 0)) if daily_stats else 0
        last_processed_at = daily_stats.get(b'last_processed_at', b'').decode() if daily_stats else None

        # Check if background processor is running (simplified check)
        processor_status_key = "background_processor:status"
        processor_info = redis_client.hgetall(processor_status_key)
        processor_running = False
        processing_interval = 60  # Default 60 seconds

        if processor_info:
            last_heartbeat = processor_info.get(b'last_heartbeat', b'').decode()
            if last_heartbeat:
                try:
                    heartbeat_time = datetime.fromisoformat(last_heartbeat.replace('Z', '+00:00'))
                    # Consider processor running if heartbeat is within last 2 minutes
                    processor_running = (datetime.now(timezone.utc) - heartbeat_time).total_seconds() < 120
                except:
                    pass
            processing_interval = int(processor_info.get(b'interval_seconds', 60))

        # Get retention policy from config
        retention_policy = {
            "raw_transcripts_days": 30,  # Keep raw transcripts for 30 days
            "processed_sessions_days": 365,  # Keep processed sessions for 1 year
            "cleanup_interval_hours": 24  # Run cleanup daily
        }

        return BackgroundProcessorStatus(
            success=True,
            processor_running=processor_running,
            queue_size=queue_size,
            processed_today=processed_today,
            last_processed_at=last_processed_at,
            processing_interval_seconds=processing_interval,
            retention_policy=retention_policy
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/memory/{vectorstore_name}/hierarchy')
async def api_get_memory_hierarchy(
    vectorstore_name: str,
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
    memory_type: Optional[str] = Query(None, description="Filter by memory type: 'discrete', 'summary', 'raw'"),
    start_date: Optional[str] = Query(None, description="Start date filter (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date filter (ISO format)"),
    limit: int = Query(50, ge=1, le=1000, description="Maximum number of results")
):
    """Retrieve memory hierarchy data including discrete memories, session summaries, and raw transcripts.

    Args:
        vectorstore_name: Name of the vectorstore to retrieve from
        request: Query parameters for filtering results

    Returns:
        JSON with hierarchical memory data organized by type and session
    """
    validate_vectorstore_name(vectorstore_name)

    try:
        if not memory_agent:
            raise HTTPException(status_code=500, detail='Memory agent not initialized')

        redis_client = memory_agent.memory_agent.core.redis_client

        # Parse date filters
        start_date_obj = None
        end_date_obj = None
        if start_date:
            start_date_obj = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        if end_date:
            end_date_obj = datetime.fromisoformat(end_date.replace('Z', '+00:00'))

        results = {
            "discrete_memories": [],
            "session_summaries": [],
            "raw_transcripts": [],
            "total_count": 0,
            "filtered_by": {
                "session_id": session_id,
                "memory_type": memory_type,
                "date_range": f"{start_date} to {end_date}" if start_date or end_date else None
            }
        }

        # Get session summaries
        if not memory_type or memory_type in ['summary', 'all']:
            summary_pattern = f"{vectorstore_name}:session_summary:*"
            if session_id:
                summary_pattern = f"{vectorstore_name}:session_summary:{session_id}"

            for key in redis_client.scan_iter(match=summary_pattern):
                key_str = key.decode() if isinstance(key, bytes) else key
                try:
                    summary_data = redis_client.get(key_str)
                    if summary_data:
                        summary = json.loads(summary_data.decode())

                        # Apply date filter
                        if start_date_obj or end_date_obj:
                            created_at = datetime.fromisoformat(summary["created_at"].replace('Z', '+00:00'))
                            if start_date_obj and created_at < start_date_obj:
                                continue
                            if end_date_obj and created_at > end_date_obj:
                                continue

                        results["session_summaries"].append(summary)

                        if len(results["session_summaries"]) >= limit:
                            break
                except:
                    continue

        # Get raw transcripts
        if not memory_type or memory_type in ['raw', 'all']:
            raw_pattern = f"{vectorstore_name}:raw_memory:*"

            for key in redis_client.scan_iter(match=raw_pattern):
                key_str = key.decode() if isinstance(key, bytes) else key
                try:
                    raw_data = redis_client.get(key_str)
                    if raw_data:
                        raw_memory = json.loads(raw_data.decode())

                        # Apply session filter
                        if session_id and raw_memory.get("session_id") != session_id:
                            continue

                        # Apply date filter
                        if start_date_obj or end_date_obj:
                            created_at = datetime.fromisoformat(raw_memory["created_at"].replace('Z', '+00:00'))
                            if start_date_obj and created_at < start_date_obj:
                                continue
                            if end_date_obj and created_at > end_date_obj:
                                continue

                        # Remove session_data from response to keep it lightweight
                        raw_summary = {k: v for k, v in raw_memory.items() if k != "session_data"}
                        raw_summary["has_session_data"] = bool(raw_memory.get("session_data"))
                        raw_summary["session_data_length"] = len(raw_memory.get("session_data", ""))

                        results["raw_transcripts"].append(raw_summary)

                        if len(results["raw_transcripts"]) >= limit:
                            break
                except:
                    continue

        # Get discrete memories (from vectorset)
        if not memory_type or memory_type in ['discrete', 'all']:
            # Use existing search functionality to get recent memories
            search_query = f"session_id:{session_id}" if session_id else "*"

            try:
                search_result = memory_agent.memory_agent.search_memories_with_filtering_info(
                    query=search_query,
                    top_k=min(limit, 100),
                    min_similarity=0.0,  # Get all memories
                    vectorset_key=vectorstore_name
                )

                discrete_memories = search_result.get('memories', [])

                # Apply date filter to discrete memories
                if start_date_obj or end_date_obj:
                    filtered_memories = []
                    for memory in discrete_memories:
                        try:
                            created_at = datetime.fromisoformat(memory["created_at"].replace('Z', '+00:00'))
                            if start_date_obj and created_at < start_date_obj:
                                continue
                            if end_date_obj and created_at > end_date_obj:
                                continue
                            filtered_memories.append(memory)
                        except:
                            filtered_memories.append(memory)  # Include if date parsing fails
                    discrete_memories = filtered_memories

                results["discrete_memories"] = discrete_memories[:limit]

            except Exception as e:
                print(f"⚠️ Error retrieving discrete memories: {e}")
                results["discrete_memories"] = []

        # Calculate total count
        results["total_count"] = (len(results["discrete_memories"]) +
                                len(results["session_summaries"]) +
                                len(results["raw_transcripts"]))

        # Sort results by creation date (newest first)
        for memory_type in ["discrete_memories", "session_summaries", "raw_transcripts"]:
            try:
                results[memory_type].sort(
                    key=lambda x: x.get("created_at", ""),
                    reverse=True
                )
            except:
                pass

        return {
            "success": True,
            "vectorstore_name": vectorstore_name,
            "results": results,
            "query_params": {
                "session_id": session_id,
                "memory_type": memory_type,
                "start_date": start_date,
                "end_date": end_date,
                "limit": limit
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/memory/{vectorstore_name}/session/{session_id}')
async def api_get_session_details(vectorstore_name: str, session_id: str):
    """Get detailed information about a specific session including all related memories.

    Args:
        vectorstore_name: Name of the vectorstore
        session_id: ID of the session to retrieve

    Returns:
        JSON with complete session information and related memories
    """
    validate_vectorstore_name(vectorstore_name)

    try:
        if not memory_agent:
            raise HTTPException(status_code=500, detail='Memory agent not initialized')

        redis_client = memory_agent.memory_agent.core.redis_client

        # Get session summary
        summary_key = f"{vectorstore_name}:session_summary:{session_id}"
        summary_data = redis_client.get(summary_key)
        session_summary = None
        if summary_data:
            session_summary = json.loads(summary_data.decode())

        # Get raw transcript
        raw_transcript = None
        raw_pattern = f"{vectorstore_name}:raw_memory:*"
        for key in redis_client.scan_iter(match=raw_pattern):
            key_str = key.decode() if isinstance(key, bytes) else key
            try:
                raw_data = redis_client.get(key_str)
                if raw_data:
                    raw_memory = json.loads(raw_data.decode())
                    if raw_memory.get("session_id") == session_id:
                        raw_transcript = raw_memory
                        break
            except:
                continue

        # Get related discrete memories
        discrete_memories = []
        memories_key = f"{vectorstore_name}:session_memories:{session_id}"
        memory_ids = redis_client.smembers(memories_key)

        if memory_ids:
            # Get memory details from vectorset
            for memory_id in memory_ids:
                memory_id_str = memory_id.decode() if isinstance(memory_id, bytes) else memory_id
                try:
                    # Search for this specific memory
                    search_result = memory_agent.memory_agent.search_memories_with_filtering_info(
                        query=memory_id_str,
                        top_k=1,
                        min_similarity=0.0,
                        vectorset_key=vectorstore_name
                    )

                    memories = search_result.get('memories', [])
                    for memory in memories:
                        if memory.get('memory_id') == memory_id_str:
                            discrete_memories.append(memory)
                            break
                except:
                    continue

        if not session_summary and not raw_transcript and not discrete_memories:
            raise HTTPException(status_code=404, detail=f'Session {session_id} not found')

        return {
            "success": True,
            "session_id": session_id,
            "vectorstore_name": vectorstore_name,
            "session_summary": session_summary,
            "raw_transcript": raw_transcript,
            "discrete_memories": discrete_memories,
            "memory_count": len(discrete_memories),
            "has_raw_data": bool(raw_transcript),
            "has_summary": bool(session_summary)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/api/memory/{vectorstore_name}/cleanup')
async def api_cleanup_memory_data(vectorstore_name: str, retention_days: int = Query(30, ge=1, le=365)):
    """Manually trigger cleanup of expired memory data.

    Args:
        vectorstore_name: Name of the vectorstore to clean up
        retention_days: Days to retain raw transcripts (default: 30)

    Returns:
        JSON with cleanup statistics
    """
    validate_vectorstore_name(vectorstore_name)

    try:
        if not memory_agent:
            raise HTTPException(status_code=500, detail='Memory agent not initialized')

        # Import the async processor for cleanup functionality
        from memory.async_processor import AsyncMemoryProcessor

        # Create processor instance for cleanup
        processor = AsyncMemoryProcessor(
            redis_host=memory_agent.memory_agent.core.redis_client.connection_pool.connection_kwargs.get('host'),
            redis_port=memory_agent.memory_agent.core.redis_client.connection_pool.connection_kwargs.get('port'),
            redis_db=memory_agent.memory_agent.core.redis_client.connection_pool.connection_kwargs.get('db'),
            retention_days=retention_days
        )

        # Run cleanup
        cleanup_result = processor.cleanup_expired_data()

        return {
            "success": True,
            "vectorstore_name": vectorstore_name,
            "retention_days": retention_days,
            "cleanup_result": cleanup_result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/memory/{vectorstore_name}/stats')
async def api_get_memory_stats(vectorstore_name: str):
    """Get comprehensive memory statistics including hierarchy breakdown.

    Args:
        vectorstore_name: Name of the vectorstore to get stats for

    Returns:
        JSON with detailed memory statistics
    """
    validate_vectorstore_name(vectorstore_name)

    try:
        if not memory_agent:
            raise HTTPException(status_code=500, detail='Memory agent not initialized')

        redis_client = memory_agent.memory_agent.core.redis_client

        # Get basic memory info
        basic_stats = await _get_memory_info_impl(vectorstore_name=vectorstore_name)

        # Count session summaries
        summary_pattern = f"{vectorstore_name}:session_summary:*"
        summary_count = 0
        for _ in redis_client.scan_iter(match=summary_pattern):
            summary_count += 1

        # Count raw memories
        raw_pattern = f"{vectorstore_name}:raw_memory:*"
        raw_count = 0
        processed_count = 0
        queued_count = 0
        error_count = 0

        for key in redis_client.scan_iter(match=raw_pattern):
            raw_count += 1
            try:
                raw_data = redis_client.get(key)
                if raw_data:
                    raw_memory = json.loads(raw_data.decode())
                    status = raw_memory.get("status", "unknown")
                    if status == "processed":
                        processed_count += 1
                    elif status == "queued":
                        queued_count += 1
                    elif status == "error":
                        error_count += 1
            except:
                continue

        # Get queue size
        queue_size = redis_client.zcard("RAW_MEMORY_QUEUE")

        # Get processing stats for today
        today = datetime.now(timezone.utc).date()
        stats_key = f"processing_stats:{today.isoformat()}"
        daily_stats = redis_client.hgetall(stats_key)
        processed_today = int(daily_stats.get(b'processed_count', 0)) if daily_stats else 0

        # Calculate storage usage (approximate)
        total_keys = 0
        for pattern in [f"{vectorstore_name}:*", "RAW_MEMORY_QUEUE", f"processing_stats:*"]:
            for _ in redis_client.scan_iter(match=pattern):
                total_keys += 1

        return {
            "success": True,
            "vectorstore_name": vectorstore_name,
            "basic_stats": basic_stats,
            "hierarchy_stats": {
                "discrete_memories": basic_stats.get("memory_count", 0),
                "session_summaries": summary_count,
                "raw_transcripts": raw_count,
                "total_items": basic_stats.get("memory_count", 0) + summary_count + raw_count
            },
            "processing_stats": {
                "processed": processed_count,
                "queued": queued_count,
                "errors": error_count,
                "queue_size": queue_size,
                "processed_today": processed_today
            },
            "storage_stats": {
                "total_redis_keys": total_keys,
                "estimated_memory_usage": "Use Redis MEMORY USAGE command for precise measurements"
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# CHAT APIs - General purpose chat interface
# =============================================================================

@app.post('/api/agent/{vectorstore_name}/session')
async def api_create_agent_session(vectorstore_name: str, request: AgentSessionCreateRequest):
    """Create a new agent session with integrated memory capabilities.

    Args:
        vectorstore_name: Name of the vectorstore to use for memory operations

    Returns:
        JSON with session_id and confirmation
    """
    try:
        validate_vectorstore_name(vectorstore_name)

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
            'use_memory': use_memory,
            'vectorstore_name': vectorstore_name
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

@app.post('/api/agent/{vectorstore_name}/session/{session_id}')
async def api_agent_session_message(vectorstore_name: str, session_id: str = Path(..., description="The agent session ID"), request: AgentSessionMessageRequest = Body(...)):
    """Send a message to an agent session with full cognitive architecture.

    This endpoint provides the complete agent experience:
    - Searches relevant Memories for context
    - Constructs K-lines (mental states)
    - Generates contextually-aware responses
    - Automatically extracts new memories from conversations

    Args:
        vectorstore_name: Name of the vectorstore to use for memory operations
        session_id: The agent session ID
        request: Request body with message and options

    Returns:
        JSON with the assistant's response and conversation context.
        For memory-enabled sessions, includes 'memory_context' with relevant memories used in the response.
    """
    try:
        validate_vectorstore_name(vectorstore_name)

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

        # Verify the vectorstore matches the session's vectorstore
        session_vectorstore = session.get('vectorstore_name')
        if session_vectorstore and session_vectorstore != vectorstore_name:
            raise HTTPException(status_code=400, detail=f'Session was created with vectorstore "{session_vectorstore}" but request uses "{vectorstore_name}"')

        if use_memory:
            print(f"1) User said: '{message[:80]}{'...' if len(message) > 80 else ''}' (vectorstore: {vectorstore_name})")
        else:
            print(f"💬 [{session_id}] AGENT API: Agent session (memory: disabled): User: {message}")

        # Handle memory-enabled sessions
        if use_memory:
            return _handle_memory_enabled_message(session_id, session, message, stream, store_memory, top_k, min_similarity, vectorstore_name)
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


def _handle_memory_enabled_message(session_id, session, user_message, stream, store_memory=True, top_k=10, min_similarity=0.9, vectorstore_name="memories"):
    """Handle message processing for sessions with memory enabled."""
    if not memory_agent:
        raise HTTPException(status_code=500, detail='Memory agent not initialized but session requires memory')
    

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
                min_similarity=min_similarity,
                vectorset_key=vectorstore_name
            )
            memory_results = search_result['memories']
            filtering_info = search_result['filtering_info']
        else:
        # For help queries, still search but with original message
            search_result = memory_agent.memory_agent.search_memories_with_filtering_info(
                query=user_message,
                top_k=top_k,
                min_similarity=min_similarity,
                vectorset_key=vectorstore_name
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
            max_tokens=session['config'].get('max_tokens', 1000)
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
        _check_and_extract_memories(session_id, session, vectorstore_name)
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
    



def _check_and_extract_memories(session_id, session, vectorstore_name="memories"):
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
        print(f"1) Searching for existing memories related to: '{latest_user_message['content'][:50]}...' (vectorstore: {vectorstore_name})")
        existing_memories = memory_agent.memory_agent.search_memories(
            latest_user_message['content'],
            top_k=10,
            min_similarity=0.7,
            vectorset_key=vectorstore_name
        )

        if existing_memories:
            print(f"2) Found {len(existing_memories)} existing relevant memories")
        else:
            print(f"2) No existing relevant memories found")

        # STEP 2: Extract memories using context-aware approach
        # Temporarily change the memory agent's default vectorset if needed
        original_vectorset = memory_agent.memory_agent.core.VECTORSET_KEY
        if vectorstore_name != original_vectorset:
            memory_agent.memory_agent.core.VECTORSET_KEY = vectorstore_name
            print(f"3) Temporarily switched vectorset from '{original_vectorset}' to '{vectorstore_name}' for extraction")

        try:
            result = memory_agent.memory_agent.extract_and_store_memories(
                raw_input=conversation_text,
                context_prompt=session.get('memory_context', 'Extract ONLY user preferences, constraints, facts, and important personal details from the user messages. Do NOT extract assistant suggestions or recommendations.'),
                apply_grounding=True,
                existing_memories=existing_memories  # Pass existing memories for context-aware extraction
            )
        finally:
            # Restore original vectorset
            if vectorstore_name != original_vectorset:
                memory_agent.memory_agent.core.VECTORSET_KEY = original_vectorset
                print(f"4) Restored vectorset to '{original_vectorset}'")


        if result["total_extracted"] > 0:
            extracted_memories = result.get("extracted_memories", [])
            memory_texts = [mem.get("final_text", mem.get("raw_text", "Unknown")) for mem in extracted_memories]
            print(f"5) Identified {result['total_extracted']} memories: {', '.join([f'"{text[:40]}{"..." if len(text) > 40 else ""}"' for text in memory_texts])}")
            print(f"6) Saved {result['total_extracted']} memories to vectorstore '{vectorstore_name}'")
            session['last_extraction'] = datetime.now(timezone.utc).isoformat()
        else:
            print(f"5) No memories identified for extraction")

        # Since we process each message individually, we can keep a smaller buffer
        # Keep only the last 4 messages (2 exchanges) for context
        session['conversation_buffer'] = buffer[-4:]

    except Exception as e:
        print(f"⚠️ Memory extraction failed: {e}")




@app.get('/api/agent/{vectorstore_name}/session/{session_id}')
async def api_get_agent_session(vectorstore_name: str, session_id: str = Path(..., description="The agent session ID")):
    """Get agent session information and conversation history.

    Args:
        vectorstore_name: Name of the vectorstore the session uses
        session_id: The agent session ID

    Returns:
        JSON with session details and message history
    """
    try:
        validate_vectorstore_name(vectorstore_name)

        if not hasattr(app, 'chat_sessions') or session_id not in app.chat_sessions:
            raise HTTPException(status_code=404, detail='Agent session not found')

        session = app.chat_sessions[session_id]

        # Verify the vectorstore matches the session's vectorstore
        session_vectorstore = session.get('vectorstore_name')
        if session_vectorstore != vectorstore_name:
            raise HTTPException(status_code=400, detail=f'Session was created with vectorstore "{session_vectorstore}" but request uses "{vectorstore_name}"')

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

@app.delete('/api/agent/{vectorstore_name}/session/{session_id}')
async def api_delete_agent_session(vectorstore_name: str, session_id: str = Path(..., description="The agent session ID")):
    """Delete an agent session.

    Args:
        vectorstore_name: Name of the vectorstore the session uses
        session_id: The agent session ID

    Returns:
        JSON confirmation of deletion
    """
    try:
        validate_vectorstore_name(vectorstore_name)

        if not hasattr(app, 'chat_sessions') or session_id not in app.chat_sessions:
            raise HTTPException(status_code=404, detail='Agent session not found')

        session = app.chat_sessions[session_id]

        # Verify the vectorstore matches the session's vectorstore
        session_vectorstore = session.get('vectorstore_name')
        if session_vectorstore != vectorstore_name:
            raise HTTPException(status_code=400, detail=f'Session was created with vectorstore "{session_vectorstore}" but request uses "{vectorstore_name}"')

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

@app.get('/api/agent/{vectorstore_name}/sessions')
async def api_list_agent_sessions(vectorstore_name: str):
    """List all active agent sessions for a specific vectorstore.

    Args:
        vectorstore_name: Name of the vectorstore to filter sessions by

    Returns:
        JSON with list of active sessions for the vectorstore
    """
    try:
        validate_vectorstore_name(vectorstore_name)

        if not hasattr(app, 'chat_sessions'):
            app.chat_sessions = {}

        sessions = []
        for session_id, session in app.chat_sessions.items():
            # Only include sessions that match the requested vectorstore
            if session.get('vectorstore_name') == vectorstore_name:
                sessions.append({
                    'session_id': session_id,
                    'created_at': session['created_at'],
                    'last_activity': session['last_activity'],
                    'message_count': len(session['messages']),
                    'vectorstore_name': session['vectorstore_name'],
                    'system_prompt_preview': session['system_prompt'][:100] + '...' if len(session['system_prompt']) > 100 else session['system_prompt']
                })

        return {
            'success': True,
            'vectorstore_name': vectorstore_name,
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
            success, message = reinitialize_llm_manager()
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
        print(f"🦙 LLM API: Fetching available models from Ollama at {base_url}")
        
        # Make request to Ollama /api/tags endpoint
        ollama_url = f"{base_url}/api/tags"
        
        try:
            request = urllib.request.Request(ollama_url)
            with urllib.request.urlopen(request, timeout=10) as response:
                response_data = response.read()
        except urllib.error.URLError as e:
            if hasattr(e, 'reason'):
                # Connection error
                raise HTTPException(
                    status_code=503,
                    detail={
                        'success': False,
                        'error': f'Unable to connect to Ollama server at {base_url}',
                        'message': 'Please ensure Ollama is running and accessible',
                        'base_url': base_url
                    }
                )
            elif hasattr(e, 'code'):
                # HTTP error
                raise HTTPException(
                    status_code=502,
                    detail={
                        'success': False,
                        'error': f'HTTP error from Ollama server: {e.code}',
                        'message': f'Ollama server returned an error: {e.reason}',
                        'base_url': base_url
                    }
                )
        except Exception as e:
            if 'timeout' in str(e).lower():
                raise HTTPException(
                    status_code=504,
                    detail={
                        'success': False,
                        'error': f'Timeout connecting to Ollama server at {base_url}',
                        'message': 'Ollama server did not respond within 10 seconds',
                        'base_url': base_url
                    }
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail={
                        'success': False,
                        'error': f'Unexpected error connecting to Ollama: {str(e)}',
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
            ollama_data = json.loads(response_data.decode('utf-8'))
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

        print(f"🦙 LLM API: Found {len(models)} models")

        return {
            'success': True,
            'models': models,
            'count': len(models),
            'base_url': base_url
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

        # Update LangCache configuration
        if 'langcache' in data:
            langcache_config = data['langcache']

            # Update master enabled flag
            if 'enabled' in langcache_config:
                old_value = app_config['langcache'].get('enabled')
                new_value = bool(langcache_config['enabled'])

                if old_value != new_value:
                    app_config['langcache']['enabled'] = new_value
                    changes_made.append(f"langcache.enabled: {old_value} → {new_value}")

            # Update individual cache type settings
            if 'cache_types' in langcache_config:
                cache_types = langcache_config['cache_types']
                if isinstance(cache_types, dict):
                    for cache_type, enabled in cache_types.items():
                        if cache_type in app_config['langcache']['cache_types']:
                            old_value = app_config['langcache']['cache_types'].get(cache_type)
                            new_value = bool(enabled)

                            if old_value != new_value:
                                app_config['langcache']['cache_types'][cache_type] = new_value
                                changes_made.append(f"langcache.cache_types.{cache_type}: {old_value} → {new_value}")

        # Check if LLM configuration was changed and reinitialize if needed
        llm_reinitialized = False
        llm_reinit_error = None
        llm_config_changed = any('llm.' in change for change in changes_made)
        
        if llm_config_changed:
            print(f"🔄 LLM configuration changed, reinitializing LLM manager...")
            success, message = reinitialize_llm_manager()
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

        # Test LangCache configuration
        if 'langcache' in data:
            langcache_config = data['langcache']
            langcache_test = {
                'valid': True,
                'errors': [],
                'warnings': []
            }

            # Validate enabled flag
            if 'enabled' in langcache_config:
                if not isinstance(langcache_config['enabled'], bool):
                    langcache_test['errors'].append('enabled must be a boolean')
                    langcache_test['valid'] = False

            # Validate cache_types
            if 'cache_types' in langcache_config:
                cache_types = langcache_config['cache_types']
                if not isinstance(cache_types, dict):
                    langcache_test['errors'].append('cache_types must be an object')
                    langcache_test['valid'] = False
                else:
                    valid_cache_types = ['memory_extraction', 'query_optimization', 'embedding_optimization', 'context_analysis', 'memory_grounding']
                    for cache_type, enabled in cache_types.items():
                        if cache_type not in valid_cache_types:
                            langcache_test['warnings'].append(f'Unknown cache type: {cache_type}')
                        if not isinstance(enabled, bool):
                            langcache_test['errors'].append(f'cache_types.{cache_type} must be a boolean')
                            langcache_test['valid'] = False

            test_results['tests']['langcache'] = langcache_test
            if not langcache_test['valid']:
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
    global memory_agent

    # Check if already initialized to prevent double initialization
    if memory_agent is not None:
        print("ℹ️ Memory agent already initialized, skipping startup initialization")
        return

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

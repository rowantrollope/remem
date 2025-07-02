#!/usr/bin/env python3
"""
REST API for Memory Agent - FastAPI Version
"""

import os
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from memory.agent import LangGraphMemoryAgent
from memory.core_agent import MemoryAgent
from llm_manager import LLMManager, LLMConfig, init_llm_manager as initialize_llm_manager, get_llm_manager

app = FastAPI(title="Memory Agent API", description="REST API for Memory Agent", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
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
    }
}

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

def init_memory_agent():
    """Initialize the LangGraph memory agent with current configuration."""
    global memory_agent
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
# Pydantic Models for Request/Response Bodies
# =============================================================================

class StoreMemoryRequest(BaseModel):
    text: str
    apply_grounding: Optional[bool] = True
    vectorstore_name: Optional[str] = None

class SearchMemoryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    filter: Optional[str] = None
    optimize_query: Optional[bool] = False
    min_similarity: Optional[float] = 0.7
    vectorstore_name: Optional[str] = None

class DeleteMemoryRequest(BaseModel):
    vectorstore_name: Optional[str] = None

class SetContextRequest(BaseModel):
    location: Optional[str] = None
    activity: Optional[str] = None
    people_present: Optional[List[str]] = None

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    stream: Optional[bool] = False

class SessionRequest(BaseModel):
    session_id: Optional[str] = None
    system_prompt: Optional[str] = None
    memory_enabled: Optional[bool] = True
    temperature: Optional[float] = 0.1

class SessionMessageRequest(BaseModel):
    message: str
    stream: Optional[bool] = False
    store_memory: Optional[bool] = True
    top_k: Optional[int] = 10
    min_similarity: Optional[float] = 0.9

class KlineRecallRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10
    min_similarity: Optional[float] = 0.85

class KlineAskRequest(BaseModel):
    question: str
    context_memories: Optional[int] = 10
    min_similarity: Optional[float] = 0.85
    include_context: Optional[bool] = True

class ConfigUpdateRequest(BaseModel):
    redis: Optional[Dict[str, Any]] = None
    openai: Optional[Dict[str, Any]] = None
    llm: Optional[Dict[str, Any]] = None
    langgraph: Optional[Dict[str, Any]] = None
    memory_agent: Optional[Dict[str, Any]] = None
    web_server: Optional[Dict[str, Any]] = None

# =============================================================================
# NEME API - Fundamental Memory Operations (Inspired by Minsky's "Nemes")
# =============================================================================

@app.post('/api/memory')
async def api_store_neme(request: StoreMemoryRequest):
    """Store a new atomic memory (Neme).

    In Minsky's framework, a Neme is a fundamental unit of memory - an atomic
    piece of knowledge that can be contextually grounded and later recalled
    by higher-level cognitive processes.
    """
    try:
        memory_text = request.text.strip()

        if not memory_text:
            raise HTTPException(status_code=400, detail='Memory text is required')

        print(f"💾 NEME API: Storing atomic memory - '{memory_text[:60]}{'...' if len(memory_text) > 60 else ''}'")
        if request.vectorstore_name:
            print(f"📦 Vectorstore: {request.vectorstore_name}")

        # Use existing memory agent with vectorstore parameter
        if not memory_agent:
            raise HTTPException(status_code=500, detail='Memory agent not initialized')

        storage_result = memory_agent.memory_agent.store_memory(
            memory_text,
            apply_grounding=request.apply_grounding if request.apply_grounding is not None else True,
            vectorset_key=request.vectorstore_name or app_config["redis"]["vectorset_key"]
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
            'vectorstore_name': request.vectorstore_name or app_config["redis"]["vectorset_key"]
        }

        # Include grounding information if available
        if 'grounding_info' in storage_result:
            response_data['grounding_info'] = storage_result['grounding_info']

        # Include context snapshot if available
        if 'context_snapshot' in storage_result:
            response_data['context_snapshot'] = storage_result['context_snapshot']

        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/api/memory/search')
async def api_search_nemes(request: SearchMemoryRequest):
    """Search atomic memories (Nemes) using vector similarity.

    This performs direct vector similarity search across stored Nemes,
    returning the most relevant atomic memories for a given query.
    """
    try:
        query = request.query.strip()

        if not query:
            raise HTTPException(status_code=400, detail='Query is required')

        print(f"🔍 NEME API: Searching memories: {query} (top_k: {request.top_k}, min_similarity: {request.min_similarity})")
        if request.vectorstore_name:
            print(f"📦 Vectorstore: {request.vectorstore_name}")
        if request.filter:
            print(f"🔍 Filter: {request.filter}")
        if request.optimize_query:
            print(f"🔍 Query optimization: enabled")

        # Use existing memory agent with vectorstore parameter
        if not memory_agent:
            raise HTTPException(status_code=500, detail='Memory agent not initialized')

        # Use the memory agent for search operations with optional optimization
        if request.optimize_query:
            validation_result = memory_agent.memory_agent.processing.validate_and_preprocess_question(query)
            if validation_result["type"] == "search":
                search_query = validation_result.get("embedding_query") or validation_result["content"]
                print(f"🔍 Using optimized search query: '{search_query}'")
                search_result = memory_agent.memory_agent.search_memories_with_filtering_info(
                    search_query, 
                    request.top_k or 5, 
                    request.filter or "", 
                    request.min_similarity or 0.7, 
                    vectorset_key=request.vectorstore_name or app_config["redis"]["vectorset_key"]
                )
            else:
                search_result = memory_agent.memory_agent.search_memories_with_filtering_info(
                    query, 
                    request.top_k or 5, 
                    request.filter or "", 
                    request.min_similarity or 0.7, 
                    vectorset_key=request.vectorstore_name or app_config["redis"]["vectorset_key"]
                )
        else:
            search_result = memory_agent.memory_agent.search_memories_with_filtering_info(
                query, 
                request.top_k or 5, 
                request.filter or "", 
                request.min_similarity or 0.7, 
                vectorset_key=request.vectorstore_name or app_config["redis"]["vectorset_key"]
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
            'vectorstore_name': request.vectorstore_name or app_config["redis"]["vectorset_key"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/memory')
async def api_get_neme_info():
    """Get atomic memory (Neme) statistics and system information."""
    try:
        if not memory_agent:
            raise HTTPException(status_code=500, detail='Memory agent not initialized')

        # Get comprehensive memory information from underlying agent
        memory_info = memory_agent.memory_agent.get_memory_info()

        if 'error' in memory_info:
            raise HTTPException(status_code=500, detail=memory_info['error'])

        return {
            'success': True,
            **memory_info
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete('/api/memory/{memory_id}')
async def api_delete_neme(memory_id: str, request: Optional[DeleteMemoryRequest] = None):
    """Delete a specific atomic memory (Neme) by ID."""
    try:
        if not memory_id or not memory_id.strip():
            raise HTTPException(status_code=400, detail='Memory ID is required')

        print(f"🗑️ NEME API: Deleting atomic memory: {memory_id}")
        vectorstore_name = request.vectorstore_name if request and request.vectorstore_name else None
        if vectorstore_name:
            print(f"📦 Vectorstore: {vectorstore_name}")

        # Use existing memory agent with vectorstore parameter
        if not memory_agent:
            raise HTTPException(status_code=500, detail='Memory agent not initialized')

        success = memory_agent.memory_agent.delete_memory(
            memory_id.strip(),
            vectorset_key=vectorstore_name or app_config["redis"]["vectorset_key"]
        )

        if success:
            return {
                'success': True,
                'message': f'Neme {memory_id} deleted successfully',
                'memory_id': memory_id,
                'vectorstore_name': vectorstore_name or app_config["redis"]["vectorset_key"]
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

@app.delete('/api/memory')
async def api_delete_all_nemes(request: Optional[DeleteMemoryRequest] = None):
    """Clear all atomic memories (Nemes) from the system."""
    try:
        print("🗑️ NEME API: Clearing all atomic memories...")
        vectorstore_name = request.vectorstore_name if request and request.vectorstore_name else None
        if vectorstore_name:
            print(f"📦 Vectorstore: {vectorstore_name}")

        # Use existing memory agent with vectorstore parameter
        if not memory_agent:
            raise HTTPException(status_code=500, detail='Memory agent not initialized')

        result = memory_agent.memory_agent.clear_all_memories(vectorset_key=vectorstore_name or app_config["redis"]["vectorset_key"])

        if result['success']:
            return {
                'success': True,
                'message': result['message'],
                'memories_deleted': result['memories_deleted'],
                'vectorset_existed': result['vectorset_existed'],
                'vectorstore_name': vectorstore_name or app_config["redis"]["vectorset_key"]
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
async def api_set_neme_context(request: SetContextRequest):
    """Set current context for memory grounding."""
    try:
        if not memory_agent:
            raise HTTPException(status_code=500, detail='Memory agent not initialized')

        print(f"🌍 NEME API: Setting context - Location: {request.location}, Activity: {request.activity}, People: {request.people_present}")

        # Set context on underlying memory agent
        memory_agent.memory_agent.set_context(
            location=request.location or "",
            activity=request.activity or "",
            people_present=request.people_present or []
        )

        return {
            'success': True,
            'message': 'Context updated successfully',
            'context': {
                'location': request.location,
                'activity': request.activity,
                'people_present': request.people_present
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/memory/context')
async def api_get_neme_context():
    """Get current context information for memory grounding."""
    try:
        if not memory_agent:
            raise HTTPException(status_code=500, detail='Memory agent not initialized')

        current_context = memory_agent.memory_agent.core._get_current_context()

        return {
            'success': True,
            'context': current_context
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# K-Lines API - Mental State Construction
# =============================================================================

@app.post('/api/klines/recall')
async def api_kline_recall(request: KlineRecallRequest):
    """Construct a mental state (K-line) by recalling relevant memories."""
    try:
        query = request.query.strip()

        if not query:
            raise HTTPException(status_code=400, detail='Query is required')

        print(f"🧠 K-LINE API: Constructing mental state for: {query} (top_k: {request.top_k})")

        if not memory_agent:
            raise HTTPException(status_code=500, detail='Memory agent not initialized')

        # Search for relevant memories with embedding optimization
        validation_result = memory_agent.memory_agent.processing.validate_and_preprocess_question(query)

        if validation_result["type"] == "search":
            search_query = validation_result.get("embedding_query") or validation_result["content"]
            print(f"🔍 Using optimized search query: '{search_query}'")
            search_result = memory_agent.memory_agent.search_memories_with_filtering_info(
                search_query, request.top_k or 10, "", request.min_similarity or 0.85, vectorset_key=""
            )
        else:
            search_result = memory_agent.memory_agent.search_memories_with_filtering_info(
                query, request.top_k or 10, "", request.min_similarity or 0.85, vectorset_key=""
            )

        memories = search_result['memories']
        filtering_info = search_result['filtering_info']

        # Construct K-line (mental state) from memories
        if memories:
            kline_result = memory_agent.memory_agent.construct_kline(query, memories)
            mental_state = kline_result.get('mental_state', 'No mental state could be constructed.')
            coherence_score = kline_result.get('coherence_score', 0.0)
        else:
            mental_state = 'No relevant memories found to construct mental state.'
            coherence_score = 0.0

        return {
            'success': True,
            'query': query,
            'mental_state': mental_state,
            'coherence_score': coherence_score,
            'memories': memories,
            'memory_count': len(memories),
            'filtering_info': filtering_info
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/api/klines/ask')
async def api_kline_answer(request: KlineAskRequest):
    """Answer a question using K-line construction and reasoning."""
    try:
        question = request.question.strip()

        if not question:
            raise HTTPException(status_code=400, detail='Question is required')

        if not memory_agent:
            raise HTTPException(status_code=500, detail='Memory agent not initialized')

        print(f"🤔 K-LINE API: Answering question via mental state construction: {question}")

        # Use the memory agent's sophisticated answer_question method
        answer_response = memory_agent.memory_agent.answer_question(
            question, 
            top_k=request.context_memories or 10, 
            filterBy="", 
            min_similarity=request.min_similarity or 0.85
        )

        # Construct K-line (mental state) from the supporting memories
        supporting_memories = answer_response.get('supporting_memories', [])
        if supporting_memories:
            kline_result = memory_agent.memory_agent.construct_kline(
                query=question,
                memories=supporting_memories,
                answer=answer_response.get('answer') or '',
                confidence=str(answer_response.get('confidence') or 0),
                reasoning=answer_response.get('reasoning') or ''
            )
            print(f"🧠 Constructed K-line with coherence score: {kline_result.get('coherence_score', 0):.3f}")
        else:
            kline_result = {
                'mental_state': 'No relevant memories found to construct mental state.',
                'coherence_score': 0.0,
                'summary': 'Empty mental state'
            }

        return {
            'success': True,
            'question': question,
            **answer_response,
            'kline': kline_result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# Agent API - High-Level Orchestration
# =============================================================================

@app.post('/api/agent/chat')
async def api_agent_chat(request: ChatRequest):
    """Full conversational agent with integrated memory architecture."""
    try:
        message = request.message.strip()

        if not message:
            raise HTTPException(status_code=400, detail='Message is required')

        if not memory_agent:
            raise HTTPException(status_code=500, detail='Memory agent not initialized')

        print(f"💬 AGENT API: Processing chat message - '{message}'")

        # Use the LangGraph agent's run method
        response = memory_agent.run(message)

        return {
            'success': True,
            'message': message,
            'response': response
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Initialize sessions storage (in production, use Redis or database)
chat_sessions = {}

@app.post('/api/agent/session')
async def api_create_agent_session(request: SessionRequest):
    """Create a new agent session with integrated memory capabilities."""
    try:
        # For this conversion, we'll use a simple system prompt
        system_prompt = "You are a helpful AI assistant with access to memory."
        
        session_id = request.session_id or str(uuid.uuid4())

        session_data = {
            'system_prompt': system_prompt,
            'messages': [],
            'memory_enabled': request.memory_enabled if request.memory_enabled is not None else True,
            'temperature': request.temperature or 0.1,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'last_activity': datetime.now(timezone.utc).isoformat()
        }

        chat_sessions[session_id] = session_data

        print(f"🆕 AGENT API: Created agent session {session_id}")

        return {
            'success': True,
            'session_id': session_id,
            'memory_enabled': session_data['memory_enabled'],
            'created_at': session_data['created_at']
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/api/agent/session/{session_id}')
async def api_agent_session_message(session_id: str, request: SessionMessageRequest):
    """Send a message to an agent session."""
    try:
        if session_id not in chat_sessions:
            raise HTTPException(status_code=404, detail='Agent session not found')

        message = request.message.strip()
        if not message:
            raise HTTPException(status_code=400, detail='message is required')

        session = chat_sessions[session_id]

        # Add user message to session
        user_message = {
            'role': 'user',
            'content': message,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        session['messages'].append(user_message)

        # For simplicity in this conversion, we'll use the memory agent directly
        if not memory_agent:
            raise HTTPException(status_code=500, detail='Memory agent not initialized')

        print(f"💬 [{session_id}] AGENT API: Processing message: {message}")

        # Get response using the memory agent
        response = memory_agent.run(message)

        # Add assistant response to session
        assistant_message = {
            'role': 'assistant',
            'content': response,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        session['messages'].append(assistant_message)
        session['last_activity'] = datetime.now(timezone.utc).isoformat()

        return {
            'success': True,
            'session_id': session_id,
            'message': response,
            'conversation_length': len(session['messages']),
            'timestamp': assistant_message['timestamp']
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/agent/session/{session_id}')
async def api_get_agent_session(session_id: str):
    """Get agent session information."""
    try:
        if session_id not in chat_sessions:
            raise HTTPException(status_code=404, detail='Agent session not found')

        session = chat_sessions[session_id]

        return {
            'success': True,
            'session_id': session_id,
            'memory_enabled': session.get('memory_enabled', False),
            'created_at': session['created_at'],
            'last_activity': session['last_activity'],
            'message_count': len(session['messages'])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete('/api/agent/session/{session_id}')
async def api_delete_agent_session(session_id: str):
    """Delete an agent session."""
    try:
        if session_id not in chat_sessions:
            raise HTTPException(status_code=404, detail='Agent session not found')

        del chat_sessions[session_id]

        return {
            'success': True,
            'message': f'Session {session_id} deleted successfully',
            'session_id': session_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/agent/sessions')
async def api_list_agent_sessions():
    """List all agent sessions."""
    try:
        sessions = []
        for session_id, session_data in chat_sessions.items():
            sessions.append({
                'session_id': session_id,
                'memory_enabled': session_data.get('memory_enabled', False),
                'created_at': session_data['created_at'],
                'last_activity': session_data['last_activity'],
                'message_count': len(session_data['messages'])
            })

        return {
            'success': True,
            'sessions': sessions,
            'total_sessions': len(sessions)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# Health and Configuration APIs
# =============================================================================

@app.get('/api/health')
async def api_health():
    """Health check endpoint."""
    try:
        health_status = {
            'success': True,
            'status': 'healthy',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'memory_agent_initialized': memory_agent is not None
        }

        # Check Redis connection if memory agent is available
        if memory_agent:
            try:
                memory_info = memory_agent.memory_agent.get_memory_info()
                health_status['redis_connected'] = True
                health_status['memory_count'] = memory_info.get('memory_count', 0)
            except Exception as e:
                health_status['redis_connected'] = False
                health_status['redis_error'] = str(e)

        return health_status

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/config')
async def api_get_config():
    """Get current configuration."""
    try:
        return {
            'success': True,
            'config': app_config
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put('/api/config')
async def api_update_config(request: ConfigUpdateRequest):
    """Update configuration."""
    try:
        changes_made = []
        requires_restart = False
        warnings = []

        request_dict = request.dict(exclude_unset=True)

        # Update Redis configuration
        if 'redis' in request_dict:
            redis_config = request_dict['redis']
            for key, value in redis_config.items():
                if key in app_config['redis']:
                    old_value = app_config['redis'][key]
                    if old_value != value:
                        app_config['redis'][key] = value
                        changes_made.append(f"redis.{key}: {old_value} → {value}")
                        requires_restart = True

        # Update other configurations similarly...
        # (Simplified for this conversion)

        response_data = {
            'success': True,
            'changes_made': changes_made,
            'requires_restart': requires_restart,
            'warnings': warnings
        }

        if requires_restart:
            response_data['message'] = 'Configuration updated. Memory agent restart required for changes to take effect.'
        elif changes_made:
            response_data['message'] = 'Configuration updated successfully.'
        else:
            response_data['message'] = 'No changes were made to the configuration.'

        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/api/config/reload')
async def api_reload_config():
    """Reload configuration and restart the memory agent."""
    try:
        global memory_agent

        print("🔄 Reloading configuration and restarting memory agent...")

        # Reinitialize the LLM manager
        llm_success = init_llm_manager()

        # Reinitialize the memory agent
        success = init_memory_agent() and llm_success

        if success:
            return {
                'success': True,
                'message': 'Configuration reloaded and memory agent restarted successfully',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail='Failed to reinitialize memory agent with new configuration'
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    
    print("🚀 Starting Memory Agent Web Server...")

    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment variables.")
        print("Please set your OpenAI API key in the .env file or environment.")
        exit(1)

    # Initialize LLM manager
    if init_llm_manager():
        print("✅ LLM manager ready")
    else:
        print("❌ Failed to initialize LLM manager")
        exit(1)

    # Initialize LangGraph memory agent
    if init_memory_agent():
        print("✅ Memory agent ready")
        print("🌐 Server running at http://localhost:5001")
        print("📖 API docs available at http://localhost:5001/docs")
        print()
        
        uvicorn.run(app, host="0.0.0.0", port=5001)
    else:
        print("❌ Failed to initialize memory agent")
        exit(1)
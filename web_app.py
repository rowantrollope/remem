#!/usr/bin/env python3
"""
REST API for Memory Agent

"""

import os
import json
import uuid
from datetime import datetime
from typing import Dict, Any
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from memory.agent import LangGraphMemoryAgent
from memory.core_agent import MemoryAgent

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

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

def init_memory_agent():
    """Initialize the LangGraph memory agent with current configuration."""
    global memory_agent
    try:
        # Create memory agent with current Redis configuration
        base_memory_agent = MemoryAgent(
            redis_host=app_config["redis"]["host"],
            redis_port=app_config["redis"]["port"],
            redis_db=app_config["redis"]["db"]
        )

        # Create LangGraph agent with current OpenAI configuration
        memory_agent = LangGraphMemoryAgent(
            model_name=app_config["langgraph"]["model_name"],
            temperature=app_config["langgraph"]["temperature"]
        )

        # Replace the underlying memory agent with our configured one
        memory_agent.memory_agent = base_memory_agent

        return True
    except Exception as e:
        print(f"Failed to initialize LangGraph memory agent: {e}")
        return False

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/chat-demo')
def chat_demo():
    """Chat API demo page."""
    return render_template('chat_demo.html')

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

@app.route('/api/memory', methods=['POST'])
def api_store_neme():
    """Store a new atomic memory (Neme).

    In Minsky's framework, a Neme is a fundamental unit of memory - an atomic
    piece of knowledge that can be contextually grounded and later recalled
    by higher-level cognitive processes.

    Body:
        text (str): Memory text to store
        apply_grounding (bool, optional): Whether to apply contextual grounding (default: true)

    Returns:
        JSON with success status, memory_id, message, and grounding information:
        - success (bool): Whether the operation succeeded
        - memory_id (str): UUID of the stored memory
        - message (str): Success message
        - original_text (str): Original memory text (after parsing)
        - final_text (str): Final stored text (grounded or original)
        - grounding_applied (bool): Whether grounding was applied
        - tags (list): Extracted tags from the memory
        - timestamp (float): Unix timestamp
        - formatted_time (str): Human-readable timestamp
        - grounding_info (dict, optional): Details about grounding changes if applied
        - context_snapshot (dict, optional): Context used for grounding if applied
    """
    try:
        data = request.get_json()

        if not memory_agent:
            return jsonify({'error': 'Memory agent not initialized'}), 500

        memory_text = data.get('text', '').strip()
        apply_grounding = data.get('apply_grounding', True)

        if not memory_text:
            return jsonify({'error': 'Memory text is required'}), 400

        print(f"💾 NEME API: Storing atomic memory - '{memory_text[:60]}{'...' if len(memory_text) > 60 else ''}'")

        # Use the underlying memory agent for storage operations
        storage_result = memory_agent.memory_agent.store_memory(memory_text, apply_grounding=apply_grounding)

        # Prepare response with grounding information
        response_data = {
            'success': True,
            'memory_id': storage_result['memory_id'],
            'message': 'Memory stored successfully',
            'original_text': storage_result['original_text'],
            'final_text': storage_result['final_text'],
            'grounding_applied': storage_result['grounding_applied'],
            'tags': storage_result['tags'],
            'timestamp': storage_result['timestamp'],
            'formatted_time': storage_result['formatted_time']
        }

        # Include grounding information if available
        if 'grounding_info' in storage_result:
            response_data['grounding_info'] = storage_result['grounding_info']

        # Include context snapshot if available
        if 'context_snapshot' in storage_result:
            response_data['context_snapshot'] = storage_result['context_snapshot']

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/memory/search', methods=['POST'])
def api_search_nemes():
    """Search atomic memories (Nemes) using vector similarity.

    This performs direct vector similarity search across stored Nemes,
    returning the most relevant atomic memories for a given query.

    Body:
        query (str): Search query text
        top_k (int, optional): Number of results to return (default: 5)
        filter (str, optional): Filter expression for Redis VSIM command
        optimize_query (bool, optional): Whether to optimize query for embedding search (default: false)
        min_similarity (float, optional): Minimum similarity score threshold (0.0-1.0, default: 0.7)

    Returns:
        JSON with success status, memories array, count, and memory breakdown by type
    """
    try:
        data = request.get_json()

        if not memory_agent:
            return jsonify({'error': 'Memory agent not initialized'}), 500

        query = data.get('query', '').strip()
        top_k = data.get('top_k', 5)
        filter_expr = data.get('filter')
        optimize_query = data.get('optimize_query', False)  # Optional query optimization
        min_similarity = data.get('min_similarity', 0.7)  # Default to 0.7

        if not query:
            return jsonify({'error': 'Query is required'}), 400

        print(f"🔍 NEME API: Searching memories: {query} (top_k: {top_k}, min_similarity: {min_similarity})")
        if filter_expr:
            print(f"🔍 Filter: {filter_expr}")
        if optimize_query:
            print(f"🔍 Query optimization: enabled")

        # Use the underlying memory agent for search operations with optional optimization
        if optimize_query:
            validation_result = memory_agent.memory_agent.processing.validate_and_preprocess_question(query)
            if validation_result["type"] == "search":
                search_query = validation_result.get("embedding_query") or validation_result["content"]
                print(f"🔍 Using optimized search query: '{search_query}'")
                search_result = memory_agent.memory_agent.search_memories_with_filtering_info(search_query, top_k, filter_expr, min_similarity)
            else:
                search_result = memory_agent.memory_agent.search_memories_with_filtering_info(query, top_k, filter_expr, min_similarity)
        else:
            search_result = memory_agent.memory_agent.search_memories_with_filtering_info(query, top_k, filter_expr, min_similarity)

        memories = search_result['memories']
        filtering_info = search_result['filtering_info']
        print(f"🔍 NEME API: Search result type: {type(search_result)}")
        print(f"🔍 NEME API: Filtering info: {filtering_info}")

        return jsonify({
            'success': True,
            'query': query,
            'memories': memories,
            'count': len(memories),
            'filtering_info': filtering_info
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/memory', methods=['GET'])
def api_get_neme_info():
    """Get atomic memory (Neme) statistics and system information.

    Returns:
        JSON with memory count, vector dimension, embedding model, and system info
    """
    try:
        if not memory_agent:
            return jsonify({'error': 'Memory agent not initialized'}), 500

        # Get comprehensive memory information from underlying agent
        memory_info = memory_agent.memory_agent.get_memory_info()

        if 'error' in memory_info:
            return jsonify(memory_info), 500

        return jsonify({
            'success': True,
            **memory_info
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/memory/<memory_id>', methods=['DELETE'])
def api_delete_neme(memory_id):
    """Delete a specific atomic memory (Neme) by ID.

    Path Parameters:
        memory_id (str): UUID of the memory to delete

    Returns:
        JSON with success status and deletion details
    """
    try:
        if not memory_agent:
            return jsonify({'error': 'Memory agent not initialized'}), 500

        if not memory_id or not memory_id.strip():
            return jsonify({'error': 'Memory ID is required'}), 400

        print(f"🗑️ NEME API: Deleting atomic memory: {memory_id}")
        success = memory_agent.memory_agent.delete_memory(memory_id.strip())

        if success:
            return jsonify({
                'success': True,
                'message': f'Neme {memory_id} deleted successfully',
                'memory_id': memory_id
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Neme {memory_id} not found or could not be deleted'
            }), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/memory', methods=['DELETE'])
def api_delete_all_nemes():
    """Clear all atomic memories (Nemes) from the system.

    Returns:
        JSON with success status, deletion count, and operation details
    """
    try:
        if not memory_agent:
            return jsonify({'error': 'Memory agent not initialized'}), 500

        print("🗑️ NEME API: Clearing all atomic memories...")
        result = memory_agent.memory_agent.clear_all_memories()

        if result['success']:
            return jsonify({
                'success': True,
                'message': result['message'],
                'memories_deleted': result['memories_deleted'],
                'vectorset_existed': result['vectorset_existed']
            })
        else:
            return jsonify({
                'success': False,
                'error': result['error'],
                'memories_deleted': result.get('memories_deleted', 0)
            }), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/memory/context', methods=['POST'])
def api_set_neme_context():
    """Set current context for memory grounding.

    Body:
        location (str, optional): Current location
        activity (str, optional): Current activity
        people_present (list, optional): List of people present
        Additional fields will be stored as environment context

    Returns:
        JSON with success status and updated context
    """
    try:
        data = request.get_json()

        if not memory_agent:
            return jsonify({'error': 'Memory agent not initialized'}), 500

        # Extract context parameters
        location = data.get('location')
        activity = data.get('activity')
        people_present = data.get('people_present', [])

        # Extract additional environment context
        environment_context = {}
        for key, value in data.items():
            if key not in ['location', 'activity', 'people_present']:
                environment_context[key] = value

        print(f"🌍 NEME API: Setting context - Location: {location}, Activity: {activity}, People: {people_present}")

        # Set context on underlying memory agent
        memory_agent.memory_agent.set_context(
            location=location,
            activity=activity,
            people_present=people_present if people_present else None,
            **environment_context
        )

        return jsonify({
            'success': True,
            'message': 'Context updated successfully',
            'context': {
                'location': location,
                'activity': activity,
                'people_present': people_present,
                'environment': environment_context
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/memory/context', methods=['GET'])
def api_get_neme_context():
    """Get current context information for memory grounding.

    Returns:
        JSON with success status and current context (temporal, spatial, social, environmental)
    """
    try:
        if not memory_agent:
            return jsonify({'error': 'Memory agent not initialized'}), 500

        current_context = memory_agent.memory_agent._get_current_context()

        return jsonify({
            'success': True,
            'context': current_context
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

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

@app.route('/api/klines/recall', methods=['POST'])
def api_kline_recall():
    """Construct a mental state (K-line) by recalling relevant memories.

    This operation searches for and filters relevant Nemes to construct
    a coherent mental state for a specific query or context. The result
    is a formatted collection of memories that can be used for reasoning.

    Body:
        query (str): Query to construct mental state around
        top_k (int, optional): Number of memories to include (default: 5)
        filter (str, optional): Filter expression for Redis VSIM command
        use_llm_filtering (bool, optional): Apply LLM-based relevance filtering (default: true)
        min_similarity (float, optional): Minimum similarity score threshold (0.0-1.0, default: 0.7)

    Returns:
        JSON with formatted mental state, supporting memories, and memory breakdown by type
    """
    try:
        data = request.get_json()

        if not memory_agent:
            return jsonify({'error': 'Memory agent not initialized'}), 500

        query = data.get('query', '').strip()
        top_k = data.get('top_k', 5)
        filter_expr = data.get('filter')
        use_llm_filtering = data.get('use_llm_filtering', False)
        min_similarity = data.get('min_similarity', 0.7)

        if not query:
            return jsonify({'error': 'Query is required'}), 400

        print(f"🧠 K-LINE API: Constructing mental state for: {query} (top_k: {top_k})")
        if filter_expr:
            print(f"🔍 Filter: {filter_expr}")
        if use_llm_filtering:
            print(f"🤖 LLM filtering: enabled")

        # Search for relevant memories with embedding optimization
        validation_result = memory_agent.memory_agent.processing.validate_and_preprocess_question(query)

        if validation_result["type"] == "search":
            # Use the embedding-optimized query for vector search
            search_query = validation_result.get("embedding_query") or validation_result["content"]
            print(f"🔍 Using optimized search query: '{search_query}'")
            search_result = memory_agent.memory_agent.search_memories_with_filtering_info(search_query, top_k, filter_expr, min_similarity)
        else:
            # For help queries, still search but with original query
            search_result = memory_agent.memory_agent.search_memories_with_filtering_info(query, top_k, filter_expr, min_similarity)

        memories = search_result['memories']
        filtering_info = search_result['filtering_info']

        # Apply LLM filtering if requested (use original query for relevance filtering)
        if use_llm_filtering and memories:
            print(f"🤖 K-LINE API: Applying LLM filtering to {len(memories)} memories")
            filtered_memories = memory_agent.memory_agent.processing.filter_relevant_memories(query, memories)

            # Track filtering statistics
            original_count = len(memories)
            filtered_count = len(filtered_memories)
            print(f"🤖 K-LINE API: LLM filtering kept {filtered_count}/{original_count} memories")

            memories = filtered_memories

        # Construct K-line (mental state) from memories
        if memories:
            kline_result = memory_agent.memory_agent.construct_kline(query, memories)
            mental_state = kline_result.get('mental_state', 'No mental state could be constructed.')
            coherence_score = kline_result.get('coherence_score', 0.0)
        else:
            mental_state = 'No relevant memories found to construct mental state.'
            coherence_score = 0.0

        response_data = {
            'success': True,
            'query': query,
            'mental_state': mental_state,
            'coherence_score': coherence_score,
            'memories': memories,
            'memory_count': len(memories),
            'filtering_info': filtering_info
        }

        # Add filtering information if LLM filtering was used
        if use_llm_filtering:
            response_data['llm_filtering_applied'] = True
            if 'original_count' in locals():
                response_data['original_memory_count'] = original_count
                response_data['filtered_memory_count'] = filtered_count

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/klines/ask', methods=['POST'])
def api_kline_answer():
    """Answer a question using K-line construction and reasoning.

    This operation constructs a mental state from relevant Nemes and applies
    sophisticated reasoning to answer questions with confidence scoring.
    It represents the full cognitive process of memory recall + reasoning.

    K-lines are constructed but NOT stored - they exist only as temporary mental states.

    Body:
        question (str): Question to answer
        top_k (int, optional): Number of memories to retrieve for context (default: 5)
        filter (str, optional): Filter expression for Redis VSIM command
        min_similarity (float, optional): Minimum similarity score threshold (0.0-1.0, default: 0.7)

    Returns:
        JSON with structured response including answer, confidence, reasoning, supporting memories,
        and the constructed mental state (K-line)
    """
    try:
        data = request.get_json()

        if not memory_agent:
            return jsonify({'error': 'Memory agent not initialized'}), 500

        question = data.get('question', '').strip()
        top_k = data.get('top_k', 5)
        filter_expr = data.get('filter')
        min_similarity = data.get('min_similarity', 0.7)

        if not question:
            return jsonify({'error': 'Question is required'}), 400

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

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/klines/extract', methods=['POST'])
def api_kline_extract():
    """Extract and store memories from conversational data using K-line analysis.

    This operation uses sophisticated LLM analysis to identify valuable
    information in conversations and extract it as new Nemes. It represents
    the process of converting raw experience into structured memory.

    Body:
        raw_input (str): The full conversational data to analyze
        context_prompt (str): Application-specific context for extraction guidance
        extraction_examples (list, optional): Examples to guide LLM extraction
        apply_grounding (bool, optional): Whether to apply contextual grounding (default: True)

    Returns:
        JSON with extracted memories and extraction summary
    """
    try:
        data = request.get_json()

        if not memory_agent:
            return jsonify({'error': 'Memory agent not initialized'}), 500

        # Validate required fields
        raw_input = data.get('raw_input', '').strip()
        context_prompt = data.get('context_prompt', '').strip()

        if not raw_input:
            return jsonify({'error': 'raw_input is required'}), 400

        if not context_prompt:
            return jsonify({'error': 'context_prompt is required'}), 400

        # Optional parameters with defaults
        extraction_examples = data.get('extraction_examples', None)
        apply_grounding = data.get('apply_grounding', True)

        print(f"🔍 K-LINE API: Extracting memories from {len(raw_input)} characters of input")
        print(f"📋 Context: {context_prompt[:100]}...")

        # STEP 1: Search for existing relevant memories first (context-aware approach)
        print(f"🔍 K-LINE API: Searching for existing relevant memories...")
        existing_memories = memory_agent.memory_agent.search_memories(
            raw_input,
            top_k=10,
            min_similarity=0.7
        )

        if existing_memories:
            print(f"📚 K-LINE API: Found {len(existing_memories)} existing relevant memories")
        else:
            print(f"📚 K-LINE API: No existing relevant memories found")

        # STEP 2: Call the extract_and_store_memories method with context
        result = memory_agent.memory_agent.extract_and_store_memories(
            raw_input=raw_input,
            context_prompt=context_prompt,
            extraction_examples=extraction_examples,
            apply_grounding=apply_grounding,
            existing_memories=existing_memories  # Pass existing memories for context-aware extraction
        )

        return jsonify({
            'success': True,
            **result
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500



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

@app.route('/api/agent/chat', methods=['POST'])
def api_agent_chat():
    """Full conversational agent with integrated memory architecture.

    This endpoint orchestrates the complete cognitive architecture:
    - Searches relevant Nemes (atomic memories)
    - Constructs K-lines (mental states) for context
    - Applies sophisticated reasoning and language generation
    - Optionally extracts new memories from the conversation

    Body:
        message (str): User message/question
        system_prompt (str, optional): Custom system prompt to override default behavior

    Returns:
        JSON with success status, original message, and agent response
    """
    try:
        data = request.get_json()

        if not memory_agent:
            return jsonify({'error': 'Memory agent not initialized'}), 500

        message = data.get('message', '').strip()
        system_prompt = data.get('system_prompt', '').strip()

        if not message:
            return jsonify({'error': 'Message is required'}), 400

        print(f"💬 AGENT API: Processing chat message - '{message}'")
        if system_prompt:
            print(f"🎯 AGENT API: Using custom system prompt - '{system_prompt[:60]}{'...' if len(system_prompt) > 60 else ''}'")

        # Use the LangGraph agent's run method with optional custom system prompt
        response = memory_agent.run(message, system_prompt=system_prompt if system_prompt else None)

        return jsonify({
            'success': True,
            'message': message,
            'response': response,
            'system_prompt_used': bool(system_prompt)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500





# =============================================================================
# CHAT APIs - General purpose chat interface
# =============================================================================

@app.route('/api/agent/session', methods=['POST'])
def api_create_agent_session():
    """Create a new agent session with integrated memory capabilities.

    Body:
        system_prompt (str): Custom system prompt for the agent session
        session_id (str, optional): Custom session ID, auto-generated if not provided
        config (dict, optional): Additional configuration options
            - use_memory (bool, optional): Whether to enable memory features (default: true)
            - model (str, optional): OpenAI model to use (default: gpt-3.5-turbo)
            - temperature (float, optional): Response creativity (default: 0.7)
            - max_tokens (int, optional): Maximum response length (default: 1000)

    Returns:
        JSON with session_id and confirmation
    """
    try:
        data = request.get_json()

        system_prompt = data.get('system_prompt', '').strip()
        if not system_prompt:
            return jsonify({'error': 'system_prompt is required'}), 400

        session_id = data.get('session_id', str(uuid.uuid4()))
        config = data.get('config', {})

        # Extract memory configuration with default to true for agent sessions
        use_memory = config.get('use_memory', True)

        # Store session in memory (in production, use Redis or database)
        if not hasattr(app, 'chat_sessions'):
            app.chat_sessions = {}

        session_data = {
            'system_prompt': system_prompt,
            'messages': [],
            'config': config,
            'created_at': datetime.now().isoformat(),
            'last_activity': datetime.now().isoformat(),
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

        return jsonify({
            'success': True,
            'session_id': session_id,
            'system_prompt': system_prompt,
            'use_memory': use_memory,
            'created_at': session_data['created_at']
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/agent/session/<session_id>', methods=['POST'])
def api_agent_session_message(session_id):
    """Send a message to an agent session with full cognitive architecture.

    This endpoint provides the complete agent experience:
    - Searches relevant Nemes for context
    - Constructs K-lines (mental states)
    - Generates contextually-aware responses
    - Automatically extracts new memories from conversations

    Args:
        session_id: The agent session ID

    Body:
        message (str): The user's message
        stream (bool, optional): Whether to stream the response (default: False)
        store_memory (bool, optional): Whether to extract and store memories from this conversation (default: True for memory-enabled sessions)
        top_k (int, optional): Number of memories to search and return (default: 10)
        min_similarity (float, optional): Minimum similarity score threshold (0.0-1.0, default: 0.7)

    Returns:
        JSON with the assistant's response and conversation context.
        For memory-enabled sessions, includes 'memory_context' with relevant memories used in the response.
    """
    try:
        data = request.get_json()

        if not hasattr(app, 'chat_sessions') or session_id not in app.chat_sessions:
            return jsonify({'error': 'Agent session not found'}), 404

        message = data.get('message', '').strip()
        if not message:
            return jsonify({'error': 'message is required'}), 400

        stream = data.get('stream', False)
        store_memory = data.get('store_memory', True)  # Default to True for backward compatibility
        top_k = data.get('top_k', 10)  # Default to 10 memories
        min_similarity = data.get('min_similarity', 0.7)  # Default to 0.7

        # Validate top_k parameter
        if not isinstance(top_k, int) or top_k < 1:
            return jsonify({'error': 'top_k must be a positive integer'}), 400

        # Validate min_similarity parameter
        if not isinstance(min_similarity, (int, float)) or min_similarity < 0.0 or min_similarity > 1.0:
            return jsonify({'error': 'min_similarity must be a number between 0.0 and 1.0'}), 400

        session = app.chat_sessions[session_id]
        use_memory = session.get('use_memory', False)

        # Add user message to session
        user_message = {
            'role': 'user',
            'content': message,
            'timestamp': datetime.now().isoformat()
        }
        session['messages'].append(user_message)

        if use_memory:
            print(f"1) User said: '{message[:80]}{'...' if len(message) > 80 else ''}'")
        else:
            print(f"💬 [{session_id}] AGENT API: Agent session (memory: disabled): User: {message}")

        # Handle memory-enabled sessions
        if use_memory:
            return _handle_memory_enabled_message(session_id, session, message, stream, store_memory, top_k, min_similarity)
        else:
            return _handle_standard_message(session_id, session, stream)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


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

    # Get response from OpenAI
    if not openai_client:
        return jsonify({'error': 'OpenAI client not initialized'}), 500

    response = openai_client.chat.completions.create(
        model=session['config'].get('model', 'gpt-3.5-turbo'),
        messages=llm_messages,
        temperature=session['config'].get('temperature', 0.7),
        max_tokens=session['config'].get('max_tokens', 1000),
        stream=stream
    )

    if stream:
        # TODO: Implement streaming response
        return jsonify({'error': 'Streaming not yet implemented'}), 501

    assistant_response = response.choices[0].message.content

    # Add assistant response to session
    assistant_message = {
        'role': 'assistant',
        'content': assistant_response,
        'timestamp': datetime.now().isoformat()
    }
    session['messages'].append(assistant_message)
    session['last_activity'] = datetime.now().isoformat()

    return jsonify({
        'success': True,
        'session_id': session_id,
        'message': assistant_response,
        'conversation_length': len(session['messages']),
        'timestamp': assistant_message['timestamp']
    })


def _handle_memory_enabled_message(session_id, session, user_message, stream, store_memory=True, top_k=10, min_similarity=0.9):
    """Handle message processing for sessions with memory enabled."""
    if not memory_agent:
        return jsonify({'error': 'Memory agent not initialized but session requires memory'}), 500

    # Add to conversation buffer for memory extraction
    if 'conversation_buffer' not in session:
        session['conversation_buffer'] = []

    session['conversation_buffer'].append({
        'role': 'user',
        'content': user_message,
        'timestamp': datetime.now().isoformat()
    })

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
            print(f"3) Found {len(memory_results)} memories, applying LLM filtering...")
            # Apply LLM filtering for better relevance using original user message (not optimized query)
            filtered_memories = memory_agent.memory_agent.processing.filter_relevant_memories(user_message, memory_results)

            # Track filtering statistics
            original_count = len(memory_results)
            filtered_count = len(filtered_memories)
            print(f"3) LLM filtering kept {filtered_count}/{original_count} memories")

            relevant_memories = filtered_memories
        else:
            print(f"3) No relevant memories found")
    except Exception as e:
        print(f"⚠️ Memory search failed: {e}")

    # Prepare enhanced system prompt with memory context
    enhanced_system_prompt = session['system_prompt']
    if relevant_memories:
        memory_context = "\n\\Possibly relevant information from previous interactions:\n"
        for i, memory in enumerate(relevant_memories, 1):
            # All memories are now nemes (atomic memories)
            memory_context += f"{i}. {memory['text']}\n"

        memory_context += "\nEvaluate whether the memory is relevant to this prompt or not (discard if not relevant) if relevant, use this information to provide more personalized and contextual responses."
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

    # Get response from OpenAI
    if not openai_client:
        return jsonify({'error': 'OpenAI client not initialized'}), 500

    response = openai_client.chat.completions.create(
        model=session['config'].get('model', 'gpt-3.5-turbo'),
        messages=llm_messages,
        temperature=session['config'].get('temperature', 0.7),
        max_tokens=session['config'].get('max_tokens', 1000),
        stream=stream
    )

    if stream:
        # TODO: Implement streaming response
        return jsonify({'error': 'Streaming not yet implemented'}), 501

    assistant_response = response.choices[0].message.content

    # Add assistant response to session and buffer
    assistant_message = {
        'role': 'assistant',
        'content': assistant_response,
        'timestamp': datetime.now().isoformat()
    }
    session['messages'].append(assistant_message)
    session['conversation_buffer'].append(assistant_message)
    session['last_activity'] = datetime.now().isoformat()

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
    return jsonify(response_data)


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

    # Check if this message contains extractable information
    if not _contains_extractable_info([latest_user_message]):
        print(f"5) No extractable information found - skipping extraction")
        return

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
            session['last_extraction'] = datetime.now().isoformat()
        else:
            print(f"5) No memories identified for extraction")

        # Since we process each message individually, we can keep a smaller buffer
        # Keep only the last 4 messages (2 exchanges) for context
        session['conversation_buffer'] = buffer[-4:]

    except Exception as e:
        print(f"⚠️ Memory extraction failed: {e}")


def _contains_extractable_info(messages):
    """Check if messages contain information worth extracting using LLM evaluation."""
    if not openai_client:
        # Fallback to keyword-based approach if OpenAI client not available
        return _contains_extractable_info_fallback(messages)

    # Combine messages into conversation text
    conversation_text = ' '.join([msg['content'] for msg in messages])

    # Skip very short messages
    if len(conversation_text.strip()) < 10:
        return False

    # Use LLM to evaluate if the text contains extractable information
    try:
        evaluation_prompt = f"""You are an intelligent memory evaluation system. Your task is to determine if the following conversational text contains information that would be valuable to remember for future interactions.

VALUABLE INFORMATION INCLUDES:
- Personal preferences (likes, dislikes, habits)
- Constraints and requirements (budget, time, accessibility needs)
- Personal details (family, dietary restrictions, important dates)
- Factual information about people, places, or things
- Goals and intentions
- Important contextual details

STRICTLY IGNORE:
- Temporary information (current weather, today's schedule, immediate tasks)
- Conversational filler or pleasantries ("Hi there", "How are you?")
- General questions without personal context ("What's the best way to...")
- Information requests that don't reveal user preferences
- Time-sensitive information that won't be relevant later
- Assistant responses or suggestions

CONVERSATIONAL TEXT TO EVALUATE:
"{conversation_text}"

Respond with ONLY "YES" if the text contains valuable information worth remembering, or "NO" if it doesn't. Do not include any explanation."""

        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": evaluation_prompt}],
            temperature=0.1,  # Low temperature for consistent evaluation
            max_tokens=10     # We only need YES or NO
        )

        result = response.choices[0].message.content.strip().upper()
        return result == "YES"

    except Exception as e:
        print(f"⚠️ LLM evaluation failed, falling back to keyword approach: {e}")
        return _contains_extractable_info_fallback(messages)


def _contains_extractable_info_fallback(messages):
    """Fallback keyword-based approach for checking extractable information."""
    extractable_keywords = [
        'prefer', 'like', 'love', 'hate', 'dislike', 'always', 'never', 'usually',
        'budget', 'family', 'wife', 'husband', 'kids', 'children', 'allergic', 'allergy',
        'need', 'want', 'can\'t', 'cannot', 'must', 'have to', 'dietary', 'vegetarian',
        'vegan', 'gluten', 'accessibility', 'wheelchair', 'mobility', 'window seat',
        'aisle seat', 'michelin', 'restaurant', 'hotel', 'flight', 'travel', 'remember',
        'important', 'note', 'remind', 'don\'t forget'
    ]

    conversation_text = ' '.join([msg['content'] for msg in messages]).lower()
    return any(keyword in conversation_text for keyword in extractable_keywords)

@app.route('/api/agent/session/<session_id>', methods=['GET'])
def api_get_agent_session(session_id):
    """Get agent session information and conversation history.

    Args:
        session_id: The agent session ID

    Returns:
        JSON with session details and message history
    """
    try:
        if not hasattr(app, 'chat_sessions') or session_id not in app.chat_sessions:
            return jsonify({'error': 'Agent session not found'}), 404

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

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/agent/session/<session_id>', methods=['DELETE'])
def api_delete_agent_session(session_id):
    """Delete an agent session.

    Args:
        session_id: The agent session ID

    Returns:
        JSON confirmation of deletion
    """
    try:
        if not hasattr(app, 'chat_sessions') or session_id not in app.chat_sessions:
            return jsonify({'error': 'Agent session not found'}), 404

        del app.chat_sessions[session_id]

        print(f"🗑️ AGENT API: Deleted agent session {session_id}")

        return jsonify({
            'success': True,
            'message': f'Agent session {session_id} deleted'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/agent/sessions', methods=['GET'])
def api_list_agent_sessions():
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

        return jsonify({
            'success': True,
            'sessions': sessions,
            'total_sessions': len(sessions)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================================================
# SYSTEM APIs - Health checks and status
# =============================================================================

@app.route('/api/health')
def api_health():
    """System health check."""
    return jsonify({
        'status': 'healthy' if memory_agent else 'unhealthy',
        'service': 'LangGraph Memory Agent API',
        'timestamp': datetime.now().isoformat()
    })



# =============================================================================
# CONFIGURATION MANAGEMENT APIs - Runtime configuration management
# =============================================================================

@app.route('/api/config', methods=['GET'])
def api_get_config():
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

        # Add runtime information
        runtime_info = {
            "memory_agent_initialized": memory_agent is not None,
            "timestamp": datetime.now().isoformat()
        }

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

        return jsonify({
            'success': True,
            'config': safe_config,
            'runtime': runtime_info
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/config', methods=['PUT'])
def api_update_config():
    """Update system configuration.

    Body:
        Configuration object with any subset of configuration categories:
        - redis: {host, port, db, vectorset_key}
        - openai: {api_key, organization, embedding_model, embedding_dimension, chat_model, temperature}
        - langgraph: {model_name, temperature, system_prompt_enabled}
        - memory_agent: {default_top_k, apply_grounding_default, validation_enabled}
        - web_server: {host, port, debug, cors_enabled}

    Returns:
        JSON with success status, updated configuration, and any warnings
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'Configuration data is required'}), 400

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
                            return jsonify({'error': f'Redis {key} must be an integer'}), 400

                    if old_value != new_value:
                        app_config['redis'][key] = new_value
                        changes_made.append(f"redis.{key}: {old_value} → {new_value}")
                        requires_restart = True

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
                        return jsonify({'error': f'OpenAI {key} must be a number'}), 400

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
                    return jsonify({'error': 'LangGraph temperature must be a number'}), 400

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
                    return jsonify({'error': 'Memory agent default_top_k must be an integer'}), 400

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
                    return jsonify({'error': 'Web server port must be an integer'}), 400

            for key in ['debug', 'cors_enabled']:
                if key in web_config:
                    old_value = app_config['web_server'].get(key)
                    new_value = bool(web_config[key])

                    if old_value != new_value:
                        app_config['web_server'][key] = new_value
                        changes_made.append(f"web_server.{key}: {old_value} → {new_value}")
                        warnings.append(f"Web server {key} change requires application restart to take effect")

        # Prepare response
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

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/config/reload', methods=['POST'])
def api_reload_config():
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

        # Reinitialize the memory agent with current configuration
        success = init_memory_agent()

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

            return jsonify({
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
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
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
            }), 500

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Error occurred during configuration reload'
        }), 500

@app.route('/api/config/test', methods=['POST'])
def api_test_config():
    """Test configuration without applying it.

    Body:
        Configuration object to test (same format as PUT /api/config)

    Returns:
        JSON with test results for each configuration component
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'Configuration data is required for testing'}), 400

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

        return jsonify({
            'success': True,
            'test_results': test_results,
            'message': 'Configuration test completed',
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Memory Agent Web Server...")

    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment variables.")
        print("Please set your OpenAI API key in the .env file or environment.")
        exit(1)

    # Initialize LangGraph memory agent
    if init_memory_agent():
        print("✅ Memory agent ready")
        print("🌐 Server running at http://localhost:5001")
        print("📖 API docs available at http://localhost:5001")
        print()
        # Disable Flask's debug output by setting debug=False and configuring logging
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.WARNING)
        app.run(debug=False, host='0.0.0.0', port=5001)
    else:
        print("❌ Failed to initialize memory agent")
        exit(1)

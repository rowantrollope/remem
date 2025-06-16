#!/usr/bin/env python3
"""
Web UI for Memory Agent

A minimalist Flask web interface for the memory agent.
"""

import os
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from langgraph_memory_agent import LangGraphMemoryAgent

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the memory agent
memory_agent = None

def init_memory_agent():
    """Initialize the LangGraph memory agent."""
    global memory_agent
    try:
        memory_agent = LangGraphMemoryAgent()
        return True
    except Exception as e:
        print(f"Failed to initialize LangGraph memory agent: {e}")
        return False

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

# =============================================================================
# DEVELOPER MEMORY APIs - Core memory operations for agent developers
# =============================================================================

@app.route('/api/memory', methods=['POST'])
def api_store_memory():
    """Store a new memory.

    Body:
        text (str): Memory text to store
        apply_grounding (bool, optional): Whether to apply contextual grounding (default: true)

    Returns:
        JSON with success status, memory_id, and message
    """
    try:
        data = request.get_json()

        if not memory_agent:
            return jsonify({'error': 'Memory agent not initialized'}), 500

        memory_text = data.get('text', '').strip()
        apply_grounding = data.get('apply_grounding', True)

        if not memory_text:
            return jsonify({'error': 'Memory text is required'}), 400

        print(f"📝 Storing memory: {memory_text}")
        print(f"🔧 Apply grounding: {apply_grounding}")

        # Use the underlying memory agent for storage operations
        memory_id = memory_agent.memory_agent.store_memory(memory_text, apply_grounding=apply_grounding)

        return jsonify({
            'success': True,
            'memory_id': memory_id,
            'message': 'Memory stored successfully'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/memory/search', methods=['POST'])
def api_search_memories():
    """Search memories using vector similarity.

    Body:
        query (str): Search query text
        top_k (int, optional): Number of results to return (default: 5)
        filter (str, optional): Filter expression for Redis VSIM command

    Returns:
        JSON with success status, memories array, and count
    """
    try:
        data = request.get_json()

        if not memory_agent:
            return jsonify({'error': 'Memory agent not initialized'}), 500

        query = data.get('query', '').strip()
        top_k = data.get('top_k', 5)
        filter_expr = data.get('filter')

        if not query:
            return jsonify({'error': 'Query is required'}), 400

        print(f"🔍 Searching memories: {query} (top_k: {top_k})")
        if filter_expr:
            print(f"🔍 Filter: {filter_expr}")

        # Use the underlying memory agent for search operations
        memories = memory_agent.memory_agent.search_memories(query, top_k, filter_expr)

        return jsonify({
            'success': True,
            'query': query,
            'memories': memories,
            'count': len(memories)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/memory/answer', methods=['POST'])
def api_answer_question():
    """Answer a question using advanced memory analysis with confidence scoring.

    This endpoint calls memory_agent.answer_question() directly for sophisticated
    confidence analysis and structured responses with supporting memories.

    Body:
        question (str): Question to answer
        top_k (int, optional): Number of memories to retrieve for context (default: 5)
        filter (str, optional): Filter expression for Redis VSIM command

    Returns:
        JSON with structured response including answer, confidence, reasoning, and supporting memories
    """
    try:
        data = request.get_json()

        if not memory_agent:
            return jsonify({'error': 'Memory agent not initialized'}), 500

        question = data.get('question', '').strip()
        top_k = data.get('top_k', 5)
        filter_expr = data.get('filter')

        if not question:
            return jsonify({'error': 'Question is required'}), 400

        print(f"🤔 Answering question: {question} (top_k: {top_k})")
        if filter_expr:
            print(f"🔍 Filter: {filter_expr}")

        # Use the memory agent's sophisticated answer_question method directly
        # This preserves the high-quality confidence scoring and structured responses
        answer_response = memory_agent.memory_agent.answer_question(question, top_k=top_k, filterBy=filter_expr)

        return jsonify({
            'success': True,
            'question': question,
            **answer_response  # Spread the structured response (answer, confidence, supporting_memories, etc.)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/memory', methods=['GET'])
def api_get_memory_info():
    """Get memory statistics and system information.

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

# =============================================================================
# CHAT APPLICATION API - Conversational interface for demo/UI applications
# =============================================================================

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """Conversational interface using LangGraph workflow for complex multi-step reasoning.

    This endpoint uses the full LangGraph agent workflow which can intelligently
    orchestrate memory tools and provide sophisticated conversational capabilities.

    Body:
        message (str): User message/question

    Returns:
        JSON with success status, original message, and agent response
    """
    try:
        data = request.get_json()

        if not memory_agent:
            return jsonify({'error': 'Memory agent not initialized'}), 500

        message = data.get('message', '').strip()

        if not message:
            return jsonify({'error': 'Message is required'}), 400

        print(f"💬 Chat message: {message}")

        # Use the LangGraph agent's run method for full workflow orchestration
        response = memory_agent.run(message)

        return jsonify({
            'success': True,
            'message': message,
            'response': response
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

@app.route('/api/memory/context', methods=['POST'])
def api_set_context():
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

        print(f"🌍 Setting context - Location: {location}, Activity: {activity}, People: {people_present}")

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
def api_get_context():
    """Get current context information.

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

@app.route('/api/memory/<memory_id>', methods=['DELETE'])
def api_delete_memory(memory_id):
    """Delete a specific memory by ID.

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

        print(f"🗑️ Deleting memory: {memory_id}")
        success = memory_agent.memory_agent.delete_memory(memory_id.strip())

        if success:
            return jsonify({
                'success': True,
                'message': f'Memory {memory_id} deleted successfully',
                'memory_id': memory_id
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Memory {memory_id} not found or could not be deleted'
            }), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/memory', methods=['DELETE'])
def api_delete_all_memories():
    """Clear all memories from the system.

    Returns:
        JSON with success status, deletion count, and operation details
    """
    try:
        if not memory_agent:
            return jsonify({'error': 'Memory agent not initialized'}), 500

        print("🗑️ Clearing all memories...")
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

if __name__ == '__main__':
    print("🚀 Starting LangGraph Memory Agent Web UI...")
    
    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment variables.")
        print("Please set your OpenAI API key in the .env file or environment.")
        exit(1)
    
    # Initialize LangGraph memory agent
    if init_memory_agent():
        print("✅ LangGraph memory agent initialized successfully")
        print("🌐 Starting web server...")
        app.run(debug=True, host='0.0.0.0', port=5001)
    else:
        print("❌ Failed to initialize LangGraph memory agent")
        exit(1)

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
from memory_agent import MemoryAgent

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the memory agent
memory_agent = None

def init_memory_agent():
    """Initialize the memory agent."""
    global memory_agent
    try:
        memory_agent = MemoryAgent()
        return True
    except Exception as e:
        print(f"Failed to initialize memory agent: {e}")
        return False

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/api/remember', methods=['POST'])
def api_remember():
    """API endpoint to store a memory."""
    try:
        data = request.get_json()
        memory_text = data.get('memory', '').strip()
        apply_grounding = data.get('apply_grounding', True)

        if not memory_text:
            return jsonify({'error': 'Memory text is required'}), 400

        if not memory_agent:
            return jsonify({'error': 'Memory agent not initialized'}), 500

        memory_id = memory_agent.store_memory(memory_text, apply_grounding=apply_grounding)

        return jsonify({
            'success': True,
            'memory_id': memory_id,
            'message': 'Memory stored successfully',
            'grounding_applied': apply_grounding
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/recall', methods=['POST'])
def api_recall():
    """API endpoint to search memories."""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        top_k = data.get('top_k', 3)
        filterBy = data.get('filterBy')

        if not query:
            return jsonify({'error': 'Query is required'}), 400

        if not memory_agent:
            return jsonify({'error': 'Memory agent not initialized'}), 500

        if filterBy:
            print(f"🔍 Recall with filter: {filterBy}")

        memories = memory_agent.search_memories(query, top_k, filterBy)

        return jsonify({
            'success': True,
            'memories': memories,
            'count': len(memories)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ask', methods=['POST'])
def api_ask():
    """API endpoint to ask a question."""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()

        if not question:
            return jsonify({'error': 'Question is required'}), 400

        if not memory_agent:
            return jsonify({'error': 'Memory agent not initialized'}), 500

        top_k = data.get('top_k', 5)
        filterBy = data.get('filterBy')
        print(f"top_k: {top_k}")

        if filterBy:
            print(f"🔍 Ask with filter: {filterBy}")

        answer_response = memory_agent.answer_question(question, top_k=top_k, filterBy=filterBy)

        return jsonify({
            'success': True,
            'question': question,
            **answer_response  # Spread the structured response (answer, confidence, supporting_memories, etc.)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def api_status():
    """API endpoint to check system status."""
    return jsonify({
        'status': 'ready' if memory_agent else 'not_initialized',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/memory-info')
def api_memory_info():
    """API endpoint to get information about all stored memories."""
    try:
        if not memory_agent:
            return jsonify({'error': 'Memory agent not initialized'}), 500

        # Get comprehensive memory information
        memory_info = memory_agent.get_memory_info()

        if 'error' in memory_info:
            return jsonify(memory_info), 500

        return jsonify({
            'success': True,
            **memory_info
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/context', methods=['POST'])
def api_set_context():
    """API endpoint to set current context for memory grounding."""
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

        # Set context
        memory_agent.set_context(
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

@app.route('/api/context', methods=['GET'])
def api_get_context():
    """API endpoint to get current context information."""
    try:
        if not memory_agent:
            return jsonify({'error': 'Memory agent not initialized'}), 500

        current_context = memory_agent._get_current_context()

        return jsonify({
            'success': True,
            'context': current_context
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete/<memory_id>', methods=['DELETE'])
def api_delete_memory(memory_id):
    """API endpoint to delete a specific memory."""
    try:
        if not memory_agent:
            return jsonify({'error': 'Memory agent not initialized'}), 500

        if not memory_id or not memory_id.strip():
            return jsonify({'error': 'Memory ID is required'}), 400

        success = memory_agent.delete_memory(memory_id.strip())

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

@app.route('/api/delete-all', methods=['DELETE'])
def api_delete_all_memories():
    """API endpoint to clear all memories."""
    try:
        if not memory_agent:
            return jsonify({'error': 'Memory agent not initialized'}), 500

        result = memory_agent.clear_all_memories()

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
    print("🚀 Starting Memory Agent Web UI...")
    
    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment variables.")
        print("Please set your OpenAI API key in the .env file or environment.")
        exit(1)
    
    # Initialize memory agent
    if init_memory_agent():
        print("✅ Memory agent initialized successfully")
        print("🌐 Starting web server...")
        app.run(debug=True, host='0.0.0.0', port=5001)
    else:
        print("❌ Failed to initialize memory agent")
        exit(1)

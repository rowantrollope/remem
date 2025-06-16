#!/usr/bin/env python3
"""
Web UI for Memory Agent

A minimalist Flask web interface for the memory agent.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from langgraph_memory_agent import LangGraphMemoryAgent
from memory_agent import MemoryAgent

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the memory agent
memory_agent = None

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

        print(f"📝 Storing memory: {memory_text}")
        print(f"🔧 Apply grounding: {apply_grounding}")

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

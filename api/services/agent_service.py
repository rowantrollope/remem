"""
Agent service for managing conversational sessions and memory-enabled interactions.
"""

import uuid
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from ..models.agent import AgentSessionCreateRequest, AgentSessionMessageRequest
from ..core.exceptions import server_error, validation_error, not_found_error
from ..core.utils import get_current_timestamp


class AgentService:
    """Service for agent session management and conversational interactions."""
    
    def __init__(self, memory_agent, chat_sessions: Dict[str, Any]):
        self.memory_agent = memory_agent
        self.chat_sessions = chat_sessions
    
    async def create_session(self, vectorstore_name: str, request: AgentSessionCreateRequest) -> Dict[str, Any]:
        """Create a new agent session with integrated memory capabilities."""
        try:
            system_prompt = request.system_prompt.strip()
            if not system_prompt:
                raise validation_error('system_prompt is required')

            session_id = request.session_id or str(uuid.uuid4())
            config = request.config or {}

            # Extract memory configuration with default to true for agent sessions
            use_memory = config.get('use_memory', True)

            session_data = {
                'system_prompt': system_prompt,
                'messages': [],
                'config': config,
                'created_at': get_current_timestamp(),
                'last_activity': get_current_timestamp(),
                'use_memory': use_memory,
                'vectorstore_name': vectorstore_name
            }

            # Add memory-specific fields if memory is enabled
            if use_memory:
                session_data.update({
                    'conversation_buffer': [],  # Buffer for memory extraction
                    'extraction_threshold': 2,  # Number of messages before extraction
                    'last_extraction': None,    # Timestamp of last memory extraction
                    'memory_context': f"Chat session for: {system_prompt[:100]}..."  # Context for memory extraction
                })

            self.chat_sessions[session_id] = session_data

            memory_status = "enabled" if use_memory else "disabled"
            print(f"🆕 AGENT API: Created agent session {session_id} (memory: {memory_status})")

            return {
                'success': True,
                'session_id': session_id,
                'system_prompt': system_prompt,
                'use_memory': use_memory,
                'created_at': session_data['created_at']
            }

        except Exception as e:
            raise server_error(str(e))
    
    async def get_session(self, vectorstore_name: str, session_id: str) -> Dict[str, Any]:
        """Get agent session information and conversation history."""
        try:
            if session_id not in self.chat_sessions:
                raise not_found_error('Agent session not found')

            session = self.chat_sessions[session_id]

            # Verify the vectorstore matches the session's vectorstore
            session_vectorstore = session.get('vectorstore_name')
            if session_vectorstore != vectorstore_name:
                raise validation_error(
                    f'Session was created with vectorstore "{session_vectorstore}" but request uses "{vectorstore_name}"'
                )

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

        except Exception as e:
            raise server_error(str(e))
    
    async def delete_session(self, vectorstore_name: str, session_id: str) -> Dict[str, Any]:
        """Delete an agent session."""
        try:
            if session_id not in self.chat_sessions:
                raise not_found_error('Agent session not found')

            session = self.chat_sessions[session_id]

            # Verify the vectorstore matches the session's vectorstore
            session_vectorstore = session.get('vectorstore_name')
            if session_vectorstore != vectorstore_name:
                raise validation_error(
                    f'Session was created with vectorstore "{session_vectorstore}" but request uses "{vectorstore_name}"'
                )

            del self.chat_sessions[session_id]

            print(f"🗑️ AGENT API: Deleted agent session {session_id}")

            return {
                'success': True,
                'message': f'Agent session {session_id} deleted'
            }

        except Exception as e:
            raise server_error(str(e))
    
    async def list_sessions(self, vectorstore_name: str) -> Dict[str, Any]:
        """List all active agent sessions for a specific vectorstore."""
        try:
            sessions = []
            for session_id, session in self.chat_sessions.items():
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
            raise server_error(str(e))

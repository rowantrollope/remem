"""
Agent API endpoints - High-Level Orchestration (Full Cognitive Architecture).

The Agent API represents the highest level of cognitive architecture,
orchestrating both Memories (atomic memories) and K-lines (mental states)
to provide sophisticated conversational and reasoning capabilities.

These APIs handle:
- Full conversational agents with memory integration
- Session management with persistent context
- Complex multi-step reasoning workflows
- Integration of memory operations with language generation
"""

from fastapi import APIRouter, HTTPException, Depends, Path, Body
from typing import Dict, Any
from ..models.agent import AgentChatRequest, AgentSessionCreateRequest, AgentSessionMessageRequest
from ..services.agent_service import AgentService
from ..dependencies import get_memory_agent
from ..core.utils import validate_vectorstore_name
from ..core.exceptions import server_error

router = APIRouter(prefix="/api/agent", tags=["agent"])

# Global chat sessions store (in production, use Redis or database)
chat_sessions: Dict[str, Any] = {}


def get_agent_service(memory_agent=Depends(get_memory_agent)) -> AgentService:
    """Get agent service dependency."""
    return AgentService(memory_agent, chat_sessions)


@router.post('/chat')
async def agent_chat(
    request: AgentChatRequest,
    memory_agent=Depends(get_memory_agent)
) -> Dict[str, Any]:
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

    except Exception as e:
        raise server_error(str(e))


@router.post('/{vectorstore_name}/session')
async def create_agent_session(
    vectorstore_name: str,
    request: AgentSessionCreateRequest,
    agent_service: AgentService = Depends(get_agent_service)
) -> Dict[str, Any]:
    """Create a new agent session with integrated memory capabilities.

    Args:
        vectorstore_name: Name of the vectorstore to use for memory operations
        request: Session creation request with system prompt and configuration

    Returns:
        JSON with session_id and confirmation
    """
    validate_vectorstore_name(vectorstore_name)
    return await agent_service.create_session(vectorstore_name, request)


@router.get('/{vectorstore_name}/session/{session_id}')
async def get_agent_session(
    vectorstore_name: str,
    session_id: str = Path(..., description="The agent session ID"),
    agent_service: AgentService = Depends(get_agent_service)
) -> Dict[str, Any]:
    """Get agent session information and conversation history.

    Args:
        vectorstore_name: Name of the vectorstore the session uses
        session_id: The agent session ID

    Returns:
        JSON with session details and message history
    """
    validate_vectorstore_name(vectorstore_name)
    return await agent_service.get_session(vectorstore_name, session_id)


@router.delete('/{vectorstore_name}/session/{session_id}')
async def delete_agent_session(
    vectorstore_name: str,
    session_id: str = Path(..., description="The agent session ID"),
    agent_service: AgentService = Depends(get_agent_service)
) -> Dict[str, Any]:
    """Delete an agent session.

    Args:
        vectorstore_name: Name of the vectorstore the session uses
        session_id: The agent session ID

    Returns:
        JSON confirmation of deletion
    """
    validate_vectorstore_name(vectorstore_name)
    return await agent_service.delete_session(vectorstore_name, session_id)


@router.get('/{vectorstore_name}/sessions')
async def list_agent_sessions(
    vectorstore_name: str,
    agent_service: AgentService = Depends(get_agent_service)
) -> Dict[str, Any]:
    """List all active agent sessions for a specific vectorstore.

    Args:
        vectorstore_name: Name of the vectorstore to filter sessions by

    Returns:
        JSON with list of active sessions for the vectorstore
    """
    validate_vectorstore_name(vectorstore_name)
    return await agent_service.list_sessions(vectorstore_name)

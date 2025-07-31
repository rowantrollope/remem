"""
Configuration management API endpoints.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
from ..models.config import ConfigUpdateRequest
from ..services.config_service import ConfigService
from ..dependencies import get_memory_agent_optional
from ..core.exceptions import server_error
from ..core.utils import get_current_timestamp

router = APIRouter(prefix="/api/config", tags=["configuration"])


def get_config_service(memory_agent=Depends(get_memory_agent_optional)) -> ConfigService:
    """Get configuration service dependency."""
    return ConfigService(memory_agent)


@router.get('')
async def get_configuration(
    config_service: ConfigService = Depends(get_config_service)
) -> Dict[str, Any]:
    """Get current system configuration.

    Returns:
        JSON with complete system configuration including Redis, OpenAI, LangGraph,
        memory agent, and web server settings
    """
    return await config_service.get_configuration()


@router.put('')
async def update_configuration(
    request: ConfigUpdateRequest,
    config_service: ConfigService = Depends(get_config_service)
) -> Dict[str, Any]:
    """Update system configuration.

    Args:
        request: Configuration object with any subset of configuration categories:
            - redis: {host, port, db, vectorset_key}
            - llm: {tier1: {provider, model, temperature, max_tokens, base_url, api_key, timeout}, tier2: {...}}
            - embedding: {provider, model, dimension, base_url, api_key, timeout}
            - openai: {api_key, organization, embedding_model, embedding_dimension, chat_model, temperature}
            - langgraph: {model_name, temperature, system_prompt_enabled}
            - memory_agent: {default_top_k, apply_grounding_default, validation_enabled}
            - web_server: {host, port, debug, cors_enabled}
            - langcache: {enabled, cache_types}

    Returns:
        JSON with success status, updated configuration, and any warnings
    """
    updates = request.dict(exclude_unset=True)
    return await config_service.update_configuration(updates)


@router.post('/test')
async def test_configuration(
    request: ConfigUpdateRequest,
    config_service: ConfigService = Depends(get_config_service)
) -> Dict[str, Any]:
    """Test configuration without applying changes.

    Args:
        request: Configuration object to test (same format as update)

    Returns:
        JSON with test results including validation status and connection tests
    """
    test_config = request.dict(exclude_unset=True)
    return await config_service.test_configuration(test_config)


@router.post('/reload')
async def reload_configuration(
    config_service: ConfigService = Depends(get_config_service)
) -> Dict[str, Any]:
    """Reload configuration and restart the memory agent.

    This endpoint reinitializes the memory agent with the current configuration.
    Useful after making configuration changes that require a restart.

    Returns:
        JSON with success status and reload details
    """
    try:
        from ..startup import startup
        
        print("🔄 Reloading configuration and restarting memory agent...")
        
        # Reinitialize services
        startup()
        
        return {
            'success': True,
            'message': 'Configuration reloaded and services restarted successfully',
            'timestamp': get_current_timestamp()
        }
        
    except Exception as e:
        raise server_error(f'Failed to reload configuration: {str(e)}')

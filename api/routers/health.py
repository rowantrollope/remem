"""
Health check API endpoints.
"""

from fastapi import APIRouter, Depends
from datetime import datetime, timezone
from ..models.responses import HealthResponse
from ..dependencies import get_memory_agent

router = APIRouter(prefix="/api", tags=["health"])


@router.get('/health', response_model=HealthResponse)
async def api_health(memory_agent=Depends(get_memory_agent)):
    """System health check."""
    return HealthResponse(
        status='healthy' if memory_agent else 'unhealthy',
        service='LangGraph Memory Agent API',
        timestamp=datetime.now(timezone.utc).isoformat()
    )

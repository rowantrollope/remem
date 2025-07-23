"""
FastAPI dependencies for dependency injection.
"""

from fastapi import Depends, HTTPException
from typing import Optional
from .core.exceptions import MemoryAgentNotInitializedError

# Global memory agent instance (will be set during startup)
_memory_agent = None


def set_memory_agent(agent):
    """Set the global memory agent instance."""
    global _memory_agent
    _memory_agent = agent


def get_memory_agent():
    """Get the memory agent dependency."""
    if _memory_agent is None:
        raise HTTPException(
            status_code=500, 
            detail="Memory agent not initialized"
        )
    return _memory_agent


def get_memory_agent_optional() -> Optional[object]:
    """Get the memory agent dependency (optional)."""
    return _memory_agent

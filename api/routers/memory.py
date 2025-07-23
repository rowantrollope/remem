"""
Memory API endpoints - NEME (atomic memory) operations.

These endpoints handle fundamental memory operations inspired by Minsky's "Memories":
- Store atomic memories with contextual grounding
- Vector similarity search across stored memories
- Memory lifecycle management (delete, clear)
- Context management for grounding operations
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
from ..models.memory import MemoryStoreRequest, MemorySearchRequest, ContextSetRequest
from ..models.responses import MemoryInfoResponse
from ..services.memory_service import MemoryService
from ..dependencies import get_memory_agent
from ..core.utils import validate_vectorstore_name

router = APIRouter(prefix="/api/memory", tags=["memory"])


def get_memory_service(memory_agent=Depends(get_memory_agent)) -> MemoryService:
    """Get memory service dependency."""
    return MemoryService(memory_agent)


@router.post('/{vectorstore_name}')
async def store_memory(
    vectorstore_name: str,
    request: MemoryStoreRequest,
    memory_service: MemoryService = Depends(get_memory_service)
) -> Dict[str, Any]:
    """Store a new atomic memory (Neme) in a specific vectorstore.

    Args:
        vectorstore_name: Name of the vectorstore to store in
        request: Memory store request with text and grounding options

    Returns:
        JSON with success status, memory_id, message, and grounding information
    """
    validate_vectorstore_name(vectorstore_name)
    return await memory_service.store_memory(vectorstore_name, request)


@router.post('/{vectorstore_name}/search')
async def search_memories(
    vectorstore_name: str,
    request: MemorySearchRequest,
    memory_service: MemoryService = Depends(get_memory_service)
) -> Dict[str, Any]:
    """Search atomic memories (Memories) in a specific vectorstore using vector similarity.

    Args:
        vectorstore_name: Name of the vectorstore to search in
        request: Search request with query and filtering options

    Returns:
        JSON with success status, memories array, count, and memory breakdown by type
    """
    validate_vectorstore_name(vectorstore_name)
    return await memory_service.search_memories(vectorstore_name, request)


@router.get('/{vectorstore_name}')
async def get_memory_info(
    vectorstore_name: str,
    memory_service: MemoryService = Depends(get_memory_service)
) -> Dict[str, Any]:
    """Get atomic memory (Neme) statistics and system information for a specific vectorstore.

    Args:
        vectorstore_name: Name of the vectorstore to get info for

    Returns:
        JSON with memory count, vector dimension, embedding model, and system info
    """
    validate_vectorstore_name(vectorstore_name)
    return await memory_service.get_memory_info(vectorstore_name)


@router.delete('/{vectorstore_name}/{memory_id}')
async def delete_memory(
    vectorstore_name: str,
    memory_id: str,
    memory_service: MemoryService = Depends(get_memory_service)
) -> Dict[str, Any]:
    """Delete a specific atomic memory (Neme) by ID from a specific vectorstore.

    Args:
        vectorstore_name: Name of the vectorstore to delete from
        memory_id: UUID of the memory to delete

    Returns:
        JSON with success status and deletion details
    """
    validate_vectorstore_name(vectorstore_name)
    return await memory_service.delete_memory(vectorstore_name, memory_id)


@router.delete('/{vectorstore_name}/all')
async def delete_all_memories(
    vectorstore_name: str,
    memory_service: MemoryService = Depends(get_memory_service)
) -> Dict[str, Any]:
    """Clear all atomic memories (Memories) from a specific vectorstore.

    Args:
        vectorstore_name: Name of the vectorstore to clear

    Returns:
        JSON with success status, deletion count, and operation details
    """
    validate_vectorstore_name(vectorstore_name)
    return await memory_service.delete_all_memories(vectorstore_name)


@router.post('/{vectorstore_name}/context')
async def set_context(
    vectorstore_name: str,
    request: ContextSetRequest,
    memory_service: MemoryService = Depends(get_memory_service)
) -> Dict[str, Any]:
    """Set current context for memory grounding in a specific vectorstore.

    Args:
        vectorstore_name: Name of the vectorstore to set context for
        request: Context parameters (location, activity, people_present)

    Returns:
        JSON with success status and updated context
    """
    validate_vectorstore_name(vectorstore_name)
    return await memory_service.set_context(vectorstore_name, request)


@router.get('/{vectorstore_name}/context')
async def get_context(
    vectorstore_name: str,
    memory_service: MemoryService = Depends(get_memory_service)
) -> Dict[str, Any]:
    """Get current context information for memory grounding from a specific vectorstore.

    Args:
        vectorstore_name: Name of the vectorstore to get context for

    Returns:
        JSON with success status and current context (temporal, spatial, social, environmental)
    """
    validate_vectorstore_name(vectorstore_name)
    return await memory_service.get_context(vectorstore_name)

"""
Async memory processing Pydantic models for API request/response validation.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class RawMemoryStoreRequest(BaseModel):
    session_data: str = Field(..., description="Complete chat session text or conversation history")
    session_id: Optional[str] = Field(None, description="Optional session ID for tracking")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata (user_id, session_type, etc.)")


class RawMemoryStoreResponse(BaseModel):
    success: bool
    raw_memory_id: str
    queued_at: str
    estimated_processing_time: str
    queue_position: Optional[int] = None


class ProcessedMemoryResponse(BaseModel):
    success: bool
    session_id: str
    discrete_memories: List[Dict[str, Any]]
    session_summary: Dict[str, Any]
    processing_stats: Dict[str, Any]


class MemoryHierarchyRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Filter by session ID")
    memory_type: Optional[str] = Field(None, description="Filter by memory type: 'discrete', 'summary', 'raw'")
    start_date: Optional[str] = Field(None, description="Start date filter (ISO format)")
    end_date: Optional[str] = Field(None, description="End date filter (ISO format)")
    limit: int = Field(50, ge=1, le=1000, description="Maximum number of results")


class BackgroundProcessorStatus(BaseModel):
    success: bool
    processor_running: bool
    queue_size: int
    processed_today: int
    last_processed_at: Optional[str]
    processing_interval_seconds: int
    retention_policy: Dict[str, Any]

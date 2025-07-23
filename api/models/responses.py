"""
Common response Pydantic models for API responses.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class SuccessResponse(BaseModel):
    success: bool = True
    message: str
    timestamp: Optional[str] = None


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None
    timestamp: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: str


class MemoryInfoResponse(BaseModel):
    success: bool
    memory_count: int
    vector_dimension: int
    vectorset_name: str
    vectorset_info: Dict[str, Any]
    embedding_model: str
    redis_host: str
    redis_port: str
    timestamp: str
    note: Optional[str] = None


class ConfigResponse(BaseModel):
    success: bool
    config: Dict[str, Any]
    runtime: Dict[str, Any]

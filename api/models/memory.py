"""
Memory-related Pydantic models for API request/response validation.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class MemoryStoreRequest(BaseModel):
    text: str = Field(..., description="Memory text to store")
    apply_grounding: bool = Field(True, description="Whether to apply contextual grounding")


class MemorySearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    top_k: int = Field(5, ge=1, description="Number of results to return")
    filter: Optional[str] = Field(None, description="Filter expression for Redis VSIM command")
    optimize_query: bool = Field(False, description="Whether to optimize query for embedding search")
    min_similarity: float = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity score threshold")


class MemoryDeleteRequest(BaseModel):
    vectorstore_name: Optional[str] = Field(None, description="Name of the vectorstore to delete from")


class ContextSetRequest(BaseModel):
    location: Optional[str] = Field(None, description="Current location")
    activity: Optional[str] = Field(None, description="Current activity")
    people_present: Optional[List[str]] = Field(None, description="List of people present")


class KLineAnswerRequest(BaseModel):
    question: str = Field(..., description="Question to answer")
    top_k: int = Field(5, ge=1, description="Number of memories to retrieve for context")
    filter: Optional[str] = Field(None, description="Filter expression for Redis VSIM command")
    min_similarity: float = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity score threshold")

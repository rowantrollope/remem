"""
Agent-related Pydantic models for API request/response validation.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class AgentChatRequest(BaseModel):
    message: str = Field(..., description="User message/question")
    system_prompt: Optional[str] = Field(None, description="Custom system prompt to override default behavior")


class AgentSessionCreateRequest(BaseModel):
    system_prompt: str = Field(..., description="Custom system prompt for the agent session")
    session_id: Optional[str] = Field(None, description="Custom session ID, auto-generated if not provided")
    config: Optional[Dict[str, Any]] = Field(None, description="Additional configuration options")


class AgentSessionMessageRequest(BaseModel):
    message: str = Field(..., description="The user's message")
    stream: bool = Field(False, description="Whether to stream the response")
    store_memory: bool = Field(True, description="Whether to extract and store memories from this conversation")
    top_k: int = Field(10, ge=1, description="Number of memories to search and return")
    min_similarity: float = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity score threshold")

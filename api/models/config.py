"""
Configuration-related Pydantic models for API request/response validation.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class LLMConfigTier(BaseModel):
    provider: str = Field(..., description="LLM provider: 'openai' or 'ollama'")
    model: str = Field(..., description="Model name")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Response creativity")
    max_tokens: int = Field(1000, ge=1, description="Maximum response length")
    base_url: Optional[str] = Field(None, description="Base URL for Ollama")
    api_key: Optional[str] = Field(None, description="API key")
    timeout: int = Field(30, ge=1, description="Request timeout in seconds")


class LLMConfigUpdate(BaseModel):
    tier1: Optional[LLMConfigTier] = None
    tier2: Optional[LLMConfigTier] = None


class ConfigUpdateRequest(BaseModel):
    redis: Optional[Dict[str, Any]] = Field(None, description="Redis configuration")
    llm: Optional[Dict[str, Any]] = Field(None, description="LLM configuration")
    openai: Optional[Dict[str, Any]] = Field(None, description="OpenAI configuration")
    langgraph: Optional[Dict[str, Any]] = Field(None, description="LangGraph configuration")
    memory_agent: Optional[Dict[str, Any]] = Field(None, description="Memory agent configuration")
    web_server: Optional[Dict[str, Any]] = Field(None, description="Web server configuration")
    langcache: Optional[Dict[str, Any]] = Field(None, description="LangCache configuration")

"""
Custom exception classes for the Memory Agent API.
"""

from fastapi import HTTPException
from typing import Optional, Dict, Any


class MemoryAgentException(Exception):
    """Base exception for Memory Agent operations."""
    pass


class VectorstoreValidationError(MemoryAgentException):
    """Raised when vectorstore name validation fails."""
    pass


class MemoryAgentNotInitializedError(MemoryAgentException):
    """Raised when memory agent is not initialized."""
    pass


class ConfigurationError(MemoryAgentException):
    """Raised when configuration is invalid."""
    pass


class LLMError(MemoryAgentException):
    """Raised when LLM operations fail."""
    pass


def create_http_exception(
    status_code: int,
    message: str,
    detail: Optional[str] = None,
    **kwargs
) -> HTTPException:
    """Create a standardized HTTP exception."""
    error_detail = {"error": message}
    if detail:
        error_detail["detail"] = detail
    error_detail.update(kwargs)
    
    return HTTPException(status_code=status_code, detail=error_detail)


def validation_error(message: str, detail: Optional[str] = None) -> HTTPException:
    """Create a validation error (400)."""
    return create_http_exception(400, message, detail)


def not_found_error(message: str, detail: Optional[str] = None) -> HTTPException:
    """Create a not found error (404)."""
    return create_http_exception(404, message, detail)


def server_error(message: str, detail: Optional[str] = None) -> HTTPException:
    """Create a server error (500)."""
    return create_http_exception(500, message, detail)

"""
Utility functions for the Memory Agent API.
"""

from datetime import datetime, timezone
from typing import Dict, Any, Optional
from .constants import RESERVED_VECTORSTORE_NAMES
from .exceptions import validation_error


def validate_vectorstore_name(name: str) -> None:
    """Validate that vectorstore name is not reserved.
    
    Args:
        name: Vectorstore name to validate
        
    Raises:
        HTTPException: If name is reserved
    """
    if name.lower() in RESERVED_VECTORSTORE_NAMES:
        raise validation_error(
            f"'{name}' is a reserved name",
            f"Reserved names: {', '.join(RESERVED_VECTORSTORE_NAMES)}"
        )


def get_current_timestamp() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def mask_api_key(api_key: Optional[str]) -> str:
    """Mask an API key for safe logging/display."""
    if not api_key:
        return "None"
    if len(api_key) < 12:
        return "***"
    return api_key[:8] + "..." + api_key[-4:]


def safe_config_copy(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create a safe copy of configuration with masked sensitive data."""
    import json
    safe_config = json.loads(json.dumps(config))
    
    # Mask API keys in different sections
    if "openai" in safe_config and "api_key" in safe_config["openai"]:
        safe_config["openai"]["api_key"] = mask_api_key(safe_config["openai"]["api_key"])
    
    if "llm" in safe_config:
        for tier in ["tier1", "tier2"]:
            if tier in safe_config["llm"] and "api_key" in safe_config["llm"][tier]:
                safe_config["llm"][tier]["api_key"] = mask_api_key(safe_config["llm"][tier]["api_key"])
    
    return safe_config


def validate_positive_integer(value: Any, field_name: str) -> int:
    """Validate that a value is a positive integer."""
    try:
        int_value = int(value)
        if int_value <= 0:
            raise validation_error(f"{field_name} must be positive")
        return int_value
    except (ValueError, TypeError):
        raise validation_error(f"{field_name} must be an integer")


def validate_float_range(value: Any, field_name: str, min_val: float = 0.0, max_val: float = 2.0) -> float:
    """Validate that a value is a float within a specified range."""
    try:
        float_value = float(value)
        if float_value < min_val or float_value > max_val:
            raise validation_error(f"{field_name} must be between {min_val} and {max_val}")
        return float_value
    except (ValueError, TypeError):
        raise validation_error(f"{field_name} must be a number")

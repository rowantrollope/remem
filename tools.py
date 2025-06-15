"""
Custom tools for the LangGraph agent.
"""

from langchain.tools import tool
import requests
from typing import Dict, Any


@tool
def get_weather(city: str) -> str:
    """Get current weather information for a city.
    
    Args:
        city: The name of the city to get weather for
        
    Returns:
        Weather information as a string
    """
    # This is a mock implementation - in a real scenario you'd use a weather API
    return f"The weather in {city} is sunny with a temperature of 72°F (22°C)."


@tool
def calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression.
    
    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 2", "10 * 5")
        
    Returns:
        The result of the calculation as a string
    """
    try:
        # Only allow basic mathematical operations for safety
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Only basic mathematical operations are allowed"
        
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


@tool
def search_web(query: str) -> str:
    """Search the web for information (mock implementation).
    
    Args:
        query: The search query
        
    Returns:
        Search results as a string
    """
    # This is a mock implementation - in a real scenario you'd use a search API
    return f"Here are some search results for '{query}': [Mock results - integrate with real search API]"


# Memory tools would be added here in Phase 1
# from memory_agent import MemoryAgent

# @tool
# def store_memory(memory_text: str, apply_grounding: bool = True) -> str:
#     """Store a new memory with optional contextual grounding."""
#     # Implementation would use MemoryAgent instance

# @tool
# def search_memories(query: str, top_k: int = 3) -> str:
#     """Search for relevant memories using vector similarity."""
#     # Implementation would use MemoryAgent instance

# List of available tools
AVAILABLE_TOOLS = [get_weather, calculate, search_web]

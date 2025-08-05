"""
Memory system package for remem.

This package contains all memory-related functionality including:
- agent: OpenAI SDK memory agent implementation
- core: Core memory operations and Redis interface
- extraction: Memory extraction from conversations
- processing: Memory processing and filtering
- reasoning: Memory reasoning and k-line operations
- core_agent: Core memory agent implementation
- tools: OpenAI function calling tools for memory operations
"""

# Import main classes for easy access
from .core import MemoryCore
from .extraction import MemoryExtraction
from .processing import MemoryProcessing
from .reasoning import MemoryReasoning
from .core_agent import MemoryAgent

__all__ = [
    'MemoryCore',
    'MemoryExtraction',
    'MemoryProcessing',
    'MemoryReasoning',
    'MemoryAgent'
]

"""
Application constants and configuration values.
"""

# Reserved vectorstore names that cannot be used by users
RESERVED_VECTORSTORE_NAMES = [
    "all", "info", "search", "context", "health", "metrics", "config"
]

# API version
API_VERSION = "1.0.0"

# Default values
DEFAULT_TOP_K = 5
DEFAULT_MIN_SIMILARITY = 0.7
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1000

# Cache configuration
CACHE_TYPES = {
    "memory_extraction": True,
    "query_optimization": True,
    "embedding_optimization": True,
    "context_analysis": True,
    "memory_grounding": True
}

# Processing configuration
DEFAULT_EXTRACTION_THRESHOLD = 2
DEFAULT_PROCESSING_INTERVAL = 60
DEFAULT_RETENTION_DAYS = 30

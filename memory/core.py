#!/usr/bin/env python3
"""
Memory Core - Low-level memory storage and retrieval operations

This module provides the fundamental memory operations using Redis VectorSet
and OpenAI embeddings, including optional contextual grounding.
"""

import os
import sys
import uuid
import json
import re
import time
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

import redis
import openai

# Import debug utilities
from .debug_utils import (
    debug_print, verbose_print, memory_search_print, memory_result_print,
    is_debug_enabled, success_print, error_print, warning_print
)

# Load environment variables
load_dotenv()

# Import LLM manager and LangCache
sys.path.append('..')
from llm.llm_manager import get_llm_manager
from clients.langcache_client import LangCacheClient, CachedLLMClient


class RelevanceConfig:
    """Configuration for memory relevance scoring parameters."""

    def __init__(self,
                 vector_weight: float = 0.85,
                 temporal_weight: float = 0.1,
                 usage_weight: float = 0.05,
                 recency_decay_days: float = 30.0,
                 access_decay_days: float = 7.0,
                 usage_boost_factor: float = 0.1,
                 max_temporal_boost: float = 0.3,
                 max_usage_boost: float = 0.2):
        """Initialize relevance scoring configuration.

        Args:
            vector_weight: Weight for vector similarity score (0.0-1.0)
            temporal_weight: Weight for temporal recency component (0.0-1.0)
            usage_weight: Weight for usage frequency component (0.0-1.0)
            recency_decay_days: Days for creation recency to decay to ~37% (e^-1)
            access_decay_days: Days for last access recency to decay to ~37% (e^-1)
            usage_boost_factor: Multiplier for access count boost
            max_temporal_boost: Maximum boost from temporal factors
            max_usage_boost: Maximum boost from usage factors
        """
        # Validate weights sum to approximately 1.0
        total_weight = vector_weight + temporal_weight + usage_weight
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Weights should sum to 1.0, got {total_weight}")

        self.vector_weight = vector_weight
        self.temporal_weight = temporal_weight
        self.usage_weight = usage_weight
        self.recency_decay_days = recency_decay_days
        self.access_decay_days = access_decay_days
        self.usage_boost_factor = usage_boost_factor
        self.max_temporal_boost = max_temporal_boost
        self.max_usage_boost = max_usage_boost

    def to_dict(self) -> Dict[str, float]:
        """Convert configuration to dictionary for serialization."""
        return {
            'vector_weight': self.vector_weight,
            'temporal_weight': self.temporal_weight,
            'usage_weight': self.usage_weight,
            'recency_decay_days': self.recency_decay_days,
            'access_decay_days': self.access_decay_days,
            'usage_boost_factor': self.usage_boost_factor,
            'max_temporal_boost': self.max_temporal_boost,
            'max_usage_boost': self.max_usage_boost
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, float]) -> 'RelevanceConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)


class MemoryCore:
    """Low-level memory storage and retrieval with Redis VectorSet and OpenAI embeddings."""
    
    def __init__(self, redis_host: str = None, redis_port: int = None, redis_db: int = None,
                 vectorset_key: str = None, relevance_config: RelevanceConfig = None):
        """Initialize the memory core.

        Args:
            redis_host: Redis server host (defaults to REDIS_HOST env var or "localhost")
            redis_port: Redis server port (defaults to REDIS_PORT env var or 6379)
            redis_db: Redis database number (defaults to REDIS_DB env var or 0)
            vectorset_key: Name of the vectorset to use (defaults to "memories")
            relevance_config: Configuration for relevance scoring (uses defaults if None)
        """
        # Get Redis connection details from environment variables with fallbacks
        redis_host = redis_host or os.getenv("REDIS_HOST", "localhost")
        redis_port = redis_port or int(os.getenv("REDIS_PORT", "6379"))
        redis_db = redis_db or int(os.getenv("REDIS_DB", "0"))

        # Debug logging (only in verbose mode)
        if os.getenv("MEMORY_DEBUG", "false").lower() == "true":
            print(f"🔍 MEMORY: Redis connection details:")
            print(f"  Host: {redis_host}")
            print(f"  Port: {redis_port} (from param: {redis_port}, from env: {os.getenv('REDIS_PORT', 'not set')})")
            print(f"  DB: {redis_db}")

        # Initialize OpenAI client
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Initialize LangCache if environment variables are available
        self.langcache_client = None
        try:
            if all([os.getenv("LANGCACHE_HOST"), os.getenv("LANGCACHE_API_KEY"), os.getenv("LANGCACHE_CACHE_ID")]):
                self.langcache_client = LangCacheClient()
                if os.getenv("MEMORY_DEBUG", "false").lower() == "true":
                    print("✅ LangCache enabled for memory core")
            elif os.getenv("MEMORY_DEBUG", "false").lower() == "true":
                print("ℹ️ LangCache not configured for memory core (missing environment variables)")
        except Exception as e:
            if os.getenv("MEMORY_DEBUG", "false").lower() == "true":
                print(f"⚠️ Failed to initialize LangCache for memory core: {e}")

        # Initialize Redis connection
        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=False  # Keep as bytes for vector operations
            )
            # Test connection
            self.redis_client.ping()
            if os.getenv("MEMORY_DEBUG", "false").lower() == "true":
                print(f"✅ MEMORY: Connected to Redis at {redis_host}:{redis_port}")
        except redis.ConnectionError as e:
            print(f"❌ MEMORY: Failed to connect to Redis: {e}")
            print("Make sure Redis is running and accessible.")
            sys.exit(1)

        # Constants
        self.VECTORSET_KEY = vectorset_key or "memories"
        self.EMBEDDING_DIM = 1536  # OpenAI text-embedding-ada-002 dimension

        # Relevance scoring configuration
        self.relevance_config = relevance_config or RelevanceConfig()

        # Context tracking for grounding
        self.current_context = {
            "location": None,
            "activity": None,
            "people_present": [],
            "environment": {}
        }

        # Initialize the vector set
        self._initialize_vectorset()

    def _calculate_relevance_score(self, memory: Dict[str, Any], vector_score: float) -> float:
        """Calculate enhanced relevance score combining vector similarity, temporal recency, and usage frequency.

        Args:
            memory: Memory object with metadata including temporal and usage fields
            vector_score: Original vector similarity score (0.0-1.0)

        Returns:
            Enhanced relevance score (0.0-1.0+)
        """
        config = self.relevance_config
        now = datetime.now(timezone.utc)

        # Base vector similarity component
        vector_component = vector_score * config.vector_weight

        # Temporal recency component
        temporal_component = 0.0
        if config.temporal_weight > 0:
            # Creation recency factor
            creation_recency = 0.0
            if memory.get("created_at"):
                try:
                    created_at = datetime.fromisoformat(memory["created_at"].replace('Z', '+00:00'))
                    days_since_creation = (now - created_at).total_seconds() / (24 * 3600)
                    # Exponential decay: e^(-days/decay_period)
                    creation_recency = np.exp(-days_since_creation / config.recency_decay_days)
                except (ValueError, TypeError):
                    creation_recency = 0.0

            # Last access recency factor
            access_recency = 0.0
            if memory.get("last_accessed_at"):
                try:
                    last_accessed = datetime.fromisoformat(memory["last_accessed_at"].replace('Z', '+00:00'))
                    days_since_access = (now - last_accessed).total_seconds() / (24 * 3600)
                    # Exponential decay: e^(-days/decay_period)
                    access_recency = np.exp(-days_since_access / config.access_decay_days)
                except (ValueError, TypeError):
                    access_recency = 0.0

            # Combine temporal factors (weighted average)
            temporal_factor = (creation_recency * 0.3 + access_recency * 0.7)  # Favor recent access
            temporal_component = min(temporal_factor, config.max_temporal_boost) * config.temporal_weight

        # Usage frequency component
        usage_component = 0.0
        if config.usage_weight > 0:
            access_count = memory.get("access_count", 0)
            # Logarithmic scaling to prevent excessive boost for very high access counts
            if access_count > 0:
                usage_factor = np.log1p(access_count) * config.usage_boost_factor
                usage_component = min(usage_factor, config.max_usage_boost) * config.usage_weight

        # Combine all components
        total_score = vector_component + temporal_component + usage_component

        # Ensure score doesn't exceed reasonable bounds (allow slight boost above 1.0)
        return min(total_score, 1.5)

    def _update_memory_access(self, memory_id: str, vectorset_key: str = None) -> None:
        """Update access tracking for a memory (increment count and update last accessed time).

        Args:
            memory_id: ID of the memory to update
            vectorset_key: Optional vectorset key to use instead of the instance default
        """
        try:
            # Get current metadata
            vectorset_to_use = vectorset_key or self.VECTORSET_KEY
            metadata_json = self.redis_client.execute_command("VGETATTR", vectorset_to_use, memory_id)
            if not metadata_json:
                return

            metadata_str = metadata_json.decode('utf-8') if isinstance(metadata_json, bytes) else metadata_json
            metadata = json.loads(metadata_str)

            # Update access tracking
            current_time_iso = datetime.now(timezone.utc).isoformat()
            metadata["last_accessed_at"] = current_time_iso
            metadata["access_count"] = metadata.get("access_count", 0) + 1

            # Update metadata in Redis
            updated_metadata_json = json.dumps(metadata)
            self.redis_client.execute_command("VSETATTR", vectorset_to_use, memory_id, updated_metadata_json)

        except Exception as e:
            # Don't fail the search if access tracking fails
            print(f"⚠️ Failed to update access tracking for memory {memory_id}: {e}")

    def _initialize_vectorset(self):
        """Initialize the VectorSet for memories."""
        try:
            # Check if vector set exists by getting info
            info = self.redis_client.execute_command("VINFO", self.VECTORSET_KEY)
            # VectorSet exists - no need to log
        except redis.ResponseError:
            # VectorSet doesn't exist yet, it will be created on first VADD
            print(f"🔧 MEMORY: VectorSet '{self.VECTORSET_KEY}' will be created on first memory")
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get OpenAI embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
        """
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ MEMORY: Error getting embedding: {e}")
            raise
    
    def _parse_memory_text(self, text: str) -> Tuple[str, List[str]]:
        """Parse memory text to extract content and tags.
        
        Args:
            text: Raw memory text
            
        Returns:
            Tuple of (cleaned_text, tags_list)
        """
        # Remove "Remember:" prefix if present
        cleaned_text = re.sub(r'^remember:\s*', '', text, flags=re.IGNORECASE).strip()
        
        # Extract potential tags (simple heuristic: proper nouns, quoted strings, etc.)
        tags = []
        
        # Find quoted strings
        quoted_matches = re.findall(r'"([^"]+)"', cleaned_text)
        tags.extend(quoted_matches)
        
        # Find capitalized words (potential proper nouns)
        capitalized_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', cleaned_text)
        tags.extend(capitalized_words)
        
        # Find common location/business keywords
        location_keywords = ['mall', 'store', 'restaurant', 'cafe', 'shop', 'market', 'plaza']
        for keyword in location_keywords:
            if keyword.lower() in cleaned_text.lower():
                tags.append(keyword)
        
        # Remove duplicates and clean tags
        tags = list(set([tag.strip() for tag in tags if len(tag.strip()) > 1]))
        
        return cleaned_text, tags

    def set_context(self, location: str = None, activity: str = None, people_present: List[str] = None, **kwargs):
        """Set current context for memory grounding.

        Args:
            location: Current location (e.g., "Jakarta, Indonesia", "Home office", "Starbucks on Main St")
            activity: Current activity (e.g., "working", "traveling", "meeting")
            people_present: List of people currently present
            **kwargs: Additional context like weather, mood, etc.
        """
        if location is not None:
            self.current_context["location"] = location
        if activity is not None:
            self.current_context["activity"] = activity
        if people_present is not None:
            self.current_context["people_present"] = people_present

        # Store additional environment context
        for key, value in kwargs.items():
            self.current_context["environment"][key] = value

        # Context updated silently

    def _get_current_context(self) -> Dict[str, Any]:
        """Get comprehensive current context for memory grounding."""
        now = datetime.now()

        context = {
            "temporal": {
                "date": now.strftime("%A, %B %d, %Y"),
                "time": now.strftime("%I:%M %p"),
                "iso_date": now.isoformat(),
                "day_of_week": now.strftime("%A"),
                "month": now.strftime("%B"),
                "year": now.year
            },
            "spatial": {
                "location": self.current_context.get("location"),
                "activity": self.current_context.get("activity")
            },
            "social": {
                "people_present": self.current_context.get("people_present", [])
            },
            "environmental": self.current_context.get("environment", {})
        }

        return context

    def _analyze_context_dependencies(self, memory_text: str) -> Dict[str, List[str]]:
        """Analyze what types of contextual references exist in the memory text.

        Args:
            memory_text: The memory text to analyze

        Returns:
            Dictionary mapping context types to detected references
        """
        analysis_prompt = f"""Analyze this text for context-dependent references that would become unclear over time or in different situations:

Text: "{memory_text}"

Identify references in these categories:

TEMPORAL: Words/phrases that depend on when this was said
- Examples: today, yesterday, this morning, last week, now, currently, recently, earlier

SPATIAL: Words/phrases that depend on where this was said
- Examples: here, outside, this place, nearby, upstairs, downstairs, this building, local

PERSONAL: References to people/things without full identification
- Examples: this guy, my boss, the meeting, this restaurant, that person, he/she (without clear antecedent)

ENVIRONMENTAL: References to current conditions/situations
- Examples: this weather, the current situation, right now, the way things are

DEMONSTRATIVE: Vague references that need context
- Examples: this, that, these, those (when unclear what they refer to)

Respond with a JSON object listing the detected references:
{{
  "temporal": ["today", "this morning"],
  "spatial": ["here", "outside"],
  "personal": ["this guy"],
  "environmental": ["this weather"],
  "demonstrative": ["this"]
}}

If no context-dependent references are found, return empty arrays for all categories."""

        try:
            # Try to use Tier 1 LLM for context analysis if available
            try:
                llm_manager = get_llm_manager()
                tier1_client = llm_manager.get_tier1_client()
            except RuntimeError:
                # LLM manager not initialized (CLI/MCP context), skip context analysis
                print(f"⚠️ Context analysis skipped: LLM manager not initialized")
                return {"temporal": [], "spatial": [], "personal": [], "environmental": [], "demonstrative": []}

            # Wrap with LangCache if available
            if self.langcache_client:
                cached_client = CachedLLMClient(tier1_client, self.langcache_client)
                response = cached_client.chat_completion(
                    messages=[
                        {"role": "user", "content": analysis_prompt}
                    ],
                    operation_type='context_analysis',
                    temperature=0.1,
                    max_tokens=300
                )
            else:
                response = tier1_client.chat_completion(
                    messages=[
                        {"role": "user", "content": analysis_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=300
                )

            analysis_result = response['content'].strip()

            # Parse JSON response
            try:
                # Extract JSON from markdown code blocks if present
                json_text = analysis_result.strip()
                if json_text.startswith('```json'):
                    # Remove markdown code block formatting
                    json_text = json_text[7:]  # Remove ```json
                    if json_text.endswith('```'):
                        json_text = json_text[:-3]  # Remove ```
                elif json_text.startswith('```'):
                    # Remove generic code block formatting
                    json_text = json_text[3:]  # Remove ```
                    if json_text.endswith('```'):
                        json_text = json_text[:-3]  # Remove ```

                json_text = json_text.strip()
                dependencies = json.loads(json_text)
                return dependencies
            except json.JSONDecodeError:
                print(f"⚠️ Could not parse context analysis: {analysis_result}")
                return {"temporal": [], "spatial": [], "personal": [], "environmental": [], "demonstrative": []}

        except Exception as e:
            print(f"❌ Error analyzing context dependencies: {e}")
            return {"temporal": [], "spatial": [], "personal": [], "environmental": [], "demonstrative": []}

    def _ground_memory_with_context(self, memory_text: str, context: Dict[str, Any], dependencies: Dict[str, List[str]]) -> Dict[str, Any]:
        """Convert context-dependent references to absolute references.

        Args:
            memory_text: Original memory text
            context: Current context information
            dependencies: Detected context dependencies

        Returns:
            Dictionary with grounded memory and metadata
        """
        # Check if grounding is needed
        total_dependencies = sum(len(refs) for refs in dependencies.values())
        if total_dependencies == 0:
            return {
                "original_text": memory_text,
                "grounded_text": memory_text,
                "grounding_applied": False,
                "dependencies_resolved": {}
            }

        # Build context description for the LLM
        context_description = []

        if context["temporal"]["date"]:
            context_description.append(f"Current date: {context['temporal']['date']}")
            context_description.append(f"Current time: {context['temporal']['time']}")

        if context["spatial"]["location"]:
            context_description.append(f"Current location: {context['spatial']['location']}")

        if context["spatial"]["activity"]:
            context_description.append(f"Current activity: {context['spatial']['activity']}")

        if context["social"]["people_present"]:
            people_list = ", ".join(context["social"]["people_present"])
            context_description.append(f"People present: {people_list}")

        if context["environmental"]:
            env_items = [f"{k}: {v}" for k, v in context["environmental"].items()]
            context_description.append(f"Environmental context: {', '.join(env_items)}")

        context_text = "\n".join(context_description) if context_description else "No additional context available"

        grounding_prompt = f"""Convert context-dependent references in this memory to absolute, context-independent references.

Current Context:
{context_text}

Original Memory: "{memory_text}"

Detected context dependencies:
{json.dumps(dependencies, indent=2)}

Instructions:
1. Replace temporal references (today, yesterday, now, etc.) with specific dates/times
2. Replace spatial references (here, outside, this place, etc.) with specific locations
3. Replace personal references (this guy, my boss, etc.) with specific names if determinable from context
4. Replace environmental references (this weather, etc.) with specific conditions and location
5. Replace vague demonstratives (this, that) with specific references when possible
6. Preserve the original meaning and tone
7. If context is insufficient to resolve a reference, keep it as-is but note the limitation

Respond with a JSON object:
{{
  "grounded_text": "The memory with context-dependent references resolved",
  "changes_made": [
    {{"original": "today", "replacement": "January 15, 2024", "type": "temporal"}},
    {{"original": "here", "replacement": "Jakarta, Indonesia", "type": "spatial"}}
  ],
  "unresolved": ["references that couldn't be resolved due to insufficient context"]
}}"""

        try:
            # Try to use Tier 1 LLM for memory grounding if available
            try:
                llm_manager = get_llm_manager()
                tier1_client = llm_manager.get_tier1_client()
            except RuntimeError:
                # LLM manager not initialized (CLI/MCP context), skip grounding
                print(f"⚠️ Memory grounding skipped: LLM manager not initialized")
                return {
                    "original_text": memory_text,
                    "grounded_text": memory_text,
                    "grounding_applied": False,
                    "dependencies_resolved": dependencies,
                    "error": "LLM manager not initialized"
                }

            # Wrap with LangCache if available
            if self.langcache_client:
                cached_client = CachedLLMClient(tier1_client, self.langcache_client)
                response = cached_client.chat_completion(
                    messages=[
                        {"role": "user", "content": grounding_prompt}
                    ],
                    operation_type='memory_grounding',
                    temperature=0.2,
                    max_tokens=500
                )
            else:
                response = tier1_client.chat_completion(
                    messages=[
                        {"role": "user", "content": grounding_prompt}
                    ],
                    temperature=0.2,
                    max_tokens=500
                )

            grounding_result = response['content'].strip()

            try:
                # Extract JSON from markdown code blocks if present
                json_text = grounding_result.strip()
                if json_text.startswith('```json'):
                    # Remove markdown code block formatting
                    json_text = json_text[7:]  # Remove ```json
                    if json_text.endswith('```'):
                        json_text = json_text[:-3]  # Remove ```
                elif json_text.startswith('```'):
                    # Remove generic code block formatting
                    json_text = json_text[3:]  # Remove ```
                    if json_text.endswith('```'):
                        json_text = json_text[:-3]  # Remove ```

                json_text = json_text.strip()
                grounding_data = json.loads(json_text)

                return {
                    "original_text": memory_text,
                    "grounded_text": grounding_data.get("grounded_text", memory_text),
                    "grounding_applied": True,
                    "dependencies_resolved": dependencies,
                    "changes_made": grounding_data.get("changes_made", []),
                    "unresolved_references": grounding_data.get("unresolved", [])
                }

            except json.JSONDecodeError:
                print(f"⚠️ Could not parse grounding result: {grounding_result}")
                return {
                    "original_text": memory_text,
                    "grounded_text": memory_text,
                    "grounding_applied": False,
                    "dependencies_resolved": dependencies,
                    "error": "Failed to parse grounding response"
                }

        except Exception as e:
            print(f"❌ Error grounding memory: {e}")
            return {
                "original_text": memory_text,
                "grounded_text": memory_text,
                "grounding_applied": False,
                "dependencies_resolved": dependencies,
                "error": str(e)
            }

    def store_memory(self, memory_text: str, apply_grounding: bool = True, vectorset_key: str = None) -> Dict[str, Any]:
        """Store a memory using VectorSet with optional contextual grounding.

        Args:
            memory_text: The memory text to store
            apply_grounding: Whether to apply contextual grounding to resolve context-dependent references
            vectorset_key: Optional vectorset key to use instead of the instance default

        Returns:
            Dictionary containing:
            - memory_id: The UUID of the stored memory
            - original_text: The original memory text (after parsing)
            - final_text: The final stored text (grounded or original)
            - grounding_applied: Whether grounding was applied
            - grounding_info: Details about grounding changes (if applied)
        """
        # Parse the memory
        cleaned_text, tags = self._parse_memory_text(memory_text)

        # Apply contextual grounding if requested
        grounding_result = None
        final_text = cleaned_text

        if apply_grounding:
            dependencies = self._analyze_context_dependencies(cleaned_text)

            # Only apply grounding if dependencies are found
            total_deps = sum(len(refs) for refs in dependencies.values())
            if total_deps > 0:
                current_context = self._get_current_context()
                grounding_result = self._ground_memory_with_context(cleaned_text, current_context, dependencies)

                if grounding_result.get("grounding_applied"):
                    final_text = grounding_result["grounded_text"]

        # Get embedding for the final text (grounded or original)
        embedding = self._get_embedding(final_text)

        # Generate UUID for this memory
        memory_id = str(uuid.uuid4())

        # Prepare data for storage
        current_time_iso = datetime.now(timezone.utc).isoformat()

        # Convert embedding to list of string values for VADD
        embedding_values = [str(x) for x in embedding]

        # Store vector in VectorSet using VADD
        # VADD key [REDUCE dim] FP32|VALUES vector element [SETATTR json]
        try:
            # Prepare metadata as JSON with enhanced temporal and usage tracking
            metadata = {
                "raw_text": cleaned_text,
                "final_text": final_text,
                "tags": tags,
                "grounding_applied": grounding_result is not None and grounding_result.get("grounding_applied", False),
                # Temporal and usage tracking fields (ISO 8601 UTC format)
                "created_at": current_time_iso,
                "last_accessed_at": current_time_iso,  # Initialize to creation time
                "access_count": 0  # Initialize to 0, will be incremented on first access
            }

            # Add grounding information if available
            if grounding_result:
                metadata["grounding_info"] = {
                    "dependencies_found": grounding_result.get("dependencies_resolved", {}),
                    "changes_made": grounding_result.get("changes_made", []),
                    "unresolved_references": grounding_result.get("unresolved_references", [])
                }

                # Store context snapshot
                if grounding_result.get("grounding_applied"):
                    context_snapshot = self._get_current_context()
                    metadata["context_snapshot"] = context_snapshot

            metadata_json = json.dumps(metadata)

            # Add vector to VectorSet with metadata
            # Format: VADD vectorsetname VALUES LENGTH value1 value2 ... valueN element SETATTR "json string"
            vectorset_to_use = vectorset_key or self.VECTORSET_KEY
            cmd = ["VADD", vectorset_to_use, "VALUES", str(self.EMBEDDING_DIM)] + embedding_values + [memory_id, "SETATTR", metadata_json]
            self.redis_client.execute_command(*cmd)

            # Return structured response with grounding information
            response = {
                "memory_id": memory_id,
                "original_text": cleaned_text,
                "final_text": final_text,
                "grounding_applied": grounding_result is not None and grounding_result.get("grounding_applied", False),
                "tags": tags,
                "created_at": current_time_iso
            }
            print(response)
            # Add grounding information if available
            if grounding_result:
                response["grounding_info"] = {
                    "dependencies_found": grounding_result.get("dependencies_resolved", {}),
                    "changes_made": grounding_result.get("changes_made", []),
                    "unresolved_references": grounding_result.get("unresolved_references", [])
                }

                # Include context snapshot if grounding was applied
                if grounding_result.get("grounding_applied"):
                    response["context_snapshot"] = self._get_current_context()

            return response

        except Exception as e:
            print(f"❌ Memory storage error: {e}")
            raise

    def search_memories(self, query: str, top_k: int = 10, filterBy: str = None, min_similarity: float = 0.7, vectorset_key: str = None) -> Dict[str, Any]:
        """Search for relevant memories using VectorSet similarity.

        Args:
            query: Search query
            top_k: Number of top results to return
            filterBy: Optional filter expression for VSIM command
            min_similarity: Minimum similarity score threshold (0.0-1.0, default: 0.7)
            vectorset_key: Optional vectorset key to use instead of the instance default

        Returns:
            Dictionary containing:
            - memories: List of matching memories with metadata that meet the minimum similarity threshold
            - filtering_info: Information about included/excluded memories
        """
        # Get embedding for query
        query_embedding = self._get_embedding(query)
        query_vector_values = [str(x) for x in query_embedding]

        # Perform vector similarity search using VSIM
        try:
            # VSIM vectorsetname VALUES LENGTH value1 value2 ... valueN [WITHSCORES] [COUNT count] [FILTER expression]
            vectorset_to_use = vectorset_key or self.VECTORSET_KEY

            # Use formatted debug output
            memory_search_print(vectorset_to_use, query, top_k, min_similarity)

            cmd = ["VSIM", vectorset_to_use, "VALUES", str(self.EMBEDDING_DIM)] + query_vector_values + ["WITHSCORES", "COUNT", str(top_k)]

            # Add user-provided filter if specified
            if filterBy:
                cmd.extend(["FILTER", filterBy])
                debug_print(f"Using Redis filter: {filterBy}", "FILTER")

            result = self.redis_client.execute_command(*cmd)

            # Parse results - VSIM returns [element1, score1, element2, score2, ...]
            memories = []
            for i in range(0, len(result), 2):
                if i + 1 < len(result):
                    element_id = result[i].decode('utf-8') if isinstance(result[i], bytes) else result[i]
                    score = float(result[i + 1])

                    # Get metadata for this element using VGETATTR
                    try:
                        metadata_json = self.redis_client.execute_command("VGETATTR", vectorset_to_use, element_id)
                        if metadata_json:
                            metadata_str = metadata_json.decode('utf-8') if isinstance(metadata_json, bytes) else metadata_json
                            metadata = json.loads(metadata_str)
                        else:
                            metadata = {}

                        # All memories are now Memories (atomic memories)
                        display_text = metadata.get("final_text", metadata.get("raw_text", ""))
                        created_at = metadata.get("created_at")

                        # Format timestamp for display
                        formatted_time = "Unknown time"
                        if created_at:
                            try:
                                dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                                formatted_time = dt.strftime("%Y-%m-%d %H:%M")
                            except (ValueError, TypeError):
                                formatted_time = "Invalid time"

                        memory = {
                            "id": element_id,
                            "text": display_text,
                            "original_text": metadata.get("raw_text", ""),
                            "grounded_text": metadata.get("final_text", ""),
                            "tags": metadata.get("tags", []),
                            "score": score,  # Original vector similarity score
                            "grounding_applied": metadata.get("grounding_applied", False),
                            "grounding_info": metadata.get("grounding_info", {}),
                            "context_snapshot": metadata.get("context_snapshot", {}),
                            # Temporal and usage fields (ISO 8601 UTC format)
                            "created_at": created_at,
                            "last_accessed_at": metadata.get("last_accessed_at", created_at),
                            "access_count": metadata.get("access_count", 0),
                            "formatted_time": formatted_time,  # Human-readable timestamp
                            # All memories are now Memories (atomic memories)
                            "type": "neme"
                        }

                        # Calculate enhanced relevance score
                        relevance_score = self._calculate_relevance_score(memory, score)
                        memory["relevance_score"] = relevance_score

                        memories.append(memory)

                    except Exception as e:
                        # Create a basic memory entry without metadata
                        current_time_iso = datetime.now(timezone.utc).isoformat()
                        dt = datetime.now(timezone.utc)
                        formatted_time = dt.strftime("%Y-%m-%d %H:%M")
                        memory = {
                            "id": element_id,
                            "text": f"Memory {element_id} (metadata unavailable)",
                            "tags": [],
                            "score": score,
                            "created_at": current_time_iso,
                            "last_accessed_at": current_time_iso,
                            "access_count": 0,
                            "formatted_time": formatted_time,
                            "type": "neme"
                        }
                        memories.append(memory)

            # Update access tracking for all retrieved memories
            for memory in memories:
                self._update_memory_access(memory["id"], vectorset_to_use)

            # Filter by minimum similarity score threshold and track included/excluded
            included_memories = []
            excluded_memories = []

            if min_similarity > 0.0:
                original_count = len(memories)
                debug_print(f"Applying similarity threshold filter (min_similarity: {min_similarity})", "FILTER")

                for i, memory in enumerate(memories, 1):
                    score = memory.get("score", 0)
                    memory_text = memory.get("text", "")[:60] + ("..." if len(memory.get("text", "")) > 60 else "")

                    if score >= min_similarity:
                        debug_print(f"INCLUDED #{i}: Score: {score:.3f} - '{memory_text}'", "✅")
                        included_memories.append(memory)
                    else:
                        debug_print(f"EXCLUDED #{i}: Score: {score:.3f} - '{memory_text}' (below {min_similarity})", "❌")
                        excluded_memories.append(memory)

                memories = included_memories
                filtered_count = len(memories)
                memory_result_print(filtered_count, len(excluded_memories), min_similarity)
            else:
                debug_print(f"Similarity filtering: No memories removed (min_similarity: {min_similarity})", "FILTER")
                included_memories = memories
                excluded_memories = []

            # Sort by relevance score (highest first)
            memories.sort(key=lambda m: m.get("relevance_score", m.get("score", 0)), reverse=True)

            # Return memories with filtering information
            return {
                'memories': memories,
                'filtering_info': {
                    'included_count': len(included_memories),
                    'excluded_count': len(excluded_memories),
                    'min_similarity_threshold': min_similarity,
                    'excluded_memories': excluded_memories
                }
            }

        except Exception as e:
            print(f"❌ Memory search error: {e}")
            return {
                'memories': [],
                'filtering_info': {
                    'included_count': 0,
                    'excluded_count': 0,
                    'min_similarity_threshold': min_similarity,
                    'excluded_memories': []
                }
            }

    def delete_memory(self, memory_id: str, vectorset_key: str = None) -> bool:
        """Delete a specific memory from the VectorSet.

        Args:
            memory_id: The UUID of the memory to delete
            vectorset_key: Optional vectorset key to use instead of the instance default

        Returns:
            True if memory was deleted successfully, False otherwise
        """
        try:
            # Use VREM to remove the memory from the VectorSet
            vectorset_to_use = vectorset_key or self.VECTORSET_KEY
            result = self.redis_client.execute_command("VREM", vectorset_to_use, memory_id)

            if result == 1:
                print(f"✅ Deleted memory with ID: {memory_id}")
                return True
            else:
                print(f"⚠️ Memory with ID {memory_id} not found")
                return False

        except Exception as e:
            print(f"❌ Error deleting memory {memory_id}: {e}")
            return False

    def clear_all_memories(self, vectorset_key: str = None) -> Dict[str, Any]:
        """Clear all memories from the VectorSet.

        Args:
            vectorset_key: Optional vectorset key to use instead of the instance default

        Returns:
            Dictionary with success status and details about the operation
        """
        try:
            print("🗑️ Clearing all memories...")
            vectorset_to_use = vectorset_key or self.VECTORSET_KEY

            # First, check if vectorset exists and get current count
            try:
                initial_count = self.redis_client.execute_command("VCARD", vectorset_to_use)
                initial_count = int(initial_count) if initial_count else 0
            except redis.ResponseError:
                # VectorSet doesn't exist
                print("⚠️ No memories to clear - VectorSet doesn't exist")
                return {
                    'success': True,
                    'message': 'No memories to clear',
                    'memories_deleted': 0,
                    'vectorset_existed': False
                }

            if initial_count == 0:
                print("⚠️ No memories to clear - VectorSet is empty")
                return {
                    'success': True,
                    'message': 'No memories to clear',
                    'memories_deleted': 0,
                    'vectorset_existed': True
                }

            # Delete the entire VectorSet - this is the most efficient way to clear all data
            # The VectorSet will be recreated automatically on the next VADD operation
            result = self.redis_client.execute_command("DEL", vectorset_to_use)

            if result == 1:
                print(f"✅ Successfully cleared all memories (deleted {initial_count} memories)")
                return {
                    'success': True,
                    'message': f'Successfully cleared all memories',
                    'memories_deleted': initial_count,
                    'vectorset_existed': True
                }
            else:
                print("⚠️ VectorSet deletion returned unexpected result")
                return {
                    'success': False,
                    'error': 'VectorSet deletion returned unexpected result',
                    'memories_deleted': 0,
                    'vectorset_existed': True
                }

        except Exception as e:
            print(f"❌ Error clearing all memories: {e}")
            return {
                'success': False,
                'error': f'Failed to clear memories: {str(e)}',
                'memories_deleted': 0,
                'vectorset_existed': None
            }

    def get_memory_info(self) -> Dict[str, Any]:
        """Get comprehensive information about stored memories.

        Returns:
            Dictionary containing memory statistics and vector set information
        """
        try:
            # Get memory count using VCARD
            memory_count = self.redis_client.execute_command("VCARD", self.VECTORSET_KEY)

            # Get vector dimension using VDIM
            dimension = self.redis_client.execute_command("VDIM", self.VECTORSET_KEY)

            # Get detailed vector set info using VINFO
            vinfo_result = self.redis_client.execute_command("VINFO", self.VECTORSET_KEY)

            # Parse VINFO result (returns key-value pairs)
            vinfo_dict = {}
            if vinfo_result:
                for i in range(0, len(vinfo_result), 2):
                    if i + 1 < len(vinfo_result):
                        key = vinfo_result[i].decode('utf-8') if isinstance(vinfo_result[i], bytes) else vinfo_result[i]
                        value = vinfo_result[i + 1]
                        # Try to decode bytes, but keep original type for numbers
                        if isinstance(value, bytes):
                            try:
                                value = value.decode('utf-8')
                            except:
                                value = str(value)
                        vinfo_dict[key] = value

            return {
                'memory_count': int(memory_count) if isinstance(memory_count, (int, bytes)) else 0,
                'vector_dimension': int(dimension) if isinstance(dimension, (int, bytes)) else 0,
                'vectorset_name': self.VECTORSET_KEY,
                'vectorset_info': vinfo_dict,
                'embedding_model': 'text-embedding-ada-002',
                'redis_host': self.redis_client.connection_pool.connection_kwargs.get('host', 'unknown'),
                'redis_port': self.redis_client.connection_pool.connection_kwargs.get('port', 'unknown'),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        except redis.ResponseError:
            # VectorSet doesn't exist yet - this is normal when no memories have been stored
            return {
                'memory_count': 0,
                'vector_dimension': 0,
                'vectorset_name': self.VECTORSET_KEY,
                'vectorset_info': {},
                'embedding_model': 'text-embedding-ada-002',
                'redis_host': self.redis_client.connection_pool.connection_kwargs.get('host', 'unknown'),
                'redis_port': self.redis_client.connection_pool.connection_kwargs.get('port', 'unknown'),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'note': 'No memories stored yet - VectorSet will be created when first memory is added'
            }
        except Exception as e:
            return {
                'error': f"Failed to get memory info: {str(e)}",
                'memory_count': 0,
                'vector_dimension': 0,
                'vectorset_name': self.VECTORSET_KEY,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

    def get_relevance_config(self) -> Dict[str, float]:
        """Get current relevance scoring configuration.

        Returns:
            Dictionary with current relevance scoring parameters
        """
        return self.relevance_config.to_dict()

    def update_relevance_config(self, **kwargs) -> Dict[str, float]:
        """Update relevance scoring configuration.

        Args:
            **kwargs: Configuration parameters to update (see RelevanceConfig for options)

        Returns:
            Dictionary with updated configuration
        """
        # Create new config with updated parameters
        current_config = self.relevance_config.to_dict()
        current_config.update(kwargs)

        # Validate and set new configuration
        self.relevance_config = RelevanceConfig.from_dict(current_config)

        return self.relevance_config.to_dict()

    def migrate_legacy_memories(self) -> Dict[str, Any]:
        """Migrate existing memories to include new temporal and usage fields.

        This method scans all existing memories and adds the new fields
        (created_at, last_accessed_at, access_count) for memories that don't have them.

        Returns:
            Dictionary with migration results
        """
        try:
            # Get all memory IDs using VSCAN (if available) or fallback method
            memory_ids = []
            try:
                # Try to get all elements - this is a simplified approach
                # In a real implementation, you might need to use VSCAN or similar
                info = self.redis_client.execute_command("VINFO", self.VECTORSET_KEY)
                # For now, we'll handle this during search operations
                return {
                    'success': True,
                    'message': 'Migration handled automatically during memory retrieval',
                    'note': 'Legacy memories will be updated with default values when accessed'
                }
            except redis.ResponseError:
                return {
                    'success': True,
                    'message': 'No memories to migrate - VectorSet does not exist',
                    'memories_migrated': 0
                }

        except Exception as e:
            return {
                'success': False,
                'error': f'Migration failed: {str(e)}',
                'memories_migrated': 0
            }



    def construct_kline(self, query: str, memories: List[Dict[str, Any]], answer: str = None,
                       confidence: str = None, reasoning: str = None) -> Dict[str, Any]:
        """Construct a K-line (mental state) from relevant memories.

        This creates a temporary mental state that combines relevant memories into
        a coherent, contextual representation suitable for LLM context windows.
        K-lines are NOT stored - they exist only as temporary mental states.

        Args:
            query: The original query/question
            memories: List of relevant memories to combine
            answer: Optional answer text (for question-answering scenarios)
            confidence: Optional confidence level
            reasoning: Optional reasoning text

        Returns:
            Dictionary containing:
            - mental_state: Formatted coherent text combining memories
            - constituent_memories: List of memory IDs and texts
            - coherence_score: How well memories relate to each other
            - summary: Brief summary of the mental state
        """
        try:
            if not memories:
                return {
                    "mental_state": "No relevant memories found for this query.",
                    "constituent_memories": [],
                    "coherence_score": 0.0,
                    "summary": "Empty mental state"
                }

            print(f"🧠 K-LINE: Constructing mental state from {len(memories)} memories for query: '{query[:50]}...'")

            # Extract memory information
            memory_texts = []
            memory_ids = []
            memory_tags = []

            for memory in memories:
                text = memory.get('text', '')
                if text:
                    memory_texts.append(text)
                    memory_ids.append(memory.get('id', ''))
                    memory_tags.extend(memory.get('tags', []))

            # Calculate coherence score based on tag overlap and relevance scores
            coherence_score = self._calculate_coherence_score(memories, memory_tags)

            # Format mental state as coherent narrative
            mental_state = self._format_mental_state(query, memory_texts, answer, confidence, reasoning)

            # Create summary
            summary = f"Mental state combining {len(memory_texts)} memories"
            if answer:
                summary += f" to answer: {query[:50]}..."
            else:
                summary += f" related to: {query[:50]}..."

            return {
                "mental_state": mental_state,
                "constituent_memories": [
                    {"id": memory.get('id', ''), "text": memory.get('text', '')}
                    for memory in memories
                ],
                "coherence_score": coherence_score,
                "summary": summary,
                "query": query,
                "answer": answer,
                "confidence": confidence,
                "reasoning": reasoning
            }

        except Exception as e:
            print(f"❌ K-LINE: Error constructing mental state: {e}")
            return {
                "mental_state": f"Error constructing mental state: {str(e)}",
                "constituent_memories": [],
                "coherence_score": 0.0,
                "summary": "Error in mental state construction"
            }

    def _calculate_coherence_score(self, memories: List[Dict[str, Any]], all_tags: List[str]) -> float:
        """Calculate how coherent/related the memories are to each other."""
        if len(memories) <= 1:
            return 1.0

        # Tag overlap score
        unique_tags = set(all_tags)
        tag_overlap = len(all_tags) - len(unique_tags) if len(unique_tags) > 0 else 0
        tag_score = min(tag_overlap / len(memories), 1.0)

        # Relevance score consistency
        scores = [mem.get('score', 0) for mem in memories]
        if scores:
            score_variance = sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)
            consistency_score = max(0, 1.0 - score_variance)
        else:
            consistency_score = 0.5

        return (tag_score * 0.4 + consistency_score * 0.6)

    def _format_mental_state(self, query: str, memory_texts: List[str], answer: str = None,
                            confidence: str = None, reasoning: str = None) -> str:
        """Format memories into a coherent mental state narrative."""
        lines = []

        # Header
        lines.append(f"🧠 Mental State for: {query}")
        lines.append("=" * 50)

        # Answer section if provided
        if answer:
            lines.append(f"\n💡 Answer: {answer}")
            if confidence:
                lines.append(f"🎯 Confidence: {confidence}")
            if reasoning:
                lines.append(f"💭 Reasoning: {reasoning}")

        # Memory integration
        lines.append(f"\n📚 Relevant Knowledge ({len(memory_texts)} memories):")
        for i, text in enumerate(memory_texts, 1):
            lines.append(f"{i}. {text}")

        # Synthesis
        if len(memory_texts) > 1:
            lines.append(f"\n🔗 This mental state combines {len(memory_texts)} related memories")
            lines.append("to provide contextual understanding for the query.")

        return "\n".join(lines)





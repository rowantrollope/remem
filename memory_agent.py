#!/usr/bin/env python3
"""
Memory Agent with Redis and OpenAI

A simple memory agent that can store and retrieve memories using vector embeddings.

Installation requirements:
pip install redis>=5.0.0 openai>=1.0.0 python-dotenv>=1.0.0 numpy>=1.24.0

Prerequisites:
- Redis server running with RedisSearch module (Redis Stack or Redis with RediSearch)
- OpenAI API key in environment variables or .env file
- Optional: Redis connection details in .env file (REDIS_HOST, REDIS_PORT, REDIS_DB)

Usage:
python memory_agent.py
"""

import os
import sys
import uuid
import json
import re
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

import redis

import openai

# Load environment variables
load_dotenv()

class MemoryAgent:
    """A memory agent that stores and retrieves memories using Redis and OpenAI embeddings."""
    
    def __init__(self, redis_host: str = None, redis_port: int = None, redis_db: int = None):
        """Initialize the memory agent.

        Args:
            redis_host: Redis server host (defaults to REDIS_HOST env var or "localhost")
            redis_port: Redis server port (defaults to REDIS_PORT env var or 6379)
            redis_db: Redis database number (defaults to REDIS_DB env var or 0)
        """
        # Get Redis connection details from environment variables with fallbacks
        redis_host = redis_host or os.getenv("REDIS_HOST", "localhost")
        redis_port = redis_port or int(os.getenv("REDIS_PORT", "6379"))
        redis_db = redis_db or int(os.getenv("REDIS_DB", "0"))
        # Initialize OpenAI client
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
            print(f"✅ Connected to Redis at {redis_host}:{redis_port}")
        except redis.ConnectionError as e:
            print(f"❌ Failed to connect to Redis: {e}")
            print("Make sure Redis is running and accessible.")
            sys.exit(1)

        # Constants
        self.VECTORSET_KEY = "memories"
        self.EMBEDDING_DIM = 1536  # OpenAI text-embedding-ada-002 dimension

        # Context tracking
        self.current_context = {
            "location": None,
            "activity": None,
            "people_present": [],
            "environment": {}
        }

        # Initialize the vector set
        self._initialize_vectorset()
    
    def _initialize_vectorset(self):
        """Initialize the VectorSet for memories."""
        try:
            # Check if vector set exists by getting info
            info = self.redis_client.execute_command("VINFO", self.VECTORSET_KEY)
            print(f"✅ VectorSet '{self.VECTORSET_KEY}' already exists")
        except redis.ResponseError:
            # VectorSet doesn't exist yet, it will be created on first VADD
            print(f"🔧 VectorSet '{self.VECTORSET_KEY}' will be created on first memory")
    
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
            print(f"❌ Error getting embedding: {e}")
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

        print(f"🌍 Context updated: {self.current_context}")

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
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )

            analysis_result = response.choices[0].message.content.strip()

            # Parse JSON response
            try:
                dependencies = json.loads(analysis_result)
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
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": grounding_prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )

            grounding_result = response.choices[0].message.content.strip()

            try:
                grounding_data = json.loads(grounding_result)

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

    def store_memory(self, memory_text: str, apply_grounding: bool = True) -> Dict[str, Any]:
        """Store a memory using VectorSet with optional contextual grounding.

        Args:
            memory_text: The memory text to store
            apply_grounding: Whether to apply contextual grounding to resolve context-dependent references

        Returns:
            Dictionary containing:
            - memory_id: The UUID of the stored memory
            - original_text: The original memory text (after parsing)
            - final_text: The final stored text (grounded or original)
            - grounding_applied: Whether grounding was applied
            - grounding_info: Details about grounding changes (if applied)
        """
        print(f"🧠 Storing memory: {memory_text}")

        # Parse the memory
        cleaned_text, tags = self._parse_memory_text(memory_text)

        # Apply contextual grounding if requested
        grounding_result = None
        final_text = cleaned_text

        if apply_grounding:
            print("🔍 Analyzing context dependencies...")
            dependencies = self._analyze_context_dependencies(cleaned_text)

            # Only apply grounding if dependencies are found
            total_deps = sum(len(refs) for refs in dependencies.values())
            if total_deps > 0:
                print(f"🌍 Found {total_deps} context-dependent references, applying grounding...")
                current_context = self._get_current_context()
                grounding_result = self._ground_memory_with_context(cleaned_text, current_context, dependencies)

                if grounding_result.get("grounding_applied"):
                    final_text = grounding_result["grounded_text"]
                    print(f"✅ Grounding applied: {len(grounding_result.get('changes_made', []))} changes made")

                    # Show changes made
                    for change in grounding_result.get('changes_made', []):
                        print(f"   • '{change['original']}' → '{change['replacement']}' ({change['type']})")
                else:
                    print("⚠️ Grounding was attempted but not applied")
            else:
                print("ℹ️ No context-dependent references detected")

        # Get embedding for the final text (grounded or original)
        print("🔄 Getting embedding...")
        embedding = self._get_embedding(final_text)

        # Generate UUID for this memory
        memory_id = str(uuid.uuid4())

        # Prepare data for storage
        timestamp = datetime.now().timestamp()

        # Convert embedding to list of string values for VADD
        embedding_values = [str(x) for x in embedding]

        # Store vector in VectorSet using VADD
        # VADD key [REDUCE dim] FP32|VALUES vector element [SETATTR json]
        try:
            # Prepare metadata as JSON
            metadata = {
                "raw_text": cleaned_text,
                "final_text": final_text,
                "tags": tags,
                "timestamp": timestamp,
                "formatted_time": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M"),
                "grounding_applied": grounding_result is not None and grounding_result.get("grounding_applied", False)
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
            cmd = ["VADD", self.VECTORSET_KEY, "VALUES", str(self.EMBEDDING_DIM)] + embedding_values + [memory_id, "SETATTR", metadata_json]
            self.redis_client.execute_command(*cmd)

            print(f"✅ Stored memory with ID: {memory_id}")
            if final_text != cleaned_text:
                print(f"   Original: {cleaned_text}")
                print(f"   Grounded: {final_text}")
            print(f"   Tags: {tags}")
            print(f"   Timestamp: {datetime.fromtimestamp(timestamp)}")

            # Return structured response with grounding information
            response = {
                "memory_id": memory_id,
                "original_text": cleaned_text,
                "final_text": final_text,
                "grounding_applied": grounding_result is not None and grounding_result.get("grounding_applied", False),
                "tags": tags,
                "timestamp": timestamp,
                "formatted_time": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
            }

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
            print(f"❌ Error storing memory: {e}")
            raise
    
    def search_memories(self, query: str, top_k: int = 3, filterBy: str = None) -> List[Dict[str, Any]]:
        """Search for relevant memories using VectorSet similarity.

        Args:
            query: Search query
            top_k: Number of top results to return
            filterBy: Optional filter expression for VSIM command

        Returns:
            List of matching memories with metadata
        """
        print(f"🔍 Searching for: {query}")
        if filterBy:
            print(f"🔍 Filter: {filterBy}")

        # Get embedding for query
        query_embedding = self._get_embedding(query)
        query_vector_values = [str(x) for x in query_embedding]

        # Perform vector similarity search using VSIM
        try:
            # VSIM vectorsetname VALUES LENGTH value1 value2 ... valueN [WITHSCORES] [COUNT count] [FILTER expression]
            cmd = ["VSIM", self.VECTORSET_KEY, "VALUES", str(self.EMBEDDING_DIM)] + query_vector_values + ["WITHSCORES", "COUNT", str(top_k)]

            # Add FILTER parameter if provided
            if filterBy:
                cmd.extend(["FILTER", filterBy])

            result = self.redis_client.execute_command(*cmd)

            # Parse results - VSIM returns [element1, score1, element2, score2, ...]
            memories = []
            for i in range(0, len(result), 2):
                if i + 1 < len(result):
                    element_id = result[i].decode('utf-8') if isinstance(result[i], bytes) else result[i]
                    score = float(result[i + 1])

                    # Get metadata for this element using VGETATTR
                    try:
                        metadata_json = self.redis_client.execute_command("VGETATTR", self.VECTORSET_KEY, element_id)
                        if metadata_json:
                            metadata_str = metadata_json.decode('utf-8') if isinstance(metadata_json, bytes) else metadata_json
                            metadata = json.loads(metadata_str)

                            memory = {
                                "id": element_id,
                                "text": metadata.get("final_text", metadata.get("raw_text", "")),
                                "original_text": metadata.get("raw_text", ""),
                                "grounded_text": metadata.get("final_text", ""),
                                "tags": metadata.get("tags", []),
                                "timestamp": metadata.get("timestamp", 0),
                                "score": score,
                                "formatted_time": metadata.get("formatted_time", ""),
                                "grounding_applied": metadata.get("grounding_applied", False),
                                "grounding_info": metadata.get("grounding_info", {}),
                                "context_snapshot": metadata.get("context_snapshot", {})
                            }
                            memories.append(memory)
                    except Exception as e:
                        print(f"⚠️ Warning: Could not retrieve metadata for {element_id}: {e}")
                        # Create a basic memory entry without metadata
                        memory = {
                            "id": element_id,
                            "text": f"Memory {element_id} (metadata unavailable)",
                            "tags": [],
                            "timestamp": 0,
                            "score": score,
                            "formatted_time": "Unknown"
                        }
                        memories.append(memory)

            print(f"✅ Found {len(memories)} relevant memories")
            return memories

        except Exception as e:
            print(f"❌ Error searching memories: {e}")
            return []
    
    def format_memory_results(self, memories: List[Dict[str, Any]]) -> str:
        """Format memory search results for display.
        
        Args:
            memories: List of memory dictionaries
            
        Returns:
            Formatted string of memories
        """
        if not memories:
            return "No relevant memories found."
        
        result_lines = ["Here's what I remember that might be useful:"]
        
        for i, memory in enumerate(memories, 1):
            # VectorSet VSIM returns similarity scores (higher = more similar)
            # Convert to percentage (scores are typically between 0 and 1)
            score_percent = memory["score"] * 100
            result_lines.append(
                f"{i}. {memory['text']} "
                f"(from {memory['formatted_time']}, {score_percent:.1f}% similar)"
            )
            if memory["tags"]:
                result_lines.append(f"   Tags: {', '.join(memory['tags'])}")
        
        return "\n".join(result_lines)

    def _validate_and_preprocess_question(self, user_input: str) -> Dict[str, Any]:
        """Validate user input and preprocess it for optimal vector search.

        Args:
            user_input: Raw user input to validate

        Returns:
            Dictionary with 'type' ('search' or 'help') and 'content' (optimized question or help message)
        """
        validation_prompt = f"""You are a helpful memory assistant. The user has submitted: "{user_input}"

We have memories stored in a vector database using text-embedding-ada-002 embeddings.

Analyze the user's input and respond with one of two formats:

SEARCH if the input is asking about stored personal information, including:
- Past events, learnings, meetings, or experiences: "What did I learn about Redis yesterday?"
- Information about specific people, pets, or entities: "What color is Molly?", "Tell me about Sarah", "What does John do?"
- Details about places, restaurants, or locations: "What was that restaurant I liked?", "Where did I go last week?"
- Facts or attributes about anything previously stored: "What's my favorite book?", "How old is my cat?"
- Questions seeking to recall any stored information: "Remind me about the project discussion"

HELP for everything else including:
- Greetings: "hello", "hi", "hey"
- Casual conversation: "how are you", "what's up"
- Random text: "asdfghjkl", gibberish
- Empty or very short inputs
- General knowledge questions not about personal memories (e.g., "What is the capital of France?")

Response format:
- For memory questions: "SEARCH: {{optimized_question}}"
- For everything else: "HELP: Ask me to remember anything, for example: 'What color is Molly?' or 'Tell me about my meeting with Sarah last week.'"

Be inclusive about what qualifies as a memory question - if it could be asking about stored personal information, treat it as SEARCH."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": validation_prompt}
                ],
                temperature=0.1,  # Very low temperature for consistent validation
                max_tokens=200
            )

            validation_result = response.choices[0].message.content.strip()
            print(f"🔍 Validation result: {validation_result}")

            if validation_result.startswith("SEARCH:"):
                optimized_question = validation_result[7:].strip()
                return {
                    "type": "search",
                    "content": optimized_question
                }
            elif validation_result.startswith("HELP:"):
                help_message = validation_result[5:].strip()
                return {
                    "type": "help",
                    "content": help_message
                }
            else:
                # Fallback: treat as search if format is unexpected
                print(f"⚠️ Unexpected validation format, treating as search: {validation_result}")
                return {
                    "type": "search",
                    "content": user_input
                }

        except Exception as e:
            print(f"❌ Error in question validation: {e}")
            # Fallback: proceed with original input
            return {
                "type": "search",
                "content": user_input
            }

    def _filter_relevant_memories(self, question: str, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter memories by relevance using LLM as judge.

        Args:
            question: The original question being asked
            memories: List of memories from vector search

        Returns:
            List of memories that are actually relevant to the question
        """
        if not memories:
            return memories

        print(f"🔍 Filtering {len(memories)} memories for relevance to: {question}")

        relevant_memories = []

        for memory in memories:
            relevance_prompt = f"""You are a relevance judge. Your task is to determine if a memory contains information that could help answer a specific question.

Question: {question}

Memory: {memory['text']}

Instructions:
1. Analyze if this memory contains ANY information that could help answer the question
2. Consider both direct and indirect relevance
3. Be strict - only consider it relevant if it actually relates to what's being asked
4. Respond with ONLY "RELEVANT" or "NOT_RELEVANT" followed by a brief reason

Examples:
- Question: "When are my kids' birthdays?" Memory: "My daughter is Penelope" → NOT_RELEVANT (mentions daughter but no birthday information)
- Question: "When are my kids' birthdays?" Memory: "Emma's birthday is March 15th" → RELEVANT (directly answers the question)
- Question: "What did I eat yesterday?" Memory: "Had pizza for dinner on Tuesday" → RELEVANT (if today is Wednesday)

Your response:"""

            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": relevance_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=100
                )

                relevance_result = response.choices[0].message.content.strip()
                print(f"   Memory relevance check: {relevance_result[:50]}...")

                if relevance_result.startswith("RELEVANT"):
                    # Add relevance reasoning to memory metadata
                    memory["relevance_reasoning"] = relevance_result[8:].strip() if len(relevance_result) > 8 else "Deemed relevant"
                    relevant_memories.append(memory)
                    print(f"   ✅ Memory kept: {memory['text'][:50]}...")
                else:
                    print(f"   ❌ Memory filtered out: {memory['text'][:50]}...")

            except Exception as e:
                print(f"   ⚠️ Error in relevance check, keeping memory: {e}")
                # If relevance check fails, keep the memory to be safe
                memory["relevance_reasoning"] = "Relevance check failed, kept by default"
                relevant_memories.append(memory)

        print(f"🔍 Relevance filtering: {len(memories)} → {len(relevant_memories)} memories")
        return relevant_memories

    def answer_question(self, question: str, top_k: int = 5, filterBy: str = None) -> Dict[str, Any]:
        """Answer a question using relevant memories and OpenAI.

        Args:
            question: The question to answer
            top_k: Number of memories to retrieve for context
            filterBy: Optional filter expression for VSIM command

        Returns:
            Dictionary with structured response containing:
            - answer: The main answer text
            - confidence: Confidence level (high/medium/low)
            - supporting_memories: List of memory citations that support the answer
            - type: Response type ('answer' or 'help')
        """
        print(f"🤔 Processing user input: {question}")
        if filterBy:
            print(f"🔍 Filter: {filterBy}")

        # First, validate and preprocess the question
        validation_result = self._validate_and_preprocess_question(question)

        if validation_result["type"] == "help":
            print("ℹ️ Returning help message for invalid input")
            return {
                "type": "help",
                "answer": validation_result["content"],
                "confidence": "n/a",
                "supporting_memories": []
            }

        # Use the optimized question for search
        optimized_question = validation_result["content"]
        print(f"🔍 Using optimized question: {optimized_question}")

        # Search for relevant memories
        memories = self.search_memories(optimized_question, top_k, filterBy)

        if not memories:
            return {
                "type": "answer",
                "answer": "I don't have any relevant memories to help answer that question.",
                "confidence": "low",
                "supporting_memories": []
            }

        # Filter memories for relevance using LLM as judge
        relevant_memories = self._filter_relevant_memories(question, memories)

        if not relevant_memories:
            return {
                "type": "answer",
                "answer": "I found some memories but none of them contain information relevant to your question.",
                "confidence": "low",
                "supporting_memories": [],
                "filtered_count": len(memories)
            }

        # Prepare context from relevant memories
        memory_context = []
        for i, memory in enumerate(relevant_memories, 1):
            score_percent = memory["score"] * 100
            relevance_note = f" (Relevance: {memory.get('relevance_reasoning', 'N/A')})" if memory.get('relevance_reasoning') else ""
            memory_context.append(
                f"Memory {i} ({score_percent:.1f}% similarity, from {memory['formatted_time']}): {memory['text']}{relevance_note}"
            )

        context_text = "\n".join(memory_context)

        # Create enhanced system prompt for strict confidence analysis
        system_prompt = """You are an expert memory analyst with a focus on accuracy and preventing hallucination. Your job is to answer questions based ONLY on the information explicitly provided in the memories.

CRITICAL INSTRUCTIONS:
1. Only use information that is explicitly stated in the memories
2. Do NOT infer, assume, or extrapolate beyond what is directly stated
3. If memories don't contain sufficient information to answer the question, be honest about it
4. Be extremely conservative with confidence levels
5. The memories have already been filtered for relevance, but still verify they actually answer the question

Your task is to analyze the provided memories and respond with a JSON object containing:
1. "answer": A clear, direct answer based ONLY on explicit information in memories, or state what information is missing
2. "confidence": One of "high", "medium", or "low" based on how directly the memories answer the question
3. "reasoning": Detailed explanation of your confidence level and exactly which memories support your answer

STRICT Confidence levels:
- "high": Memories contain explicit, direct, and complete information that fully answers the question
- "medium": Memories contain some relevant information but may be incomplete or require minor interpretation
- "low": Memories are related but don't contain enough explicit information to answer the question confidently, OR no memories are truly relevant

IMPORTANT: If the memories don't actually answer the question, say so clearly. Don't try to piece together an answer from unrelated information.

Your response must be valid JSON in this exact format:
{
  "answer": "Direct answer based only on explicit memory content, or explanation of what information is missing",
  "confidence": "high|medium|low",
  "reasoning": "Detailed explanation citing specific memories and why confidence level was chosen"
}

Be honest and conservative. It's better to say "I don't have enough information" than to hallucinate an answer."""

        # Use the original question in the prompt to maintain user context
        user_prompt = f"""Question: {question}

Available memories (already filtered for relevance):
{context_text}

Please answer the question based ONLY on these memories. If the memories don't contain sufficient information to answer the question, be honest about it. Do not make assumptions or inferences beyond what is explicitly stated."""

        try:
            # Call OpenAI to generate the answer
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent answers
                max_tokens=500
            )

            ai_response = response.choices[0].message.content.strip()
            print(f"✅ Generated answer with {len(relevant_memories)} relevant memory context(s) (filtered from {len(memories)} total)")

            # Parse the JSON response from OpenAI
            try:
                ai_json = json.loads(ai_response)

                # Prepare supporting memories list with citations (only relevant memories)
                supporting_memories = []
                for i, memory in enumerate(relevant_memories, 1):
                    supporting_memories.append({
                        "id": memory["id"],
                        "text": memory["text"],
                        "relevance_score": round(memory["score"] * 100, 1),
                        "timestamp": memory["formatted_time"],
                        "tags": memory["tags"],
                        "relevance_reasoning": memory.get("relevance_reasoning", "N/A")
                    })

                return {
                    "type": "answer",
                    "answer": ai_json.get("answer", "Unable to parse answer"),
                    "confidence": ai_json.get("confidence", "low"),
                    "reasoning": ai_json.get("reasoning", ""),
                    "supporting_memories": supporting_memories,
                    "total_memories_searched": len(memories),
                    "relevant_memories_used": len(relevant_memories)
                }

            except json.JSONDecodeError as e:
                print(f"⚠️ Failed to parse JSON response from OpenAI: {e}")
                print(f"Raw response: {ai_response}")

                # Fallback: return the raw response in our structure (use relevant memories)
                supporting_memories = []
                for i, memory in enumerate(relevant_memories, 1):
                    supporting_memories.append({
                        "id": memory["id"],
                        "text": memory["text"],
                        "relevance_score": round(memory["score"] * 100, 1),
                        "timestamp": memory["formatted_time"],
                        "tags": memory["tags"],
                        "relevance_reasoning": memory.get("relevance_reasoning", "N/A")
                    })

                return {
                    "type": "answer",
                    "answer": ai_response,
                    "confidence": "low",  # Lower confidence due to parsing failure
                    "reasoning": "Response format could not be parsed",
                    "supporting_memories": supporting_memories,
                    "total_memories_searched": len(memories),
                    "relevant_memories_used": len(relevant_memories)
                }

        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            return {
                "type": "answer",
                "answer": f"I found {len(relevant_memories)} relevant memories (from {len(memories)} searched) but couldn't generate an answer due to an error: {str(e)}",
                "confidence": "low",
                "reasoning": "Error occurred during answer generation",
                "supporting_memories": [],
                "total_memories_searched": len(memories),
                "relevant_memories_used": len(relevant_memories)
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
                'timestamp': datetime.now().isoformat()
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
                'timestamp': datetime.now().isoformat(),
                'note': 'No memories stored yet - VectorSet will be created when first memory is added'
            }
        except Exception as e:
            return {
                'error': f"Failed to get memory info: {str(e)}",
                'memory_count': 0,
                'vector_dimension': 0,
                'vectorset_name': self.VECTORSET_KEY,
                'timestamp': datetime.now().isoformat()
            }

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory from the VectorSet.

        Args:
            memory_id: The UUID of the memory to delete

        Returns:
            True if memory was deleted successfully, False otherwise
        """
        try:
            # Use VREM to remove the memory from the VectorSet
            result = self.redis_client.execute_command("VREM", self.VECTORSET_KEY, memory_id)

            if result == 1:
                print(f"✅ Deleted memory with ID: {memory_id}")
                return True
            else:
                print(f"⚠️ Memory with ID {memory_id} not found")
                return False

        except Exception as e:
            print(f"❌ Error deleting memory {memory_id}: {e}")
            return False

    def clear_all_memories(self) -> Dict[str, Any]:
        """Clear all memories from the VectorSet.

        Returns:
            Dictionary with success status and details about the operation
        """
        try:
            print("🗑️ Clearing all memories...")

            # First, check if vectorset exists and get current count
            try:
                initial_count = self.redis_client.execute_command("VCARD", self.VECTORSET_KEY)
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
            result = self.redis_client.execute_command("DEL", self.VECTORSET_KEY)

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


def main():
    """Main CLI interface for the memory agent."""
    print("🧠 Memory Agent with Redis and OpenAI")
    print("=" * 50)
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment variables.")
        print("Please set your OpenAI API key in the .env file or environment.")
        sys.exit(1)
    
    # Initialize the agent
    try:
        agent = MemoryAgent()
    except Exception as e:
        print(f"❌ Failed to initialize memory agent: {e}")
        sys.exit(1)
    
    print("\nCommands:")
    print("- remember \"<memory>\" - Store a new memory (with contextual grounding)")
    print("- remember-raw \"<memory>\" - Store a memory without contextual grounding")
    print("- recall \"<query>\" - Search for relevant memories")
    print("- ask \"<question>\" - Ask a question based on memories")
    print("- context location=\"<location>\" activity=\"<activity>\" people=\"<person1,person2>\" - Set current context")
    print("- context-info - Show current context")
    print("- delete <memory_id> - Delete a specific memory by ID")
    print("- info - Show memory statistics and system information")
    print("- quit - Exit the program")
    print()
    
    while True:
        try:
            user_input = input("Memory Agent> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
            
            if not user_input:
                continue
            
            # Parse commands
            if user_input.lower().startswith('remember-raw '):
                memory_text = user_input[13:].strip()
                if memory_text.startswith('"') and memory_text.endswith('"'):
                    memory_text = memory_text[1:-1]

                if memory_text:
                    storage_result = agent.store_memory(memory_text, apply_grounding=False)
                    print(f"✅ Stored memory with ID: {storage_result['memory_id']}")
                else:
                    print("❌ Please provide memory text after 'remember-raw'")

            elif user_input.lower().startswith('remember '):
                memory_text = user_input[9:].strip()
                if memory_text.startswith('"') and memory_text.endswith('"'):
                    memory_text = memory_text[1:-1]

                if memory_text:
                    storage_result = agent.store_memory(memory_text, apply_grounding=True)
                    print(f"✅ Stored memory with ID: {storage_result['memory_id']}")
                    if storage_result['grounding_applied']:
                        print(f"🌍 Grounding applied:")
                        print(f"   Original: {storage_result['original_text']}")
                        print(f"   Grounded: {storage_result['final_text']}")
                        if 'grounding_info' in storage_result and storage_result['grounding_info']['changes_made']:
                            changes = storage_result['grounding_info']['changes_made']
                            print(f"   Changes: {len(changes)} modifications made")
                else:
                    print("❌ Please provide memory text after 'remember'")
            
            elif user_input.lower().startswith('recall '):
                query_text = user_input[7:].strip()
                if query_text.startswith('"') and query_text.endswith('"'):
                    query_text = query_text[1:-1]

                if query_text:
                    memories = agent.search_memories(query_text)
                    result = agent.format_memory_results(memories)
                    print(result)
                else:
                    print("❌ Please provide query text after 'recall'")

            elif user_input.lower().startswith('ask '):
                question_text = user_input[4:].strip()
                if question_text.startswith('"') and question_text.endswith('"'):
                    question_text = question_text[1:-1]

                if question_text:
                    answer_response = agent.answer_question(question_text)

                    # Format the structured response for CLI display
                    if answer_response["type"] == "help":
                        print(f"\n💡 {answer_response['answer']}")
                    else:
                        print(f"\n🤖 Answer: {answer_response['answer']}")
                        print(f"🎯 Confidence: {answer_response['confidence']}")

                        if answer_response.get('reasoning'):
                            print(f"💭 Reasoning: {answer_response['reasoning']}")

                        if answer_response.get('supporting_memories'):
                            print(f"\n📚 Supporting Memories ({len(answer_response['supporting_memories'])}):")
                            for i, memory in enumerate(answer_response['supporting_memories'], 1):
                                print(f"   {i}. {memory['text']} ({memory['relevance_score']}% relevant, {memory['timestamp']})")
                                if memory['tags']:
                                    print(f"      Tags: {', '.join(memory['tags'])}")
                else:
                    print("❌ Please provide a question after 'ask'")

            elif user_input.lower().startswith('context '):
                context_args = user_input[8:].strip()

                # Parse context arguments (simple key=value format)
                context_params = {}
                people_present = []

                # Split by spaces but handle quoted values
                import shlex
                try:
                    args = shlex.split(context_args)
                    for arg in args:
                        if '=' in arg:
                            key, value = arg.split('=', 1)
                            key = key.strip()
                            value = value.strip()

                            if key == 'people':
                                people_present = [p.strip() for p in value.split(',') if p.strip()]
                            elif key in ['location', 'activity']:
                                context_params[key] = value
                            else:
                                # Store as environment context
                                context_params[key] = value

                    # Apply context
                    if 'location' in context_params or 'activity' in context_params or people_present:
                        agent.set_context(
                            location=context_params.get('location'),
                            activity=context_params.get('activity'),
                            people_present=people_present if people_present else None,
                            **{k: v for k, v in context_params.items() if k not in ['location', 'activity']}
                        )
                    else:
                        print("❌ Please provide context parameters like: location=\"Jakarta\" activity=\"working\" people=\"John,Sarah\"")

                except Exception as e:
                    print(f"❌ Error parsing context: {e}")
                    print("Example: context location=\"Jakarta, Indonesia\" activity=\"traveling\" people=\"John,Sarah\" weather=\"hot\"")

            elif user_input.lower() == 'context-info':
                print("\n🌍 Current Context:")
                print("=" * 30)
                context = agent._get_current_context()

                print(f"📅 Date: {context['temporal']['date']}")
                print(f"🕐 Time: {context['temporal']['time']}")

                if context['spatial']['location']:
                    print(f"📍 Location: {context['spatial']['location']}")
                else:
                    print("📍 Location: Not set")

                if context['spatial']['activity']:
                    print(f"🎯 Activity: {context['spatial']['activity']}")
                else:
                    print("🎯 Activity: Not set")

                if context['social']['people_present']:
                    people = ", ".join(context['social']['people_present'])
                    print(f"👥 People Present: {people}")
                else:
                    print("👥 People Present: None specified")

                if context['environmental']:
                    print("🌡️  Environment:")
                    for key, value in context['environmental'].items():
                        print(f"   {key}: {value}")
                else:
                    print("🌡️  Environment: No additional context")

            elif user_input.lower().startswith('delete '):
                memory_id = user_input[7:].strip()

                if memory_id:
                    success = agent.delete_memory(memory_id)
                    if not success:
                        print("❌ Failed to delete memory. Check the memory ID and try again.")
                else:
                    print("❌ Please provide a memory ID after 'delete'")

            elif user_input.lower() == 'info':
                print("\n📊 Memory Information:")
                print("=" * 40)

                memory_info = agent.get_memory_info()

                if 'error' in memory_info:
                    print(f"❌ Error: {memory_info['error']}")
                else:
                    print(f"📝 Total Memories: {memory_info['memory_count']}")
                    print(f"🔢 Vector Dimension: {memory_info['vector_dimension']}")
                    print(f"🗃️  VectorSet Name: {memory_info['vectorset_name']}")
                    print(f"🤖 Embedding Model: {memory_info['embedding_model']}")
                    print(f"🔗 Redis Connection: {memory_info['redis_host']}:{memory_info['redis_port']}")
                    print(f"⏰ Last Updated: {memory_info['timestamp']}")

                    if 'vectorset_info' in memory_info and memory_info['vectorset_info']:
                        print(f"\n🔧 VectorSet Details:")
                        for key, value in memory_info['vectorset_info'].items():
                            print(f"   {key}: {value}")

            else:
                print("❌ Unknown command. Available commands:")
                print("   remember \"<text>\" | remember-raw \"<text>\" | recall \"<query>\" | ask \"<question>\"")
                print("   context location=\"<loc>\" activity=\"<act>\" people=\"<names>\" | context-info | delete <id> | info")
        
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()

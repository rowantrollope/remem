#!/usr/bin/env python3
"""
Optimized Memory Processing - Reduced LLM calls through batching and caching

This module provides optimized versions of memory processing functions that
reduce LLM API calls through intelligent batching, caching, and merged operations.
"""

import json
import sys
from typing import List, Dict, Any, Optional, Tuple
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import LLM manager and cache
sys.path.append('..')
from llm_manager import get_llm_manager
from optimizations.llm_cache import CachedLLMClient


class OptimizedMemoryProcessing:
    """Optimized memory analysis and filtering utilities with reduced LLM calls."""
    
    def __init__(self, cache_enabled: bool = True):
        """Initialize the optimized memory processing utilities."""
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.cache_enabled = cache_enabled
    
    def validate_and_optimize_query_batch(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple queries in a single LLM call for better efficiency.
        
        Args:
            queries: List of user queries to process
            
        Returns:
            List of validation results for each query
        """
        if not queries:
            return []
        
        # Single query optimization (existing behavior)
        if len(queries) == 1:
            return [self.validate_and_preprocess_question(queries[0])]
        
        # Batch processing for multiple queries
        batch_prompt = f"""You are a helpful memory assistant. Process the following {len(queries)} user queries and for each one, determine if it's a search request or needs help.

For each query, respond with either:
- "SEARCH: <optimized_question>" if it's a valid search/question
- "HELP: <helpful_message>" if it needs clarification or help

Queries to process:
"""
        for i, query in enumerate(queries, 1):
            batch_prompt += f"{i}. {query}\n"
        
        batch_prompt += """
Respond with exactly one line per query in the format:
1. SEARCH: <optimized_question> OR HELP: <message>
2. SEARCH: <optimized_question> OR HELP: <message>
...and so on."""

        try:
            llm_manager = get_llm_manager()
            tier2_client = llm_manager.get_tier2_client()
            
            # Use caching if available
            if hasattr(tier2_client, 'chat_completion'):
                response = tier2_client.chat_completion(
                    messages=[{"role": "user", "content": batch_prompt}],
                    operation_type='query_optimization_batch',
                    cache_context={'query_count': len(queries)},
                    temperature=0.1,
                    max_tokens=200 * len(queries)
                )
            else:
                response = tier2_client.chat_completion(
                    messages=[{"role": "user", "content": batch_prompt}],
                    temperature=0.1,
                    max_tokens=200 * len(queries)
                )

            validation_results = response['content'].strip().split('\n')
            
            results = []
            for i, (query, result_line) in enumerate(zip(queries, validation_results)):
                # Parse the result line
                if result_line.strip():
                    # Remove numbering if present
                    clean_result = result_line.split('.', 1)[-1].strip()
                    
                    if clean_result.startswith("SEARCH:"):
                        optimized_question = clean_result[7:].strip()
                        embedding_query = self.optimize_query_for_embedding_search(query, optimized_question)
                        results.append({
                            "type": "search",
                            "content": optimized_question,
                            "embedding_query": embedding_query
                        })
                    elif clean_result.startswith("HELP:"):
                        help_message = clean_result[5:].strip()
                        results.append({
                            "type": "help",
                            "content": help_message,
                            "embedding_query": None
                        })
                    else:
                        # Fallback for malformed response
                        embedding_query = self.optimize_query_for_embedding_search(query, query)
                        results.append({
                            "type": "search",
                            "content": query,
                            "embedding_query": embedding_query
                        })
                else:
                    # Fallback for missing response
                    embedding_query = self.optimize_query_for_embedding_search(query, query)
                    results.append({
                        "type": "search",
                        "content": query,
                        "embedding_query": embedding_query
                    })
            
            return results
            
        except Exception as e:
            print(f"⚠️ Batch query validation failed: {e}")
            # Fallback to individual processing
            return [self.validate_and_preprocess_question(query) for query in queries]
    
    def validate_and_preprocess_question(self, user_input: str) -> Dict[str, Any]:
        """
        Validate user input and preprocess it for optimal vector search.
        
        Args:
            user_input: Raw user input to validate
            
        Returns:
            Dictionary with validation and optimization results
        """
        validation_prompt = f"""You are a helpful memory assistant. The user has submitted: "{user_input}"

Determine if this is:
1. A search/question that should query the memory system
2. A request that needs help or clarification

If it's a valid search/question, respond with: "SEARCH: <optimized_question>"
If it needs help, respond with: "HELP: <helpful_message>"

The optimized question should be clear, specific, and good for semantic search."""

        try:
            llm_manager = get_llm_manager()
            tier2_client = llm_manager.get_tier2_client()
            
            # Use caching if available
            if hasattr(tier2_client, 'chat_completion'):
                response = tier2_client.chat_completion(
                    messages=[{"role": "user", "content": validation_prompt}],
                    operation_type='query_optimization',
                    cache_context={'input_length': len(user_input)},
                    temperature=0.1,
                    max_tokens=100
                )
            else:
                response = tier2_client.chat_completion(
                    messages=[{"role": "user", "content": validation_prompt}],
                    temperature=0.1,
                    max_tokens=100
                )

            validation_result = response['content'].strip()

            if validation_result.startswith("SEARCH:"):
                optimized_question = validation_result[7:].strip()
                embedding_query = self.optimize_query_for_embedding_search(user_input, optimized_question)
                return {
                    "type": "search",
                    "content": optimized_question,
                    "embedding_query": embedding_query
                }
            elif validation_result.startswith("HELP:"):
                help_message = validation_result[5:].strip()
                return {
                    "type": "help",
                    "content": help_message,
                    "embedding_query": None
                }
            else:
                # Fallback: treat as search if format is unexpected
                embedding_query = self.optimize_query_for_embedding_search(user_input, user_input)
                return {
                    "type": "search",
                    "content": user_input,
                    "embedding_query": embedding_query
                }

        except Exception as e:
            print(f"⚠️ Query validation failed: {e}")
            # Fallback: treat as search
            embedding_query = self.optimize_query_for_embedding_search(user_input, user_input)
            return {
                "type": "search",
                "content": user_input,
                "embedding_query": embedding_query
            }
    
    def optimize_query_for_embedding_search(self, original_query: str, optimized_query: str) -> str:
        """
        Optimize query for vector embedding search with caching.
        
        Args:
            original_query: Original user query
            optimized_query: Pre-optimized query from validation
            
        Returns:
            Query optimized for embedding similarity search
        """
        # Simple heuristic optimization to reduce LLM calls
        # This can be enhanced with more sophisticated rules
        
        # If queries are very similar, use the optimized one
        if len(optimized_query) > len(original_query) * 0.8:
            return optimized_query
        
        # For short queries, use original
        if len(original_query) < 20:
            return original_query
        
        # For longer queries, prefer the optimized version
        return optimized_query
    
    def filter_relevant_memories_batch(self, query: str, memories_list: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
        """
        Filter multiple memory lists for relevance in a single LLM call.
        
        Args:
            query: The user query
            memories_list: List of memory lists to filter
            
        Returns:
            List of filtered memory lists
        """
        if not memories_list or not any(memories_list):
            return memories_list
        
        # Flatten all memories for batch processing
        all_memories = []
        memory_indices = []  # Track which list each memory belongs to
        
        for list_idx, memories in enumerate(memories_list):
            for memory in memories:
                all_memories.append(memory)
                memory_indices.append(list_idx)
        
        if not all_memories:
            return memories_list
        
        # Process in batch
        filtered_memories = self.filter_relevant_memories(query, all_memories)
        
        # Reconstruct the original structure
        result = [[] for _ in memories_list]
        filtered_set = {mem['id'] for mem in filtered_memories}
        
        for memory, list_idx in zip(all_memories, memory_indices):
            if memory['id'] in filtered_set:
                result[list_idx].append(memory)
        
        return result
    
    def filter_relevant_memories(self, query: str, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter memories for relevance to the query using optimized LLM calls.
        
        Args:
            query: The user query
            memories: List of memories to filter
            
        Returns:
            List of relevant memories
        """
        if not memories:
            return []
        
        # For small lists, use existing logic
        if len(memories) <= 3:
            return self._filter_memories_individual(query, memories)
        
        # For larger lists, use batch processing
        return self._filter_memories_batch(query, memories)
    
    def _filter_memories_batch(self, query: str, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter memories in batch to reduce LLM calls."""
        batch_prompt = f"""Analyze the relevance of the following memories to the user query: "{query}"

For each memory, respond with either "RELEVANT" or "NOT_RELEVANT" followed by a brief reason.

Memories to analyze:
"""
        
        for i, memory in enumerate(memories, 1):
            memory_text = memory.get('text', '')[:100]  # Truncate for efficiency
            batch_prompt += f"{i}. {memory_text}\n"
        
        batch_prompt += f"""
Respond with exactly one line per memory in the format:
1. RELEVANT: <reason> OR NOT_RELEVANT: <reason>
2. RELEVANT: <reason> OR NOT_RELEVANT: <reason>
...and so on for all {len(memories)} memories."""

        try:
            llm_manager = get_llm_manager()
            tier2_client = llm_manager.get_tier2_client()
            
            if hasattr(tier2_client, 'chat_completion'):
                response = tier2_client.chat_completion(
                    messages=[{"role": "user", "content": batch_prompt}],
                    operation_type='memory_relevance_batch',
                    cache_context={'memory_count': len(memories), 'query_hash': hash(query)},
                    temperature=0.1,
                    max_tokens=50 * len(memories)
                )
            else:
                response = tier2_client.chat_completion(
                    messages=[{"role": "user", "content": batch_prompt}],
                    temperature=0.1,
                    max_tokens=50 * len(memories)
                )

            relevance_results = response['content'].strip().split('\n')
            
            relevant_memories = []
            for i, (memory, result_line) in enumerate(zip(memories, relevance_results)):
                if result_line.strip():
                    # Remove numbering if present
                    clean_result = result_line.split('.', 1)[-1].strip()
                    
                    if clean_result.startswith("RELEVANT"):
                        reason = clean_result[8:].strip() if len(clean_result) > 8 else "Deemed relevant"
                        memory["relevance_reasoning"] = reason.lstrip(':').strip()
                        relevant_memories.append(memory)
            
            print(f"🤖 Batch filtering kept {len(relevant_memories)}/{len(memories)} memories")
            return relevant_memories
            
        except Exception as e:
            print(f"⚠️ Batch memory filtering failed: {e}")
            # Fallback to individual processing
            return self._filter_memories_individual(query, memories)
    
    def _filter_memories_individual(self, query: str, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter memories individually (fallback method)."""
        relevant_memories = []
        
        for memory in memories:
            memory_text = memory.get('text', '')
            
            relevance_prompt = f"""Is the following memory relevant to the user query: "{query}"?

Memory: "{memory_text}"

Respond with either:
- "RELEVANT: <brief reason why it's relevant>"
- "NOT_RELEVANT: <brief reason why it's not relevant>"

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

                if relevance_result.startswith("RELEVANT"):
                    memory["relevance_reasoning"] = relevance_result[8:].strip() if len(relevance_result) > 8 else "Deemed relevant"
                    relevant_memories.append(memory)

            except Exception as e:
                print(f"⚠️ Individual memory filtering failed: {e}")
                # Include memory if filtering fails
                relevant_memories.append(memory)

        return relevant_memories

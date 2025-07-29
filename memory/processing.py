#!/usr/bin/env python3
"""
Memory Processing - Mid-level memory analysis and filtering utilities

This module provides utilities for processing memories and questions,
including validation, relevance filtering, and formatting.
"""

import json
import sys
from typing import List, Dict, Any
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import LLM manager and LangCache
sys.path.append('..')
from llm.llm_manager import get_llm_manager
from clients.langcache_client import LangCacheClient, CachedLLMClient


class MemoryProcessing:
    """Memory analysis and filtering utilities."""
    
    def __init__(self):
        """Initialize the memory processing utilities."""
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Initialize LangCache if environment variables are available
        self.langcache_client = None
        try:
            if all([os.getenv("LANGCACHE_HOST"), os.getenv("LANGCACHE_API_KEY"), os.getenv("LANGCACHE_CACHE_ID")]):
                self.langcache_client = LangCacheClient()
                print("✅ LangCache enabled for memory processing")
            else:
                print("ℹ️ LangCache not configured for memory processing (missing environment variables)")
        except Exception as e:
            print(f"⚠️ Failed to initialize LangCache for memory processing: {e}")

    def validate_and_preprocess_question(self, user_input: str) -> Dict[str, Any]:
        """Validate user input and preprocess it for optimal vector search.

        Args:
            user_input: Raw user input to validate

        Returns:
            Dictionary with 'type' ('search' or 'help'), 'content' (optimized question or help message),
            and 'embedding_query' (optimized query for vector similarity search)
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
            # Try to use Tier 2 LLM for input validation if available
            try:
                llm_manager = get_llm_manager()
                tier2_client = llm_manager.get_tier2_client()
            except RuntimeError:
                # LLM manager not initialized (CLI/MCP context), skip validation optimization
                print(f"⚠️ Query validation skipped: LLM manager not initialized, using original input")
                embedding_query = self.optimize_query_for_embedding_search(user_input, user_input)
                return {
                    "type": "search",
                    "content": user_input,
                    "embedding_query": embedding_query
                }

            # Wrap with LangCache if available
            if self.langcache_client:
                cached_client = CachedLLMClient(tier2_client, self.langcache_client)
                response = cached_client.chat_completion(
                    messages=[
                        {"role": "user", "content": validation_prompt}
                    ],
                    operation_type='query_optimization',
                    temperature=0.1,  # Very low temperature for consistent validation
                    max_tokens=200
                )
            else:
                response = tier2_client.chat_completion(
                    messages=[
                        {"role": "user", "content": validation_prompt}
                    ],
                    temperature=0.1,  # Very low temperature for consistent validation
                    max_tokens=200
                )

            validation_result = response['content'].strip()

            if validation_result.startswith("SEARCH:"):
                optimized_question = validation_result[7:].strip()

                # Generate embedding-optimized query for vector search
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
            # Fallback: proceed with original input
            try:
                embedding_query = self.optimize_query_for_embedding_search(user_input, user_input)
            except:
                embedding_query = user_input
            return {
                "type": "search",
                "content": user_input,
                "embedding_query": embedding_query
            }

    def optimize_query_for_embedding_search(self, original_input: str, processed_question: str) -> str:
        """Optimize a query specifically for vector embedding similarity search.

        This method extracts the semantic intent and creates a query optimized for finding
        relevant memories via vector similarity, removing verbose content that dilutes embeddings.

        Args:
            original_input: The raw user input
            processed_question: The already-processed question from validation

        Returns:
            Optimized query string for vector embedding search
        """
        optimization_prompt = f"""You are a memory search optimization expert. Your task is to create the optimal search query for finding relevant memories using vector similarity search.

Original user input: "{original_input}"
Processed question: "{processed_question}"

The user has a personal memory system with embeddings created using text-embedding-ada-002. Your job is to create a search query that will have high vector similarity with the types of memories needed to help with this request.

IMPORTANT GUIDELINES:
1. Extract the core semantic intent and memory types needed
2. Remove verbose content, code blocks, long examples that dilute the embedding
3. Focus on the TYPE of information needed rather than specific details
4. Use keywords that would likely appear in relevant stored memories
5. Keep it concise but semantically rich

EXAMPLES:
- Input: "Review my code [500 lines of code]" → Output: "code review preferences programming style guidelines"
- Input: "What did I learn about Redis yesterday?" → Output: "Redis learning database knowledge"
- Input: "Tell me about my meeting with Sarah last week" → Output: "Sarah meeting discussion conversation"
- Input: "What's my favorite restaurant in Paris?" → Output: "favorite restaurant Paris dining preferences"
- Input: "Help me plan a trip to Italy" → Output: "Italy travel planning preferences destinations"
- Input: "What color is my cat Molly?" → Output: "Molly cat color pet description"

Your response should be ONLY the optimized search query (no explanations, no quotes, just the query):"""

        try:
            # Try to use Tier 2 LLM for query optimization if available
            try:
                llm_manager = get_llm_manager()
                tier2_client = llm_manager.get_tier2_client()
            except RuntimeError:
                # LLM manager not initialized (CLI/MCP context), skip optimization
                print(f"⚠️ Embedding optimization skipped: LLM manager not initialized, using processed question")
                return processed_question

            # Wrap with LangCache if available
            if self.langcache_client:
                cached_client = CachedLLMClient(tier2_client, self.langcache_client)
                response = cached_client.chat_completion(
                    messages=[
                        {"role": "user", "content": optimization_prompt}
                    ],
                    operation_type='embedding_optimization',
                    temperature=0.1,  # Low temperature for consistent optimization
                    max_tokens=50  # Keep it concise
                )
            else:
                response = tier2_client.chat_completion(
                    messages=[
                        {"role": "user", "content": optimization_prompt}
                    ],
                    temperature=0.1,  # Low temperature for consistent optimization
                    max_tokens=50  # Keep it concise
                )

            optimized_query = response['content'].strip()

            # Remove any quotes that might have been added
            optimized_query = optimized_query.strip('"\'')

            # Fallback to processed question if optimization result is too short or empty
            if len(optimized_query.strip()) < 3:
                return processed_question

            print(f"🔍 EMBEDDING OPTIMIZATION: '{original_input[:50]}...' → '{optimized_query}'")
            return optimized_query

        except Exception as e:
            print(f"⚠️ Embedding optimization failed: {e}, using processed question")
            return processed_question

    def filter_relevant_memories(self, question: str, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter memories by relevance using LLM as judge.

        Args:
            question: The original question being asked
            memories: List of memories from vector search

        Returns:
            List of memories that are actually relevant to the question
        """
        if not memories:
            return memories

        relevant_memories = []

        for memory in memories:
            relevance_prompt = f"""You are a relevance judge. Your task is to determine if a memory contains information that could help answer a specific question.

Question: {question}

Memory: {memory['text']}

Instructions:
1. Analyze if this memory contains ANY information that could help answer the question or perform the task
2. Consider both direct and indirect relevance
3. Be strict - only consider it relevant if it actually relates to what's being asked OR may be used logically to deduce or infer the answer
4. Respond with ONLY "RELEVANT" or "NOT_RELEVANT" followed by a brief reason

Examples:
- Question: "When are my kids' birthdays?" Memory: "My daughter is Penelope" → RELEVANT (mentions daughter which may be useful to peice together who are the kids)
- Question: "When are my kids' birthdays?" Memory: "Emma's birthday is March 15th" → RELEVANT (directly answers the question)
- Question: "When are my Kids' birthdays?" Memory: "My birthday is on October 7th" - NOT_RELEVANT (refers to the users birthday but not his/her kids)
- Question: "When are my kids' birthdays?" Memory: "I went to an important and special birthday on July 1st" - RELEVANT (Birthday may have been related to kids?)
- Question: "When are my kids' birthdays?" Memory: "I was with my daughter on July 1st" - RELEVANT (Celebration may have been the referenced birthday of the kids)
- Question: "Help me plan a trip to paris" Memory: "Had french food for dinner on Tuesday" → NOT RELEVANT (Trip planning is unrelated to french food)
- Question: "Help me plan a trip to paris" Memory: "I like to travel" → RELEVANT (General travel interest may be useful for trip planning)
- Question: "Help me plan a trip to paris" Memory: "I have never been to Paris" → RELEVANT (Indicates first-time traveler, may need more help)
- Question: "Help me plan a trip to paris" Memory: "I prefer to stay in 5-star hotels" → RELEVANT (Trip planning is related to hotel preferences)
- Task: "Please review my code" Memory: "I like to travel" → NOT_RELEVANT (Traveling is unrelated to code review)
- Task: "Review my code" Memory: "I like lots of comments" → RELEVANT (Code review is related to commenting style)

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
                    # Add relevance reasoning to memory metadata
                    memory["relevance_reasoning"] = relevance_result[8:].strip() if len(relevance_result) > 8 else "Deemed relevant"
                    relevant_memories.append(memory)
                else:
                    print(f"   ❌ EXCLUDED: {memory['text'][:60] - memory["relevance_reasoning"]}")
                    pass  # Memory filtered out silently

            except Exception as e:
                # If relevance check fails, keep the memory to be safe
                memory["relevance_reasoning"] = "Relevance check failed, kept by default"
                relevant_memories.append(memory)

        print(f"🔍 MEMORY: Relevance filtering: {len(memories)} → {len(relevant_memories)} memories")
        return relevant_memories

    def format_memory_results(self, memories: List[Dict[str, Any]]) -> str:
        """Format memory search results for display, handling both Memories and k-lines.

        Args:
            memories: List of memory dictionaries (can include both Memories and k-lines)

        Returns:
            Formatted string of memories with appropriate formatting for each type
        """
        if not memories:
            return "No relevant memories found."

        result_lines = ["Here's what I remember that might be useful:"]

        for i, memory in enumerate(memories, 1):
            # VectorSet VSIM returns similarity scores (higher = more similar)
            # Convert to percentage (scores are typically between 0 and 1)
            score_percent = memory["score"] * 100

            # Check if this is a k-line or regular neme
            memory_type = memory.get("type", "neme")

            if memory_type == "k-line":
                # Format k-line with its reasoning structure
                question = memory.get("original_question", "Unknown question")
                answer = memory.get("answer", memory.get("text", "No answer available"))
                confidence = memory.get("confidence", "unknown")

                result_lines.append(
                    f"{i}. [K-LINE] Q: {question}"
                )
                result_lines.append(
                    f"   A: {answer} (confidence: {confidence})"
                )
                result_lines.append(
                    f"   (from {memory['formatted_time']}, {score_percent:.1f}% similar)"
                )

                # Show reasoning if available
                reasoning = memory.get("reasoning", "")
                if reasoning and len(reasoning) > 0:
                    # Truncate long reasoning for display
                    reasoning_preview = reasoning[:100] + "..." if len(reasoning) > 100 else reasoning
                    result_lines.append(f"   Reasoning: {reasoning_preview}")

            else:
                # Format regular neme
                text = memory.get("text", memory.get("final_text", memory.get("raw_text", "No text available")))
                result_lines.append(
                    f"{i}. {text} "
                    f"(from {memory['formatted_time']}, {score_percent:.1f}% similar)"
                )

            # Show tags for both types
            if memory.get("tags"):
                result_lines.append(f"   Tags: {', '.join(memory['tags'])}")

        return "\n".join(result_lines)

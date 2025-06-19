#!/usr/bin/env python3
"""
Memory Processing - Mid-level memory analysis and filtering utilities

This module provides utilities for processing memories and questions,
including validation, relevance filtering, and formatting.
"""

import json
from typing import List, Dict, Any
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class MemoryProcessing:
    """Memory analysis and filtering utilities."""
    
    def __init__(self):
        """Initialize the memory processing utilities."""
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def validate_and_preprocess_question(self, user_input: str) -> Dict[str, Any]:
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
                return {
                    "type": "search",
                    "content": user_input
                }

        except Exception as e:
            # Fallback: proceed with original input
            return {
                "type": "search",
                "content": user_input
            }

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
1. Analyze if this memory contains ANY information that could help answer the question
2. Consider both direct and indirect relevance
3. Be strict - only consider it relevant if it actually relates to what's being asked OR may be used logically to deduce or infer the answer
4. Respond with ONLY "RELEVANT" or "NOT_RELEVANT" followed by a brief reason

Examples:
- Question: "When are my kids' birthdays?" Memory: "My daughter is Penelope" → RELEVANT (mentions daughter which may be useful to peice together who are the kids)
- Question: "When are my kids' birthdays?" Memory: "Emma's birthday is March 15th" → RELEVANT (directly answers the question)
- Question: "When are my Kids' birthdays?" Memory: "My birthday is on October 7th" - NOT_RELEVANT (refers to the users birthday but not his/her kids)
- Question: "When are my kids' birthdays?" Memory: "I went to an important and special birthday on July 1st" - RELEVANT (Birthday may have been related to kids?)
- Question: "When are my kids' birthdays?" Memory: "I was with my daughter on July 1st" - RELEVANT (Celebration may have been the referenced birthday of the kids)
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

                if relevance_result.startswith("RELEVANT"):
                    # Add relevance reasoning to memory metadata
                    memory["relevance_reasoning"] = relevance_result[8:].strip() if len(relevance_result) > 8 else "Deemed relevant"
                    relevant_memories.append(memory)
                else:
                    pass  # Memory filtered out silently

            except Exception as e:
                # If relevance check fails, keep the memory to be safe
                memory["relevance_reasoning"] = "Relevance check failed, kept by default"
                relevant_memories.append(memory)

        print(f"🔍 MEMORY: Relevance filtering: {len(memories)} → {len(relevant_memories)} memories")
        return relevant_memories

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

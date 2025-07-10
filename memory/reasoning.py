#!/usr/bin/env python3
"""
Memory Reasoning Service - High-level question answering and analysis

This module provides sophisticated question answering capabilities that
combine memory search with LLM reasoning and relevance filtering.
"""

import json
import sys
from typing import Dict, Any, List
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import LLM manager
sys.path.append('..')
from llm_manager import get_llm_manager


class MemoryReasoning:
    """High-level question answering and analysis using memories."""
    
    def __init__(self, memory_core, memory_processing):
        """Initialize the memory reasoning service.
        
        Args:
            memory_core: Instance of MemoryCore for memory operations
            memory_processing: Instance of MemoryProcessing for filtering and validation
        """
        self.memory_core = memory_core
        self.memory_processing = memory_processing
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def answer_question(self, question: str, top_k: int = 5, filterBy: str = None, min_similarity: float = 0.9, vectorset_key: str = None) -> Dict[str, Any]:
        """Answer a question using relevant memories and OpenAI.

        Args:
            question: The question to answer
            top_k: Number of memories to retrieve for context
            filterBy: Optional filter expression for VSIM command
            min_similarity: Minimum similarity score threshold (0.0-1.0, default: 0.9)
            vectorset_key: Optional vectorset key to use instead of the instance default

        Returns:
            Dictionary with structured response containing:
            - answer: The main answer text
            - confidence: Confidence level (high/medium/low)
            - supporting_memories: List of memory citations that support the answer
            - type: Response type ('answer' or 'help')
        """
        print(f"🤔 MEMORY: Processing question '{question[:60]}{'...' if len(question) > 60 else ''}'")

        # Validate and preprocess the question
        validation_result = self.memory_processing.validate_and_preprocess_question(question)
        
        if validation_result["type"] == "help":
            return {
                "type": "help",
                "answer": validation_result["content"],
                "confidence": "high",
                "supporting_memories": []
            }

        # Use the embedding-optimized query for vector search, fallback to optimized question
        embedding_query = validation_result.get("embedding_query") or validation_result["content"]

        # Search for relevant memories using embedding-optimized query
        search_result = self.memory_core.search_memories(embedding_query, top_k, filterBy, min_similarity, vectorset_key)
        memories = search_result['memories'] if isinstance(search_result, dict) else search_result

        if not memories:
            return {
                "type": "answer",
                "answer": "I don't have any relevant memories to help answer that question.",
                "confidence": "low",
                "supporting_memories": []
            }

        # Send all memories to LLM for inline relevance evaluation (no pre-filtering)
        relevant_memories = memories
        print(f"🧠 REASONING: Sending {len(memories)} memories directly to LLM for inline relevance evaluation")

        # Prepare context from all memories (let LLM evaluate relevance)
        memory_context = []
        for i, memory in enumerate(relevant_memories, 1):
            score_percent = memory["score"] * 100
            memory_context.append(
                f"Memory {i} (Similarity: {score_percent:.1f}%, from {memory['formatted_time']}):\n{memory['text']}\n"
            )

        context_text = "\n".join(memory_context)

        # Create enhanced system prompt for comprehensive memory-based assistance
        system_prompt = """You are an expert personal memory assistant with access to potentially relevant user memories. Your job is to provide helpful, personalized answers based on the user's stored memories while being accurate about what you know and don't know.

CORE MISSION:
Provide personalized, helpful assistance by leveraging ONLY the relevant memories from the user's stored information to understand their preferences, context, constraints, and needs. Use this information to give tailored advice and answers.

CRITICAL INSTRUCTIONS:
1. EVALUATE each memory for relevance to the current question - ignore memories that don't relate
2. Use ONLY relevant memories to provide personalized assistance  
3. When relevant memories contain information, use it to give specific, tailored advice
4. If no memories are relevant or contain sufficient information, be honest about what's missing
5. Consider the user's preferences, constraints, and context from relevant memories when formulating answers
6. Connect related relevant memories to provide comprehensive, personalized responses
7. Do not reference or use memories that are clearly not related to the current question

Your task is to analyze the provided memories and respond with a JSON object containing:
1. "answer": A helpful, personalized answer that leverages relevant memory information, or explanation of what information is needed
2. "confidence": One of "high", "medium", or "low" based on how well the memories support a personalized response
3. "reasoning": Detailed explanation of how you used the memories and why you chose this confidence level

ENHANCED Confidence levels:
- "high": Multiple relevant memories contain comprehensive information that enables a highly personalized and complete answer
- "medium": Some relevant memories contain useful information that enables a partially personalized answer, but some details may be missing  
- "low": Few or no relevant memories found, requiring a more general answer or indicating what information is needed

PERSONALIZATION APPROACH:
- Use family information to tailor recommendations (family size, ages, dietary needs)
- Apply budget constraints and preferences to suggestions
- Consider past experiences and preferences when making recommendations
- Factor in location, accessibility needs, and other personal context
- Connect multiple memories to provide comprehensive, personalized advice

Your response must be valid JSON in this exact format:
{
  "answer": "Personalized answer leveraging relevant memories, or explanation of what information would help provide better assistance",
  "confidence": "high|medium|low",
  "reasoning": "Detailed explanation of how memories were used and why this confidence level was chosen"
}

Be helpful and personalized while being honest about the limits of available information."""

        # Use the original question in the prompt to maintain user context
        user_prompt = f"""Question: {question}

Available user memories (may contain both relevant and irrelevant entries):
{context_text}

Please evaluate each memory for relevance to the current question. Use ONLY the relevant memories to provide a helpful, personalized answer. Ignore any memories that don't relate to the user's current question. Consider the user's preferences, constraints, family situation, past experiences, and other personal context from the relevant memories when formulating your response. If the relevant memories don't contain sufficient information for a complete answer, explain what additional information would be helpful and provide what guidance you can based on what you know about the user."""

        try:
            # Use Tier 1 LLM for main question answering
            llm_manager = get_llm_manager()
            tier1_client = llm_manager.get_tier1_client()

            response = tier1_client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                bypass_cache=True,  # Answer generation is highly context-dependent and accuracy-critical
                temperature=0.3,  # Lower temperature for more consistent answers
                max_tokens=500
            )

            ai_response = response['content'].strip()
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

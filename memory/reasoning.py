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

    def answer_question(self, question: str, top_k: int = 5, filterBy: str = None, min_similarity: float = 0.9) -> Dict[str, Any]:
        """Answer a question using relevant memories and OpenAI.

        Args:
            question: The question to answer
            top_k: Number of memories to retrieve for context
            filterBy: Optional filter expression for VSIM command
            min_similarity: Minimum similarity score threshold (0.0-1.0, default: 0.9)

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
        search_result = self.memory_core.search_memories(embedding_query, top_k, filterBy, min_similarity)
        memories = search_result['memories'] if isinstance(search_result, dict) else search_result

        if not memories:
            return {
                "type": "answer",
                "answer": "I don't have any relevant memories to help answer that question.",
                "confidence": "low",
                "supporting_memories": []
            }

        # Filter memories for relevance using LLM as judge
        relevant_memories = self.memory_processing.filter_relevant_memories(question, memories)

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

        # Create enhanced system prompt for comprehensive memory-based assistance
        system_prompt = """You are an expert personal memory assistant with access to comprehensive user profile information. Your job is to provide helpful, personalized answers based on the user's stored memories while being accurate about what you know and don't know.

CORE MISSION:
Provide personalized, helpful assistance by leveraging the user's stored memories to understand their preferences, context, constraints, and needs. Use this information to give tailored advice and answers.

CRITICAL INSTRUCTIONS:
1. Use ALL relevant information from the memories to provide personalized assistance
2. When memories contain relevant information, use it to give specific, tailored advice
3. If memories don't contain sufficient information, be honest about what's missing
4. Consider the user's preferences, constraints, and context when formulating answers
5. Connect related memories to provide comprehensive, personalized responses

Your task is to analyze the provided memories and respond with a JSON object containing:
1. "answer": A helpful, personalized answer that leverages relevant memory information, or explanation of what information is needed
2. "confidence": One of "high", "medium", or "low" based on how well the memories support a personalized response
3. "reasoning": Detailed explanation of how you used the memories and why you chose this confidence level

ENHANCED Confidence levels:
- "high": Memories contain comprehensive, relevant information that enables a highly personalized and complete answer
- "medium": Memories contain some relevant information that enables a partially personalized answer, but some details may be missing
- "low": Memories contain limited relevant information, requiring a more general answer or indicating what information is needed

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

Available user memories (already filtered for relevance):
{context_text}

Please provide a helpful, personalized answer using the relevant information from these memories. Consider the user's preferences, constraints, family situation, past experiences, and other personal context when formulating your response. If the memories don't contain sufficient information for a complete answer, explain what additional information would be helpful and provide what guidance you can based on what you know about the user."""

        try:
            # Use Tier 1 LLM for main question answering
            llm_manager = get_llm_manager()
            tier1_client = llm_manager.get_tier1_client()

            response = tier1_client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
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

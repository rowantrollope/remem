#!/usr/bin/env python3
"""
Memory Reasoning Service - High-level question answering and analysis

This module provides sophisticated question answering capabilities that
combine memory search with LLM reasoning and relevance filtering.
"""

import json
from typing import Dict, Any, List
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


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

        # Use the optimized question for search
        optimized_question = validation_result["content"]

        # Search for relevant memories
        memories = self.memory_core.search_memories(optimized_question, top_k, filterBy)

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

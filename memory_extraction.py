#!/usr/bin/env python3
"""
Memory Extraction Service - Intelligent memory extraction from conversations

This module provides services for extracting valuable memories from
conversational data using LLM analysis.
"""

import json
from typing import List, Dict, Any, Optional
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class MemoryExtraction:
    """Intelligent memory extraction from conversational data."""
    
    def __init__(self, memory_core):
        """Initialize the memory extraction service.
        
        Args:
            memory_core: Instance of MemoryCore for storing extracted memories
        """
        self.memory_core = memory_core
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def extract_and_store_memories(self,
                                 raw_input: str,
                                 context_prompt: str,
                                 extraction_examples: Optional[List[Dict[str, str]]] = None,
                                 apply_grounding: bool = True) -> Dict[str, Any]:
        """Intelligently extract and store valuable memories from conversational data.

        Args:
            raw_input: The full conversational data to analyze
            context_prompt: Application-specific context for extraction guidance
            extraction_examples: Optional examples to guide LLM extraction
            apply_grounding: Whether to apply contextual grounding to extracted memories

        Returns:
            Dictionary containing:
            - extracted_memories: List of extracted and stored memories with IDs
            - extraction_summary: Summary of extraction process
            - total_extracted: Number of memories extracted
            - total_filtered: Number of potential memories filtered out
        """
        print(f"🔍 Extracting memories from conversational input ({len(raw_input)} characters)")
        print(f"📋 Context: {context_prompt[:100]}...")

        # Build extraction prompt
        extraction_prompt = self._build_extraction_prompt(
            raw_input, context_prompt, extraction_examples
        )

        # First, check if the input contains extractable information
        if not self._contains_extractable_info_llm(raw_input):
            return {
                "extracted_memories": [],
                "extraction_summary": "No valuable memories found in the input (pre-screening)",
                "total_extracted": 0,
                "total_filtered": 0,
                "extraction_reasoning": "Input did not pass initial extractability screening"
            }

        # Extract memories using LLM
        extraction_result = self._extract_memories_with_llm(extraction_prompt)

        if not extraction_result or not extraction_result.get("extracted_facts"):
            return {
                "extracted_memories": [],
                "extraction_summary": "No valuable memories found in the input",
                "total_extracted": 0,
                "total_filtered": len(extraction_result.get("filtered_out", [])) if extraction_result else 0,
                "extraction_reasoning": extraction_result.get("reasoning", "No extraction performed") if extraction_result else "Extraction failed"
            }

        # Store each extracted memory
        stored_memories = []
        extraction_errors = []

        for i, fact in enumerate(extraction_result["extracted_facts"], 1):
            try:
                # Store the memory using existing store_memory method
                storage_result = self.memory_core.store_memory(fact["text"], apply_grounding=apply_grounding)

                # Add extraction metadata to the result
                memory_info = {
                    "memory_id": storage_result["memory_id"],
                    "extracted_text": fact["text"],
                    "confidence": fact.get("confidence", "medium"),
                    "reasoning": fact.get("reasoning", ""),
                    "category": fact.get("category", "general"),
                    "original_text": storage_result["original_text"],
                    "final_text": storage_result["final_text"],
                    "grounding_applied": storage_result["grounding_applied"],
                    "timestamp": storage_result["timestamp"],
                    "formatted_time": storage_result["formatted_time"]
                }

                # Add grounding info if available
                if storage_result.get("grounding_info"):
                    memory_info["grounding_info"] = storage_result["grounding_info"]

                stored_memories.append(memory_info)

            except Exception as e:
                error_msg = f"Failed to store memory '{fact['text'][:50]}...': {str(e)}"
                print(f"❌ {error_msg}")
                extraction_errors.append(error_msg)

        # Prepare summary
        total_extracted = len(stored_memories)
        total_filtered = len(extraction_result.get("filtered_out", []))

        extraction_summary = f"Successfully extracted and stored {total_extracted} memories"
        if total_filtered > 0:
            extraction_summary += f", filtered out {total_filtered} items"
        if extraction_errors:
            extraction_summary += f", encountered {len(extraction_errors)} storage errors"

        result = {
            "extracted_memories": stored_memories,
            "extraction_summary": extraction_summary,
            "total_extracted": total_extracted,
            "total_filtered": total_filtered,
            "extraction_reasoning": extraction_result.get("reasoning", ""),
            "filtered_items": extraction_result.get("filtered_out", []),
            "extraction_errors": extraction_errors
        }

        return result

    def _contains_extractable_info_llm(self, raw_input: str) -> bool:
        """Use LLM to pre-screen if input contains extractable information."""
        try:
            evaluation_prompt = f"""You are an intelligent memory evaluation system. Your task is to determine if the following conversational text contains information that would be valuable to remember for future interactions.

VALUABLE INFORMATION INCLUDES:
- Personal preferences (likes, dislikes, habits)
- Constraints and requirements (budget, time, accessibility needs)
- Personal details (family, dietary restrictions, important dates)
- Factual information about people, places, or things
- Goals and intentions
- Important contextual details

STRICTLY IGNORE:
- Temporary information (current weather, today's schedule, immediate tasks)
- Conversational filler or pleasantries ("Hi there", "How are you?")
- General questions without personal context ("What's the best way to...")
- Information requests that don't reveal user preferences
- Time-sensitive information that won't be relevant later
- Assistant responses or suggestions

CONVERSATIONAL TEXT TO EVALUATE:
"{raw_input}"

Respond with ONLY "YES" if the text contains valuable information worth remembering, or "NO" if it doesn't. Do not include any explanation."""

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": evaluation_prompt}],
                temperature=0.1,  # Low temperature for consistent evaluation
                max_tokens=10     # We only need YES or NO
            )

            result = response.choices[0].message.content.strip().upper()
            return result == "YES"

        except Exception as e:
            print(f"⚠️ LLM pre-screening failed, proceeding with extraction: {e}")
            return True  # Default to allowing extraction if screening fails

    def _build_extraction_prompt(self,
                               raw_input: str,
                               context_prompt: str,
                               extraction_examples: Optional[List[Dict[str, str]]] = None) -> str:
        """Build the LLM prompt for memory extraction.

        Args:
            raw_input: The conversational input to analyze
            context_prompt: Application-specific context
            extraction_examples: Optional examples to guide extraction

        Returns:
            Complete prompt for LLM extraction
        """

        # Build examples section
        examples_section = ""
        if extraction_examples:
            examples_section = "\n\nEXTRACTION EXAMPLES:\n"
            for i, example in enumerate(extraction_examples, 1):
                examples_section += f"\nExample {i}:\n"
                examples_section += f"Input: \"{example.get('input', '')}\"\n"
                examples_section += f"Extract: \"{example.get('extract', '')}\"\n"
                examples_section += f"Reason: {example.get('reason', '')}\n"

        prompt = f"""You are an intelligent memory extraction system. Your task is to analyze conversational input and extract only the most valuable, memorable facts that should be stored for long-term retention.

CONTEXT: {context_prompt}

EXTRACTION CRITERIA:
1. **Personal Preferences**: User likes, dislikes, preferences, habits
2. **Constraints & Requirements**: Budget limits, time constraints, accessibility needs
3. **Personal Details**: Family information, dietary restrictions, important dates
4. **Factual Information**: Specific facts about people, places, or things mentioned
5. **Goals & Intentions**: User's stated objectives or plans
6. **Important Context**: Significant situational or environmental details

QUALITY STANDARDS:
- Extract ONLY information that would be valuable to remember for future interactions
- Prefer specific, actionable facts over general statements
- Each extracted fact should be self-contained and clear

STRICTLY AVOID EXTRACTING:
- Temporary information (current weather, today's schedule, immediate tasks)
- Conversational filler or pleasantries ("Hi there", "How are you?")
- General questions without personal context
- Information requests that don't reveal user preferences
- Time-sensitive information that won't be relevant later
- Assistant responses or suggestions

CONFIDENCE LEVELS:
- "high": Clear, explicit statement of fact or preference
- "medium": Implied or contextual information that's likely accurate
- "low": Uncertain or ambiguous information that might be worth storing

CATEGORIES:
- "preference": User likes, dislikes, or preferences
- "constraint": Limitations, requirements, or restrictions
- "personal": Personal details, family, relationships
- "factual": Objective facts about entities
- "goal": Objectives, plans, or intentions
- "context": Important situational information{examples_section}
- You are able to create new categories if you identify a better a categorization system for data you receive

CONVERSATIONAL INPUT TO ANALYZE:
"{raw_input}"

INSTRUCTIONS:
1. Carefully read through the entire conversational input
2. Identify facts worth remembering based on the criteria above
3. For each fact, determine confidence level and category
4. Provide clear reasoning for extraction decisions
5. Also list items you considered but filtered out and why

Respond with a JSON object in this exact format:
{{
  "extracted_facts": [
    {{
      "text": "Clear, self-contained statement of the fact",
      "confidence": "high|medium|low",
      "category": "preference|constraint|personal|factual|goal|context|etc",
      "reasoning": "Why this fact is worth remembering"
    }}
  ],
  "filtered_out": [
    {{
      "text": "Information that was considered but not extracted",
      "reason": "Why this was filtered out"
    }}
  ],
  "reasoning": "Overall summary of extraction decisions and approach"
}}

Be selective and focus on quality over quantity. It's better to extract fewer high-value memories than many low-value ones."""

        return prompt

    def _extract_memories_with_llm(self, extraction_prompt: str) -> Optional[Dict[str, Any]]:
        """Use LLM to extract memories from conversational input.

        Args:
            extraction_prompt: The complete prompt for memory extraction

        Returns:
            Dictionary with extracted facts and filtering information, or None if failed
        """
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=0.2,  # Low temperature for consistent extraction
                max_tokens=1500   # Allow for detailed extraction
            )

            extraction_result = response.choices[0].message.content.strip()
            print(f"🔍 LLM extraction response received ({len(extraction_result)} characters)")

            # Parse JSON response
            try:
                extraction_data = json.loads(extraction_result)

                # Validate the response structure
                if not isinstance(extraction_data, dict):
                    print("⚠️ LLM response is not a valid JSON object")
                    return None

                extracted_facts = extraction_data.get("extracted_facts", [])
                filtered_out = extraction_data.get("filtered_out", [])
                reasoning = extraction_data.get("reasoning", "")

                # Validate extracted facts structure
                valid_facts = []
                for fact in extracted_facts:
                    if isinstance(fact, dict) and "text" in fact:
                        # Ensure required fields have defaults
                        valid_fact = {
                            "text": fact["text"],
                            "confidence": fact.get("confidence", "medium"),
                            "category": fact.get("category", "general"),
                            "reasoning": fact.get("reasoning", "")
                        }
                        valid_facts.append(valid_fact)
                    else:
                        print(f"⚠️ Skipping invalid fact structure: {fact}")

                print(f"✅ Extracted {len(valid_facts)} valid facts, filtered {len(filtered_out)} items")

                return {
                    "extracted_facts": valid_facts,
                    "filtered_out": filtered_out,
                    "reasoning": reasoning
                }

            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse LLM extraction response as JSON: {e}")
                print(f"Raw response: {extraction_result[:500]}...")
                return None

        except Exception as e:
            print(f"❌ Error during LLM memory extraction: {e}")
            return None

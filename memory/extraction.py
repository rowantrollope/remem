#!/usr/bin/env python3
"""
Memory Extraction Service - Intelligent memory extraction from conversations

This module provides services for extracting valuable memories from
conversational data using LLM analysis.
"""

import json
import sys
from typing import List, Dict, Any, Optional
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import debug utilities
from .debug_utils import debug_print, warning_print, error_print, success_print, is_debug_enabled

# Import LLM manager and LangCache
sys.path.append('..')
from llm.llm_manager import get_llm_manager
from clients.langcache_client import LangCacheClient, CachedLLMClient


class MemoryExtraction:
    """Intelligent memory extraction from conversational data."""
    
    def __init__(self, memory_core):
        """Initialize the memory extraction service.

        Args:
            memory_core: Instance of MemoryCore for storing extracted memories
        """
        self.memory_core = memory_core
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Initialize LangCache if environment variables are available
        self.langcache_client = None
        try:
            if all([os.getenv("LANGCACHE_HOST"), os.getenv("LANGCACHE_API_KEY"), os.getenv("LANGCACHE_CACHE_ID")]):
                self.langcache_client = LangCacheClient()
                debug_print("LangCache enabled for memory extraction", "CACHE")
            else:
                debug_print("LangCache not configured for memory extraction (missing environment variables)", "CACHE")
        except Exception as e:
            warning_print(f"Failed to initialize LangCache for memory extraction: {e}")

    def extract_and_store_memories(self,
                                 raw_input: str,
                                 context_prompt: str,
                                 extraction_examples: Optional[List[Dict[str, str]]] = None,
                                 apply_grounding: bool = True,
                                 existing_memories: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Intelligently extract and store valuable memories from conversational data.

        Args:
            raw_input: The full conversational data to analyze
            context_prompt: Application-specific context for extraction guidance
            extraction_examples: Optional examples to guide LLM extraction
            apply_grounding: Whether to apply contextual grounding to extracted memories
            existing_memories: Optional list of existing relevant memories to avoid duplicates

        Returns:
            Dictionary containing:
            - extracted_memories: List of extracted and stored memories with IDs
            - extraction_summary: Summary of extraction process
            - total_extracted: Number of memories extracted
            - total_filtered: Number of potential memories filtered out
        """
        debug_print(f"Extracting memories from conversational input ({len(raw_input)} characters)", "EXTRACT")
        debug_print(f"Context: {context_prompt[:100]}...", "CONTEXT")
        if existing_memories:
            debug_print(f"EXISTING MEMORIES PROVIDED: {len(existing_memories)} memories", "MEMORY")
            if is_debug_enabled():
                for i, mem in enumerate(existing_memories[:3], 1):  # Show first 3
                    mem_text = mem.get('text', mem.get('final_text', ''))
                    debug_print(f"{i}. {mem_text[:60]}...", "📝")
        else:
            debug_print("NO EXISTING MEMORIES PROVIDED", "MEMORY")

        # Build extraction prompt with existing memories context
        extraction_prompt = self._build_extraction_prompt(
            raw_input, context_prompt, extraction_examples, existing_memories
        )

        # Extract memories using LLM (single call handles both screening and extraction)
        extraction_result = self._extract_memories_with_llm(extraction_prompt)

        if not extraction_result or not extraction_result.get("extracted_facts"):
            return {
                "extracted_memories": [],
                "extraction_summary": "No valuable memories found in the input",
                "total_extracted": 0,
                "total_filtered": len(extraction_result.get("filtered_out", [])) if extraction_result else 0,
                "extraction_reasoning": extraction_result.get("reasoning", "No extraction performed") if extraction_result else "Extraction failed"
            }

        # Store each extracted memory with duplicate checking
        stored_memories = []
        extraction_errors = []
        duplicates_skipped = 0

        for fact in extraction_result["extracted_facts"]:
            try:
                # Note: Duplicate checking is now handled by LLM context awareness
                # The LLM should only extract information not already in existing_memories

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
                    "created_at": storage_result["created_at"]
                }

                # Add grounding info if available
                if storage_result.get("grounding_info"):
                    memory_info["grounding_info"] = storage_result["grounding_info"]

                stored_memories.append(memory_info)

            except Exception as e:
                error_msg = f"Failed to store memory '{fact['text'][:50]}...': {str(e)}"
                error_print(error_msg)
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
            "duplicates_skipped": duplicates_skipped,
            "extraction_reasoning": extraction_result.get("reasoning", ""),
            "filtered_items": extraction_result.get("filtered_out", []),
            "extraction_errors": extraction_errors
        }

        return result



    def _build_extraction_prompt(self,
                               raw_input: str,
                               context_prompt: str,
                               extraction_examples: Optional[List[Dict[str, str]]] = None,
                               existing_memories: Optional[List[Dict[str, Any]]] = None) -> str:
        """Build the LLM prompt for memory extraction.

        Args:
            raw_input: The conversational input to analyze
            context_prompt: Application-specific context
            extraction_examples: Optional examples to guide extraction
            existing_memories: Optional list of existing memories to avoid duplicates

        Returns:
            Complete prompt for LLM extraction
        """

        # Build existing memories section
        existing_memories_section = ""
        if existing_memories and len(existing_memories) > 0:
            existing_memories_section = "\n\nEXISTING RELEVANT MEMORIES:\n"
            existing_memories_section += "The following information is ALREADY STORED about the user. DO NOT extract information that duplicates these existing memories:\n\n"
            for i, memory in enumerate(existing_memories, 1):
                memory_text = memory.get('text', memory.get('final_text', ''))
                memory_time = memory.get('created_at', 'Unknown time')
                existing_memories_section += f"{i}. {memory_text} (stored: {memory_time})\n"
            existing_memories_section += "\nONLY extract information that is NEW and NOT already captured in the above memories.\n"

        # Build examples section
        examples_section = ""
        if extraction_examples:
            examples_section = "\n\nEXTRACTION EXAMPLES:\n"
            for i, example in enumerate(extraction_examples, 1):
                examples_section += f"\nExample {i}:\n"
                examples_section += f"Input: \"{example.get('input', '')}\"\n"
                examples_section += f"Extract: \"{example.get('extract', '')}\"\n"
                examples_section += f"Reason: {example.get('reason', '')}\n"

        prompt = f"""You are a selective memory extraction system. Your task is to analyze conversational input and extract ONLY significant NEW information that is not already stored in existing memories.

**CRITICAL WARNING**: You will see examples and reference material in this prompt. DO NOT extract information from examples, reference material, or any text marked as examples. ONLY extract from the actual conversational input that will be clearly marked.

CONTEXT: {context_prompt}

MISSION: Extract only important NEW user information that would significantly improve future assistance. Be selective and conservative - avoid extracting minor details or temporary information.

{existing_memories_section}

EXTRACTION CRITERIA (only extract if truly valuable):
1. **Important Preferences**: Significant likes, dislikes, or requirements
2. **Key Constraints**: Budget limits, accessibility needs, dietary restrictions
3. **Essential Personal Details**: Family composition, location, major life details
4. **Significant Plans**: Important goals, upcoming events, major decisions

WHAT TO EXTRACT:
- Dietary restrictions or allergies
- Budget constraints for major categories (travel, dining)
- Family composition (size, ages of children)
- Location or travel preferences
- Accessibility needs
- Important upcoming plans or goals
- Significant preferences that affect recommendations

WHAT NOT TO EXTRACT:
- Minor conversational details
- Temporary information (today's weather, current mood)
- Information already captured in existing memories
- Assistant responses or suggestions
- Vague or uncertain information
- Overly specific details that won't be useful later

QUALITY STANDARDS:
- Be smart - extract information that would improve future assistance
- Prefer clear, actionable facts over minor details
- Each extracted fact should be self-contained and valuable
- Focus on information that has lasting relevance

CONFIDENCE LEVELS:
- "high": Clear, explicit statement of important fact
- "medium": Reasonably certain information that's valuable
- "low": Uncertain information (generally avoid extracting these)

CATEGORIES:
- "preference": Important user preferences
- "constraint": Significant limitations or requirements
- "personal": Key personal details
- "goal": Important objectives or plans{examples_section}

**IMPORTANT: THE FOLLOWING ARE EXAMPLES ONLY - DO NOT EXTRACT INFORMATION FROM THESE EXAMPLES**

EXTRACTION EXAMPLES (FOR REFERENCE ONLY - DO NOT EXTRACT FROM THESE):
Example Input: "I'm planning a trip to Europe with my family of 4 next summer"
Example Extract:
- "User is planning a trip to Europe next summer" (category: future_plans)
- "User has a family of 4 people" (category: family)

Example Input: "My wife is vegetarian and my 8-year-old is picky about food"
Example Extract:
- "User's wife is vegetarian" (category: family_dietary)
- "User has an 8-year-old child" (category: family)
- "User's 8-year-old child is picky about food" (category: family_preferences)

Example Input: "I prefer window seats and usually book hotels under $200/night"
Example Extract:
- "User prefers window seats when flying" (category: travel_preferences)
- "User typically books hotels under $200 per night" (category: budget_constraints)

**END OF EXAMPLES - THESE ARE FOR REFERENCE ONLY**

NOW ANALYZE THE ACTUAL CONVERSATIONAL INPUT BELOW:
"{raw_input}"

INSTRUCTIONS:
1. Carefully read through the conversational input
2. Identify information that would improve future assistance
3. Be smart - avoid extracting minor or temporary details
4. For each fact, ensure it's valuable and determine confidence level
5. Provide clear reasoning for your extraction decisions
6. If no significant new information is present, return empty "extracted_facts" array

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

CRITICAL: Only extract information that is NEW, and NOT already captured in existing memories.

EXTRACTION GUIDELINES:
- If existing memories already capture the information, extract NOTHING
- Only extract information that adds significant new value
- Be conservative - when in doubt, don't extract
- Focus on information that would meaningfully improve future assistance

ADDITIONAL GUIDANCE:

SKIP (not significant enough):
- "I'm having a good day" → EXTRACT: [] (temporary mood)
- "That sounds interesting" → EXTRACT: [] (conversational response)
- "Thanks for the help" → EXTRACT: [] (politeness)

Take into account the type of work this assistant it trying to accomplish. For example: 
for a coding assistant, you want to extract preferences and things relating to coding
for a travel assistant, you want to extract memories relating to travel.
For a customer service assistant, you want to extract memories relating to the user's needs and preferences.
Etc. 

EXTRACT (significant new information):
- "I'm vegetarian" → EXTRACT: ["User is vegetarian"] (important dietary restriction)
- "My family of 4 is planning a trip to Europe" → EXTRACT: ["User has family of 4", "User is planning Europe trip"]
- "I have a $2000 budget for hotels" → EXTRACT: ["User has $2000 hotel budget"] (important constraint)

Be selective and focus only on information that would improve future assistance.

**FINAL REMINDER**: Only extract information from the actual conversational input marked above. Do NOT extract any information from the examples, reference material, or instructional text in this prompt."""

        # Debug: Show if existing memories section was included
        if existing_memories and len(existing_memories) > 0:
            debug_print(f"Including {len(existing_memories)} existing memories in extraction prompt", "PROMPT")
        else:
            debug_print("No existing memories included in extraction prompt", "PROMPT")

        return prompt

    def _has_semantic_overlap(self, text1: str, text2: str) -> bool:
        """Check for semantic overlap between two texts using key phrase matching.

        Args:
            text1: First text to compare
            text2: Second text to compare

        Returns:
            True if texts have significant semantic overlap
        """
        try:
            import re

            # Extract key semantic patterns
            patterns = [
                r'planning (?:a )?trip to (\w+)',
                r'wants? to (?:plan|visit|go to) (\w+)',
                r'going to (\w+)',
                r'family of (\d+)',
                r'(?:wife|husband|spouse) is (\w+)',
                r'prefer (\w+(?:\s+\w+)*)',
                r'budget (?:is|of) ([\d,]+)',
                r'allergic to (\w+)',
                r'live in (\w+)',
                r'work (?:as|in) (\w+)',
            ]

            # Extract semantic elements from both texts
            elements1 = set()
            elements2 = set()

            for pattern in patterns:
                matches1 = re.findall(pattern, text1.lower())
                matches2 = re.findall(pattern, text2.lower())
                elements1.update(matches1)
                elements2.update(matches2)

            # Check for overlap in semantic elements
            if elements1 and elements2:
                overlap = len(elements1.intersection(elements2))
                total_unique = len(elements1.union(elements2))

                if total_unique > 0:
                    semantic_overlap_ratio = overlap / total_unique
                    return semantic_overlap_ratio > 0.5  # 50% semantic overlap

            return False

        except Exception as e:
            warning_print(f"Semantic overlap check failed: {e}")
            return False

    def _extract_memories_with_llm(self, extraction_prompt: str) -> Optional[Dict[str, Any]]:
        """Use LLM to extract memories from conversational input.

        Args:
            extraction_prompt: The complete prompt for memory extraction

        Returns:
            Dictionary with extracted facts and filtering information, or None if failed
        """
        try:
            # Try to use Tier 1 LLM for memory extraction if available
            try:
                llm_manager = get_llm_manager()
                tier1_client = llm_manager.get_tier1_client()
            except RuntimeError:
                # LLM manager not initialized (CLI/MCP context), skip extraction
                error_print("Memory extraction skipped: LLM manager not initialized")
                return None

            # Wrap with LangCache if available
            if self.langcache_client:
                cached_client = CachedLLMClient(tier1_client, self.langcache_client)
                response = cached_client.chat_completion(
                    messages=[
                        {"role": "user", "content": extraction_prompt}
                    ],
                    operation_type='memory_extraction',
                    temperature=0.2,  # Low temperature for consistent extraction
                    max_tokens=1500   # Allow for detailed extraction
                )
            else:
                # Use Tier 1 LLM for memory extraction without caching
                response = tier1_client.chat_completion(
                    messages=[
                        {"role": "user", "content": extraction_prompt}
                    ],
                    temperature=0.2,  # Low temperature for consistent extraction
                    max_tokens=1500   # Allow for detailed extraction
                )

            extraction_result = response['content'].strip()
            debug_print(f"LLM extraction response received ({len(extraction_result)} characters)", "LLM")

            # Parse JSON response
            try:
                extraction_data = json.loads(extraction_result)

                # Validate the response structure
                if not isinstance(extraction_data, dict):
                    warning_print("LLM response is not a valid JSON object")
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
                        warning_print(f"Skipping invalid fact structure: {fact}")

                success_print(f"Extracted {len(valid_facts)} valid facts, filtered {len(filtered_out)} items")

                return {
                    "extracted_facts": valid_facts,
                    "filtered_out": filtered_out,
                    "reasoning": reasoning
                }

            except json.JSONDecodeError as e:
                error_print(f"Failed to parse LLM extraction response as JSON: {e}")
                debug_print(f"Raw response: {extraction_result[:500]}...", "RAW")
                return None

        except Exception as e:
            error_print(f"Error during LLM memory extraction: {e}")
            return None

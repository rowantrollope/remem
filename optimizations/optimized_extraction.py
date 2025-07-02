#!/usr/bin/env python3
"""
Optimized Memory Extraction - Merged LLM operations for efficiency

This module provides optimized memory extraction that combines multiple
LLM operations into single calls and uses intelligent caching.
"""

import json
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import LLM manager
sys.path.append('..')
from llm_manager import get_llm_manager


class OptimizedMemoryExtraction:
    """Optimized memory extraction with merged LLM operations."""
    
    def __init__(self, memory_core):
        """Initialize with memory core for storage operations."""
        self.memory_core = memory_core
    
    def extract_and_store_memories_optimized(self,
                                           raw_input: str,
                                           context_prompt: str,
                                           extraction_examples: Optional[List[Dict[str, str]]] = None,
                                           apply_grounding: bool = True,
                                           existing_memories: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Optimized memory extraction that combines evaluation and extraction in a single LLM call.
        
        Args:
            raw_input: The conversational data to analyze
            context_prompt: Application-specific context
            extraction_examples: Optional examples to guide extraction
            apply_grounding: Whether to apply contextual grounding
            existing_memories: Optional existing memories to avoid duplicates
            
        Returns:
            Dictionary with extraction results and stored memories
        """
        print(f"🧠 EXTRACTION: Processing input ({len(raw_input)} chars)")
        
        # Build comprehensive extraction prompt that handles both evaluation and extraction
        extraction_prompt = self._build_comprehensive_extraction_prompt(
            raw_input, context_prompt, extraction_examples, existing_memories
        )
        
        # Single LLM call for both evaluation and extraction
        extraction_result = self._extract_memories_comprehensive(extraction_prompt)
        
        if not extraction_result or not extraction_result.get("should_extract", False):
            return {
                "extracted_memories": [],
                "extraction_summary": extraction_result.get("reasoning", "No valuable memories found"),
                "total_extracted": 0,
                "total_filtered": 0,
                "extraction_reasoning": extraction_result.get("reasoning", "No extraction performed")
            }
        
        # Store extracted memories
        extracted_facts = extraction_result.get("extracted_facts", [])
        stored_memories = []
        
        for fact in extracted_facts:
            try:
                # Store with optional grounding
                storage_result = self.memory_core.store_memory(
                    fact, apply_grounding=apply_grounding
                )
                stored_memories.append(storage_result)
                print(f"✅ Stored: {fact[:60]}{'...' if len(fact) > 60 else ''}")
            except Exception as e:
                print(f"⚠️ Failed to store memory '{fact[:40]}...': {e}")
        
        return {
            "extracted_memories": stored_memories,
            "extraction_summary": f"Extracted and stored {len(stored_memories)} memories",
            "total_extracted": len(stored_memories),
            "total_filtered": len(extraction_result.get("filtered_out", [])),
            "extraction_reasoning": extraction_result.get("reasoning", "Extraction completed")
        }
    
    def _build_comprehensive_extraction_prompt(self,
                                             raw_input: str,
                                             context_prompt: str,
                                             extraction_examples: Optional[List[Dict[str, str]]] = None,
                                             existing_memories: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Build a comprehensive prompt that handles both evaluation and extraction.
        """
        prompt = f"""You are an intelligent memory extraction system. Your task is to:
1. EVALUATE if the input contains valuable information worth remembering
2. If valuable, EXTRACT specific facts that should be stored as memories

{context_prompt}

EVALUATION CRITERIA - Information is valuable if it contains:
- Personal preferences, constraints, or requirements
- Factual information about people, places, or things
- Important personal details or context
- Goals, intentions, or plans
- Specific constraints (budget, time, accessibility needs)

IGNORE:
- Temporary information (current weather, today's schedule)
- Conversational filler or pleasantries
- General questions without personal context
- Assistant responses or suggestions
- Time-sensitive information that won't be relevant later

INPUT TO ANALYZE:
"{raw_input}"
"""

        # Add existing memories context if provided
        if existing_memories:
            prompt += f"""
EXISTING RELATED MEMORIES (avoid duplicates):
"""
            for i, memory in enumerate(existing_memories[:5], 1):  # Limit to 5 for prompt efficiency
                prompt += f"{i}. {memory.get('text', '')[:80]}...\n"

        # Add examples if provided
        if extraction_examples:
            prompt += "\nEXTRACTION EXAMPLES:\n"
            for example in extraction_examples[:3]:  # Limit examples
                prompt += f"Input: {example.get('input', '')}\n"
                prompt += f"Output: {example.get('output', '')}\n\n"

        prompt += """
RESPONSE FORMAT (JSON):
{
    "should_extract": true/false,
    "reasoning": "Brief explanation of decision",
    "extracted_facts": ["fact1", "fact2", ...],
    "filtered_out": ["reason1", "reason2", ...]
}

If should_extract is false, set extracted_facts to empty array.
If should_extract is true, list specific, atomic facts to remember.
Each fact should be a complete, standalone statement.

Your response:"""

        return prompt
    
    def _extract_memories_comprehensive(self, extraction_prompt: str) -> Optional[Dict[str, Any]]:
        """
        Perform comprehensive extraction and evaluation in a single LLM call.
        
        Args:
            extraction_prompt: Complete prompt for extraction
            
        Returns:
            Dictionary with evaluation and extraction results, or None if failed
        """
        try:
            # Use Tier 2 LLM for memory extraction
            llm_manager = get_llm_manager()
            tier2_client = llm_manager.get_tier2_client()

            # Use caching if available
            if hasattr(tier2_client, 'chat_completion'):
                response = tier2_client.chat_completion(
                    messages=[
                        {"role": "user", "content": extraction_prompt}
                    ],
                    operation_type='extraction_comprehensive',
                    cache_context={'prompt_length': len(extraction_prompt)},
                    temperature=0.2,
                    max_tokens=1000
                )
            else:
                response = tier2_client.chat_completion(
                    messages=[
                        {"role": "user", "content": extraction_prompt}
                    ],
                    temperature=0.2,
                    max_tokens=1000
                )

            extraction_result = response['content'].strip()
            print(f"🔍 Comprehensive extraction response received ({len(extraction_result)} characters)")

            # Parse JSON response
            try:
                result = json.loads(extraction_result)
                
                # Validate response structure
                if not isinstance(result, dict):
                    raise ValueError("Response is not a dictionary")
                
                # Ensure required fields exist
                result.setdefault("should_extract", False)
                result.setdefault("reasoning", "No reasoning provided")
                result.setdefault("extracted_facts", [])
                result.setdefault("filtered_out", [])
                
                return result
                
            except json.JSONDecodeError as e:
                print(f"⚠️ Could not parse extraction JSON: {e}")
                print(f"Raw response: {extraction_result[:200]}...")
                
                # Fallback: try to extract facts from text
                return self._fallback_extraction_parsing(extraction_result)
                
        except Exception as e:
            print(f"⚠️ Comprehensive extraction failed: {e}")
            return None
    
    def _fallback_extraction_parsing(self, response_text: str) -> Dict[str, Any]:
        """
        Fallback parsing when JSON parsing fails.
        
        Args:
            response_text: Raw LLM response text
            
        Returns:
            Parsed extraction result
        """
        # Simple heuristic parsing
        should_extract = "should_extract" in response_text.lower() and "true" in response_text.lower()
        
        # Try to find facts in the response
        facts = []
        lines = response_text.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for lines that might be facts
            if (line.startswith('-') or line.startswith('*') or 
                line.startswith('•') or line[0:2].isdigit()):
                # Clean up the line
                fact = line.lstrip('-*•0123456789. ').strip()
                if len(fact) > 10 and not fact.startswith('"'):  # Basic quality filter
                    facts.append(fact)
        
        return {
            "should_extract": should_extract and len(facts) > 0,
            "reasoning": "Fallback parsing applied due to JSON parse error",
            "extracted_facts": facts[:5],  # Limit to 5 facts
            "filtered_out": ["JSON parsing failed"]
        }
    
    def extract_memories_batch(self, inputs: List[str], context_prompt: str,
                             apply_grounding: bool = True) -> List[Dict[str, Any]]:
        """
        Extract memories from multiple inputs in a single LLM call for efficiency.
        
        Args:
            inputs: List of conversational inputs to process
            context_prompt: Application context
            apply_grounding: Whether to apply grounding to extracted memories
            
        Returns:
            List of extraction results for each input
        """
        if not inputs:
            return []
        
        if len(inputs) == 1:
            return [self.extract_and_store_memories_optimized(
                inputs[0], context_prompt, apply_grounding=apply_grounding
            )]
        
        # Build batch extraction prompt
        batch_prompt = f"""You are an intelligent memory extraction system. Process the following {len(inputs)} conversational inputs and extract valuable memories from each.

{context_prompt}

INPUTS TO PROCESS:
"""
        for i, input_text in enumerate(inputs, 1):
            batch_prompt += f"\n{i}. {input_text}\n"
        
        batch_prompt += f"""
For each input, determine if it contains valuable information and extract specific facts.

RESPONSE FORMAT (JSON):
{{
    "results": [
        {{
            "input_number": 1,
            "should_extract": true/false,
            "reasoning": "Brief explanation",
            "extracted_facts": ["fact1", "fact2", ...],
            "filtered_out": ["reason1", ...]
        }},
        ...
    ]
}}

Your response:"""

        try:
            llm_manager = get_llm_manager()
            tier2_client = llm_manager.get_tier2_client()
            
            if hasattr(tier2_client, 'chat_completion'):
                response = tier2_client.chat_completion(
                    messages=[{"role": "user", "content": batch_prompt}],
                    operation_type='extraction_batch',
                    cache_context={'input_count': len(inputs)},
                    temperature=0.2,
                    max_tokens=500 * len(inputs)
                )
            else:
                response = tier2_client.chat_completion(
                    messages=[{"role": "user", "content": batch_prompt}],
                    temperature=0.2,
                    max_tokens=500 * len(inputs)
                )

            batch_result = json.loads(response['content'].strip())
            results = []
            
            for i, input_text in enumerate(inputs):
                # Find corresponding result
                input_result = None
                for result in batch_result.get("results", []):
                    if result.get("input_number") == i + 1:
                        input_result = result
                        break
                
                if not input_result or not input_result.get("should_extract", False):
                    results.append({
                        "extracted_memories": [],
                        "extraction_summary": "No valuable memories found",
                        "total_extracted": 0,
                        "total_filtered": 0,
                        "extraction_reasoning": input_result.get("reasoning", "No extraction") if input_result else "Processing failed"
                    })
                    continue
                
                # Store extracted facts
                extracted_facts = input_result.get("extracted_facts", [])
                stored_memories = []
                
                for fact in extracted_facts:
                    try:
                        storage_result = self.memory_core.store_memory(
                            fact, apply_grounding=apply_grounding
                        )
                        stored_memories.append(storage_result)
                    except Exception as e:
                        print(f"⚠️ Failed to store memory '{fact[:40]}...': {e}")
                
                results.append({
                    "extracted_memories": stored_memories,
                    "extraction_summary": f"Extracted and stored {len(stored_memories)} memories",
                    "total_extracted": len(stored_memories),
                    "total_filtered": len(input_result.get("filtered_out", [])),
                    "extraction_reasoning": input_result.get("reasoning", "Extraction completed")
                })
            
            return results
            
        except Exception as e:
            print(f"⚠️ Batch extraction failed: {e}")
            # Fallback to individual processing
            return [self.extract_and_store_memories_optimized(
                input_text, context_prompt, apply_grounding=apply_grounding
            ) for input_text in inputs]

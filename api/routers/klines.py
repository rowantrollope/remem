"""
K-line API endpoints - Reflective Operations (Inspired by Minsky's "K-lines").

K-lines (Knowledge lines) represent temporary mental states that activate
and connect relevant Memories for specific cognitive tasks. In Minsky's theory,
K-lines are the mechanism by which the mind constructs coherent mental states
from distributed memory fragments.

These APIs handle:
- Constructing mental states by recalling and filtering relevant memories
- Question answering with confidence scoring and reasoning
- Extracting valuable memories from conversational data
- Advanced cognitive operations that combine multiple Memories
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional
import json
import os
from ..models.memory import KLineAnswerRequest
from ..dependencies import get_memory_agent
from ..core.utils import validate_vectorstore_name
from ..core.exceptions import server_error, validation_error

router = APIRouter(prefix="/api/memory", tags=["klines"])


def check_early_cache_for_question(question: str, vectorstore_name: str) -> Optional[Dict[str, Any]]:
    """
    Check if a user question has been cached previously using LangCache API.

    This function implements early caching by:
    1. Extracting just the user's question from the request
    2. Checking LangCache for a previously cached response
    3. Returning the cached response immediately if found

    Args:
        question: The user's question text
        vectorstore_name: Name of the vectorstore (used as context)

    Returns:
        Cached response dict if found, None otherwise
    """
    try:
        # Check if LangCache is configured
        if not all([os.getenv("LANGCACHE_HOST"), os.getenv("LANGCACHE_API_KEY"), os.getenv("LANGCACHE_CACHE_ID")]):
            print("ℹ️ LangCache not configured for early caching")
            return None

        # Initialize LangCache client
        from langcache_client import LangCacheClient
        langcache_client = LangCacheClient()

        # Create a unique cache key for early caching that won't conflict with existing LLM caching
        # Use a very specific format that's different from internal LLM prompts
        cache_key = f"EARLY_ASK_ENDPOINT_CACHE_V1:{vectorstore_name}:{question}"
        messages = [{"role": "user", "content": cache_key}]

        # Search for cached response
        print(f"🔍 EARLY CACHE: Checking cache for question: {question[:60]}{'...' if len(question) > 60 else ''}")
        print(f"🔍 EARLY CACHE: Using cache key: '{cache_key}'")
        cached_response = langcache_client.search_cache(
            messages=messages
        )

        if cached_response:
            print(f"✅ EARLY CACHE HIT: Found cached response for question")
            print(f"✅ EARLY CACHE: Similarity score: {cached_response.get('_similarity_score', 'Unknown')}")
            # Parse the cached response back to the expected format
            try:
                cached_data = json.loads(cached_response['content'])
                # Add cache metadata
                cached_data['_cache_hit'] = True
                cached_data['_cached_at'] = cached_response.get('_cached_at')
                cached_data['_similarity_score'] = cached_response.get('_similarity_score')
                print(f"✅ EARLY CACHE: Returning cached answer: {cached_data.get('answer', 'No answer')[:50]}...")
                return cached_data
            except json.JSONDecodeError:
                print("⚠️ EARLY CACHE: Failed to parse cached response as JSON")
                return None
        else:
            print("❌ EARLY CACHE MISS: No cached response found")
            return None

    except Exception as e:
        print(f"⚠️ EARLY CACHE ERROR: {e}")
        return None


def store_early_cache_for_question(question: str, vectorstore_name: str, response_data: Dict[str, Any]) -> bool:
    """
    Store a question response in the early cache using LangCache API.

    Args:
        question: The user's question text
        vectorstore_name: Name of the vectorstore (used as context)
        response_data: The response data to cache

    Returns:
        True if stored successfully, False otherwise
    """
    try:
        # Check if LangCache is configured
        if not all([os.getenv("LANGCACHE_HOST"), os.getenv("LANGCACHE_API_KEY"), os.getenv("LANGCACHE_CACHE_ID")]):
            return False

        # Initialize LangCache client
        from langcache_client import LangCacheClient
        langcache_client = LangCacheClient()

        # Create the same unique cache key format used in the check function
        cache_key = f"EARLY_ASK_ENDPOINT_CACHE_V1:{vectorstore_name}:{question}"
        messages = [{"role": "user", "content": cache_key}]

        # Remove cache metadata from response before storing
        clean_response = {k: v for k, v in response_data.items()
                         if not k.startswith('_cache')}

        # Store as JSON string
        response_json = json.dumps(clean_response)

        print(f"💾 EARLY CACHE: Storing response for question: {question[:60]}{'...' if len(question) > 60 else ''}")
        success = langcache_client.store_cache(
            messages=messages,
            response=response_json
        )

        if success:
            print("✅ EARLY CACHE: Successfully stored response")
        else:
            print("❌ EARLY CACHE: Failed to store response")

        return success

    except Exception as e:
        print(f"⚠️ EARLY CACHE STORE ERROR: {e}")
        return False


@router.post('/{vectorstore_name}/ask')
async def answer_question_with_klines(
    vectorstore_name: str,
    request: KLineAnswerRequest,
    memory_agent=Depends(get_memory_agent)
) -> Dict[str, Any]:
    """Answer a question using K-line construction and reasoning.

    Args:
        vectorstore_name: Name of the vectorstore to search in
        request: Question and search parameters

    This operation constructs a mental state from relevant Memories and applies
    sophisticated reasoning to answer questions with confidence scoring.
    It represents the full cognitive process of memory recall + reasoning.

    K-lines are constructed but NOT stored - they exist only as temporary mental states.

    Returns:
        JSON with structured response including answer, confidence, reasoning, supporting memories,
        and the constructed mental state (K-line)
    """
    try:
        validate_vectorstore_name(vectorstore_name)

        question = request.question.strip()
        top_k = request.top_k
        filter_expr = request.filter
        min_similarity = request.min_similarity

        if not question:
            raise validation_error('Question is required')

        # EARLY CACHING: Check if this exact question has been cached before
        # This avoids redundant memory searches and LLM calls for frequently asked questions
        print(f"🔍 ENDPOINT: About to check early cache for question: '{question}'")
        cached_response = check_early_cache_for_question(question, vectorstore_name)
        if cached_response:
            print(f"🚀 ENDPOINT: EARLY CACHE HIT! Returning cached response for question")
            return cached_response
        else:
            print(f"❌ ENDPOINT: Early cache miss, proceeding with normal processing")

        print(f"🤔 K-LINE API: Answering question via mental state construction: {question} (top_k: {top_k})")
        print(f"📦 Vectorstore: {vectorstore_name}")
        if filter_expr:
            print(f"🔍 Filter: {filter_expr}")

        # Use the memory agent's sophisticated answer_question method
        # This constructs a K-line (mental state) and applies reasoning
        answer_response = memory_agent.memory_agent.answer_question(
            question, 
            top_k=top_k, 
            filterBy=filter_expr, 
            min_similarity=min_similarity, 
            vectorset_key=vectorstore_name
        )

        # Construct K-line (mental state) from the supporting memories
        supporting_memories = answer_response.get('supporting_memories', [])
        if supporting_memories:
            kline_result = memory_agent.memory_agent.construct_kline(
                query=question,
                memories=supporting_memories,
                answer=answer_response.get('answer'),
                confidence=answer_response.get('confidence'),
                reasoning=answer_response.get('reasoning')
            )
            print(f"🧠 Constructed K-line with coherence score: {kline_result.get('coherence_score', 0):.3f}")
        else:
            kline_result = {
                'mental_state': 'No relevant memories found to construct mental state.',
                'coherence_score': 0.0,
                'summary': 'Empty mental state'
            }

        # Prepare the response
        response_data = {
            'success': True,
            'question': question,
            'vectorstore_name': vectorstore_name,
            **answer_response,  # Spread the structured response (answer, confidence, supporting_memories, etc.)
            'kline': kline_result  # Include the constructed mental state
        }

        # EARLY CACHING: Store the response for future use
        # This caches the complete response to avoid redundant processing
        print(f"💾 ENDPOINT: About to store response in early cache for question: '{question}'")
        cache_stored = store_early_cache_for_question(question, vectorstore_name, response_data)
        print(f"💾 ENDPOINT: Cache storage result: {cache_stored}")

        return response_data

    except Exception as e:
        raise server_error(str(e))

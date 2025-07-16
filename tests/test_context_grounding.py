#!/usr/bin/env python3
"""
Test script for the contextual grounding system in memory_agent.py

This script demonstrates how the memory agent can resolve context-dependent
references like temporal, spatial, and personal references.
"""

import os
import sys
from datetime import datetime
from memory.core_agent import MemoryAgent

def test_contextual_grounding():
    """Test the contextual grounding functionality."""
    
    print("🧪 Testing Contextual Grounding System")
    print("=" * 50)
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found. Please set it in your environment.")
        print("This test requires OpenAI API access for contextual grounding.")
        return
    
    try:
        # Initialize the agent (this will try to connect to Redis)
        agent = MemoryAgent()
        
        print("\n🌍 Setting up test context...")
        # Set context for Jakarta scenario
        agent.set_context(
            location="Jakarta, Indonesia",
            activity="just arrived from flight",
            people_present=["travel companion"],
            weather="very hot and humid",
            temperature="32°C"
        )
        
        print("\n📝 Testing contextual memory storage...")
        
        # Test cases with different types of context dependencies
        test_memories = [
            "Darn its really hot outside",  # Spatial + environmental
            "I just arrived here and I'm exhausted",  # Temporal + spatial
            "This place is so crowded",  # Spatial + demonstrative
            "The weather today is unbearable",  # Temporal + environmental
            "My travel companion is complaining about the heat",  # Personal + environmental
            "We need to find somewhere cool to rest",  # Personal + spatial
            "This is nothing like home"  # Demonstrative + spatial comparison
        ]
        
        stored_memories = []
        
        for i, memory in enumerate(test_memories, 1):
            print(f"\n--- Test Memory {i} ---")
            memory_id = agent.store_memory(memory, apply_grounding=True)
            stored_memories.append(memory_id)
            print()
        
        print("\n🔍 Testing memory retrieval...")
        
        # Test queries that should find the grounded memories
        test_queries = [
            "What's the weather like?",
            "How do you feel about Jakarta?",
            "Tell me about your arrival",
            "What did you think of the temperature?"
        ]
        
        for query in test_queries:
            print(f"\n--- Query: '{query}' ---")
            memories = agent.search_memories(query, top_k=10)
            
            if memories:
                for j, memory in enumerate(memories, 1):
                    print(f"{j}. {memory['text']} (Score: {memory['score']:.3f})")
                    if memory.get('grounding_applied'):
                        print(f"   📍 Grounded from: {memory.get('original_text', 'N/A')}")
                        grounding_info = memory.get('grounding_info', {})
                        if grounding_info.get('changes_made'):
                            print(f"   🔄 Changes: {len(grounding_info['changes_made'])} context references resolved")
            else:
                print("   No memories found")
        
        print("\n🤖 Testing question answering...")
        
        # Test question answering with grounded memories
        answer_response = agent.answer_question("What was it like when you arrived in Jakarta?")
        
        print(f"\nAnswer: {answer_response['answer']}")
        print(f"Confidence: {answer_response['confidence']}")
        if answer_response.get('supporting_memories'):
            print(f"Supporting memories: {len(answer_response['supporting_memories'])}")
            for mem in answer_response['supporting_memories']:
                if mem.get('grounding_applied'):
                    print(f"  • Grounded memory: {mem['text']}")
        
        print("\n✅ Contextual grounding test completed!")
        print("\nKey benefits demonstrated:")
        print("• Temporal references ('today', 'just arrived') → specific dates/times")
        print("• Spatial references ('here', 'outside', 'this place') → Jakarta, Indonesia")
        print("• Environmental references ('the weather') → specific conditions")
        print("• Personal references ('my travel companion') → preserved with context")
        
        return stored_memories
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("\nNote: This test requires:")
        print("1. Redis server running on localhost:6379 with RedisSearch")
        print("2. OpenAI API key in environment")
        return None

def test_without_grounding():
    """Test storing the same memories without grounding for comparison."""
    
    print("\n\n🔄 Testing WITHOUT contextual grounding for comparison...")
    print("=" * 60)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found.")
        return
    
    try:
        agent = MemoryAgent()
        
        # Set the same context
        agent.set_context(
            location="Jakarta, Indonesia",
            activity="just arrived from flight",
            people_present=["travel companion"],
            weather="very hot and humid"
        )
        
        # Store the same memories without grounding
        test_memories = [
            "Darn its really hot outside",
            "I just arrived here and I'm exhausted",
            "This place is so crowded"
        ]
        
        print("\nStoring memories WITHOUT grounding:")
        for memory in test_memories:
            print(f"\nStoring: {memory}")
            agent.store_memory(memory, apply_grounding=False)
        
        print("\n🔍 Searching memories stored without grounding...")
        memories = agent.search_memories("What's the weather like in Jakarta?", top_k=10)
        
        print("Results from non-grounded memories:")
        for i, memory in enumerate(memories, 1):
            print(f"{i}. {memory['text']} (Score: {memory['score']:.3f})")
        
        print("\n💡 Notice: Without grounding, 'outside' doesn't connect to 'Jakarta'")
        print("The spatial context is lost, making retrieval less effective.")
        
    except Exception as e:
        print(f"❌ Comparison test failed: {e}")

if __name__ == "__main__":
    print("This script tests the contextual grounding system.")
    print("Make sure you have Redis running and OpenAI API key set.")
    print()
    
    # Run the main test
    stored_ids = test_contextual_grounding()
    
    # Run comparison test
    test_without_grounding()
    
    print("\n" + "=" * 60)
    print("🎯 Test Summary:")
    print("The contextual grounding system resolves ambiguous references")
    print("like 'here', 'today', 'outside' to specific contexts, making")
    print("memories more useful and searchable over time.")

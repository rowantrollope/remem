#!/usr/bin/env python3
"""
Demo script showing the contextual grounding system in action.

This demonstrates how memories with context-dependent references
are automatically grounded to absolute references.
"""

import os
from memory_agent import MemoryAgent

def demo_contextual_grounding():
    """Interactive demo of contextual grounding."""
    
    print("🌍 Contextual Memory Grounding Demo")
    print("=" * 40)
    print()
    print("This demo shows how the memory agent automatically converts")
    print("context-dependent references to absolute references.")
    print()
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    try:
        # Initialize agent
        agent = MemoryAgent()
        
        # Scenario 1: Travel to Jakarta
        print("📍 SCENARIO 1: You've just arrived in Jakarta")
        print("-" * 45)
        
        agent.set_context(
            location="Jakarta, Indonesia", 
            activity="just arrived from international flight",
            weather="hot and humid"
        )
        
        print("Setting context:")
        print("  Location: Jakarta, Indonesia")
        print("  Activity: just arrived from international flight") 
        print("  Weather: hot and humid")
        print()
        
        # Store some memories with context-dependent references
        memories_jakarta = [
            "It's incredibly hot outside",
            "I just got here and I'm already sweating",
            "This weather is nothing like back home",
            "The humidity here is overwhelming"
        ]
        
        print("Storing memories (watch the grounding in action):")
        for memory in memories_jakarta:
            print(f"\n💭 Original: '{memory}'")
            agent.store_memory(memory)
        
        print("\n" + "="*50)
        
        # Scenario 2: Later, at the office
        print("\n📍 SCENARIO 2: Next day at the office")
        print("-" * 40)
        
        agent.set_context(
            location="Office building, Jakarta CBD",
            activity="working at new job",
            people_present=["Sarah (colleague)", "Mike (manager)"],
            environment="air conditioned"
        )
        
        print("Context changed:")
        print("  Location: Office building, Jakarta CBD")
        print("  Activity: working at new job")
        print("  People: Sarah (colleague), Mike (manager)")
        print("  Environment: air conditioned")
        print()
        
        memories_office = [
            "Sarah showed me around the office today",
            "Mike seems like a good manager",
            "It's much cooler in here than outside",
            "This place has great air conditioning"
        ]
        
        print("Storing more memories:")
        for memory in memories_office:
            print(f"\n💭 Original: '{memory}'")
            agent.store_memory(memory)
        
        print("\n" + "="*50)
        
        # Test retrieval
        print("\n🔍 TESTING MEMORY RETRIEVAL")
        print("-" * 30)
        
        queries = [
            "What was the weather like in Jakarta?",
            "Tell me about your first day",
            "Who did you meet at the office?",
            "How was the temperature?"
        ]
        
        for query in queries:
            print(f"\n❓ Query: '{query}'")
            memories = agent.search_memories(query, top_k=2)
            
            for i, memory in enumerate(memories, 1):
                print(f"  {i}. {memory['text']} ({memory['score']*100:.1f}% relevant)")
                
                # Show grounding info if available
                if memory.get('grounding_applied'):
                    print(f"     📍 Grounded from: '{memory.get('original_text', 'N/A')}'")
        
        print("\n" + "="*50)
        
        # Test question answering
        print("\n🤖 TESTING QUESTION ANSWERING")
        print("-" * 35)
        
        question = "What was your experience arriving in Jakarta?"
        print(f"\n❓ Question: '{question}'")
        
        response = agent.answer_question(question)
        print(f"\n🤖 Answer: {response['answer']}")
        print(f"🎯 Confidence: {response['confidence']}")
        
        if response.get('supporting_memories'):
            print(f"\n📚 Based on {len(response['supporting_memories'])} memories:")
            for i, mem in enumerate(response['supporting_memories'], 1):
                print(f"  {i}. {mem['text']} ({mem['relevance_score']}% relevant)")
        
        print("\n" + "="*50)
        print("\n✅ Demo completed!")
        print("\n💡 Key Benefits of Contextual Grounding:")
        print("  • 'here' → specific locations (Jakarta, office)")
        print("  • 'today' → specific dates")
        print("  • 'outside' → weather at specific location")
        print("  • 'this place' → specific buildings/areas")
        print("  • Personal references preserved with context")
        print("\n🎯 Result: Memories remain useful and searchable over time!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("\nRequirements:")
        print("  • Redis server with RedisSearch on localhost:6381")
        print("  • OPENAI_API_KEY environment variable")

if __name__ == "__main__":
    demo_contextual_grounding()

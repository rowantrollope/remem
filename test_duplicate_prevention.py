#!/usr/bin/env python3
"""
Test script to verify duplicate prevention is working correctly.
"""

from langgraph_memory_agent import LangGraphMemoryAgent

def test_duplicate_prevention():
    """Test that duplicate memories are properly prevented."""
    
    print("🧪 Testing Duplicate Prevention")
    print("=" * 50)
    
    # Initialize the agent
    agent = LangGraphMemoryAgent()
    
    # Set travel context for consistent extraction
    agent.set_extraction_context("I am a travel agent app. Extract user preferences, constraints, and personal details.")
    
    print("\n1. First mention of Paris trip:")
    response1 = agent.run("I want to plan a trip to Paris")
    print(f"Response: {response1}")
    
    print("\n2. Second mention of Paris trip (should detect duplicate):")
    response2 = agent.run("I want to plan a trip to Paris")
    print(f"Response: {response2}")
    
    print("\n3. Slight variation (should detect duplicate):")
    response3 = agent.run("I'm planning a trip to Paris")
    print(f"Response: {response3}")
    
    print("\n4. Different information (should NOT be duplicate):")
    response4 = agent.run("I want to plan a trip to Paris with my family of 4 in June")
    print(f"Response: {response4}")
    
    print("\n5. Check for duplicates in the system:")
    duplicates = agent.find_duplicate_memories(similarity_threshold=0.85)
    print(f"Duplicate analysis: {duplicates}")
    
    print("\n6. Get user profile summary:")
    profile = agent.get_user_profile_summary()
    print(f"Profile:\n{profile}")

if __name__ == "__main__":
    test_duplicate_prevention()

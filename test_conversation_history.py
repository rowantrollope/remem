#!/usr/bin/env python3
"""
Test script to verify conversation history is working in LangGraph Memory Agent.
"""

from langgraph_memory_agent import LangGraphMemoryAgent

def test_conversation_history():
    """Test that conversation history is maintained across multiple exchanges."""
    
    print("🧪 Testing Conversation History in LangGraph Memory Agent")
    print("=" * 60)
    
    # Initialize the agent
    agent = LangGraphMemoryAgent()
    
    print("\n1. Starting fresh conversation:")
    agent.show_conversation_history()
    
    print("\n2. First exchange - introducing myself:")
    response1 = agent.run("Hi, my name is John and I'm planning a trip to Paris")
    print(f"Agent: {response1}")
    agent.show_conversation_history()
    
    print("\n3. Second exchange - asking follow-up (should remember my name):")
    response2 = agent.run("What do you think about my Paris trip idea?")
    print(f"Agent: {response2}")
    agent.show_conversation_history()
    
    print("\n4. Third exchange - testing memory of previous context:")
    response3 = agent.run("Can you remind me what we were just discussing?")
    print(f"Agent: {response3}")
    agent.show_conversation_history()
    
    print("\n5. Fourth exchange - testing if it remembers my name:")
    response4 = agent.run("What's my name again?")
    print(f"Agent: {response4}")
    agent.show_conversation_history()
    
    print("\n6. Testing conversation history clearing:")
    agent.clear_conversation_history()
    agent.show_conversation_history()
    
    print("\n7. After clearing - should not remember previous context:")
    response5 = agent.run("What's my name?")
    print(f"Agent: {response5}")
    agent.show_conversation_history()

def test_long_conversation():
    """Test conversation history with many exchanges."""
    
    print("\n" + "=" * 60)
    print("🧪 Testing Long Conversation History Management")
    print("=" * 60)
    
    agent = LangGraphMemoryAgent()
    
    # Have a longer conversation to test history trimming
    topics = [
        "Hi, I'm Sarah and I love traveling",
        "I've been to Italy, France, and Spain",
        "My favorite food is pasta",
        "I work as a software engineer",
        "I have two cats named Whiskers and Mittens",
        "I live in San Francisco",
        "I'm planning a trip to Japan next year",
        "I prefer window seats on flights",
        "I'm vegetarian",
        "I love hiking and photography",
        "What do you remember about me?",
        "Can you help me plan my Japan trip based on what you know about me?"
    ]
    
    for i, topic in enumerate(topics, 1):
        print(f"\n{i}. User: {topic}")
        response = agent.run(topic)
        print(f"   Agent: {response[:100]}{'...' if len(response) > 100 else ''}")
        
        # Show history length
        print(f"   📊 History length: {len(agent.conversation_history)} messages")
    
    print(f"\n📋 Final conversation history:")
    agent.show_conversation_history()

if __name__ == "__main__":
    try:
        test_conversation_history()
        test_long_conversation()
        print("\n🎉 Conversation history tests completed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

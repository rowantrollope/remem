#!/usr/bin/env python3
"""
Test script for the new extract_and_store_memories() API

This script demonstrates how to use the intelligent memory extraction functionality
for different use cases like travel agents, customer service, and personal assistants.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the MemoryAgent
from memory.core_agent import MemoryAgent

def test_travel_agent_extraction():
    """Test memory extraction for a travel agent scenario."""
    print("🧳 Testing Travel Agent Memory Extraction")
    print("=" * 50)
    
    # Initialize the memory agent
    agent = MemoryAgent()
    
    # Set context for travel planning
    agent.set_context(
        location="Travel Agency Office",
        activity="Trip Planning Consultation",
        people_present=["Travel Agent", "Customer"]
    )
    
    # Sample travel conversation
    travel_conversation = """
    Customer: Hi, I'm planning a family trip to Paris for next month. We're a family of four - me, my wife, and our two kids aged 8 and 12.
    
    Agent: That sounds wonderful! Tell me about your preferences.
    
    Customer: Well, I always prefer window seats when flying, and my wife is vegetarian so we'll need vegetarian meal options. Our budget is around $8000 for the whole trip. The kids love museums, especially anything with dinosaurs or science. 
    
    Agent: Great! Any other preferences?
    
    Customer: Oh yes, I have a bad back so I need aisle access, and we prefer staying in hotels with pools since the kids love swimming. Also, we're not into fancy restaurants - we prefer casual dining. My wife has a severe nut allergy, so that's really important for restaurants.
    
    Agent: Perfect, I'll keep all of that in mind. When are you thinking of traveling?
    
    Customer: We're flexible, but sometime in March would be ideal. We definitely want to avoid school holidays to save money.
    """
    
    # Context prompt for travel agent
    travel_context = """I am a travel agent application. Please extract user preferences, travel constraints, family information, and personal details that would be valuable for planning future trips and providing personalized recommendations."""
    
    # Extraction examples for travel domain
    travel_examples = [
        {
            "input": "I prefer window seats and my budget is $5000",
            "extract": "User prefers window seats when flying",
            "reason": "Specific seating preference useful for future flight bookings"
        },
        {
            "input": "We had a great time at the beach resort",
            "extract": "User enjoys beach resort vacations",
            "reason": "Preference information for future trip recommendations"
        }
    ]
    
    # Extract and store memories
    result = agent.extract_and_store_memories(
        raw_input=travel_conversation,
        context_prompt=travel_context,
        extraction_examples=travel_examples
    )
    
    # Display results
    print(f"\n✅ Extraction Summary: {result['extraction_summary']}")
    print(f"📊 Total extracted: {result['total_extracted']}, Total filtered: {result['total_filtered']}")
    
    if result['extracted_memories']:
        print(f"\n📝 Extracted Travel Memories:")
        for i, memory in enumerate(result['extracted_memories'], 1):
            print(f"   {i}. [{memory['category']}] {memory['extracted_text']}")
            print(f"      Confidence: {memory['confidence']} | ID: {memory['memory_id']}")
            if memory['reasoning']:
                print(f"      Reasoning: {memory['reasoning']}")
            print()
    

    
    print(f"\n💭 Extraction Reasoning: {result['extraction_reasoning']}")
    
    return result

def test_customer_service_extraction():
    """Test memory extraction for customer service scenario."""
    print("\n📞 Testing Customer Service Memory Extraction")
    print("=" * 50)
    
    agent = MemoryAgent()
    
    # Set context for customer service
    agent.set_context(
        location="Customer Service Center",
        activity="Customer Support Call",
        people_present=["Support Agent", "Customer"]
    )
    
    # Sample customer service conversation
    service_conversation = """
    Agent: Hello, this is Sarah from TechCorp support. How can I help you today?
    
    Customer: Hi, I'm having issues with my laptop. It's a TechCorp Pro 15 that I bought last year. The screen keeps flickering and sometimes goes completely black.
    
    Agent: I'm sorry to hear that. Can you tell me your account details?
    
    Customer: Sure, my name is John Smith, account number is TC-789456. I'm calling from my office in Seattle. This laptop is critical for my work as a graphic designer.
    
    Agent: I see your account here. You purchased the laptop in March 2023, and it's still under warranty.
    
    Customer: That's good to hear. This flickering started about two weeks ago, and it's getting worse. I've tried restarting multiple times. I really need this fixed quickly because I have client deadlines coming up.
    
    Agent: I understand the urgency. Based on your description, this sounds like a display hardware issue. I'd like to schedule a technician visit or you can bring it to our Seattle service center.
    
    Customer: I prefer the service center option. Also, just so you know, I've been a customer for over 5 years and this is my third TechCorp laptop. I really love the brand but this issue is concerning.
    """
    
    # Context prompt for customer service
    service_context = """I am a customer service application. Please extract customer information, product details, issue descriptions, customer preferences, and any important context that would help provide better support in future interactions."""
    
    # Extract memories
    result = agent.extract_and_store_memories(
        raw_input=service_conversation,
        context_prompt=service_context
    )
    
    # Display results
    print(f"\n✅ Extraction Summary: {result['extraction_summary']}")
    
    if result['extracted_memories']:
        print(f"\n📝 Extracted Customer Service Memories:")
        for i, memory in enumerate(result['extracted_memories'], 1):
            print(f"   {i}. [{memory['category']}] {memory['extracted_text']}")
            print(f"      Confidence: {memory['confidence']}")
    
    return result

def test_memory_retrieval():
    """Test retrieving the extracted memories."""
    print("\n🔍 Testing Memory Retrieval")
    print("=" * 30)
    
    agent = MemoryAgent()
    
    # Test various queries
    queries = [
        "What are the user's seating preferences?",
        "Tell me about dietary restrictions",
        "What is the customer's profession?",
        "What product issues have been reported?"
    ]
    
    for query in queries:
        print(f"\n❓ Query: {query}")
        answer = agent.answer_question(query)
        print(f"🤖 Answer: {answer['answer']}")
        print(f"🎯 Confidence: {answer['confidence']}")

def main():
    """Run all tests."""
    print("🧠 Testing Extract and Store Memories API")
    print("=" * 60)
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment variables.")
        print("Please set your OpenAI API key in the .env file or environment.")
        sys.exit(1)
    
    try:
        # Test travel agent extraction
        travel_result = test_travel_agent_extraction()
        
        # Test customer service extraction
        service_result = test_customer_service_extraction()
        
        # Test memory retrieval
        test_memory_retrieval()
        
        print("\n🎉 All tests completed successfully!")
        
        # Show final statistics
        agent = MemoryAgent()
        memory_info = agent.get_memory_info()

        print(f"\n📊 Final Statistics:")
        print(f"   Long-term memories: {memory_info['memory_count']}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

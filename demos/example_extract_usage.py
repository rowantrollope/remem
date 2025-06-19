#!/usr/bin/env python3
"""
Example usage of the extract_and_store_memories() API

This demonstrates how to use the intelligent memory extraction functionality
for different application scenarios.
"""

# Example 1: Travel Agent Memory Extraction
def travel_agent_example():
    """Example of using extract_and_store_memories for a travel agent."""
    
    # Sample conversation from travel planning session
    conversation = """
    Agent: Welcome! How can I help you plan your trip today?
    
    Customer: Hi, we're planning a family vacation to Europe. It's me, my wife Sarah, 
    and our two kids - Emma who's 10 and Jake who's 8.
    
    Agent: That sounds wonderful! Tell me about your preferences.
    
    Customer: Well, I always prefer window seats when we fly - I love looking out at 
    the clouds. Sarah is vegetarian, so we'll need vegetarian meal options on the 
    flight and at restaurants. Our budget is around $12,000 for the whole family.
    
    Agent: Great! Any other preferences for the trip?
    
    Customer: The kids love interactive museums, especially science museums. Emma is 
    really into astronomy right now. We prefer staying in family-friendly hotels 
    with pools since the kids love swimming. Oh, and Sarah has a severe nut allergy, 
    so that's really important for restaurant recommendations.
    
    Agent: Perfect! When are you thinking of traveling?
    
    Customer: We're flexible, but sometime in July would be ideal. We want to avoid 
    the peak tourist season if possible to save money. Also, I should mention I have 
    mobility issues, so we'll need accessible accommodations and transportation.
    """
    
    # Context prompt for travel agent application
    context_prompt = """I am a travel agent application. Please extract user preferences, 
    travel constraints, family information, dietary restrictions, accessibility needs, 
    and personal details that would be valuable for planning future trips and providing 
    personalized recommendations."""
    
    # Optional examples to guide extraction
    extraction_examples = [
        {
            "input": "I prefer window seats and my budget is $5000",
            "extract": "User prefers window seats when flying",
            "reason": "Specific seating preference useful for future flight bookings"
        },
        {
            "input": "My wife is vegetarian and has nut allergies",
            "extract": "Wife is vegetarian with severe nut allergy",
            "reason": "Critical dietary information for meal planning and restaurant selection"
        }
    ]
    
    # API call example
    api_call_example = f"""
    from memory_agent import MemoryAgent
    
    agent = MemoryAgent()
    
    # Set context for better grounding
    agent.set_context(
        location="Travel Agency Office",
        activity="Trip Planning Consultation",
        people_present=["Travel Agent", "Customer Family"]
    )
    
    # Extract and store memories
    result = agent.extract_and_store_memories(
        raw_input='''{conversation}''',
        context_prompt='''{context_prompt}''',
        store_raw=True,  # Store full conversation for audit trail
        extraction_examples={extraction_examples}
    )
    
    # Results would include extracted memories like:
    # - "User prefers window seats when flying" (preference, high confidence)
    # - "Wife Sarah is vegetarian with severe nut allergy" (personal, high confidence)  
    # - "Family budget is around $12,000" (constraint, high confidence)
    # - "Kids love interactive science museums" (preference, medium confidence)
    # - "User has mobility issues requiring accessible accommodations" (constraint, high confidence)
    """
    
    return {
        "conversation": conversation,
        "context_prompt": context_prompt,
        "extraction_examples": extraction_examples,
        "api_call": api_call_example
    }

# Example 2: Customer Service Memory Extraction  
def customer_service_example():
    """Example of using extract_and_store_memories for customer service."""
    
    conversation = """
    Agent: Hello, this is TechSupport. How can I help you today?
    
    Customer: Hi, I'm having issues with my laptop. It's a TechCorp Pro 15 that I 
    bought about 8 months ago. The screen keeps flickering and sometimes goes black.
    
    Agent: I'm sorry to hear that. Can you provide your account information?
    
    Customer: Sure, I'm Dr. Jennifer Martinez, account number TC-445789. I'm calling 
    from my clinic in Portland. This laptop is critical for my medical practice - 
    I use it for patient records and telemedicine consultations.
    
    Agent: I see your account. You purchased it in March 2023, correct?
    
    Customer: Yes, that's right. The flickering started about two weeks ago and it's 
    getting worse. I've tried restarting multiple times. I really need this fixed 
    urgently because I have patient appointments this week.
    
    Agent: I understand the urgency. Have you experienced any other issues?
    
    Customer: Not really, though I should mention I've been very happy with TechCorp 
    products overall. This is actually my third TechCorp laptop over the years. 
    I specifically chose the Pro 15 because I need the larger screen for medical imaging.
    """
    
    context_prompt = """I am a customer service application. Please extract customer 
    information, product details, issue descriptions, usage context, customer history, 
    and preferences that would help provide better support in future interactions."""
    
    api_call_example = f"""
    result = agent.extract_and_store_memories(
        raw_input='''{conversation}''',
        context_prompt='''{context_prompt}''',
        store_raw=True
    )
    
    # Expected extractions:
    # - "Customer Dr. Jennifer Martinez, account TC-445789" (personal, high confidence)
    # - "Customer is a medical professional using laptop for patient records" (factual, high confidence)
    # - "TechCorp Pro 15 laptop experiencing screen flickering and blackouts" (factual, high confidence)
    # - "Customer has owned three TechCorp laptops, loyal customer" (personal, medium confidence)
    # - "Customer chose Pro 15 specifically for larger screen for medical imaging" (preference, high confidence)
    """
    
    return {
        "conversation": conversation,
        "context_prompt": context_prompt,
        "api_call": api_call_example
    }

# Example 3: Personal Assistant Memory Extraction
def personal_assistant_example():
    """Example for personal assistant application."""
    
    conversation = """
    User: Hey assistant, I need to plan my week. I have a dentist appointment on 
    Wednesday at 2 PM with Dr. Smith on Main Street. 
    
    Assistant: Got it! Anything else for this week?
    
    User: Yes, I need to remember to call my mom on Friday - it's her birthday and 
    I always forget. She loves flowers, especially roses. Oh, and I'm trying to eat 
    healthier, so remind me to meal prep on Sunday. I'm avoiding gluten and dairy 
    right now.
    
    Assistant: I'll help you remember all of that. Any other preferences?
    
    User: I prefer morning workouts, usually around 7 AM. And I drive a blue Honda 
    Civic, license plate ABC-123, in case that's ever relevant for parking or anything.
    """
    
    context_prompt = """I am a personal assistant application. Please extract 
    appointments, personal preferences, dietary restrictions, family information, 
    and any details that would help me provide better personal assistance."""
    
    return {
        "conversation": conversation,
        "context_prompt": context_prompt
    }

def main():
    """Display all examples."""
    print("🧠 Extract and Store Memories API Examples")
    print("=" * 60)
    
    print("\n🧳 Example 1: Travel Agent Application")
    print("-" * 40)
    travel_example = travel_agent_example()
    print("Conversation snippet:")
    print(travel_example["conversation"][:200] + "...")
    print(f"\nContext prompt: {travel_example['context_prompt']}")
    print("\nAPI Usage:")
    print(travel_example["api_call"])
    
    print("\n📞 Example 2: Customer Service Application") 
    print("-" * 40)
    service_example = customer_service_example()
    print("Conversation snippet:")
    print(service_example["conversation"][:200] + "...")
    print(f"\nContext prompt: {service_example['context_prompt']}")
    
    print("\n🤖 Example 3: Personal Assistant Application")
    print("-" * 40)
    assistant_example = personal_assistant_example()
    print("Conversation snippet:")
    print(assistant_example["conversation"][:200] + "...")
    print(f"\nContext prompt: {assistant_example['context_prompt']}")
    
    print("\n" + "=" * 60)
    print("💡 Key Benefits:")
    print("• Intelligent filtering - only valuable information is stored")
    print("• Contextual grounding - resolves temporal/spatial references")
    print("• Confidence scoring - indicates reliability of extracted facts")
    print("• Category classification - organizes memories by type")
    print("• Working memory - optional storage of complete conversations")
    print("• Seamless integration - works with existing memory search/retrieval")

if __name__ == "__main__":
    main()

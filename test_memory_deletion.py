#!/usr/bin/env python3
"""
Test script to verify memory deletion functionality in the LangGraph agent.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_memory_deletion():
    """Test memory deletion functionality."""
    try:
        from memory.agent import LangGraphMemoryAgent
        from llm.llm_manager import init_llm_manager, LLMConfig

        # Initialize LLM manager first
        print("🔧 Initializing LLM Manager...")
        tier1_config = LLMConfig(
            provider="openai",
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=2000,
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=30
        )

        tier2_config = LLMConfig(
            provider="openai",
            model="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=1000,
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=30
        )

        init_llm_manager(tier1_config, tier2_config)
        print("✅ LLM Manager initialized")

        print("🔧 Initializing LangGraph Memory Agent...")
        agent = LangGraphMemoryAgent(vectorset_key="test_deletion")
        
        # First, store some test memories
        print("\n📝 Storing test memories...")
        test_memories = [
            "I love pizza",
            "My favorite color is blue", 
            "I went to Paris last year"
        ]
        
        for memory in test_memories:
            response = agent.run(f"Remember that {memory}")
            print(f"Stored: {memory}")
        
        # Check how many memories we have
        print("\n📊 Checking memory count...")
        stats_response = agent.run("How many memories do you have?")
        print(f"Stats response: {stats_response}")
        
        # Now test deletion
        print("\n🗑️ Testing memory deletion...")
        deletion_response = agent.run("Please remove all memories of me")
        print(f"Deletion response: {deletion_response}")
        
        # Check memory count again
        print("\n📊 Checking memory count after deletion...")
        final_stats = agent.run("How many memories do you have?")
        print(f"Final stats: {final_stats}")
        
        print("\n✅ Test completed!")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_memory_deletion()

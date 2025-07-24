#!/usr/bin/env python3
"""
Quick test script to verify async processor initialization
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_processor_init():
    """Test async processor initialization."""
    print("🧪 Testing Async Processor Initialization")
    print("=" * 50)
    
    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment variables.")
        return False
    
    try:
        from memory.async_processor import AsyncMemoryProcessor
        
        print("📦 Creating AsyncMemoryProcessor...")
        processor = AsyncMemoryProcessor(processing_interval=60, retention_days=30)
        
        print("✅ Processor created successfully!")
        
        # Test LLM manager
        from llm.llm_manager import get_llm_manager
        llm_manager = get_llm_manager()
        print("✅ LLM manager accessible!")
        
        # Test a simple processing cycle
        print("🔄 Testing single processing cycle...")
        result = processor.run_once()
        print(f"✅ Processing cycle completed: {result['processing_result']['message']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_processor_init()
    if success:
        print("\n🎉 Async processor initialization test PASSED!")
    else:
        print("\n💥 Async processor initialization test FAILED!")

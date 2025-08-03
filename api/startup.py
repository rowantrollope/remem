"""
Application startup logic and initialization.
"""

import os
from memory.agent import MemoryAgentChat
from memory.core_agent import MemoryAgent
from llm.llm_manager import LLMManager, LLMConfig, init_llm_manager as initialize_llm_manager
from .core.config import app_config
from .dependencies import set_memory_agent


def init_llm_manager():
    """Initialize the LLM manager with current configuration."""
    try:
        # Create LLM configurations for both tiers
        tier1_config = LLMConfig(
            provider=app_config["llm"]["tier1"]["provider"],
            model=app_config["llm"]["tier1"]["model"],
            temperature=app_config["llm"]["tier1"]["temperature"],
            max_tokens=app_config["llm"]["tier1"]["max_tokens"],
            base_url=app_config["llm"]["tier1"]["base_url"],
            api_key=app_config["llm"]["tier1"]["api_key"],
            timeout=app_config["llm"]["tier1"]["timeout"]
        )

        tier2_config = LLMConfig(
            provider=app_config["llm"]["tier2"]["provider"],
            model=app_config["llm"]["tier2"]["model"],
            temperature=app_config["llm"]["tier2"]["temperature"],
            max_tokens=app_config["llm"]["tier2"]["max_tokens"],
            base_url=app_config["llm"]["tier2"]["base_url"],
            api_key=app_config["llm"]["tier2"]["api_key"],
            timeout=app_config["llm"]["tier2"]["timeout"]
        )

        # Initialize the global LLM manager
        initialize_llm_manager(tier1_config, tier2_config)
        return True
    except Exception as e:
        print(f"Failed to initialize LLM manager: {e}")
        return False


def init_memory_agent():
    """Initialize the memory agent with current configuration."""
    try:
        # Create memory agent with current Redis configuration
        base_memory_agent = MemoryAgent(
            redis_host=app_config["redis"]["host"],
            redis_port=app_config["redis"]["port"],
            redis_db=app_config["redis"]["db"],
            vectorset_key=app_config["redis"]["vectorset_key"],
            app_config=app_config
        )

        # Create memory agent chat with current OpenAI configuration
        memory_agent = MemoryAgentChat(
            model_name=app_config["langgraph"]["model_name"],
            temperature=app_config["langgraph"]["temperature"],
            vectorset_key=app_config["redis"]["vectorset_key"]
        )

        # Replace the underlying memory agent with our configured one
        memory_agent.memory_agent = base_memory_agent

        # Set the global memory agent for dependency injection
        set_memory_agent(memory_agent)

        return True
    except Exception as e:
        print(f"Failed to initialize memory agent: {e}")
        return False


def startup():
    """Initialize services on startup."""
    print("🚀 Starting Memory Agent Web Server...")

    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment variables.")
        print("Please set your OpenAI API key in the .env file or environment.")
        raise RuntimeError("OPENAI_API_KEY not found")

    # Initialize LLM manager
    if init_llm_manager():
        print("✅ LLM manager ready")
    else:
        print("❌ Failed to initialize LLM manager")
        raise RuntimeError("Failed to initialize LLM manager")

    # Initialize memory agent
    if init_memory_agent():
        print("✅ Memory agent ready")
        print("🌐 Server running at http://localhost:5001")
        print("📖 API docs available at http://localhost:5001/docs")
        print()
    else:
        print("❌ Failed to initialize memory agent")
        raise RuntimeError("Failed to initialize memory agent")

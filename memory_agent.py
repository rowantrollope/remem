#!/usr/bin/env python3
"""
Memory Agent with Redis and OpenAI - Refactored Architecture

A layered memory agent system with clean separation of concerns.

Installation requirements:
pip install redis>=5.0.0 openai>=1.0.0 python-dotenv>=1.0.0 numpy>=1.24.0

Prerequisites:
- Redis server running with RedisSearch module (Redis Stack or Redis with RediSearch)
- OpenAI API key in environment variables or .env file
- Optional: Redis connection details in .env file (REDIS_HOST, REDIS_PORT, REDIS_DB)

Usage:
python memory_agent.py
"""

from memory_core import MemoryCore
from memory_processing import MemoryProcessing
from memory_extraction import MemoryExtraction
from memory_reasoning import MemoryReasoning
from typing import Dict, Any, List, Optional
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class MemoryAgent:
    """Intelligent Memory Agent based on Minsky's Society of Mind Theory.

    This agent implements a cognitive architecture inspired by Marvin Minsky's
    concepts of Nemes (fundamental memory units) and K-lines (knowledge lines
    that construct mental states from distributed memories).

    Architecture Layers:
    - MemoryCore: Manages Nemes (atomic memory storage and retrieval)
    - MemoryProcessing: Analysis and filtering utilities
    - MemoryExtraction: Intelligent conversation processing
    - MemoryReasoning: K-line construction and sophisticated reasoning

    The agent provides three levels of API:
    1. NEME API: Direct manipulation of atomic memories
    2. K-LINE API: Mental state construction and reasoning
    3. AGENT API: Full cognitive orchestration
    """

    def __init__(self, redis_host: str = None, redis_port: int = None, redis_db: int = None):
        """Initialize the memory agent with Minsky-inspired cognitive architecture.

        This creates a layered system where:
        - Core layer manages Nemes (fundamental memory units)
        - Processing layer provides cognitive utilities
        - Extraction layer converts experience to structured memory
        - Reasoning layer constructs K-lines (mental states) for complex cognition

        Args:
            redis_host: Redis server host (defaults to REDIS_HOST env var or "localhost")
            redis_port: Redis server port (defaults to REDIS_PORT env var or 6379)
            redis_db: Redis database number (defaults to REDIS_DB env var or 0)
        """
        # Initialize the core layer (Neme management)
        self.core = MemoryCore(redis_host, redis_port, redis_db)

        # Initialize processing utilities (cognitive tools)
        self.processing = MemoryProcessing()

        # Initialize extraction service (experience → memory conversion)
        self.extraction = MemoryExtraction(self.core)

        # Initialize reasoning service (K-line construction and reasoning)
        self.reasoning = MemoryReasoning(self.core, self.processing)

    # =========================================================================
    # NEME API - Fundamental Memory Operations
    # =========================================================================
    #
    # Nemes are the atomic units of memory in Minsky's Society of Mind theory.
    # They represent fundamental knowledge structures that can be:
    # - Stored with contextual grounding
    # - Retrieved through vector similarity
    # - Combined by higher-level cognitive processes
    #
    # These methods provide direct access to the memory substrate that
    # underlies all higher-level cognitive operations.
    # =========================================================================
    def store_memory(self, memory_text: str, apply_grounding: bool = True) -> Dict[str, Any]:
        """Store an atomic memory (Neme) with optional contextual grounding.

        In Minsky's framework, this creates a fundamental memory unit that
        can later be activated and combined with other Nemes to form
        complex mental states (K-lines).

        Args:
            memory_text: The memory text to store as a Neme
            apply_grounding: Whether to apply contextual grounding

        Returns:
            Dictionary with storage results and grounding information
        """
        return self.core.store_memory(memory_text, apply_grounding)

    def search_memories(self, query: str, top_k: int = 3, filterBy: str = None) -> List[Dict[str, Any]]:
        """Search for relevant Nemes using vector similarity.

        This is the fundamental operation for finding related atomic memories
        that can be activated together to form mental states (K-lines).

        Args:
            query: Search query to find relevant Nemes
            top_k: Number of Nemes to return
            filterBy: Optional filter expression

        Returns:
            List of matching Nemes with metadata and relevance scores
        """
        return self.core.search_memories(query, top_k, filterBy)

    def set_context(self, location: str = None, activity: str = None, people_present: List[str] = None, **kwargs):
        """Set current context for memory grounding.

        Args:
            location: Current location
            activity: Current activity
            people_present: List of people present
            **kwargs: Additional environmental context
        """
        return self.core.set_context(location, activity, people_present, **kwargs)

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory.

        Args:
            memory_id: UUID of memory to delete

        Returns:
            True if deleted successfully
        """
        return self.core.delete_memory(memory_id)

    def clear_all_memories(self) -> Dict[str, Any]:
        """Clear all stored memories.

        Returns:
            Dictionary with operation results
        """
        return self.core.clear_all_memories()

    def get_memory_info(self) -> Dict[str, Any]:
        """Get memory system statistics.

        Returns:
            Dictionary with memory count and system info
        """
        return self.core.get_memory_info()

    # =========================================================================
    # K-LINE API - Reflective Operations (Mental State Construction)
    # =========================================================================
    #
    # K-lines (Knowledge lines) are Minsky's concept for temporary mental states
    # that activate and connect relevant Nemes for specific cognitive tasks.
    #
    # A K-line represents the mind's ability to:
    # - Recall relevant memories for a specific context
    # - Filter and organize them into coherent mental states
    # - Apply reasoning and generate insights
    # - Extract new memories from experience
    #
    # These operations represent higher-level cognition built on the Neme substrate.
    # =========================================================================
    def answer_question(self, question: str, top_k: int = 5, filterBy: str = None) -> Dict[str, Any]:
        """Answer a question by constructing a K-line and applying reasoning.

        This method demonstrates the full K-line process:
        1. Activates relevant Nemes based on the question
        2. Constructs a mental state (K-line) from these memories
        3. Applies reasoning to generate a confident answer
        4. Returns structured results with supporting evidence

        Args:
            question: The question to answer
            top_k: Number of Nemes to activate for the mental state
            filterBy: Optional filter expression

        Returns:
            Structured answer with confidence, reasoning, and supporting Nemes
        """
        return self.reasoning.answer_question(question, top_k, filterBy)

    def extract_and_store_memories(self, raw_input: str, context_prompt: str,
                                 extraction_examples: Optional[List[Dict[str, str]]] = None,
                                 apply_grounding: bool = True) -> Dict[str, Any]:
        """Extract new Nemes from conversational experience.

        This represents the process of converting raw experience into
        structured memory units (Nemes) that can later be activated
        in K-line construction for reasoning and recall.

        Args:
            raw_input: Conversational input to analyze for valuable information
            context_prompt: Application context for extraction guidance
            extraction_examples: Optional examples to guide extraction
            apply_grounding: Whether to apply grounding to extracted Nemes

        Returns:
            Dictionary with extraction results and newly stored Nemes
        """
        return self.extraction.extract_and_store_memories(raw_input, context_prompt, extraction_examples, apply_grounding)

    def recall_memories(self, query: str, top_k: int = 3) -> str:
        """Construct and format a mental state (K-line) for display.

        This method demonstrates K-line construction by:
        1. Activating relevant Nemes based on the query
        2. Organizing them into a coherent mental state
        3. Formatting the result for human consumption

        Args:
            query: Query to construct mental state around
            top_k: Number of Nemes to activate

        Returns:
            Formatted string representation of the mental state
        """
        memories = self.search_memories(query, top_k)
        return self.processing.format_memory_results(memories)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics (alias for get_memory_info)."""
        return self.get_memory_info()


def main():
    """Main CLI interface for the Minsky-inspired Memory Agent."""
    print("🧠 Minsky Memory Agent - Nemes & K-lines")
    print("=" * 50)
    print("Based on Marvin Minsky's Society of Mind theory")
    print("• Nemes: Fundamental memory units")
    print("• K-lines: Mental states constructed from Nemes")
    print("=" * 50)

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment variables.")
        print("Please set your OpenAI API key in the .env file or environment.")
        sys.exit(1)

    # Initialize the agent
    try:
        agent = MemoryAgent()
    except Exception as e:
        print(f"❌ Failed to initialize memory agent: {e}")
        sys.exit(1)

    print("\nCommands (Minsky Framework):")
    print("NEME Operations (Fundamental Memory Units):")
    print("- remember \"<memory>\" - Store a new Neme (with contextual grounding)")
    print("- remember-raw \"<memory>\" - Store a Neme without contextual grounding")
    print("- delete <memory_id> - Delete a specific Neme by ID")
    print("- info - Show Neme statistics and system information")
    print()
    print("K-LINE Operations (Mental State Construction):")
    print("- recall \"<query>\" - Construct mental state from relevant Nemes")
    print("- ask \"<question>\" - Answer question using K-line reasoning")
    print()
    print("Context Management:")
    print("- context location=\"<location>\" activity=\"<activity>\" people=\"<person1,person2>\" - Set current context")
    print("- context-info - Show current context")
    print()
    print("- quit - Exit the program")
    print()

    while True:
        try:
            user_input = input("Memory Agent> ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break

            if not user_input:
                continue

            # Parse commands
            if user_input.lower().startswith('remember-raw '):
                memory_text = user_input[13:].strip()
                if memory_text.startswith('"') and memory_text.endswith('"'):
                    memory_text = memory_text[1:-1]

                if memory_text:
                    storage_result = agent.store_memory(memory_text, apply_grounding=False)
                    print(f"✅ Stored Neme with ID: {storage_result['memory_id']}")
                else:
                    print("❌ Please provide memory text after 'remember-raw'")

            elif user_input.lower().startswith('remember '):
                memory_text = user_input[9:].strip()
                if memory_text.startswith('"') and memory_text.endswith('"'):
                    memory_text = memory_text[1:-1]

                if memory_text:
                    storage_result = agent.store_memory(memory_text, apply_grounding=True)
                    print(f"✅ Stored Neme with ID: {storage_result['memory_id']}")
                    if storage_result['grounding_applied']:
                        print(f"🌍 Contextual grounding applied:")
                        print(f"   Original: {storage_result['original_text']}")
                        print(f"   Grounded: {storage_result['final_text']}")
                        if 'grounding_info' in storage_result and storage_result['grounding_info']['changes_made']:
                            changes = storage_result['grounding_info']['changes_made']
                            print(f"   Changes: {len(changes)} modifications made")
                else:
                    print("❌ Please provide memory text after 'remember'")

            elif user_input.lower().startswith('recall '):
                query_text = user_input[7:].strip()
                if query_text.startswith('"') and query_text.endswith('"'):
                    query_text = query_text[1:-1]

                if query_text:
                    print(f"🧠 Constructing K-line (mental state) for: '{query_text}'")
                    result = agent.recall_memories(query_text)
                    print(result)
                else:
                    print("❌ Please provide query text after 'recall'")

            elif user_input.lower().startswith('ask '):
                question_text = user_input[4:].strip()
                if question_text.startswith('"') and question_text.endswith('"'):
                    question_text = question_text[1:-1]

                if question_text:
                    print(f"🧠 Constructing K-line and reasoning for: '{question_text}'")
                    answer_response = agent.answer_question(question_text)

                    # Format the structured response for CLI display
                    if answer_response["type"] == "help":
                        print(f"\n💡 {answer_response['answer']}")
                    else:
                        print(f"\n🤖 Answer: {answer_response['answer']}")
                        print(f"🎯 Confidence: {answer_response['confidence']}")

                        if answer_response.get('reasoning'):
                            print(f"💭 K-line Reasoning: {answer_response['reasoning']}")

                        if answer_response.get('supporting_memories'):
                            print(f"\n📚 Supporting Nemes ({len(answer_response['supporting_memories'])}):")
                            for i, memory in enumerate(answer_response['supporting_memories'], 1):
                                print(f"   {i}. {memory['text']} ({memory['relevance_score']}% relevant, {memory['timestamp']})")
                                if memory['tags']:
                                    print(f"      Tags: {', '.join(memory['tags'])}")
                else:
                    print("❌ Please provide a question after 'ask'")

            elif user_input.lower().startswith('context '):
                context_args = user_input[8:].strip()

                # Parse context arguments (simple key=value format)
                context_params = {}
                people_present = []

                # Split by spaces but handle quoted values
                import shlex
                try:
                    args = shlex.split(context_args)
                    for arg in args:
                        if '=' in arg:
                            key, value = arg.split('=', 1)
                            key = key.strip()
                            value = value.strip()

                            if key == 'people':
                                people_present = [p.strip() for p in value.split(',') if p.strip()]
                            elif key in ['location', 'activity']:
                                context_params[key] = value
                            else:
                                # Store as environment context
                                context_params[key] = value

                    # Apply context
                    if 'location' in context_params or 'activity' in context_params or people_present:
                        agent.set_context(
                            location=context_params.get('location'),
                            activity=context_params.get('activity'),
                            people_present=people_present if people_present else None,
                            **{k: v for k, v in context_params.items() if k not in ['location', 'activity']}
                        )
                    else:
                        print("❌ Please provide context parameters like: location=\"Jakarta\" activity=\"working\" people=\"John,Sarah\"")

                except Exception as e:
                    print(f"❌ Error parsing context: {e}")
                    print("Example: context location=\"Jakarta, Indonesia\" activity=\"traveling\" people=\"John,Sarah\" weather=\"hot\"")

            elif user_input.lower() == 'context-info':
                print("\n🌍 Current Context:")
                print("=" * 30)
                context = agent.core._get_current_context()

                print(f"📅 Date: {context['temporal']['date']}")
                print(f"🕐 Time: {context['temporal']['time']}")

                if context['spatial']['location']:
                    print(f"📍 Location: {context['spatial']['location']}")
                else:
                    print("📍 Location: Not set")

                if context['spatial']['activity']:
                    print(f"🎯 Activity: {context['spatial']['activity']}")
                else:
                    print("🎯 Activity: Not set")

                if context['social']['people_present']:
                    people = ", ".join(context['social']['people_present'])
                    print(f"👥 People Present: {people}")
                else:
                    print("👥 People Present: None specified")

                if context['environmental']:
                    print("🌡️  Environment:")
                    for key, value in context['environmental'].items():
                        print(f"   {key}: {value}")
                else:
                    print("🌡️  Environment: No additional context")

            elif user_input.lower().startswith('delete '):
                memory_id = user_input[7:].strip()

                if memory_id:
                    success = agent.delete_memory(memory_id)
                    if success:
                        print(f"✅ Deleted Neme: {memory_id}")
                    else:
                        print("❌ Failed to delete Neme. Check the memory ID and try again.")
                else:
                    print("❌ Please provide a Neme ID after 'delete'")

            elif user_input.lower() == 'info':
                print("\n📊 Minsky Memory System Information:")
                print("=" * 40)
                info = agent.get_memory_info()

                if 'error' in info:
                    print(f"❌ {info['error']}")
                else:
                    print(f"🧠 Total Nemes (Atomic Memories): {info['memory_count']}")
                    print(f"🔢 Vector Dimension: {info['vector_dimension']}")
                    print(f"🗃️  VectorSet Name: {info['vectorset_name']}")
                    print(f"🤖 Embedding Model: {info['embedding_model']}")
                    print(f"🔗 Redis: {info['redis_host']}:{info['redis_port']}")
                    print(f"⏰ Timestamp: {info['timestamp']}")
                    print(f"📖 Framework: Minsky's Society of Mind (Nemes + K-lines)")

                    if 'note' in info:
                        print(f"ℹ️  Note: {info['note']}")

            else:
                print("❌ Unknown command. Type 'quit' to exit.")
                print("Available commands: remember, remember-raw, recall, ask, context, context-info, delete, info, quit")

        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
MCP Server for LangGraph Memory Agent

A Model Context Protocol (MCP) server that exposes sophisticated memory capabilities
for AI assistants, including storage, retrieval, question answering, and context management.

Based on Minsky's Society of Mind theory with Nemes (atomic memories) and K-lines (mental states).
"""

import asyncio
import json
import logging
import sys
from typing import Any, Dict, List, Optional, Sequence
from contextlib import asynccontextmanager

# MCP imports will be available when the package is installed
try:
    import uvicorn
    from mcp.server.models import InitializationOptions
    from mcp.server import NotificationOptions, Server
    from mcp.types import Resource, Tool, TextContent, ImageContent, EmbeddedResource
    from mcp.server.models import GetPromptResult
    from pydantic import AnyUrl
except ImportError as e:
    print(f"❌ Missing MCP dependencies: {e}")
    print("Please install with: pip install -r requirements.txt")
    sys.exit(1)

from dotenv import load_dotenv
from memory.agent import LangGraphMemoryAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-memory-server")

# Initialize the memory agent
try:
    memory_agent = LangGraphMemoryAgent()
    logger.info("Memory agent initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize memory agent: {e}")
    sys.exit(1)

# Create the MCP server
server = Server("memory-agent")

@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List available memory tools."""
    return [
        Tool(
            name="store_memory",
            description="Store a new memory with optional contextual grounding. Creates a new Neme (atomic memory unit) in the memory system.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_text": {
                        "type": "string",
                        "description": "The memory text to store"
                    },
                    "apply_grounding": {
                        "type": "boolean",
                        "description": "Whether to apply contextual grounding (adds current date/time/location context)",
                        "default": True
                    }
                },
                "required": ["memory_text"]
            }
        ),
        Tool(
            name="search_memories",
            description="Search for relevant memories using vector similarity. Finds Nemes that can be activated for cognitive tasks.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to find relevant memories"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of memories to return",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 50
                    },
                    "min_similarity": {
                        "type": "number",
                        "description": "Minimum similarity score threshold (0.0-1.0)",
                        "default": 0.7,
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "filter": {
                        "type": "string",
                        "description": "Optional filter expression for Redis VSIM command"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="answer_question",
            description="Answer a question using K-line reasoning. Constructs mental state from relevant memories and provides confident answers with supporting evidence.",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to answer"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of memories to use for answering",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    },
                    "min_similarity": {
                        "type": "number",
                        "description": "Minimum similarity score threshold for memory selection",
                        "default": 0.7,
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                },
                "required": ["question"]
            }
        ),
        Tool(
            name="recall_memories",
            description="Construct and format a mental state (K-line) from relevant memories for a specific query. Shows how memories connect to form understanding.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query to construct mental state around"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of memories to include in mental state",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 20
                    },
                    "min_similarity": {
                        "type": "number",
                        "description": "Minimum similarity score threshold",
                        "default": 0.7,
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="set_context",
            description="Set current context for memory grounding. This context will be automatically applied to new memories.",
            inputSchema={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Current location"
                    },
                    "activity": {
                        "type": "string",
                        "description": "Current activity"
                    },
                    "people_present": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of people present"
                    },
                    "additional_context": {
                        "type": "object",
                        "description": "Additional environmental context (weather, mood, etc.)",
                        "additionalProperties": {"type": "string"}
                    }
                }
            }
        ),
        Tool(
            name="get_memory_stats",
            description="Get memory system statistics including total memories, system info, and current context.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="extract_and_store_memories",
            description="Extract new memories from conversational input using LLM analysis. Converts raw experience into structured memory units.",
            inputSchema={
                "type": "object",
                "properties": {
                    "conversation_text": {
                        "type": "string",
                        "description": "Conversational text to analyze for valuable information"
                    },
                    "context_prompt": {
                        "type": "string",
                        "description": "Context for extraction (e.g., 'I am a travel assistant')",
                        "default": "I am a personal assistant. Extract significant user information that would be valuable for future assistance."
                    },
                    "apply_grounding": {
                        "type": "boolean",
                        "description": "Whether to apply contextual grounding to extracted memories",
                        "default": True
                    }
                },
                "required": ["conversation_text"]
            }
        ),
        Tool(
            name="chat_with_memory",
            description="Have a conversational interaction that automatically manages memory. The agent will search existing memories for context and optionally extract new ones.",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Your message to the memory agent"
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": "Optional custom system prompt to override the default assistant behavior"
                    }
                },
                "required": ["message"]
            }
        ),
        Tool(
            name="delete_memory",
            description="Delete a specific memory by its ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "UUID of the memory to delete"
                    }
                },
                "required": ["memory_id"]
            }
        ),
        Tool(
            name="clear_all_memories",
            description="Clear all stored memories. WARNING: This action cannot be undone!",
            inputSchema={
                "type": "object",
                "properties": {
                    "confirm": {
                        "type": "boolean",
                        "description": "Must be set to true to confirm deletion of all memories"
                    }
                },
                "required": ["confirm"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool execution."""
    try:
        if name == "store_memory":
            memory_text = arguments["memory_text"]
            apply_grounding = arguments.get("apply_grounding", True)
            
            result = memory_agent.memory_agent.store_memory(memory_text, apply_grounding)
            
            response = f"✅ Memory stored successfully!\n"
            response += f"Memory ID: {result['memory_id']}\n"
            
            if result.get('grounding_applied'):
                response += f"\n🌍 Contextual grounding applied:\n"
                response += f"Original: {result['original_text']}\n"
                response += f"Grounded: {result['final_text']}\n"
                
                if 'grounding_info' in result and result['grounding_info'].get('changes_made'):
                    changes = result['grounding_info']['changes_made']
                    response += f"Changes: {len(changes)} modifications made\n"
            
            return [TextContent(type="text", text=response)]

        elif name == "search_memories":
            query = arguments["query"]
            top_k = arguments.get("top_k", 10)
            min_similarity = arguments.get("min_similarity", 0.7)
            filter_expr = arguments.get("filter")
            
            memories = memory_agent.memory_agent.search_memories(
                query, top_k, filter_expr, min_similarity
            )
            
            if not memories:
                return [TextContent(type="text", text=f"No memories found for query: '{query}' (minimum similarity: {min_similarity})")]
            
            response = f"🔍 Found {len(memories)} relevant memories for '{query}':\n\n"
            
            for i, memory in enumerate(memories, 1):
                text = memory.get('text', memory.get('final_text', memory.get('raw_text', 'No text')))
                score = memory.get('score', memory.get('relevance_score', 0))
                timestamp = memory.get('formatted_time', memory.get('timestamp', 'unknown time'))
                memory_id = memory.get('id', 'unknown')
                
                response += f"{i}. {text}\n"
                response += f"   📊 Relevance: {score:.1%} | 🕐 {timestamp} | 🆔 {memory_id}\n"
                
                if memory.get('tags'):
                    response += f"   🏷️  Tags: {', '.join(memory['tags'])}\n"
                response += "\n"
            
            return [TextContent(type="text", text=response)]

        elif name == "answer_question":
            question = arguments["question"]
            top_k = arguments.get("top_k", 5)
            min_similarity = arguments.get("min_similarity", 0.7)
            
            answer_result = memory_agent.memory_agent.answer_question(
                question, top_k, min_similarity=min_similarity
            )
            
            response = f"🤖 Answer: {answer_result['answer']}\n"
            response += f"🎯 Confidence: {answer_result['confidence']}\n"
            
            if answer_result.get('reasoning'):
                response += f"💭 Reasoning: {answer_result['reasoning']}\n"
            
            if answer_result.get('supporting_memories'):
                response += f"\n📚 Supporting Evidence ({len(answer_result['supporting_memories'])} memories):\n"
                for i, memory in enumerate(answer_result['supporting_memories'], 1):
                    text = memory.get('text', memory.get('final_text', memory.get('raw_text', 'No text')))
                    relevance = memory.get('relevance_score', memory.get('score', 0))
                    timestamp = memory.get('formatted_time', memory.get('timestamp', 'unknown time'))
                    
                    response += f"{i}. {text}\n"
                    response += f"   📊 Relevance: {relevance:.1%} | 🕐 {timestamp}\n\n"
            
            return [TextContent(type="text", text=response)]

        elif name == "recall_memories":
            query = arguments["query"]
            top_k = arguments.get("top_k", 10)
            min_similarity = arguments.get("min_similarity", 0.7)
            
            mental_state = memory_agent.memory_agent.recall_memories(query, top_k, min_similarity)
            
            response = f"🧠 Mental State (K-line) for '{query}':\n\n{mental_state}"
            
            return [TextContent(type="text", text=response)]

        elif name == "set_context":
            location = arguments.get("location")
            activity = arguments.get("activity")
            people_present = arguments.get("people_present")
            additional_context = arguments.get("additional_context", {})
            
            memory_agent.memory_agent.set_context(
                location=location,
                activity=activity,
                people_present=people_present,
                **additional_context
            )
            
            response = "✅ Context updated successfully!\n\n"
            response += "🌍 Current Context:\n"
            if location:
                response += f"📍 Location: {location}\n"
            if activity:
                response += f"🎯 Activity: {activity}\n"
            if people_present:
                response += f"👥 People Present: {', '.join(people_present)}\n"
            if additional_context:
                response += f"🌡️  Additional Context:\n"
                for key, value in additional_context.items():
                    response += f"   {key}: {value}\n"
            
            return [TextContent(type="text", text=response)]

        elif name == "get_memory_stats":
            stats = memory_agent.memory_agent.get_memory_stats()
            
            response = f"📊 Memory System Statistics:\n\n"
            response += f"💾 Total Memories: {stats.get('memory_count', 0)}\n"
            response += f"🔢 Vector Dimension: {stats.get('vector_dimension', 'unknown')}\n"
            response += f"🏪 Vectorset Name: {stats.get('vectorset_name', 'unknown')}\n"
            response += f"🤖 Embedding Model: {stats.get('embedding_model', 'unknown')}\n"
            response += f"🔗 Redis Host: {stats.get('redis_host', 'unknown')}\n"
            response += f"🔌 Redis Port: {stats.get('redis_port', 'unknown')}\n"
            
            # Get current context
            try:
                context = memory_agent.memory_agent.core._get_current_context()
                response += f"\n🌍 Current Context:\n"
                response += f"📅 Date: {context['temporal']['date']}\n"
                response += f"🕐 Time: {context['temporal']['time']}\n"
                response += f"📍 Location: {context['spatial'].get('location', 'Not set')}\n"
                response += f"🎯 Activity: {context['spatial'].get('activity', 'Not set')}\n"
                
                if context['social'].get('people_present'):
                    response += f"👥 People Present: {', '.join(context['social']['people_present'])}\n"
                
                if context.get('environmental'):
                    response += f"🌡️  Environment: {', '.join([f'{k}={v}' for k, v in context['environmental'].items()])}\n"
                    
            except Exception as e:
                response += f"\n⚠️ Could not retrieve context: {e}\n"
            
            return [TextContent(type="text", text=response)]

        elif name == "extract_and_store_memories":
            conversation_text = arguments["conversation_text"]
            context_prompt = arguments.get("context_prompt", 
                "I am a personal assistant. Extract significant user information that would be valuable for future assistance.")
            apply_grounding = arguments.get("apply_grounding", True)
            
            # Search for existing memories to avoid duplicates
            existing_memories = memory_agent.memory_agent.search_memories(
                conversation_text, top_k=5, min_similarity=0.8
            )
            
            result = memory_agent.memory_agent.extract_and_store_memories(
                raw_input=conversation_text,
                context_prompt=context_prompt,
                apply_grounding=apply_grounding,
                existing_memories=existing_memories
            )
            
            response = f"🧠 Memory Extraction Results:\n\n"
            response += f"📊 Total Extracted: {result['total_extracted']}\n"
            response += f"💾 Successfully Stored: {result['total_stored']}\n"
            
            if result.get('extracted_memories'):
                response += f"\n📝 Extracted Memories:\n"
                for i, memory in enumerate(result['extracted_memories'], 1):
                    response += f"{i}. {memory['text']}\n"
                    if memory.get('memory_id'):
                        response += f"   🆔 ID: {memory['memory_id']}\n"
                    response += "\n"
            
            if result['total_extracted'] == 0:
                response += "\n💡 No new significant information found to extract, or information already stored.\n"
            
            return [TextContent(type="text", text=response)]

        elif name == "chat_with_memory":
            message = arguments["message"]
            system_prompt = arguments.get("system_prompt")
            
            response = memory_agent.run(message, system_prompt)
            
            return [TextContent(type="text", text=response)]

        elif name == "delete_memory":
            memory_id = arguments["memory_id"]
            
            success = memory_agent.memory_agent.delete_memory(memory_id)
            
            if success:
                response = f"✅ Memory {memory_id} deleted successfully!"
            else:
                response = f"❌ Failed to delete memory {memory_id}. It may not exist."
            
            return [TextContent(type="text", text=response)]

        elif name == "clear_all_memories":
            confirm = arguments.get("confirm", False)
            
            if not confirm:
                return [TextContent(type="text", text="❌ You must set 'confirm' to true to clear all memories. This action cannot be undone!")]
            
            result = memory_agent.memory_agent.clear_all_memories()
            
            response = f"✅ All memories cleared successfully!\n"
            response += f"📊 Memories deleted: {result.get('memories_deleted', 0)}\n"
            response += f"🏪 Vectorset existed: {result.get('vectorset_existed', False)}\n"
            
            return [TextContent(type="text", text=response)]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        error_msg = f"Error executing {name}: {str(e)}"
        logger.error(error_msg)
        return [TextContent(type="text", text=error_msg)]

@server.list_resources()
async def handle_list_resources() -> list[Resource]:
    """List available memory resources."""
    try:
        stats = memory_agent.memory_agent.get_memory_stats()
        memory_count = stats.get('memory_count', 0)
        
        return [
            Resource(
                uri=AnyUrl("memory://stats"),
                name="Memory System Statistics",
                description=f"Current memory system statistics ({memory_count} memories stored)",
                mimeType="application/json"
            ),
            Resource(
                uri=AnyUrl("memory://context"),
                name="Current Context",
                description="Current context settings for memory grounding",
                mimeType="application/json"
            ),
            Resource(
                uri=AnyUrl("memory://recent"),
                name="Recent Memories",
                description="Most recently stored memories (last 10)",
                mimeType="application/json"
            )
        ]
    except Exception as e:
        logger.error(f"Error listing resources: {e}")
        return []

@server.read_resource()
async def handle_read_resource(uri: AnyUrl) -> str:
    """Read memory system resources."""
    try:
        if uri.scheme != "memory":
            raise ValueError(f"Unsupported URI scheme: {uri.scheme}")
        
        if uri.path == "/stats":
            stats = memory_agent.memory_agent.get_memory_stats()
            return json.dumps(stats, indent=2)
        
        elif uri.path == "/context":
            context = memory_agent.memory_agent.core._get_current_context()
            return json.dumps(context, indent=2)
        
        elif uri.path == "/recent":
            # Get recent memories by searching with a broad query
            recent_memories = memory_agent.memory_agent.search_memories(
                "", top_k=10, min_similarity=0.0
            )
            
            # Format for JSON output
            formatted_memories = []
            for memory in recent_memories:
                formatted_memories.append({
                    "id": memory.get('id'),
                    "text": memory.get('text', memory.get('final_text', memory.get('raw_text'))),
                    "timestamp": memory.get('formatted_time', memory.get('timestamp')),
                    "tags": memory.get('tags', []),
                    "score": memory.get('score', memory.get('relevance_score', 0))
                })
            
            return json.dumps(formatted_memories, indent=2)
        
        else:
            raise ValueError(f"Unknown resource path: {uri.path}")
    
    except Exception as e:
        logger.error(f"Error reading resource {uri}: {e}")
        raise

@server.list_prompts()
async def handle_list_prompts() -> list[Tool]:
    """List available memory-related prompts."""
    return [
        Tool(
            name="memory_guided_conversation",
            description="Start a conversation that intelligently uses memory context",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic or context for the conversation"
                    },
                    "role": {
                        "type": "string",
                        "description": "Role for the assistant (e.g., 'travel assistant', 'cooking helper')",
                        "default": "personal assistant"
                    }
                },
                "required": ["topic"]
            }
        ),
        Tool(
            name="memory_summary",
            description="Generate a summary of stored memories related to a topic",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic to summarize memories about"
                    },
                    "max_memories": {
                        "type": "integer",
                        "description": "Maximum number of memories to include",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 50
                    }
                },
                "required": ["topic"]
            }
        )
    ]

@server.get_prompt()
async def handle_get_prompt(name: str, arguments: dict | None) -> GetPromptResult:
    """Generate memory-guided prompts."""
    try:
        if name == "memory_guided_conversation":
            topic = arguments.get("topic", "") if arguments else ""
            role = arguments.get("role", "personal assistant") if arguments else "personal assistant"
            
            # Search for relevant memories
            memories = memory_agent.memory_agent.search_memories(topic, top_k=5, min_similarity=0.7)
            
            # Build context from memories
            memory_context = ""
            if memories:
                memory_context = "\n\nRelevant memories about the user:\n"
                for i, memory in enumerate(memories, 1):
                    text = memory.get('text', memory.get('final_text', memory.get('raw_text', '')))
                    memory_context += f"{i}. {text}\n"
            
            prompt_text = f"""You are a {role} with access to the user's personal memories. 

Topic of conversation: {topic}
{memory_context}

Instructions:
1. Use the relevant memories to provide personalized assistance
2. Reference specific details from memories when appropriate
3. Ask follow-up questions based on what you know about the user
4. Be conversational and helpful

Start the conversation by acknowledging what you remember about the user related to {topic}."""

            return GetPromptResult(
                description=f"Memory-guided conversation about {topic}",
                messages=[
                    {
                        "role": "system",
                        "content": {
                            "type": "text",
                            "text": prompt_text
                        }
                    }
                ]
            )
        
        elif name == "memory_summary":
            topic = arguments.get("topic", "") if arguments else ""
            max_memories = arguments.get("max_memories", 10) if arguments else 10
            
            # Search for relevant memories
            memories = memory_agent.memory_agent.search_memories(topic, top_k=max_memories, min_similarity=0.6)
            
            if not memories:
                summary_text = f"No memories found related to '{topic}'."
            else:
                summary_text = f"Summary of memories related to '{topic}':\n\n"
                for i, memory in enumerate(memories, 1):
                    text = memory.get('text', memory.get('final_text', memory.get('raw_text', '')))
                    timestamp = memory.get('formatted_time', memory.get('timestamp', 'unknown time'))
                    relevance = memory.get('score', memory.get('relevance_score', 0))
                    
                    summary_text += f"{i}. {text}\n"
                    summary_text += f"   📊 Relevance: {relevance:.1%} | 🕐 {timestamp}\n\n"
            
            return GetPromptResult(
                description=f"Summary of memories about {topic}",
                messages=[
                    {
                        "role": "user", 
                        "content": {
                            "type": "text",
                            "text": summary_text
                        }
                    }
                ]
            )
        
        else:
            raise ValueError(f"Unknown prompt: {name}")
    
    except Exception as e:
        logger.error(f"Error generating prompt {name}: {e}")
        raise

async def main():
    """Main function to run the MCP server."""
    # Determine the transport based on command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "stdio":
        # Standard I/O transport (for Cursor and other MCP clients)
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="memory-agent",
                    server_version="1.0.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={}
                    )
                )
            )
    else:
        # HTTP transport for development/testing
        config = uvicorn.Config(
            "mcp_server:app", 
            host="localhost", 
            port=8000, 
            log_level="info"
        )
        server_instance = uvicorn.Server(config)
        await server_instance.serve()

if __name__ == "__main__":
    print("🧠 Memory Agent MCP Server")
    print("=" * 50)
    print("Model Context Protocol Server for Memory Operations")
    print("Based on Minsky's Society of Mind theory")
    print("• Nemes: Atomic memory units")
    print("• K-lines: Mental states from connected memories")
    print("=" * 50)
    print()
    
    # Check for required environment variables
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment variables.")
        print("Please set your OpenAI API key in the .env file or environment.")
        sys.exit(1)
    
    print("🚀 Starting MCP server...")
    if len(sys.argv) > 1 and sys.argv[1] == "stdio":
        print("📡 Using stdio transport (for MCP clients like Cursor)")
    else:
        print("🌐 Using HTTP transport on http://localhost:8000")
        print("💡 For MCP clients, run with 'stdio' argument")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Shutting down MCP server...")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)
#!/usr/bin/env python3
"""
Asynchronous Memory Processor - Background worker for processing raw memories

This module implements a background processor that:
1. Monitors the RAW_MEMORY_QUEUE for new raw memory entries
2. Extracts discrete memories from chat sessions
3. Generates session summaries
4. Creates hierarchical memory structures with cross-references
5. Implements data retention policies

The processor runs as a scheduled background task and can be configured
for different processing intervals and retention policies.
"""

import json
import time
import uuid
import sys
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
import redis
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.core_agent import MemoryAgent
from llm_manager import get_llm_manager, init_llm_manager, LLMConfig


class AsyncMemoryProcessor:
    """Background processor for asynchronous memory processing."""
    
    def __init__(self, 
                 redis_host: str = None, 
                 redis_port: int = None, 
                 redis_db: int = None,
                 processing_interval: int = 60,
                 retention_days: int = 30):
        """Initialize the async memory processor.
        
        Args:
            redis_host: Redis server host
            redis_port: Redis server port  
            redis_db: Redis database number
            processing_interval: Seconds between processing cycles
            retention_days: Days to retain raw transcripts
        """
        # Get Redis connection details
        self.redis_host = redis_host or os.getenv("REDIS_HOST", "localhost")
        self.redis_port = redis_port or int(os.getenv("REDIS_PORT", "6379"))
        self.redis_db = redis_db or int(os.getenv("REDIS_DB", "0"))
        
        # Processing configuration
        self.processing_interval = processing_interval
        self.retention_days = retention_days
        self.queue_key = "RAW_MEMORY_QUEUE"
        
        # Initialize Redis connection
        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                decode_responses=False
            )
            self.redis_client.ping()
            print(f"✅ ASYNC PROCESSOR: Connected to Redis at {self.redis_host}:{self.redis_port}")
        except redis.ConnectionError as e:
            print(f"❌ ASYNC PROCESSOR: Failed to connect to Redis: {e}")
            raise
            
        # Initialize LLM manager for this process
        try:
            # Create default LLM configurations
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
            print("✅ ASYNC PROCESSOR: LLM manager initialized")
        except Exception as e:
            print(f"⚠️ ASYNC PROCESSOR: LLM manager initialization error: {e}")

        # Initialize memory agents cache (one per vectorstore)
        self.memory_agents: Dict[str, MemoryAgent] = {}

        # Processing statistics
        self.stats = {
            "processed_count": 0,
            "error_count": 0,
            "last_processed_at": None,
            "start_time": datetime.now(timezone.utc)
        }
        
    def get_memory_agent(self, vectorstore_name: str) -> MemoryAgent:
        """Get or create a memory agent for the specified vectorstore."""
        if vectorstore_name not in self.memory_agents:
            self.memory_agents[vectorstore_name] = MemoryAgent(
                redis_host=self.redis_host,
                redis_port=self.redis_port,
                redis_db=self.redis_db,
                vectorset_key=vectorstore_name
            )
        return self.memory_agents[vectorstore_name]
    
    def update_heartbeat(self):
        """Update processor heartbeat for status monitoring."""
        status_key = "background_processor:status"
        heartbeat_data = {
            "last_heartbeat": datetime.now(timezone.utc).isoformat(),
            "interval_seconds": self.processing_interval,
            "processed_count": self.stats["processed_count"],
            "error_count": self.stats["error_count"],
            "start_time": self.stats["start_time"].isoformat()
        }
        
        for key, value in heartbeat_data.items():
            self.redis_client.hset(status_key, key, str(value))
    
    def process_raw_memory(self, raw_memory_key: str) -> Dict[str, Any]:
        """Process a single raw memory entry.
        
        Args:
            raw_memory_key: Redis key for the raw memory record
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Get raw memory data
            raw_data = self.redis_client.get(raw_memory_key)
            if not raw_data:
                return {"success": False, "error": "Raw memory not found"}
                
            raw_memory = json.loads(raw_data.decode())
            vectorstore_name = raw_memory["vectorstore_name"]
            session_id = raw_memory["session_id"]
            session_data = raw_memory["session_data"]
            
            print(f"🔄 ASYNC PROCESSOR: Processing raw memory {raw_memory['raw_memory_id']}")
            print(f"📦 Vectorstore: {vectorstore_name}, Session: {session_id}")
            
            # Get memory agent for this vectorstore
            memory_agent = self.get_memory_agent(vectorstore_name)
            
            # Extract discrete memories from session data
            extraction_result = memory_agent.extract_and_store_memories(
                raw_input=session_data,
                context_prompt="Extract valuable memories from this chat session. Focus on user preferences, insights, facts, and important information that would be useful for future conversations.",
                apply_grounding=True
            )
            
            discrete_memories = extraction_result.get("memories", [])
            
            # Generate session summary
            session_summary = self._generate_session_summary(session_data, discrete_memories)
            
            # Store session summary
            summary_key = f"{vectorstore_name}:session_summary:{session_id}"
            summary_data = {
                "session_id": session_id,
                "summary": session_summary,
                "memory_count": len(discrete_memories),
                "memory_ids": [mem.get("memory_id") for mem in discrete_memories],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "raw_memory_id": raw_memory["raw_memory_id"]
            }
            self.redis_client.set(summary_key, json.dumps(summary_data))
            
            # Create cross-references
            if discrete_memories:
                memories_key = f"{vectorstore_name}:session_memories:{session_id}"
                memory_ids = [mem.get("memory_id") for mem in discrete_memories if mem.get("memory_id")]
                if memory_ids:
                    self.redis_client.sadd(memories_key, *memory_ids)
            
            # Update raw memory status
            raw_memory["status"] = "processed"
            raw_memory["processed_at"] = datetime.now(timezone.utc).isoformat()
            raw_memory["processing_result"] = {
                "discrete_memories_count": len(discrete_memories),
                "session_summary_created": True
            }
            self.redis_client.set(raw_memory_key, json.dumps(raw_memory))
            
            print(f"✅ ASYNC PROCESSOR: Successfully processed {len(discrete_memories)} memories from session {session_id}")
            
            return {
                "success": True,
                "session_id": session_id,
                "discrete_memories_count": len(discrete_memories),
                "session_summary_created": True,
                "memory_ids": [mem.get("memory_id") for mem in discrete_memories]
            }
            
        except Exception as e:
            print(f"❌ ASYNC PROCESSOR: Error processing {raw_memory_key}: {e}")
            
            # Update raw memory with error status
            try:
                raw_data = self.redis_client.get(raw_memory_key)
                if raw_data:
                    raw_memory = json.loads(raw_data.decode())
                    raw_memory["status"] = "error"
                    raw_memory["error"] = str(e)
                    raw_memory["processing_attempts"] = raw_memory.get("processing_attempts", 0) + 1
                    raw_memory["last_error_at"] = datetime.now(timezone.utc).isoformat()
                    self.redis_client.set(raw_memory_key, json.dumps(raw_memory))
            except:
                pass
                
            return {"success": False, "error": str(e)}
    
    def _generate_session_summary(self, session_data: str, discrete_memories: List[Dict]) -> str:
        """Generate a summary of the chat session.
        
        Args:
            session_data: Raw session conversation text
            discrete_memories: List of extracted discrete memories
            
        Returns:
            Generated session summary text
        """
        try:
            llm_manager = get_llm_manager()
            if not llm_manager:
                return "Summary generation failed: LLM manager not available"
                
            tier2_client = llm_manager.get_tier2_client()
            
            # Create summary prompt
            memory_list = "\n".join([f"- {mem.get('text', mem.get('final_text', ''))}" for mem in discrete_memories])
            
            prompt = f"""Generate a concise summary of this chat session. Focus on the main topics discussed, key insights, and overall purpose.

Session Content:
{session_data[:2000]}...

Extracted Memories:
{memory_list}

Provide a 2-3 sentence summary that captures the essence of this conversation:"""

            response = tier2_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            
            return response.get('content', 'Summary generation failed').strip()
            
        except Exception as e:
            print(f"⚠️ ASYNC PROCESSOR: Summary generation failed: {e}")
            return f"Chat session with {len(discrete_memories)} extracted memories. Summary generation failed."
    
    def process_queue(self) -> Dict[str, Any]:
        """Process all items in the raw memory queue.
        
        Returns:
            Dictionary with processing statistics
        """
        processed_count = 0
        error_count = 0
        
        try:
            # Get all items from queue (oldest first)
            queue_items = self.redis_client.zrange(self.queue_key, 0, -1, withscores=True)
            
            if not queue_items:
                return {"processed": 0, "errors": 0, "message": "Queue is empty"}
            
            print(f"🔄 ASYNC PROCESSOR: Processing {len(queue_items)} items from queue")
            
            for raw_memory_key, timestamp in queue_items:
                raw_memory_key = raw_memory_key.decode() if isinstance(raw_memory_key, bytes) else raw_memory_key
                
                # Process the raw memory
                result = self.process_raw_memory(raw_memory_key)
                
                if result["success"]:
                    processed_count += 1
                    # Remove from queue after successful processing
                    self.redis_client.zrem(self.queue_key, raw_memory_key)
                else:
                    error_count += 1
                    # Keep in queue for retry (could implement max retry logic here)
                
                # Small delay between processing items
                time.sleep(0.1)
            
            # Update statistics
            self.stats["processed_count"] += processed_count
            self.stats["error_count"] += error_count
            self.stats["last_processed_at"] = datetime.now(timezone.utc)
            
            # Update daily statistics
            today = datetime.now(timezone.utc).date()
            stats_key = f"processing_stats:{today.isoformat()}"
            self.redis_client.hincrby(stats_key, "processed_count", processed_count)
            self.redis_client.hset(stats_key, "last_processed_at", self.stats["last_processed_at"].isoformat())
            self.redis_client.expire(stats_key, 86400 * 7)  # Keep for 7 days
            
            return {
                "processed": processed_count,
                "errors": error_count,
                "message": f"Processed {processed_count} items, {error_count} errors"
            }
            
        except Exception as e:
            print(f"❌ ASYNC PROCESSOR: Queue processing failed: {e}")
            return {"processed": 0, "errors": 1, "message": f"Queue processing failed: {e}"}

    def cleanup_expired_data(self) -> Dict[str, Any]:
        """Clean up expired raw transcripts and old data.

        Returns:
            Dictionary with cleanup statistics
        """
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
            cutoff_timestamp = cutoff_date.timestamp()

            # Find expired raw memories
            expired_keys = []

            # Scan for raw memory keys
            for key in self.redis_client.scan_iter(match="*:raw_memory:*"):
                key_str = key.decode() if isinstance(key, bytes) else key
                try:
                    raw_data = self.redis_client.get(key_str)
                    if raw_data:
                        raw_memory = json.loads(raw_data.decode())
                        created_at = datetime.fromisoformat(raw_memory["created_at"].replace('Z', '+00:00'))

                        # Only delete if processed and older than retention period
                        if (raw_memory.get("status") == "processed" and
                            created_at.timestamp() < cutoff_timestamp):
                            expired_keys.append(key_str)
                except:
                    continue

            # Delete expired raw memories
            deleted_count = 0
            for key in expired_keys:
                self.redis_client.delete(key)
                deleted_count += 1

            # Clean up old queue entries (remove processed items older than retention)
            removed_from_queue = self.redis_client.zremrangebyscore(
                self.queue_key, 0, cutoff_timestamp
            )

            print(f"🧹 ASYNC PROCESSOR: Cleanup completed - deleted {deleted_count} raw memories, removed {removed_from_queue} queue entries")

            return {
                "deleted_raw_memories": deleted_count,
                "removed_queue_entries": removed_from_queue,
                "cutoff_date": cutoff_date.isoformat()
            }

        except Exception as e:
            print(f"❌ ASYNC PROCESSOR: Cleanup failed: {e}")
            return {"error": str(e)}

    def run_once(self) -> Dict[str, Any]:
        """Run one processing cycle.

        Returns:
            Dictionary with cycle results
        """
        cycle_start = datetime.now(timezone.utc)

        # Update heartbeat
        self.update_heartbeat()

        # Process queue
        processing_result = self.process_queue()

        # Run cleanup every hour (3600 seconds) - but not on first run
        cleanup_result = None
        should_cleanup = False

        if self.stats["last_processed_at"]:
            time_since_last = (cycle_start - self.stats["last_processed_at"]).total_seconds()
            should_cleanup = time_since_last > 3600
        else:
            # On first run, only cleanup if there are actually old items
            should_cleanup = False

        if should_cleanup:
            cleanup_result = self.cleanup_expired_data()

        cycle_duration = (datetime.now(timezone.utc) - cycle_start).total_seconds()

        return {
            "cycle_start": cycle_start.isoformat(),
            "cycle_duration_seconds": cycle_duration,
            "processing_result": processing_result,
            "cleanup_result": cleanup_result,
            "stats": self.stats.copy()
        }

    def run_continuous(self):
        """Run the processor continuously with the configured interval."""
        print(f"🚀 ASYNC PROCESSOR: Starting continuous processing (interval: {self.processing_interval}s)")
        print(f"📅 Retention policy: {self.retention_days} days")

        try:
            while True:
                cycle_result = self.run_once()

                if cycle_result["processing_result"]["processed"] > 0:
                    print(f"✅ ASYNC PROCESSOR: Cycle completed - {cycle_result['processing_result']['message']}")

                # Sleep until next cycle
                time.sleep(self.processing_interval)

        except KeyboardInterrupt:
            print("🛑 ASYNC PROCESSOR: Stopping due to keyboard interrupt")
        except Exception as e:
            print(f"❌ ASYNC PROCESSOR: Fatal error: {e}")
            raise


def main():
    """Main entry point for running the async memory processor."""
    import argparse

    parser = argparse.ArgumentParser(description="Asynchronous Memory Processor")
    parser.add_argument("--interval", type=int, default=60,
                       help="Processing interval in seconds (default: 60)")
    parser.add_argument("--retention-days", type=int, default=30,
                       help="Days to retain raw transcripts (default: 30)")
    parser.add_argument("--redis-host", default=None,
                       help="Redis host (default: from env or localhost)")
    parser.add_argument("--redis-port", type=int, default=None,
                       help="Redis port (default: from env or 6379)")
    parser.add_argument("--redis-db", type=int, default=None,
                       help="Redis database (default: from env or 0)")
    parser.add_argument("--run-once", action="store_true",
                       help="Run one processing cycle and exit")

    args = parser.parse_args()

    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ASYNC PROCESSOR: OPENAI_API_KEY not found in environment variables.")
        print("Please set your OpenAI API key in the .env file or environment.")
        return 1

    # Create processor
    processor = AsyncMemoryProcessor(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_db=args.redis_db,
        processing_interval=args.interval,
        retention_days=args.retention_days
    )

    if args.run_once:
        print("🔄 ASYNC PROCESSOR: Running single processing cycle")
        result = processor.run_once()
        print(f"✅ ASYNC PROCESSOR: Cycle completed: {result['processing_result']['message']}")
        if result["cleanup_result"]:
            print(f"🧹 ASYNC PROCESSOR: Cleanup: {result['cleanup_result']}")
    else:
        processor.run_continuous()


if __name__ == "__main__":
    main()

"""
Memory operations service for business logic.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timezone
import redis
from ..models.memory import MemoryStoreRequest, MemorySearchRequest, ContextSetRequest
from ..core.exceptions import server_error, validation_error
from ..core.utils import get_current_timestamp


class MemoryService:
    """Service for memory operations."""
    
    def __init__(self, memory_agent):
        self.memory_agent = memory_agent
    
    async def store_memory(self, vectorstore_name: str, request: MemoryStoreRequest) -> Dict[str, Any]:
        """Store a memory in the specified vectorstore."""
        try:
            memory_text = request.text.strip()
            apply_grounding = request.apply_grounding

            if not memory_text:
                raise validation_error('Memory text is required')

            print(f"💾 MEMORY API: Storing atomic memory - '{memory_text[:60]}{'...' if len(memory_text) > 60 else ''}'")
            print(f"📦 Vectorstore: {vectorstore_name}")

            storage_result = self.memory_agent.memory_agent.store_memory(
                memory_text,
                apply_grounding=apply_grounding,
                vectorset_key=vectorstore_name
            )

            # Prepare response with grounding information
            response_data = {
                'success': True,
                'memory_id': storage_result['memory_id'],
                'message': 'Memory stored successfully',
                'original_text': storage_result['original_text'],
                'final_text': storage_result['final_text'],
                'grounding_applied': storage_result['grounding_applied'],
                'tags': storage_result['tags'],
                'created_at': storage_result['created_at'],
                'vectorstore_name': vectorstore_name
            }

            # Include grounding information if available
            if 'grounding_info' in storage_result:
                response_data['grounding_info'] = storage_result['grounding_info']

            # Include context snapshot if available
            if 'context_snapshot' in storage_result:
                response_data['context_snapshot'] = storage_result['context_snapshot']

            return response_data

        except Exception as e:
            raise server_error(str(e))
    
    async def search_memories(self, vectorstore_name: str, request: MemorySearchRequest) -> Dict[str, Any]:
        """Search memories in the specified vectorstore."""
        try:
            query = request.query.strip()
            top_k = request.top_k
            filter_expr = request.filter
            optimize_query = request.optimize_query
            min_similarity = request.min_similarity

            if not query:
                raise validation_error('Query is required')

            print(f"🔍 MEMORY API Searching memories: {query} (top_k: {top_k}, min_similarity: {min_similarity})")
            print(f"📦 Vectorstore: {vectorstore_name}")
            if filter_expr:
                print(f"🔍 Filter: {filter_expr}")
            if optimize_query:
                print(f"🔍 Query optimization: enabled")

            # Use the memory agent for search operations with optional optimization
            if optimize_query:
                validation_result = self.memory_agent.memory_agent.processing.validate_and_preprocess_question(query)
                if validation_result["type"] == "search":
                    search_query = validation_result.get("embedding_query") or validation_result["content"]
                    print(f"🔍 Using optimized search query: '{search_query}'")
                    search_result = self.memory_agent.memory_agent.search_memories_with_filtering_info(
                        search_query, top_k, filter_expr, min_similarity, vectorset_key=vectorstore_name
                    )
                else:
                    search_result = self.memory_agent.memory_agent.search_memories_with_filtering_info(
                        query, top_k, filter_expr, min_similarity, vectorset_key=vectorstore_name
                    )
            else:
                search_result = self.memory_agent.memory_agent.search_memories_with_filtering_info(
                    query, top_k, filter_expr, min_similarity, vectorset_key=vectorstore_name
                )

            memories = search_result['memories']
            filtering_info = search_result['filtering_info']
            print(f"🔍 MEMORY API Search result type: {type(search_result)}")
            print(f"🔍 MEMORY API Filtering info: {filtering_info}")

            return {
                'success': True,
                'query': query,
                'memories': memories,
                'count': len(memories),
                'filtering_info': filtering_info,
                'vectorstore_name': vectorstore_name
            }

        except Exception as e:
            raise server_error(str(e))

    async def get_memory_info(self, vectorstore_name: str) -> Dict[str, Any]:
        """Get memory statistics and system information for a vectorstore."""
        try:
            redis_client = self.memory_agent.memory_agent.core.redis_client

            try:
                # Get memory count using VCARD for the specific vectorstore
                memory_count = redis_client.execute_command("VCARD", vectorstore_name)
                memory_count = int(memory_count) if memory_count else 0

                # Get vector dimension using VDIM for the specific vectorstore
                dimension = redis_client.execute_command("VDIM", vectorstore_name)
                dimension = int(dimension) if dimension else 0

                # Get detailed vector set info using VINFO for the specific vectorstore
                vinfo_result = redis_client.execute_command("VINFO", vectorstore_name)

                # Parse VINFO result (returns key-value pairs)
                vinfo_dict = {}
                if vinfo_result:
                    for i in range(0, len(vinfo_result), 2):
                        if i + 1 < len(vinfo_result):
                            key = vinfo_result[i].decode('utf-8') if isinstance(vinfo_result[i], bytes) else vinfo_result[i]
                            value = vinfo_result[i + 1]
                            # Try to decode bytes, but keep original type for numbers
                            if isinstance(value, bytes):
                                try:
                                    value = value.decode('utf-8')
                                except:
                                    value = str(value)
                            vinfo_dict[key] = value

                memory_info = {
                    'memory_count': memory_count,
                    'vector_dimension': dimension,
                    'vectorset_name': vectorstore_name,
                    'vectorset_info': vinfo_dict,
                    'embedding_model': 'text-embedding-ada-002',
                    'redis_host': redis_client.connection_pool.connection_kwargs.get('host', 'unknown'),
                    'redis_port': redis_client.connection_pool.connection_kwargs.get('port', 'unknown'),
                    'timestamp': get_current_timestamp()
                }

            except redis.ResponseError:
                # VectorSet doesn't exist yet - this is normal when no memories have been stored
                memory_info = {
                    'memory_count': 0,
                    'vector_dimension': 0,
                    'vectorset_name': vectorstore_name,
                    'vectorset_info': {},
                    'embedding_model': 'text-embedding-ada-002',
                    'redis_host': redis_client.connection_pool.connection_kwargs.get('host', 'unknown'),
                    'redis_port': redis_client.connection_pool.connection_kwargs.get('port', 'unknown'),
                    'timestamp': get_current_timestamp(),
                    'note': 'No memories stored yet - VectorSet will be created when first memory is added'
                }

            return {
                'success': True,
                **memory_info
            }

        except Exception as e:
            raise server_error(str(e))

    async def delete_memory(self, vectorstore_name: str, memory_id: str) -> Dict[str, Any]:
        """Delete a specific memory by ID from a vectorstore."""
        try:
            if not memory_id or not memory_id.strip():
                raise validation_error('Memory ID is required')

            print(f"🗑️ MEMORY API Deleting atomic memory: {memory_id}")
            print(f"📦 Vectorstore: {vectorstore_name}")

            success = self.memory_agent.memory_agent.delete_memory(
                memory_id.strip(),
                vectorset_key=vectorstore_name
            )

            if success:
                return {
                    'success': True,
                    'message': f'Memory {memory_id} deleted successfully',
                    'memory_id': memory_id,
                    'vectorstore_name': vectorstore_name
                }
            else:
                raise validation_error(
                    f'Memory {memory_id} not found or could not be deleted'
                )

        except Exception as e:
            raise server_error(str(e))

    async def delete_all_memories(self, vectorstore_name: str) -> Dict[str, Any]:
        """Clear all memories from a vectorstore."""
        try:
            print("🗑️ MEMORY API Clearing all atomic memories...")
            print(f"📦 Vectorstore: {vectorstore_name}")

            result = self.memory_agent.memory_agent.clear_all_memories(vectorset_key=vectorstore_name)

            if result['success']:
                return {
                    'success': True,
                    'message': result['message'],
                    'memories_deleted': result['memories_deleted'],
                    'vectorset_existed': result['vectorset_existed'],
                    'vectorstore_name': vectorstore_name
                }
            else:
                raise server_error(result['error'])

        except Exception as e:
            raise server_error(str(e))

    async def set_context(self, vectorstore_name: str, request: ContextSetRequest) -> Dict[str, Any]:
        """Set current context for memory grounding in a vectorstore."""
        try:
            # Extract context parameters
            location = request.location
            activity = request.activity
            people_present = request.people_present or []

            print(f"🌍 MEMORY API Setting context - Location: {location}, Activity: {activity}, People: {people_present}")
            print(f"📦 Vectorstore: {vectorstore_name}")

            # Set context on underlying memory agent
            self.memory_agent.memory_agent.set_context(
                location=location,
                activity=activity,
                people_present=people_present if people_present else None
            )

            return {
                'success': True,
                'message': 'Context updated successfully',
                'context': {
                    'location': location,
                    'activity': activity,
                    'people_present': people_present
                },
                'vectorstore_name': vectorstore_name
            }

        except Exception as e:
            raise server_error(str(e))

    async def get_context(self, vectorstore_name: str) -> Dict[str, Any]:
        """Get current context information for memory grounding from a vectorstore."""
        try:
            current_context = self.memory_agent.memory_agent.core._get_current_context()

            return {
                'success': True,
                'context': current_context,
                'vectorstore_name': vectorstore_name
            }

        except Exception as e:
            raise server_error(str(e))

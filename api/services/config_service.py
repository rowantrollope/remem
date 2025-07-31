"""
Configuration service for managing system configuration.
"""

from typing import Dict, Any, List, Tuple
from ..core.config import get_config, update_config
from ..core.utils import safe_config_copy, get_current_timestamp, validate_positive_integer, validate_float_range
from ..core.exceptions import server_error, validation_error


class ConfigService:
    """Service for configuration management."""
    
    def __init__(self, memory_agent=None):
        self.memory_agent = memory_agent
    
    async def get_configuration(self) -> Dict[str, Any]:
        """Get current system configuration with masked sensitive data."""
        try:
            config = get_config()
            safe_config = safe_config_copy(config)
            
            # Add runtime information
            runtime_info = {
                "memory_agent_initialized": self.memory_agent is not None,
                "timestamp": get_current_timestamp()
            }

            # Add LLM manager status
            try:
                from llm.llm_manager import get_llm_manager
                llm_mgr = get_llm_manager()
                runtime_info["llm_manager_initialized"] = True
                runtime_info["llm_tier1_provider"] = llm_mgr.tier1_config.provider
                runtime_info["llm_tier2_provider"] = llm_mgr.tier2_config.provider
            except Exception:
                runtime_info["llm_manager_initialized"] = False

            if self.memory_agent:
                try:
                    memory_info = self.memory_agent.memory_agent.get_memory_info()
                    runtime_info["memory_count"] = memory_info.get("memory_count", 0)
                    runtime_info["redis_connected"] = True
                    runtime_info["actual_redis_host"] = memory_info.get("redis_host", "unknown")
                    runtime_info["actual_redis_port"] = memory_info.get("redis_port", "unknown")
                except Exception as e:
                    runtime_info["redis_connected"] = False
                    runtime_info["redis_error"] = str(e)

            return {
                'success': True,
                'config': safe_config,
                'runtime': runtime_info
            }

        except Exception as e:
            raise server_error(str(e))
    
    async def update_configuration(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update system configuration."""
        try:
            if not updates:
                raise validation_error('Configuration data is required')

            warnings = []
            changes_made = []
            requires_restart = False

            config = get_config()

            # Update Redis configuration
            if 'redis' in updates:
                redis_config = updates['redis']
                redis_changes = self._update_redis_config(config, redis_config)
                changes_made.extend(redis_changes)
                if redis_changes:
                    requires_restart = True

            # Update LLM configuration
            if 'llm' in updates:
                llm_config = updates['llm']
                llm_changes = self._update_llm_config(config, llm_config)
                changes_made.extend(llm_changes)

            # Update OpenAI configuration
            if 'openai' in updates:
                openai_config = updates['openai']
                openai_changes = self._update_openai_config(config, openai_config)
                changes_made.extend(openai_changes)
                if openai_changes:
                    requires_restart = True

            # Update Embedding configuration
            if 'embedding' in updates:
                embedding_config = updates['embedding']
                embedding_changes = self._update_embedding_config(config, embedding_config)
                changes_made.extend(embedding_changes)
                if embedding_changes:
                    requires_restart = True

            # Update LangGraph configuration
            if 'langgraph' in updates:
                langgraph_config = updates['langgraph']
                langgraph_changes = self._update_langgraph_config(config, langgraph_config)
                changes_made.extend(langgraph_changes)
                if langgraph_changes:
                    requires_restart = True

            # Update Memory Agent configuration
            if 'memory_agent' in updates:
                memory_config = updates['memory_agent']
                memory_changes = self._update_memory_agent_config(config, memory_config)
                changes_made.extend(memory_changes)

            # Update Web Server configuration
            if 'web_server' in updates:
                web_config = updates['web_server']
                web_changes, web_warnings = self._update_web_server_config(config, web_config)
                changes_made.extend(web_changes)
                warnings.extend(web_warnings)

            # Update LangCache configuration
            if 'langcache' in updates:
                langcache_config = updates['langcache']
                langcache_changes = self._update_langcache_config(config, langcache_config)
                changes_made.extend(langcache_changes)

            # Check if LLM configuration was changed and reinitialize if needed
            llm_reinitialized = False
            llm_reinit_error = None
            llm_config_changed = any('llm.' in change for change in changes_made)
            
            if llm_config_changed:
                print(f"🔄 LLM configuration changed, reinitializing LLM manager...")
                success, message = self._reinitialize_llm_manager()
                llm_reinitialized = success
                if not success:
                    llm_reinit_error = message
                    warnings.append(f"LLM reinitialization failed: {message}")
                else:
                    # If LLM reinitialization was successful, we don't need a full restart for LLM changes
                    requires_restart = any(change for change in changes_made if not change.startswith('llm.'))

            # Prepare response
            response_data = {
                'success': True,
                'changes_made': changes_made,
                'requires_restart': requires_restart,
                'warnings': warnings
            }

            # Add LLM reinitialization info if LLM config was changed
            if llm_config_changed:
                response_data['llm_reinitialized'] = llm_reinitialized
                if llm_reinit_error:
                    response_data['llm_reinit_error'] = llm_reinit_error

            if requires_restart:
                response_data['message'] = 'Configuration updated. Memory agent restart required for changes to take effect.'
            elif changes_made and llm_config_changed and llm_reinitialized:
                response_data['message'] = 'Configuration updated and LLM changes applied successfully.'
            elif changes_made and llm_config_changed and not llm_reinitialized:
                response_data['message'] = 'Configuration updated but LLM changes failed to apply. Manual restart may be required.'
            elif changes_made:
                response_data['message'] = 'Configuration updated successfully.'
            else:
                response_data['message'] = 'No changes were made to the configuration.'

            return response_data

        except Exception as e:
            raise server_error(str(e))

    async def test_configuration(self, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Test configuration without applying changes."""
        try:
            test_results = {
                'valid': True,
                'tests': {},
                'errors': [],
                'warnings': []
            }

            # Test embedding configuration if provided
            if 'embedding' in test_config:
                embedding_test = await self._test_embedding_config(test_config['embedding'])
                test_results['tests']['embedding'] = embedding_test
                if not embedding_test['valid']:
                    test_results['valid'] = False
                    test_results['errors'].extend(embedding_test.get('errors', []))

            return {
                'success': True,
                'test_results': test_results,
                'timestamp': get_current_timestamp()
            }

        except Exception as e:
            raise server_error(str(e))
    
    def _update_redis_config(self, config: Dict[str, Any], redis_config: Dict[str, Any]) -> List[str]:
        """Update Redis configuration section."""
        changes = []
        for key in ['host', 'port', 'db', 'vectorset_key']:
            if key in redis_config:
                old_value = config['redis'].get(key)
                new_value = redis_config[key]

                # Validate port and db are integers
                if key in ['port', 'db']:
                    new_value = validate_positive_integer(new_value, f'Redis {key}')

                if old_value != new_value:
                    config['redis'][key] = new_value
                    changes.append(f"redis.{key}: {old_value} → {new_value}")
        
        return changes
    
    def _update_llm_config(self, config: Dict[str, Any], llm_config: Dict[str, Any]) -> List[str]:
        """Update LLM configuration section."""
        changes = []
        for tier in ['tier1', 'tier2']:
            if tier in llm_config:
                tier_config = llm_config[tier]

                # Handle string fields
                for key in ['provider', 'model', 'base_url']:
                    if key in tier_config:
                        old_value = config['llm'][tier].get(key)
                        new_value = tier_config[key]

                        if old_value != new_value:
                            config['llm'][tier][key] = new_value
                            changes.append(f"llm.{tier}.{key}: {old_value} → {new_value}")

                # Handle API key with masking
                if 'api_key' in tier_config:
                    old_value = config['llm'][tier].get('api_key')
                    new_value = tier_config['api_key']

                    if old_value != new_value:
                        config['llm'][tier]['api_key'] = new_value
                        from ..core.utils import mask_api_key
                        masked_old = mask_api_key(old_value)
                        masked_new = mask_api_key(new_value)
                        changes.append(f"llm.{tier}.api_key: {masked_old} → {masked_new}")

                # Handle numeric fields
                for key in ['temperature', 'max_tokens', 'timeout']:
                    if key in tier_config:
                        old_value = config['llm'][tier].get(key)
                        if key == 'temperature':
                            new_value = validate_float_range(tier_config[key], f'LLM {tier}.{key}')
                        else:
                            new_value = validate_positive_integer(tier_config[key], f'LLM {tier}.{key}')

                        if old_value != new_value:
                            config['llm'][tier][key] = new_value
                            changes.append(f"llm.{tier}.{key}: {old_value} → {new_value}")

                # Validate provider
                if 'provider' in tier_config:
                    provider = tier_config['provider'].lower()
                    if provider not in ['openai', 'ollama']:
                        raise validation_error(f'LLM provider must be "openai" or "ollama", got "{provider}"')
        
        return changes

    def _update_openai_config(self, config: Dict[str, Any], openai_config: Dict[str, Any]) -> List[str]:
        """Update OpenAI configuration section."""
        changes = []
        for key in ['api_key', 'organization', 'embedding_model', 'chat_model']:
            if key in openai_config:
                old_value = config['openai'].get(key)
                new_value = openai_config[key]

                if old_value != new_value:
                    config['openai'][key] = new_value
                    if key == 'api_key':
                        # Mask API key in logs
                        from ..core.utils import mask_api_key
                        masked_old = mask_api_key(old_value)
                        masked_new = mask_api_key(new_value)
                        changes.append(f"openai.{key}: {masked_old} → {masked_new}")
                    else:
                        changes.append(f"openai.{key}: {old_value} → {new_value}")

        # Handle numeric fields
        for key in ['embedding_dimension', 'temperature']:
            if key in openai_config:
                old_value = config['openai'].get(key)
                if key == 'temperature':
                    new_value = validate_float_range(openai_config[key], f'OpenAI {key}')
                else:
                    new_value = validate_positive_integer(openai_config[key], f'OpenAI {key}')

                if old_value != new_value:
                    config['openai'][key] = new_value
                    changes.append(f"openai.{key}: {old_value} → {new_value}")

        return changes

    def _update_embedding_config(self, config: Dict[str, Any], embedding_config: Dict[str, Any]) -> List[str]:
        """Update Embedding configuration section."""
        changes = []

        # Handle string fields
        for key in ['provider', 'model', 'base_url', 'api_key']:
            if key in embedding_config:
                old_value = config['embedding'].get(key)
                new_value = embedding_config[key]

                if old_value != new_value:
                    config['embedding'][key] = new_value
                    if key == 'api_key':
                        # Mask API key in logs
                        from ..core.utils import mask_api_key
                        masked_old = mask_api_key(old_value)
                        masked_new = mask_api_key(new_value)
                        changes.append(f"embedding.{key}: {masked_old} → {masked_new}")
                    else:
                        changes.append(f"embedding.{key}: {old_value} → {new_value}")

        # Handle numeric fields
        for key in ['dimension', 'timeout']:
            if key in embedding_config:
                old_value = config['embedding'].get(key)
                new_value = validate_positive_integer(embedding_config[key], f'Embedding {key}')

                if old_value != new_value:
                    config['embedding'][key] = new_value
                    changes.append(f"embedding.{key}: {old_value} → {new_value}")

        # Validate provider
        if 'provider' in embedding_config:
            provider = embedding_config['provider'].lower()
            if provider not in ['openai', 'ollama']:
                raise validation_error(f'Embedding provider must be "openai" or "ollama", got "{provider}"')

        return changes

    async def _test_embedding_config(self, embedding_config: Dict[str, Any]) -> Dict[str, Any]:
        """Test embedding configuration."""
        test_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'connection_test': None
        }

        try:
            # Import embedding modules
            from embedding import EmbeddingConfig, create_embedding_client

            # Create embedding config from test data
            config = EmbeddingConfig(
                provider=embedding_config.get('provider', 'openai'),
                model=embedding_config.get('model', 'text-embedding-ada-002'),
                dimension=embedding_config.get('dimension', 1536),
                base_url=embedding_config.get('base_url'),
                api_key=embedding_config.get('api_key'),
                timeout=embedding_config.get('timeout', 30)
            )

            # Validate provider
            if config.provider.lower() not in ['openai', 'ollama']:
                test_result['valid'] = False
                test_result['errors'].append(f'Invalid embedding provider: {config.provider}')
                return test_result

            # Create and test embedding client
            embedding_client = create_embedding_client(config)
            connection_test = embedding_client.test_connection()
            test_result['connection_test'] = connection_test

            if not connection_test['success']:
                test_result['valid'] = False
                test_result['errors'].append(f"Embedding connection failed: {connection_test.get('error', 'Unknown error')}")

            # Check dimension mismatch
            if connection_test.get('dimension_match') is False:
                test_result['warnings'].append(
                    f"Dimension mismatch: configured {config.dimension}, actual {connection_test.get('actual_dimension')}"
                )

        except Exception as e:
            test_result['valid'] = False
            test_result['errors'].append(f"Embedding test error: {str(e)}")

        return test_result

    def _update_langgraph_config(self, config: Dict[str, Any], langgraph_config: Dict[str, Any]) -> List[str]:
        """Update LangGraph configuration section."""
        changes = []
        for key in ['model_name']:
            if key in langgraph_config:
                old_value = config['langgraph'].get(key)
                new_value = langgraph_config[key]

                if old_value != new_value:
                    config['langgraph'][key] = new_value
                    changes.append(f"langgraph.{key}: {old_value} → {new_value}")

        # Handle numeric and boolean fields
        if 'temperature' in langgraph_config:
            old_value = config['langgraph'].get('temperature')
            new_value = validate_float_range(langgraph_config['temperature'], 'LangGraph temperature')

            if old_value != new_value:
                config['langgraph']['temperature'] = new_value
                changes.append(f"langgraph.temperature: {old_value} → {new_value}")

        if 'system_prompt_enabled' in langgraph_config:
            old_value = config['langgraph'].get('system_prompt_enabled')
            new_value = bool(langgraph_config['system_prompt_enabled'])

            if old_value != new_value:
                config['langgraph']['system_prompt_enabled'] = new_value
                changes.append(f"langgraph.system_prompt_enabled: {old_value} → {new_value}")

        return changes

    def _update_memory_agent_config(self, config: Dict[str, Any], memory_config: Dict[str, Any]) -> List[str]:
        """Update Memory Agent configuration section."""
        changes = []

        if 'default_top_k' in memory_config:
            old_value = config['memory_agent'].get('default_top_k')
            new_value = validate_positive_integer(memory_config['default_top_k'], 'Memory agent default_top_k')

            if old_value != new_value:
                config['memory_agent']['default_top_k'] = new_value
                changes.append(f"memory_agent.default_top_k: {old_value} → {new_value}")

        for key in ['apply_grounding_default', 'validation_enabled']:
            if key in memory_config:
                old_value = config['memory_agent'].get(key)
                new_value = bool(memory_config[key])

                if old_value != new_value:
                    config['memory_agent'][key] = new_value
                    changes.append(f"memory_agent.{key}: {old_value} → {new_value}")

        return changes

    def _update_web_server_config(self, config: Dict[str, Any], web_config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Update Web Server configuration section."""
        changes = []
        warnings = []

        for key in ['host']:
            if key in web_config:
                old_value = config['web_server'].get(key)
                new_value = web_config[key]

                if old_value != new_value:
                    config['web_server'][key] = new_value
                    changes.append(f"web_server.{key}: {old_value} → {new_value}")
                    warnings.append(f"Web server {key} change requires application restart to take effect")

        if 'port' in web_config:
            old_value = config['web_server'].get('port')
            new_value = validate_positive_integer(web_config['port'], 'Web server port')

            if old_value != new_value:
                config['web_server']['port'] = new_value
                changes.append(f"web_server.port: {old_value} → {new_value}")
                warnings.append("Web server port change requires application restart to take effect")

        for key in ['debug', 'cors_enabled']:
            if key in web_config:
                old_value = config['web_server'].get(key)
                new_value = bool(web_config[key])

                if old_value != new_value:
                    config['web_server'][key] = new_value
                    changes.append(f"web_server.{key}: {old_value} → {new_value}")
                    warnings.append(f"Web server {key} change requires application restart to take effect")

        return changes, warnings

    def _update_langcache_config(self, config: Dict[str, Any], langcache_config: Dict[str, Any]) -> List[str]:
        """Update LangCache configuration section."""
        changes = []

        # Update master enabled flag
        if 'enabled' in langcache_config:
            old_value = config['langcache'].get('enabled')
            new_value = bool(langcache_config['enabled'])

            if old_value != new_value:
                config['langcache']['enabled'] = new_value
                changes.append(f"langcache.enabled: {old_value} → {new_value}")

        # Update individual cache type settings
        if 'cache_types' in langcache_config:
            cache_types = langcache_config['cache_types']
            if isinstance(cache_types, dict):
                for cache_type, enabled in cache_types.items():
                    if cache_type in config['langcache']['cache_types']:
                        old_value = config['langcache']['cache_types'].get(cache_type)
                        new_value = bool(enabled)

                        if old_value != new_value:
                            config['langcache']['cache_types'][cache_type] = new_value
                            changes.append(f"langcache.cache_types.{cache_type}: {old_value} → {new_value}")

        # Update minimum similarity threshold
        if 'minimum_similarity' in langcache_config:
            old_value = config['langcache'].get('minimum_similarity')
            new_value = float(langcache_config['minimum_similarity'])

            # Validate range
            if not (0.0 <= new_value <= 1.0):
                raise ValueError(f"minimum_similarity must be between 0.0 and 1.0, got {new_value}")

            if old_value != new_value:
                config['langcache']['minimum_similarity'] = new_value
                changes.append(f"langcache.minimum_similarity: {old_value} → {new_value}")

        # Update TTL setting
        if 'ttl_minutes' in langcache_config:
            old_value = config['langcache'].get('ttl_minutes')
            new_value = float(langcache_config['ttl_minutes'])

            # Validate range (allow 0.1 minutes to 1440 minutes = 24 hours)
            if not (0.1 <= new_value <= 1440):
                raise ValueError(f"ttl_minutes must be between 0.1 and 1440 (24 hours), got {new_value}")

            if old_value != new_value:
                config['langcache']['ttl_minutes'] = new_value
                changes.append(f"langcache.ttl_minutes: {old_value} → {new_value}")

        return changes

    def _reinitialize_llm_manager(self) -> Tuple[bool, str]:
        """Reinitialize LLM manager with current configuration."""
        try:
            from ..startup import init_llm_manager
            if not init_llm_manager():
                return False, "Failed to reinitialize LLM manager"

            print("✅ LLM manager reinitialized successfully")
            return True, "LLM manager reinitialized successfully"
        except Exception as e:
            error_msg = f"Failed to reinitialize LLM manager: {e}"
            print(f"❌ {error_msg}")
            return False, error_msg

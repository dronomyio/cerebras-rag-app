"""
LLM Provider Factory
------------------
This module provides a factory for creating LLM provider instances.
"""

import os
import logging
import importlib
from typing import Dict, List, Any, Optional, Type

from .base import BaseLLMProvider

logger = logging.getLogger(__name__)

class LLMProviderFactory:
    """
    Factory class for creating and managing LLM provider instances.
    
    This class handles provider registration, instantiation, and fallback logic.
    """
    
    # Registry of available provider classes
    _provider_registry: Dict[str, Type[BaseLLMProvider]] = {}
    
    # Cache of provider instances
    _provider_instances: Dict[str, BaseLLMProvider] = {}
    
    # Current active provider name
    _active_provider: str = None
    
    # Fallback provider order
    _fallback_order: List[str] = []
    
    @classmethod
    def register_provider(cls, name: str, provider_class: Type[BaseLLMProvider]) -> None:
        """
        Register a provider class with the factory.
        
        Args:
            name: Provider name (e.g., 'cerebras', 'openai')
            provider_class: Provider class that inherits from BaseLLMProvider
        """
        cls._provider_registry[name] = provider_class
        logger.info(f"Registered LLM provider: {name}")
    
    @classmethod
    def create_provider(cls, name: str, config: Dict[str, Any] = None) -> Optional[BaseLLMProvider]:
        """
        Create a provider instance.
        
        Args:
            name: Provider name
            config: Provider configuration
            
        Returns:
            Provider instance or None if creation fails
        """
        if name not in cls._provider_registry:
            logger.error(f"Unknown LLM provider: {name}")
            return None
        
        try:
            provider_class = cls._provider_registry[name]
            provider = provider_class(config)
            cls._provider_instances[name] = provider
            logger.info(f"Created LLM provider instance: {name}")
            return provider
        except Exception as e:
            logger.error(f"Failed to create LLM provider {name}: {e}")
            return None
    
    @classmethod
    def get_provider(cls, name: str = None, config: Dict[str, Any] = None) -> Optional[BaseLLMProvider]:
        """
        Get a provider instance, creating it if necessary.
        
        Args:
            name: Provider name (uses active provider if None)
            config: Provider configuration for creation or update
            
        Returns:
            Provider instance or None if unavailable
        """
        # Use active provider if name is not specified
        if name is None:
            name = cls._active_provider or cls._get_default_provider_name()
            if name is None:
                logger.error("No active or default provider available")
                return None
        
        # Return cached instance if available and no new config
        if name in cls._provider_instances and config is None:
            return cls._provider_instances[name]
        
        # Create new instance if not cached
        if name not in cls._provider_instances:
            provider = cls.create_provider(name, config)
            if provider:
                return provider
        
        # Update existing instance with new config
        if name in cls._provider_instances and config is not None:
            cls._provider_instances[name].update_config(config)
            return cls._provider_instances[name]
        
        return None
    
    @classmethod
    def set_active_provider(cls, name: str, config: Dict[str, Any] = None) -> bool:
        """
        Set the active provider.
        
        Args:
            name: Provider name
            config: Provider configuration
            
        Returns:
            True if successful, False otherwise
        """
        provider = cls.get_provider(name, config)
        if provider and provider.is_available():
            cls._active_provider = name
            logger.info(f"Set active LLM provider to: {name}")
            return True
        
        logger.error(f"Failed to set active provider to {name}: Provider unavailable")
        return False
    
    @classmethod
    def get_active_provider(cls) -> Optional[BaseLLMProvider]:
        """
        Get the current active provider instance.
        
        Returns:
            Active provider instance or None if unavailable
        """
        return cls.get_provider(cls._active_provider)
    
    @classmethod
    def set_fallback_order(cls, provider_names: List[str]) -> None:
        """
        Set the fallback order for providers.
        
        Args:
            provider_names: List of provider names in fallback order
        """
        # Validate provider names
        valid_names = [name for name in provider_names if name in cls._provider_registry]
        cls._fallback_order = valid_names
        logger.info(f"Set fallback order: {valid_names}")
    
    @classmethod
    def generate_with_fallback(cls, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a response using the active provider with fallback.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Returns:
            Response dictionary with provider info and result
        """
        # Try active provider first
        active_provider = cls.get_active_provider()
        if active_provider and active_provider.is_available():
            try:
                result = active_provider.generate(prompt, **kwargs)
                return {
                    "provider": cls._active_provider,
                    "fallback_used": False,
                    "result": result
                }
            except Exception as e:
                logger.error(f"Active provider {cls._active_provider} failed: {e}")
        
        # Try fallback providers in order
        for provider_name in cls._fallback_order:
            if provider_name == cls._active_provider:
                continue  # Skip active provider that already failed
                
            provider = cls.get_provider(provider_name)
            if provider and provider.is_available():
                try:
                    result = provider.generate(prompt, **kwargs)
                    return {
                        "provider": provider_name,
                        "fallback_used": True,
                        "result": result
                    }
                except Exception as e:
                    logger.error(f"Fallback provider {provider_name} failed: {e}")
        
        # All providers failed
        return {
            "provider": None,
            "fallback_used": True,
            "result": {
                "text": "All LLM providers failed to generate a response.",
                "error": "No available providers"
            }
        }
    
    @classmethod
    def _get_default_provider_name(cls) -> Optional[str]:
        """
        Get the default provider name from environment or first registered.
        
        Returns:
            Default provider name or None if no providers available
        """
        # Check environment variable
        default_provider = os.getenv("DEFAULT_LLM_PROVIDER")
        if default_provider and default_provider in cls._provider_registry:
            return default_provider
        
        # Use first registered provider
        if cls._provider_registry:
            return next(iter(cls._provider_registry))
        
        return None
    
    @classmethod
    def list_available_providers(cls) -> List[str]:
        """
        List all available provider names.
        
        Returns:
            List of available provider names
        """
        return list(cls._provider_registry.keys())
    
    @classmethod
    def load_providers_from_config(cls) -> None:
        """
        Load and initialize all providers from environment variables.
        """
        # Load provider configurations from environment
        providers_to_load = {
            "cerebras": {
                "api_key": os.getenv("CEREBRAS_API_KEY"),
                "api_url": os.getenv("CEREBRAS_API_URL", "https://api.cerebras.ai/v1/completions"),
                "model": os.getenv("CEREBRAS_MODEL", "cerebras/Cerebras-GPT-4.5-8B"),
                "max_tokens": int(os.getenv("CEREBRAS_MAX_TOKENS", "1024")),
                "temperature": float(os.getenv("CEREBRAS_TEMPERATURE", "0.7")),
            },
            "openai": {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "api_url": os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions"),
                "model": os.getenv("OPENAI_MODEL", "gpt-4"),
                "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "1024")),
                "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
            },
            "anthropic": {
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
                "api_url": os.getenv("ANTHROPIC_API_URL", "https://api.anthropic.com/v1/messages"),
                "model": os.getenv("ANTHROPIC_MODEL", "claude-3-opus-20240229"),
                "max_tokens": int(os.getenv("ANTHROPIC_MAX_TOKENS", "1024")),
                "temperature": float(os.getenv("ANTHROPIC_TEMPERATURE", "0.7")),
            },
            "huggingface": {
                "api_key": os.getenv("HUGGINGFACE_API_KEY"),
                "api_url": os.getenv("HUGGINGFACE_API_URL", "https://api-inference.huggingface.co/models"),
                "model": os.getenv("HUGGINGFACE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2"),
                "max_tokens": int(os.getenv("HUGGINGFACE_MAX_TOKENS", "1024")),
                "temperature": float(os.getenv("HUGGINGFACE_TEMPERATURE", "0.7")),
            }
        }
        
        # Create provider instances
        for name, config in providers_to_load.items():
            if name in cls._provider_registry and config.get("api_key"):
                cls.create_provider(name, config)
        
        # Set active provider from environment
        default_provider = os.getenv("DEFAULT_LLM_PROVIDER", "cerebras")
        if default_provider in cls._provider_instances:
            cls.set_active_provider(default_provider)
        
        # Set fallback order from environment
        fallback_str = os.getenv("LLM_FALLBACK_ORDER", "cerebras,openai,anthropic,huggingface")
        fallback_providers = [p.strip() for p in fallback_str.split(",")]
        cls.set_fallback_order(fallback_providers)

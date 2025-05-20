"""
Base LLM Provider Interface
--------------------------
This module defines the abstract base class for all LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    All LLM provider implementations must inherit from this class
    and implement the required methods.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the LLM provider with configuration.
        
        Args:
            config: Provider-specific configuration dictionary
        """
        self.config = config or {}
        self.api_key = self.config.get('api_key')
        self.api_url = self.config.get('api_url')
        self.model = self.config.get('model')
        self.max_tokens = self.config.get('max_tokens', 1024)
        self.temperature = self.config.get('temperature', 0.7)
        self.top_p = self.config.get('top_p', 0.9)
        
        # Initialize provider-specific settings
        self._initialize()
    
    @abstractmethod
    def _initialize(self) -> None:
        """
        Initialize provider-specific settings.
        Must be implemented by each provider.
        """
        pass
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Dictionary containing the generated response and metadata
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the provider is available and properly configured.
        
        Returns:
            True if the provider is available, False otherwise
        """
        pass
    
    def validate_config(self) -> Dict[str, Any]:
        """
        Validate the provider configuration.
        
        Returns:
            Dictionary with validation results
        """
        missing = []
        if not self.api_key:
            missing.append('api_key')
        if not self.api_url:
            missing.append('api_url')
        if not self.model:
            missing.append('model')
            
        return {
            'valid': len(missing) == 0,
            'missing': missing
        }
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update the provider configuration.
        
        Args:
            config: New configuration dictionary
        """
        self.config.update(config)
        
        # Update common parameters
        if 'api_key' in config:
            self.api_key = config['api_key']
        if 'api_url' in config:
            self.api_url = config['api_url']
        if 'model' in config:
            self.model = config['model']
        if 'max_tokens' in config:
            self.max_tokens = config['max_tokens']
        if 'temperature' in config:
            self.temperature = config['temperature']
        if 'top_p' in config:
            self.top_p = config['top_p']

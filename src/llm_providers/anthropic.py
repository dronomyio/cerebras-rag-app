"""
Anthropic LLM Provider
-------------------
Implementation of the Anthropic LLM provider.
"""

import logging
import requests
import json
from typing import Dict, List, Any, Optional

from .base import BaseLLMProvider

logger = logging.getLogger(__name__)

class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic LLM provider implementation.
    """
    
    def _initialize(self) -> None:
        """
        Initialize Anthropic-specific settings.
        """
        # Set default values if not provided
        if not self.api_url:
            self.api_url = "https://api.anthropic.com/v1/messages"
        if not self.model:
            self.model = "claude-3-opus-20240229"
        
        # Anthropic-specific parameters
        self.system_prompt = self.config.get('system_prompt', '')
    
    def is_available(self) -> bool:
        """
        Check if the Anthropic provider is available.
        
        Returns:
            True if available, False otherwise
        """
        if not self.api_key:
            logger.warning("Anthropic API key not provided")
            return False
            
        try:
            # Make a minimal API call to check availability
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"
            }
            
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 1
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=5
            )
            
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Anthropic provider not available: {e}")
            return False
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a response using the Anthropic API.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with generated text and metadata
        """
        # Override default parameters with kwargs
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        temperature = kwargs.get('temperature', self.temperature)
        top_p = kwargs.get('top_p', self.top_p)
        model = kwargs.get('model', self.model)
        system_prompt = kwargs.get('system_prompt', self.system_prompt)
        
        # Check for conversation history in kwargs
        messages = kwargs.get('messages', None)
        
        # If no messages provided, create a simple user message
        if not messages:
            messages = [{"role": "user", "content": prompt}]
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        
        # Add system prompt if provided
        if system_prompt:
            payload["system"] = system_prompt
        
        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in ['max_tokens', 'temperature', 'top_p', 'model', 'messages', 'system_prompt']:
                payload[key] = value
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get("content", [{}])[0].get("text", "").strip()
                
                return {
                    "text": text,
                    "model": model,
                    "provider": "anthropic",
                    "usage": {
                        "input_tokens": result.get("usage", {}).get("input_tokens", 0),
                        "output_tokens": result.get("usage", {}).get("output_tokens", 0)
                    },
                    "raw_response": result
                }
            else:
                error_msg = f"Anthropic API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return {
                    "text": f"Error: {error_msg}",
                    "error": error_msg,
                    "provider": "anthropic"
                }
                
        except Exception as e:
            error_msg = f"Error calling Anthropic API: {str(e)}"
            logger.error(error_msg)
            return {
                "text": f"Error: {error_msg}",
                "error": error_msg,
                "provider": "anthropic"
            }

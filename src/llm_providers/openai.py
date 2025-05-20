"""
OpenAI LLM Provider
-----------------
Implementation of the OpenAI LLM provider.
"""

import logging
import requests
from typing import Dict, List, Any, Optional

from .base import BaseLLMProvider

logger = logging.getLogger(__name__)

class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI LLM provider implementation.
    """
    
    def _initialize(self) -> None:
        """
        Initialize OpenAI-specific settings.
        """
        # Set default values if not provided
        if not self.api_url:
            self.api_url = "https://api.openai.com/v1/chat/completions"
        if not self.model:
            self.model = "gpt-4"
    
    def is_available(self) -> bool:
        """
        Check if the OpenAI provider is available.
        
        Returns:
            True if available, False otherwise
        """
        if not self.api_key:
            logger.warning("OpenAI API key not provided")
            return False
            
        try:
            # Make a minimal API call to check availability
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 1
                },
                timeout=5
            )
            
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"OpenAI provider not available: {e}")
            return False
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a response using the OpenAI API.
        
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
        
        # Check for conversation history in kwargs
        messages = kwargs.get('messages', None)
        
        # If no messages provided, create a simple user message
        if not messages:
            messages = [{"role": "user", "content": prompt}]
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        
        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in ['max_tokens', 'temperature', 'top_p', 'model', 'messages']:
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
                text = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                
                return {
                    "text": text,
                    "model": model,
                    "provider": "openai",
                    "usage": result.get("usage", {}),
                    "raw_response": result
                }
            else:
                error_msg = f"OpenAI API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return {
                    "text": f"Error: {error_msg}",
                    "error": error_msg,
                    "provider": "openai"
                }
                
        except Exception as e:
            error_msg = f"Error calling OpenAI API: {str(e)}"
            logger.error(error_msg)
            return {
                "text": f"Error: {error_msg}",
                "error": error_msg,
                "provider": "openai"
            }

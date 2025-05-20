"""
Cerebras LLM Provider
-------------------
Implementation of the Cerebras LLM provider.
"""

import logging
import requests
from typing import Dict, List, Any, Optional

from .base import BaseLLMProvider

logger = logging.getLogger(__name__)

class CerebrasProvider(BaseLLMProvider):
    """
    Cerebras LLM provider implementation.
    """
    
    def _initialize(self) -> None:
        """
        Initialize Cerebras-specific settings.
        """
        # Set default values if not provided
        if not self.api_url:
            self.api_url = "https://api.cerebras.ai/v1/completions"
        if not self.model:
            self.model = "cerebras/Cerebras-GPT-4.5-8B"
    
    def is_available(self) -> bool:
        """
        Check if the Cerebras provider is available.
        
        Returns:
            True if available, False otherwise
        """
        if not self.api_key:
            logger.warning("Cerebras API key not provided")
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
                    "prompt": "Hello",
                    "max_tokens": 1
                },
                timeout=5
            )
            
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Cerebras provider not available: {e}")
            return False
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a response using the Cerebras API.
        
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
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        
        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in ['max_tokens', 'temperature', 'top_p', 'model']:
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
                text = result.get("choices", [{}])[0].get("text", "").strip()
                
                return {
                    "text": text,
                    "model": model,
                    "provider": "cerebras",
                    "usage": result.get("usage", {}),
                    "raw_response": result
                }
            else:
                error_msg = f"Cerebras API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return {
                    "text": f"Error: {error_msg}",
                    "error": error_msg,
                    "provider": "cerebras"
                }
                
        except Exception as e:
            error_msg = f"Error calling Cerebras API: {str(e)}"
            logger.error(error_msg)
            return {
                "text": f"Error: {error_msg}",
                "error": error_msg,
                "provider": "cerebras"
            }

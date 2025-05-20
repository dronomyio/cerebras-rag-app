"""
Hugging Face LLM Provider
----------------------
Implementation of the Hugging Face LLM provider.
"""

import logging
import requests
from typing import Dict, List, Any, Optional

from .base import BaseLLMProvider

logger = logging.getLogger(__name__)

class HuggingFaceProvider(BaseLLMProvider):
    """
    Hugging Face LLM provider implementation.
    """
    
    def _initialize(self) -> None:
        """
        Initialize Hugging Face-specific settings.
        """
        # Set default values if not provided
        if not self.api_url:
            self.api_url = "https://api-inference.huggingface.co/models"
        if not self.model:
            self.model = "mistralai/Mistral-7B-Instruct-v0.2"
    
    def is_available(self) -> bool:
        """
        Check if the Hugging Face provider is available.
        
        Returns:
            True if available, False otherwise
        """
        if not self.api_key:
            logger.warning("Hugging Face API key not provided")
            return False
            
        try:
            # Make a minimal API call to check availability
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # Construct the full URL with the model
            url = f"{self.api_url}/{self.model}"
            
            response = requests.post(
                url,
                headers=headers,
                json={"inputs": "Hello", "parameters": {"max_new_tokens": 1}},
                timeout=5
            )
            
            # 200 means ready, 503 means model is loading but available
            return response.status_code in [200, 503]
        except Exception as e:
            logger.warning(f"Hugging Face provider not available: {e}")
            return False
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a response using the Hugging Face API.
        
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
        
        # Construct the full URL with the model
        url = f"{self.api_url}/{model}"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Format the prompt based on model type
        # This is a simplified approach - different models may need different formats
        if "mistral" in model.lower() or "llama" in model.lower():
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        elif "falcon" in model.lower():
            formatted_prompt = f"User: {prompt}\nAssistant:"
        else:
            formatted_prompt = prompt
        
        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "return_full_text": False
            }
        }
        
        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in ['max_tokens', 'temperature', 'top_p', 'model']:
                if key == 'stop':
                    payload["parameters"]["stop"] = value
                elif key == 'presence_penalty':
                    payload["parameters"]["repetition_penalty"] = value
                else:
                    payload["parameters"][key] = value
        
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Handle different response formats
                if isinstance(result, list) and len(result) > 0:
                    if "generated_text" in result[0]:
                        text = result[0]["generated_text"].strip()
                    else:
                        text = str(result[0]).strip()
                elif isinstance(result, dict) and "generated_text" in result:
                    text = result["generated_text"].strip()
                else:
                    text = str(result).strip()
                
                return {
                    "text": text,
                    "model": model,
                    "provider": "huggingface",
                    "raw_response": result
                }
            elif response.status_code == 503:
                # Model is loading
                error_msg = "Hugging Face model is loading. Please try again in a few moments."
                logger.warning(error_msg)
                return {
                    "text": error_msg,
                    "error": error_msg,
                    "provider": "huggingface"
                }
            else:
                error_msg = f"Hugging Face API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return {
                    "text": f"Error: {error_msg}",
                    "error": error_msg,
                    "provider": "huggingface"
                }
                
        except Exception as e:
            error_msg = f"Error calling Hugging Face API: {str(e)}"
            logger.error(error_msg)
            return {
                "text": f"Error: {error_msg}",
                "error": error_msg,
                "provider": "huggingface"
            }

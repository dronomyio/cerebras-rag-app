"""
Validation script for LLM provider switching and fallback.
"""

import os
import sys
import logging
import json
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import LLM provider modules
from llm_providers import LLMProviderFactory, BaseLLMProvider
from llm_providers.cerebras import CerebrasProvider
from llm_providers.openai import OpenAIProvider
from llm_providers.anthropic import AnthropicProvider
from llm_providers.huggingface import HuggingFaceProvider

def initialize_providers():
    """Initialize and register all LLM providers."""
    # Register provider classes
    LLMProviderFactory.register_provider("cerebras", CerebrasProvider)
    LLMProviderFactory.register_provider("openai", OpenAIProvider)
    LLMProviderFactory.register_provider("anthropic", AnthropicProvider)
    LLMProviderFactory.register_provider("huggingface", HuggingFaceProvider)
    
    # Load providers from environment variables
    LLMProviderFactory.load_providers_from_config()
    
    # Log available providers
    available_providers = LLMProviderFactory.list_available_providers()
    logger.info(f"Available LLM providers: {available_providers}")

def test_provider_switching():
    """Test switching between providers."""
    providers = ["cerebras", "openai", "anthropic", "huggingface"]
    results = {}
    
    for provider_name in providers:
        logger.info(f"Testing provider: {provider_name}")
        
        # Try to set active provider
        success = LLMProviderFactory.set_active_provider(provider_name)
        
        if success:
            logger.info(f"Successfully switched to {provider_name}")
            
            # Get active provider
            active_provider = LLMProviderFactory.get_active_provider()
            
            if active_provider:
                logger.info(f"Active provider is {provider_name}")
                results[provider_name] = {
                    "switching": "success",
                    "available": True
                }
            else:
                logger.error(f"Failed to get active provider {provider_name}")
                results[provider_name] = {
                    "switching": "success",
                    "available": False
                }
        else:
            logger.warning(f"Failed to switch to {provider_name}")
            results[provider_name] = {
                "switching": "failed",
                "available": False
            }
    
    return results

def test_provider_responses():
    """Test responses from each provider."""
    providers = ["cerebras", "openai", "anthropic", "huggingface"]
    test_prompt = "What is the capital of France?"
    results = {}
    
    for provider_name in providers:
        logger.info(f"Testing response from provider: {provider_name}")
        
        # Try to set active provider
        success = LLMProviderFactory.set_active_provider(provider_name)
        
        if success:
            try:
                # Get provider
                provider = LLMProviderFactory.get_provider(provider_name)
                
                # Generate response
                response = provider.generate(test_prompt)
                
                logger.info(f"Response from {provider_name}: {response.get('text', '')[:50]}...")
                results[provider_name] = {
                    "response": "success",
                    "text": response.get("text", "")[:100],
                    "provider": response.get("provider", "unknown")
                }
            except Exception as e:
                logger.error(f"Error generating response from {provider_name}: {e}")
                results[provider_name] = {
                    "response": "error",
                    "error": str(e)
                }
        else:
            logger.warning(f"Provider {provider_name} not available for testing")
            results[provider_name] = {
                "response": "unavailable"
            }
    
    return results

def test_fallback_mechanism():
    """Test the fallback mechanism."""
    # Set fallback order
    LLMProviderFactory.set_fallback_order(["cerebras", "openai", "anthropic", "huggingface"])
    
    # Set an unavailable provider as active
    LLMProviderFactory._active_provider = "unavailable_provider"
    
    # Try to generate with fallback
    logger.info("Testing fallback mechanism with unavailable primary provider")
    test_prompt = "What is the capital of France?"
    
    try:
        response = LLMProviderFactory.generate_with_fallback(test_prompt)
        
        logger.info(f"Fallback result: {json.dumps(response, indent=2)}")
        
        if response.get("fallback_used", False):
            logger.info(f"Fallback successfully used provider: {response.get('provider')}")
            return {
                "fallback": "success",
                "provider_used": response.get("provider"),
                "text": response.get("result", {}).get("text", "")[:100]
            }
        else:
            logger.warning("Fallback not used when it should have been")
            return {
                "fallback": "failed",
                "provider_used": response.get("provider")
            }
    except Exception as e:
        logger.error(f"Error testing fallback: {e}")
        return {
            "fallback": "error",
            "error": str(e)
        }

def test_parameter_consistency():
    """Test that parameters are respected across providers."""
    test_prompt = "What is the capital of France?"
    providers = ["cerebras", "openai", "anthropic", "huggingface"]
    results = {}
    
    for provider_name in providers:
        logger.info(f"Testing parameter consistency for provider: {provider_name}")
        
        # Try to set active provider
        success = LLMProviderFactory.set_active_provider(provider_name)
        
        if success:
            try:
                # Get provider
                provider = LLMProviderFactory.get_provider(provider_name)
                
                # Test with different temperatures
                temps = [0.2, 0.8]
                temp_responses = []
                
                for temp in temps:
                    response = provider.generate(test_prompt, temperature=temp)
                    temp_responses.append(response.get("text", "")[:50])
                
                # Check if responses are different (indicating temperature had an effect)
                different_responses = temp_responses[0] != temp_responses[1]
                
                results[provider_name] = {
                    "parameter_test": "success",
                    "temperature_effect": different_responses,
                    "responses": temp_responses
                }
            except Exception as e:
                logger.error(f"Error testing parameters for {provider_name}: {e}")
                results[provider_name] = {
                    "parameter_test": "error",
                    "error": str(e)
                }
        else:
            logger.warning(f"Provider {provider_name} not available for parameter testing")
            results[provider_name] = {
                "parameter_test": "unavailable"
            }
    
    return results

def run_validation():
    """Run all validation tests."""
    initialize_providers()
    
    results = {
        "provider_switching": test_provider_switching(),
        "provider_responses": test_provider_responses(),
        "fallback_mechanism": test_fallback_mechanism(),
        "parameter_consistency": test_parameter_consistency()
    }
    
    logger.info("Validation complete")
    logger.info(f"Results: {json.dumps(results, indent=2)}")
    
    return results

if __name__ == "__main__":
    run_validation()

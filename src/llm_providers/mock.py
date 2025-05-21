"""
Mock LLM Provider
-------------------
Implementation of a Mock LLM provider for development purposes.
"""

import logging
import random
from typing import Dict, List, Any, Optional

from .base import BaseLLMProvider

logger = logging.getLogger(__name__)

class MockProvider(BaseLLMProvider):
    """
    Mock LLM provider implementation for development and testing.
    """
    
    def _initialize(self) -> None:
        """
        Initialize Mock-specific settings.
        """
        # No specific initialization needed for mock
        self.api_key = "mock_key"  # Always set a dummy key so is_available() returns True
    
    def is_available(self) -> bool:
        """
        Check if the Mock provider is available.
        
        Returns:
            True always since this is a mock
        """
        return True
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a mock response. If the prompt contains specific keywords related to
        financial engineering, provide relevant mock answers.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with generated text and metadata
        """
        # Handle messages format if present
        if 'messages' in kwargs:
            messages = kwargs.get('messages', [])
            # Extract the user's query from messages
            user_message = next((m.get('content', '') for m in messages if m.get('role') == 'user'), '')
            # If the user message is not empty, use it as the prompt
            if user_message:
                prompt = user_message
        
        # Log the prompt for debugging
        logger.info(f"Mock provider generating response for prompt: {prompt[:100]}...")
        
        # Generate a relevant response based on the query
        response_text = self._generate_mock_response(prompt)
        
        # Simulate variable token count
        token_count = len(response_text.split()) + random.randint(10, 50)
        
        return {
            "text": response_text,
            "model": "mock-model",
            "provider": "mock",
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": token_count,
                "total_tokens": len(prompt.split()) + token_count
            }
        }
    
    def _generate_mock_response(self, prompt: str) -> str:
        """
        Generate a mock response based on keywords in the prompt.
        
        Args:
            prompt: The user's query
            
        Returns:
            A mock response
        """
        prompt_lower = prompt.lower()
        
        # Check for chapter-related queries
        if "chapter 3" in prompt_lower:
            return """
Chapter 3 of Ruppert's "Statistics and Financial Engineering" covers Time Series Analysis. 
The main topics include:

1. Basic time series concepts and stationarity
2. Autoregressive (AR) models
3. Moving Average (MA) models
4. ARMA and ARIMA models
5. Forecasting methods
6. Applications to financial data

The chapter emphasizes how time series models can be used to analyze financial markets and make predictions about future price movements. It includes several case studies with stock market data.
"""
        
        elif "chapter 1" in prompt_lower:
            return """
Chapter 1 of Ruppert's book introduces the fundamental concepts of financial engineering and statistics. It covers:

1. The role of probability in finance
2. Basic statistical measures (mean, variance, etc.)
3. Introduction to financial markets
4. Risk and return relationships
5. Overview of financial instruments

The chapter sets the foundation for understanding the statistical techniques used in modern finance.
"""
        
        elif "chapter 2" in prompt_lower:
            return """
Chapter 2 focuses on probability distributions in finance. Key topics include:

1. Normal distribution and its applications
2. Log-normal distribution for asset prices
3. Heavy-tailed distributions
4. Copulas for modeling dependencies
5. Statistical inference with financial data

The chapter explains why certain distributions are particularly relevant in financial modeling.
"""
        
        elif "black-scholes" in prompt_lower or "options" in prompt_lower:
            return """
The Black-Scholes model is covered in detail in Chapter 7. The model provides a mathematical framework for pricing European-style options.

Key components of the Black-Scholes model:
1. Geometric Brownian motion for stock prices
2. Risk-neutral valuation
3. The Black-Scholes formula
4. Greeks (delta, gamma, theta, vega, rho)
5. Model limitations and extensions

The chapter includes practical examples of option pricing and hedging strategies.
"""
        
        elif "portfolio" in prompt_lower or "markowitz" in prompt_lower:
            return """
Portfolio theory is discussed in Chapter 5, with a focus on the Markowitz model of portfolio optimization.

Main concepts:
1. Expected return and risk of portfolios
2. Diversification benefits
3. Efficient frontier
4. Capital Asset Pricing Model (CAPM)
5. Factor models

The chapter demonstrates how to construct optimal portfolios balancing risk and return.
"""
        
        # Default response for other queries
        else:
            return """
Based on Ruppert's book on Financial Engineering and Statistics, I can provide information on various topics including time series analysis, option pricing, portfolio optimization, risk management, and statistical methods in finance.

Could you please specify which particular topic or chapter from the book you're interested in learning about?
"""
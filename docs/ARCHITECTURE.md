# Pluggable LLM Architecture

This document provides an overview of the pluggable LLM architecture implemented for the Cerebras RAG application.

## Architecture Overview

The pluggable LLM architecture allows the Cerebras RAG application to use different LLM providers interchangeably, with support for runtime switching and fallback mechanisms.

```
┌─────────────────────┐
│                     │
│  Web Application    │
│                     │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│                     │
│  LLM Provider       │
│  Factory            │
│                     │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  ┌───────────┐   ┌───────────┐   ┌───────────────┐      │
│  │           │   │           │   │               │      │
│  │ Cerebras  │   │  OpenAI   │   │  Anthropic   │  ... │
│  │ Provider  │   │ Provider  │   │  Provider    │      │
│  │           │   │           │   │               │      │
│  └───────────┘   └───────────┘   └───────────────┘      │
│                                                         │
│                 LLM Providers                           │
└─────────────────────────────────────────────────────────┘
```

## Key Components

1. **BaseLLMProvider**: Abstract base class that defines the interface for all LLM providers
2. **LLMProviderFactory**: Factory class that manages provider registration, instantiation, and fallback logic
3. **Provider Implementations**: Concrete implementations for Cerebras, OpenAI, Anthropic, and Hugging Face
4. **Configuration System**: Environment variable-based configuration for all providers

## Provider Interface

The `BaseLLMProvider` interface defines the following key methods:

- `_initialize()`: Initialize provider-specific settings
- `is_available()`: Check if the provider is available and properly configured
- `generate(prompt, **kwargs)`: Generate a response from the LLM
- `validate_config()`: Validate the provider configuration
- `update_config(config)`: Update the provider configuration

## Factory Pattern

The `LLMProviderFactory` implements a factory pattern with these key features:

- Provider registration and instantiation
- Active provider management
- Fallback order configuration
- Response generation with automatic fallback

## Runtime Provider Switching

The architecture supports runtime switching between providers:

1. The web application exposes an API endpoint for switching providers
2. The factory maintains the active provider state
3. Switching validates the new provider's availability before changing

## Fallback Mechanism

If the active provider fails or is unavailable:

1. The factory attempts to use the active provider
2. If it fails, it tries providers in the configured fallback order
3. It returns the result along with metadata about which provider was used

## Configuration

All provider settings are configurable via environment variables:

- Default provider selection
- Fallback order
- Provider-specific API keys and URLs
- Model selection and parameters (temperature, max tokens, etc.)

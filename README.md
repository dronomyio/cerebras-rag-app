# Cerebras RAG LLM Plugin

This plugin adds a pluggable LLM architecture to the Cerebras RAG application, allowing you to easily switch between different LLM providers (Cerebras, OpenAI, Anthropic, and Hugging Face) without changing the core application code.

![final_architecture_diagram](https://github.com/user-attachments/assets/71aa09c6-b919-494c-9b8c-84fae0b4d001)


## Features

- **Multiple LLM Provider Support**: Use Cerebras, OpenAI, Anthropic, or Hugging Face models
- **Runtime Provider Switching**: Change LLM providers on-the-fly without restarting the application
- **Fallback Mechanism**: Automatically fall back to alternative providers if the primary provider is unavailable
- **Configurable Parameters**: Control temperature, max tokens, and other parameters for each provider
- **Environment Variable Configuration**: Easy setup through environment variables
- **Consistent Response Format**: Uniform response structure across all providers

## Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/cerebras-rag-llm-plugin.git
cd cerebras-rag-llm-plugin
```

2. Create a `.env` file from the example:
```bash
cp .env.example .env
```

3. Edit the `.env` file to add your API keys and configure providers.

4. Install the plugin in your Cerebras RAG application:
```bash
# Copy the llm_providers directory to your application
cp -r src/llm_providers /path/to/your/cerebras-rag-app/src/
```

## Configuration

### Environment Variables

Configure the LLM providers using environment variables in your `.env` file:

```
# Default LLM provider to use
DEFAULT_LLM_PROVIDER=cerebras

# Fallback order if the primary provider fails
LLM_FALLBACK_ORDER=cerebras,openai,anthropic,huggingface

# Enable runtime switching between providers
ENABLE_RUNTIME_SWITCHING=true

# Cerebras Configuration
CEREBRAS_API_KEY=your_cerebras_api_key_here
CEREBRAS_API_URL=https://api.cerebras.ai/v1/completions
CEREBRAS_MODEL=cerebras/Cerebras-GPT-4.5-8B
CEREBRAS_MAX_TOKENS=1024
CEREBRAS_TEMPERATURE=0.7
CEREBRAS_TOP_P=0.9

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_API_URL=https://api.openai.com/v1/chat/completions
OPENAI_MODEL=gpt-4
OPENAI_MAX_TOKENS=1024
OPENAI_TEMPERATURE=0.7
OPENAI_TOP_P=0.9

# Anthropic Configuration
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ANTHROPIC_API_URL=https://api.anthropic.com/v1/messages
ANTHROPIC_MODEL=claude-3-opus-20240229
ANTHROPIC_MAX_TOKENS=1024
ANTHROPIC_TEMPERATURE=0.7
ANTHROPIC_TOP_P=0.9
ANTHROPIC_SYSTEM_PROMPT=You are an AI assistant for answering questions about financial engineering and statistics based on Ruppert's book.

# Hugging Face Configuration
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
HUGGINGFACE_API_URL=https://api-inference.huggingface.co/models
HUGGINGFACE_MODEL=mistralai/Mistral-7B-Instruct-v0.2
HUGGINGFACE_MAX_TOKENS=1024
HUGGINGFACE_TEMPERATURE=0.7
HUGGINGFACE_TOP_P=0.9
```

## Usage

### Initializing the LLM Providers

Add the following code to your application to initialize the LLM providers:

```python
from llm_providers import LLMProviderFactory, BaseLLMProvider
from llm_providers.cerebras import CerebrasProvider
from llm_providers.openai import OpenAIProvider
from llm_providers.anthropic import AnthropicProvider
from llm_providers.huggingface import HuggingFaceProvider

def initialize_llm_providers():
    """Initialize and register all LLM providers."""
    # Register provider classes
    LLMProviderFactory.register_provider("cerebras", CerebrasProvider)
    LLMProviderFactory.register_provider("openai", OpenAIProvider)
    LLMProviderFactory.register_provider("anthropic", AnthropicProvider)
    LLMProviderFactory.register_provider("huggingface", HuggingFaceProvider)
    
    # Load providers from environment variables
    LLMProviderFactory.load_providers_from_config()

# Initialize LLM providers
initialize_llm_providers()
```

### Generating Responses

Use the LLM provider factory to generate responses:

```python
def generate_response(prompt, provider_name=None):
    """Generate a response using the configured LLM provider."""
    if provider_name:
        # Use specified provider
        provider = LLMProviderFactory.get_provider(provider_name)
        if provider and provider.is_available():
            return provider.generate(prompt)
    
    # Use factory with fallback
    response = LLMProviderFactory.generate_with_fallback(prompt)
    return response.get("result", {})
```

### Switching Providers at Runtime

To switch the active provider at runtime:

```python
# Switch to a different provider
success = LLMProviderFactory.set_active_provider("openai")

if success:
    print("Successfully switched to OpenAI")
else:
    print("Failed to switch to OpenAI")
```

### Getting the Active Provider

To get the current active provider:

```python
provider = LLMProviderFactory.get_active_provider()
if provider:
    print(f"Active provider: {LLMProviderFactory._active_provider}")
else:
    print("No active provider")
```

### Error Handling and Fallback

The plugin includes built-in error handling and fallback mechanisms:

```python
# Generate with fallback
response = LLMProviderFactory.generate_with_fallback(prompt)

# Check if fallback was used
if response.get("fallback_used", False):
    print(f"Fallback to provider: {response.get('provider')}")

# Get the result
result = response.get("result", {})
text = result.get("text", "No response")
```

## Provider-Specific Considerations

### Cerebras

- Uses the completions API
- Requires a valid Cerebras API key
- Default model: `cerebras/Cerebras-GPT-4.5-8B`

### OpenAI

- Uses the chat completions API
- Requires a valid OpenAI API key
- Default model: `gpt-4`
- Supports conversation history through the `messages` parameter

### Anthropic

- Uses the messages API
- Requires a valid Anthropic API key
- Default model: `claude-3-opus-20240229`
- Supports system prompts and conversation history

### Hugging Face

- Uses the inference API
- Requires a valid Hugging Face API key
- Default model: `mistralai/Mistral-7B-Instruct-v0.2`
- May have longer response times for first request (model loading)

## Extending with New Providers

To add a new LLM provider:

1. Create a new provider class that inherits from `BaseLLMProvider`
2. Implement the required methods: `_initialize`, `is_available`, and `generate`
3. Register the provider with the factory

Example:

```python
from llm_providers.base import BaseLLMProvider

class MyCustomProvider(BaseLLMProvider):
    def _initialize(self):
        # Initialize provider-specific settings
        pass
        
    def is_available(self):
        # Check if the provider is available
        return True
        
    def generate(self, prompt, **kwargs):
        # Generate a response
        return {"text": "My response", "provider": "custom"}

# Register the provider
LLMProviderFactory.register_provider("custom", MyCustomProvider)
```

## Troubleshooting

### Provider Not Available

If a provider is not available:

1. Check that you've provided the correct API key in your `.env` file
2. Verify that the API URL is correct
3. Ensure the model name is valid
4. Check network connectivity to the provider's API

### Switching Providers Fails

If switching providers fails:

1. Check that the provider is registered and initialized
2. Verify that the provider is available (API key, connectivity)
3. Ensure `ENABLE_RUNTIME_SWITCHING` is set to `true` in your `.env` file

### Inconsistent Responses

If responses are inconsistent across providers:

1. Check that you're using compatible models across providers
2. Verify that temperature and other parameters are set appropriately
3. Consider using provider-specific prompt formatting for best results

## License

MIT

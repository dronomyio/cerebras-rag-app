# Cerebras RAG Application with Pluggable Components

This application provides a comprehensive RAG (Retrieval-Augmented Generation) system for Ruppert's book with three key pluggable architectures:

1. **Pluggable LLM Architecture**: Switch between different LLM providers (Cerebras, OpenAI, Anthropic, and Hugging Face)
2. **Pluggable Document Processing**: Use native processors or Unstructured.io for enhanced document handling
3. **Autonomous Agent System**: Leverage AI agent capabilities for planning, memory, decision-making, and monitoring

![Integrated Architecture Diagram](https://private-us-east-1.manuscdn.com/sessionFile/o7vMEc8MH6iovTRE17cWPO/sandbox/le2bWsENXTtOdPdq80TZkN-images_1747761625053_na1fn_L2hvbWUvdWJ1bnR1L2NlcmVicmFzLXJhZy1sbG0tcGx1Z2luL2RvY3MvaW50ZWdyYXRlZF9hcmNoaXRlY3R1cmVfZGlhZ3JhbQ.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvbzd2TUVjOE1INmlvdlRSRTE3Y1dQTy9zYW5kYm94L2xlMmJXc0VOWFR0T2RQZHE4MFRaa04taW1hZ2VzXzE3NDc3NjE2MjUwNTNfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyTmxjbVZpY21GekxYSmhaeTFzYkcwdGNHeDFaMmx1TDJSdlkzTXZhVzUwWldkeVlYUmxaRjloY21Ob2FYUmxZM1IxY21WZlpHbGhaM0poYlEucG5nIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzY3MjI1NjAwfX19XX0_&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=h5gpwSLwOd8OpfND9wQQZw6jZjwNw37loFDqL~-czagY4vwTvYQ6dGCBWCp1s4ZoV8Plaj0Zwm9-EIPBfjB~7JFoMI~8HKjLZCOjxKcS59mixxBAwRzKDz652K6ESuPfy0-d7Y7cXo~GUCR-0bSTGVRDTzs1qCzDQNTOQXm8bHKFMnwW1URhKn2M3mTWdkCxwNnRWyvEOE0JPQQvzOQO4bDwffNANlieeqTojVastGFTBFCCsrx4BYcIGU~QgGQ7X11GlngAw4L7Fm8awl-FthzDPfMJrG-JfIM-C62DDVRbaNyGufSlSD3EE~wamk3KRNO3cMVM7Gexj4fjRPzHcQ__)

## Key Features

### Pluggable LLM Architecture
- **Multiple LLM Provider Support**: Use Cerebras, OpenAI, Anthropic, or Hugging Face models
- **Runtime Provider Switching**: Change LLM providers on-the-fly without restarting the application
- **Fallback Mechanism**: Automatically fall back to alternative providers if the primary provider is unavailable
- **Configurable Parameters**: Control temperature, max tokens, and other parameters for each provider

### Pluggable Document Processing
- **Native Document Processors**: Built-in support for PDF, DOCX, and text files
- **Unstructured.io Integration**: Enhanced document processing with configurable options
- **Extensible Plugin System**: Add new document processors through the plugin interface
- **Configurable Processing Pipeline**: Customize document chunking, embedding, and storage

### Autonomous Agent System
- **Planning Module**: Strategic task planning and execution monitoring
- **Memory Module**: Short and long-term memory for context retention
- **Decision Module**: Intelligent decision-making based on context and goals
- **Monitoring Module**: System health and performance monitoring

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

3. Edit the `.env` file to add your API keys and configure components.

4. Install the application:
```bash
# Install dependencies
pip install -r requirements.txt

# Set up the application
docker-compose up -d
```

## Configuration

### LLM Provider Configuration

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

### Document Processing Configuration

Configure document processing in the `config/document_processor.yaml` file:

```yaml
document_processor:
  chunk_size: 1000
  chunk_overlap: 200
  embedding_model: "text-embedding-ada-002"
  
  # Enable/disable Unstructured.io integration
  use_unstructured: true
  
  # Unstructured.io configuration
  unstructured:
    api_key: "your_unstructured_api_key_here"
    api_url: "https://api.unstructured.io/general/v0/general"
    strategy: "hi_res"
    ocr_languages: ["eng"]
    
  # Plugin configuration
  plugins:
    pdf:
      enabled: true
      ocr_enabled: true
    docx:
      enabled: true
    text:
      enabled: true
```

### Autonomous Agent Configuration

Configure the autonomous agent in the `config/agent.yaml` file:

```yaml
agent:
  enabled: true
  
  # Planning module configuration
  planning:
    max_steps: 10
    planning_model: "gpt-4"
    
  # Memory module configuration
  memory:
    short_term_capacity: 10
    long_term_enabled: true
    vector_db: "weaviate"
    
  # Decision module configuration
  decision:
    model: "gpt-4"
    temperature: 0.2
    
  # Monitoring module configuration
  monitoring:
    log_level: "info"
    metrics_enabled: true
    alert_threshold: 0.8
```

## Usage

### Using the Pluggable LLM Architecture

```python
from llm_providers import LLMProviderFactory, BaseLLMProvider
from llm_providers.cerebras import CerebrasProvider
from llm_providers.openai import OpenAIProvider
from llm_providers.anthropic import AnthropicProvider
from llm_providers.huggingface import HuggingFaceProvider

# Initialize providers
LLMProviderFactory.register_provider("cerebras", CerebrasProvider)
LLMProviderFactory.register_provider("openai", OpenAIProvider)
LLMProviderFactory.register_provider("anthropic", AnthropicProvider)
LLMProviderFactory.register_provider("huggingface", HuggingFaceProvider)
LLMProviderFactory.load_providers_from_config()

# Generate response with current provider
response = LLMProviderFactory.generate_with_fallback("What is the capital of France?")
print(response.get("result", {}).get("text", ""))

# Switch provider at runtime
LLMProviderFactory.set_active_provider("openai")
```

### Using the Pluggable Document Processing

```python
from document_processor.processor import DocumentProcessorService
from document_processor.plugins.pdf_processor import PDFProcessor
from document_processor.plugins.docx_processor import DOCXProcessor

# Initialize document processor
processor = DocumentProcessorService()

# Process a document
chunks = processor.process_document("/path/to/document.pdf")

# Ingest to vector database
processor.ingest_to_weaviate(chunks)

# Use Unstructured.io for processing
processor.config.use_unstructured = True
enhanced_chunks = processor.process_document("/path/to/complex_document.pdf")
```

### Using the Autonomous Agent

```python
from agent_core.agent import Agent
from agent_core.planning.planner import Planner
from agent_core.memory.memory import Memory

# Initialize agent
agent = Agent()

# Set a goal for the agent
agent.set_goal("Research and summarize Chapter 3 of Ruppert's book")

# Execute the goal
result = agent.execute()

# Access agent memory
relevant_context = agent.memory.retrieve("statistical models")
```

## Provider-Specific Considerations

### LLM Providers

#### Cerebras
- Uses the completions API
- Requires a valid Cerebras API key
- Default model: `cerebras/Cerebras-GPT-4.5-8B`

#### OpenAI
- Uses the chat completions API
- Requires a valid OpenAI API key
- Default model: `gpt-4`
- Supports conversation history through the `messages` parameter

#### Anthropic
- Uses the messages API
- Requires a valid Anthropic API key
- Default model: `claude-3-opus-20240229`
- Supports system prompts and conversation history

#### Hugging Face
- Uses the inference API
- Requires a valid Hugging Face API key
- Default model: `mistralai/Mistral-7B-Instruct-v0.2`
- May have longer response times for first request (model loading)

### Document Processing

#### Native Processors
- PDF Processor: Handles PDF files with optional OCR
- DOCX Processor: Handles Microsoft Word documents
- Text Processor: Handles plain text files

#### Unstructured.io
- Enhanced document understanding
- Better handling of tables, forms, and complex layouts
- Requires API key for production use

### Autonomous Agent

The agent system provides:
- Strategic planning for complex tasks
- Memory for context retention across interactions
- Decision-making based on goals and constraints
- Monitoring for system health and performance

## Extending the System

### Adding New LLM Providers

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

### Adding New Document Processors

```python
from document_processor.plugins.base_processor import BaseDocumentProcessor

class CSVProcessor(BaseDocumentProcessor):
    def __init__(self, config=None):
        super().__init__(config)
        self.supported_extensions = [".csv"]
        
    def process(self, file_path):
        # Process CSV file
        chunks = []
        # ... processing logic ...
        return chunks

# Register the processor
processor_service.register_processor("csv", CSVProcessor)
```

### Extending the Agent System

```python
from agent_core.decision.decision import DecisionModule

class EnhancedDecisionModule(DecisionModule):
    def __init__(self, config=None):
        super().__init__(config)
        self.risk_threshold = config.get("risk_threshold", 0.5)
        
    def make_decision(self, context, options):
        # Enhanced decision-making logic
        # ... decision logic ...
        return selected_option

# Replace the default module
agent.decision_module = EnhancedDecisionModule(config)
```

## Troubleshooting

### LLM Provider Issues

If a provider is not available:
1. Check that you've provided the correct API key in your `.env` file
2. Verify that the API URL is correct
3. Ensure the model name is valid
4. Check network connectivity to the provider's API

### Document Processing Issues

If document processing fails:
1. Check file permissions and format
2. Verify Unstructured.io API key if using that integration
3. Check log files for specific error messages
4. Try processing with a different plugin or configuration

### Agent System Issues

If the agent is not performing as expected:
1. Check the agent logs for decision points and reasoning
2. Verify the goal is clear and achievable
3. Adjust configuration parameters for planning and decision-making
4. Ensure the agent has access to necessary tools and information

## License

MIT

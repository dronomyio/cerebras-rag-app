# Pluggable HuggingGPT Integration Guide

This document provides comprehensive documentation for the pluggable HuggingGPT integration in the Cerebras RAG application. The integration allows you to leverage specialized AI models for different tasks, enhancing the capabilities of the RAG system with multimodal understanding and processing.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Key Components](#key-components)
3. [Configuration Guide](#configuration-guide)
4. [Usage Examples](#usage-examples)
5. [Extending the System](#extending-the-system)
6. [Troubleshooting](#troubleshooting)

## Architecture Overview

The pluggable HuggingGPT integration combines the strategic planning capabilities of the Cerebras RAG system with the model orchestration approach pioneered by HuggingGPT. This integration enables:

- **Hierarchical Planning**: Strategic planning for high-level goals and tactical planning for model-specific subtasks
- **Multimodal Processing**: Handling of text, images, code, math, and audio through specialized models
- **Dynamic Model Selection**: Intelligent selection of the most appropriate models for each subtask
- **Enhanced Decision-Making**: Combined strategic decisions and model selection intelligence
- **Pluggable Components**: Easily swappable and extensible components for customization

The architecture follows a modular design where each component can be independently configured, replaced, or extended without affecting other parts of the system.

## Key Components

### 1. Hierarchical Planner (`planning/planner.py`)

The hierarchical planner combines strategic planning (Cerebras RAG style) with tactical planning (HuggingGPT style):

- **Strategic Planner**: Creates high-level plans with sequential steps to achieve user goals
- **Tactical Planner**: Breaks down strategic steps into subtasks that can be executed by specialized models
- **Capability Identification**: Automatically identifies required capabilities for each subtask

```python
# Example: Creating a hierarchical plan
from agent_core.planning.planner import HierarchicalPlanner

planner = HierarchicalPlanner(config)
planner.set_llm_factory(llm_factory)

# Create strategic plan
strategic_plan = planner.create_strategic_plan(user_goal, memory)

# Create tactical plan for a strategic step
tactical_plan = planner.create_tactical_plan(strategic_step, memory)
```

### 2. Model Selector (`model_orchestration/model_selector.py`)

The model selector chooses the most appropriate specialized models for different tasks:

- **Capability-Based Selection**: Matches task requirements to model capabilities
- **Performance-Based Selection**: Considers past performance metrics
- **Hybrid Selection**: Combines capability matching with performance history
- **Pluggable Strategies**: Easily swappable selection strategies

```python
# Example: Selecting models for a tactical plan
from agent_core.model_orchestration.model_selector import HybridModelSelector

selector = HybridModelSelector(config)
selected_models = selector.select_models(tactical_plan, available_models)
```

### 3. Task Router (`model_orchestration/task_router.py`)

The task router formats inputs appropriately for different specialized models:

- **Input Formatting**: Prepares task inputs based on model requirements
- **Adaptive Routing**: Adjusts formatting based on past performance
- **Multimodal Support**: Handles different input types (text, images, code, etc.)
- **Pluggable Routers**: Easily swappable routing strategies

```python
# Example: Routing a task to a model
from agent_core.model_orchestration.task_router import AdaptiveTaskRouter

router = AdaptiveTaskRouter(config)
routing_info = router.route_task(subtask, selected_model)
```

### 4. Model Orchestrator (`model_orchestration/__init__.py`)

The main orchestrator coordinates model selection, task routing, and execution:

- **Model Management**: Initializes and manages available specialized models
- **Execution Coordination**: Executes tasks using selected models
- **Performance Tracking**: Monitors and records model performance
- **Tactical Plan Execution**: Executes complete tactical plans with multiple models

```python
# Example: Executing a tactical plan
from agent_core.model_orchestration import ModelOrchestrator

orchestrator = ModelOrchestrator(config)
results = orchestrator.execute_tactical_plan(tactical_plan, selected_models)
```

### 5. Enhanced Decision Module (`decision/decision.py`)

The decision module combines strategic decision-making with model selection:

- **Strategic Decisions**: Evaluates different approaches to achieving goals
- **Model Selection Decisions**: Chooses appropriate models for specific tasks
- **LLM-Powered Reasoning**: Uses LLMs for complex decision-making
- **Pluggable Decision Strategies**: Easily swappable decision approaches

```python
# Example: Making strategic decisions and selecting models
from agent_core.decision.decision import EnhancedDecisionModule

decision_module = EnhancedDecisionModule(config)
decision_module.set_llm_factory(llm_factory)

# Make strategic decision
selected_option = decision_module.make_strategic_decision(options, context, goal)

# Select models for a tactical plan
selected_models = decision_module.select_models_for_plan(tactical_plan, model_orchestrator)
```

## Configuration Guide

### Basic Configuration

The pluggable HuggingGPT integration is configured through the `agent.yaml` file and environment variables:

1. **Copy the example files**:
   ```bash
   cp config/agent.yaml.example config/agent.yaml
   cp .env.example .env
   ```

2. **Edit the configuration files** to set your preferences and API keys.

### Component Selection

You can select which implementation to use for each component:

```yaml
# In agent.yaml
agent:
  default_components:
    planner: "hierarchical"  # Options: strategic, tactical, hierarchical
    decision: "enhanced"     # Options: enhanced
    model_orchestrator: "standard"  # Options: standard
```

### Model Orchestration Configuration

Configure the model orchestration system:

```yaml
# In agent.yaml
model_orchestration:
  # Model selector configuration
  selector_type: "hybrid"  # Options: capability, performance, hybrid
  selector:
    capability_weight: 0.6  # Higher values favor capability matching
    
  # Task router configuration
  router_type: "adaptive"  # Options: standard, adaptive
  router:
    format_history_size: 100
```

### Specialized Model Configuration

Configure the specialized models available to the system:

```yaml
# In agent.yaml
model_orchestration:
  models:
    vision:
      - id: "vision_model_1"
        name: "Image Analyzer Pro"
        capabilities: ["image_understanding", "object_detection"]
        api_config:
          endpoint: "${VISION_MODEL_1_ENDPOINT}"
          api_key: "${VISION_MODEL_1_API_KEY}"
    # Additional model types: code, math, audio, text
```

### Environment Variables

Set the required environment variables in your `.env` file:

```
# Model Orchestration
VISION_MODEL_1_ENDPOINT=https://api.example.com/vision/v1
VISION_MODEL_1_API_KEY=your_vision_model_1_api_key_here

# Additional model endpoints and API keys
```

## Usage Examples

### Basic Usage

```python
from agent_core.agent import Agent

# Initialize the agent
agent = Agent()

# Set a goal
agent.set_goal("Analyze the statistical concepts in Chapter 3 of Ruppert's book")

# Execute the goal
results = agent.execute()
```

### Customizing Model Selection

```python
from agent_core.model_orchestration.model_selector import PerformanceBasedModelSelector

# Create a custom model selector
custom_selector = PerformanceBasedModelSelector(config)

# Update the agent's model orchestrator to use the custom selector
agent.model_orchestrator.model_selector = custom_selector

# Execute with the custom selector
results = agent.execute()
```

### Adding a New Specialized Model

```python
# Add a new model to the available models
new_model = {
    "id": "custom_vision_model",
    "name": "Custom Vision Analyzer",
    "type": "vision",
    "capabilities": ["image_understanding", "custom_capability"],
    "api_config": {
        "endpoint": "https://api.example.com/custom_vision",
        "api_key": "your_api_key_here"
    }
}

agent.model_orchestrator.available_models["vision"].append(new_model)
```

## Extending the System

### Creating a Custom Model Selector

```python
from agent_core.model_orchestration.model_selector import BaseModelSelector

class CustomModelSelector(BaseModelSelector):
    def select_models(self, tactical_plan, available_models):
        # Your custom selection logic here
        selected_models = {}
        
        # Example: Always select the first available model of the appropriate type
        for subtask in tactical_plan.get("subtasks", []):
            subtask_id = subtask.get("id")
            subtask_type = subtask.get("type", "text")
            
            models_of_type = available_models.get(subtask_type, [])
            selected_models[subtask_id] = models_of_type[0] if models_of_type else None
        
        return selected_models
```

### Creating a Custom Task Router

```python
from agent_core.model_orchestration.task_router import BaseTaskRouter

class CustomTaskRouter(BaseTaskRouter):
    def route_task(self, task, selected_model):
        # Your custom routing logic here
        formatted_inputs = {
            "custom_format": True,
            "task_description": task.get("description", ""),
            # Additional formatting
        }
        
        return {
            "success": True,
            "task_id": task.get("id"),
            "model_id": selected_model.get("id"),
            "formatted_inputs": formatted_inputs
        }
```

### Adding Support for a New Model Type

1. **Update the configuration**:
   ```yaml
   # In agent.yaml
   model_orchestration:
     models:
       new_model_type:
         - id: "new_model_1"
           name: "New Model Type"
           capabilities: ["capability_1", "capability_2"]
           api_config:
             endpoint: "${NEW_MODEL_1_ENDPOINT}"
             api_key: "${NEW_MODEL_1_API_KEY}"
   ```

2. **Update the task router**:
   ```python
   def _format_new_model_type_inputs(self, inputs, input_format):
       """Format inputs for the new model type."""
       formatted = {}
       
       # Add appropriate formatting logic
       if "specific_input" in inputs:
           formatted["specific_input"] = inputs["specific_input"]
       
       return formatted
   ```

3. **Update the model orchestrator**:
   ```python
   def _initialize_models(self):
       available_models = {
           # Existing model types
           "vision": [],
           "code": [],
           "math": [],
           "audio": [],
           "text": [],
           # New model type
           "new_model_type": []
       }
       # Rest of initialization
   ```

## Troubleshooting

### Common Issues

1. **Model Selection Failures**:
   - Check that the required capabilities are correctly identified in the tactical planner
   - Verify that models with the necessary capabilities are configured
   - Ensure API keys and endpoints are correctly set in the environment variables

2. **Task Routing Errors**:
   - Check the input format requirements for the selected model
   - Verify that the task inputs contain the necessary information
   - Check for formatting errors in the task router

3. **Model Execution Failures**:
   - Verify API connectivity to the specialized model endpoints
   - Check API key validity and permissions
   - Ensure the formatted inputs meet the model's requirements

### Debugging

Enable debug logging to get more detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check the execution history and performance metrics:

```python
# Get execution history
history = agent.model_orchestrator.get_execution_history()

# Get performance metrics
metrics = agent.model_orchestrator.get_performance_metrics()
```

### Getting Help

If you encounter issues not covered in this guide:

1. Check the logs for detailed error messages
2. Review the configuration for any misconfigurations
3. Consult the API documentation for the specialized models
4. File an issue in the project repository with detailed information about the problem

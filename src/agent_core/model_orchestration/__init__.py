"""
Model Orchestration - Main Orchestrator Module

This module implements the main model orchestration system that integrates
with the agent core to provide pluggable model selection and task routing.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple

from .model_selector import ModelSelectorFactory
from .task_router import TaskRouterFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelOrchestrator:
    """
    Main orchestrator for model selection and task routing.
    Integrates HuggingGPT-style model orchestration with the agent core.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model orchestrator.
        
        Args:
            config: Configuration dictionary for the orchestrator
        """
        self.config = config or {}
        
        # Create model selector
        selector_type = self.config.get("selector_type", "hybrid")
        selector_config = self.config.get("selector", {})
        self.model_selector = ModelSelectorFactory.create_model_selector(selector_type, selector_config)
        
        # Create task router
        router_type = self.config.get("router_type", "standard")
        router_config = self.config.get("router", {})
        self.task_router = TaskRouterFactory.create_task_router(router_type, router_config)
        
        # Initialize available models
        self.available_models = self._initialize_models()
        
        # Initialize execution history
        self.execution_history = []
        
        logger.info(f"Model Orchestrator initialized with {selector_type} selector and {router_type} router")
    
    def _initialize_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Initialize available models from configuration.
        
        Returns:
            Dictionary of available models by type
        """
        models_config = self.config.get("models", {})
        available_models = {
            "vision": [],
            "code": [],
            "math": [],
            "audio": [],
            "text": []
        }
        
        # Load models from configuration
        for model_type, models in models_config.items():
            if model_type in available_models:
                for model_config in models:
                    model = {
                        "id": model_config.get("id", f"{model_type}_{len(available_models[model_type])}"),
                        "name": model_config.get("name", "Unknown Model"),
                        "type": model_type,
                        "capabilities": model_config.get("capabilities", []),
                        "api_config": model_config.get("api_config", {}),
                        "input_format": model_config.get("input_format", {}),
                        "output_format": model_config.get("output_format", {})
                    }
                    available_models[model_type].append(model)
        
        # Log available models
        for model_type, models in available_models.items():
            logger.info(f"Loaded {len(models)} {model_type} models")
        
        return available_models
    
    def select_models(self, tactical_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select appropriate models for each subtask in the tactical plan.
        
        Args:
            tactical_plan: Tactical plan with subtasks
            
        Returns:
            Dictionary mapping subtask IDs to selected models
        """
        return self.model_selector.select_models(tactical_plan, self.available_models)
    
    def route_task(self, task: Dict[str, Any], selected_model: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a task to the appropriate model with formatted inputs.
        
        Args:
            task: Task information
            selected_model: Selected model for the task
            
        Returns:
            Routing information including formatted inputs
        """
        return self.task_router.route_task(task, selected_model)
    
    def execute_task(self, task: Dict[str, Any], selected_model: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task using the selected model.
        
        Args:
            task: Task information
            selected_model: Selected model for the task
            
        Returns:
            Execution results
        """
        # Route the task to get formatted inputs
        routing_info = self.route_task(task, selected_model)
        
        if not routing_info.get("success", False):
            return {
                "success": False,
                "error": routing_info.get("error", "Unknown routing error"),
                "task_id": task.get("id"),
                "execution_time": 0
            }
        
        # Execute the model
        start_time = time.time()
        try:
            result = self._execute_model(selected_model, routing_info.get("formatted_inputs", {}))
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
            logger.error(f"Error executing model {selected_model.get('name')}: {error}")
        
        execution_time = time.time() - start_time
        
        # Update performance metrics
        if hasattr(self.model_selector, "update_model_performance"):
            self.model_selector.update_model_performance(
                selected_model.get("id"),
                task.get("type", "text"),
                success,
                execution_time
            )
        
        if hasattr(self.task_router, "update_format_performance"):
            self.task_router.update_format_performance(
                selected_model.get("id"),
                task.get("type", "text"),
                "standard",  # For now, we only have one format approach
                success
            )
        
        # Record execution in history
        execution_record = {
            "task_id": task.get("id"),
            "model_id": selected_model.get("id"),
            "success": success,
            "execution_time": execution_time,
            "timestamp": time.time()
        }
        self.execution_history.append(execution_record)
        
        return {
            "success": success,
            "result": result,
            "error": error,
            "execution_time": execution_time,
            "task_id": task.get("id"),
            "model": selected_model.get("id")
        }
    
    def _execute_model(self, model: Dict[str, Any], formatted_inputs: Dict[str, Any]) -> Any:
        """
        Execute a model with the given inputs.
        
        Args:
            model: Model information
            formatted_inputs: Formatted inputs for the model
            
        Returns:
            Model execution results
        """
        model_type = model.get("type", "text")
        model_name = model.get("name", "Unknown Model")
        api_config = model.get("api_config", {})
        
        logger.info(f"Executing {model_type} model {model_name}")
        
        # In a real implementation, this would call the actual model API
        # For this example, we'll simulate model execution
        
        if model_type == "vision":
            return self._simulate_vision_model(formatted_inputs, api_config)
        elif model_type == "code":
            return self._simulate_code_model(formatted_inputs, api_config)
        elif model_type == "math":
            return self._simulate_math_model(formatted_inputs, api_config)
        elif model_type == "audio":
            return self._simulate_audio_model(formatted_inputs, api_config)
        else:  # Default to text
            return self._simulate_text_model(formatted_inputs, api_config)
    
    def _simulate_vision_model(self, inputs: Dict[str, Any], api_config: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate execution of a vision model."""
        # In a real implementation, this would call a vision model API
        return {
            "description": f"Simulated vision analysis of image: {inputs.get('image_url', 'unknown')}",
            "objects_detected": ["object1", "object2"],
            "confidence": 0.95
        }
    
    def _simulate_code_model(self, inputs: Dict[str, Any], api_config: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate execution of a code model."""
        # In a real implementation, this would call a code model API
        code = inputs.get("code", "print('Hello, World!')")
        return {
            "execution_result": "Hello, World!",
            "output": "Simulated code execution completed successfully",
            "runtime": 0.1
        }
    
    def _simulate_math_model(self, inputs: Dict[str, Any], api_config: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate execution of a math model."""
        # In a real implementation, this would call a math model API
        equation = inputs.get("equation", "1 + 1")
        return {
            "result": "2",
            "steps": ["Parse equation", "Evaluate expression", "Return result"],
            "confidence": 1.0
        }
    
    def _simulate_audio_model(self, inputs: Dict[str, Any], api_config: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate execution of an audio model."""
        # In a real implementation, this would call an audio model API
        return {
            "transcription": "Simulated audio transcription",
            "confidence": 0.85,
            "duration": 5.0
        }
    
    def _simulate_text_model(self, inputs: Dict[str, Any], api_config: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate execution of a text model."""
        # In a real implementation, this would call a text model API
        prompt = inputs.get("prompt", inputs.get("text", ""))
        return {
            "text": f"Simulated response to: {prompt[:50]}...",
            "tokens": len(prompt.split()),
            "model": "simulated-text-model"
        }
    
    def execute_tactical_plan(self, tactical_plan: Dict[str, Any], selected_models: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a complete tactical plan using selected models.
        
        Args:
            tactical_plan: Tactical plan with subtasks
            selected_models: Dictionary mapping subtask IDs to selected models
            
        Returns:
            Execution results for the entire tactical plan
        """
        results = {
            "success": True,
            "subtask_results": {},
            "start_time": time.time(),
            "end_time": None,
            "execution_time": 0,
            "strategic_step_id": tactical_plan.get("strategic_step_id")
        }
        
        # Execute each subtask
        for subtask in tactical_plan.get("subtasks", []):
            subtask_id = subtask.get("id")
            selected_model = selected_models.get(subtask_id)
            
            # Execute the subtask
            subtask_result = self.execute_task(subtask, selected_model)
            
            # Store the result
            results["subtask_results"][subtask_id] = subtask_result
            
            # Update overall success
            if not subtask_result.get("success", False):
                results["success"] = False
        
        # Calculate total execution time
        results["end_time"] = time.time()
        results["execution_time"] = results["end_time"] - results["start_time"]
        
        return results
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """
        Get the execution history.
        
        Returns:
            List of execution records
        """
        return self.execution_history
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for all models.
        
        Returns:
            Dictionary of performance metrics
        """
        metrics = {
            "models": {},
            "overall": {
                "total_executions": len(self.execution_history),
                "successful_executions": sum(1 for record in self.execution_history if record.get("success", False)),
                "average_execution_time": sum(record.get("execution_time", 0) for record in self.execution_history) / len(self.execution_history) if self.execution_history else 0
            }
        }
        
        # Calculate metrics for each model
        for record in self.execution_history:
            model_id = record.get("model_id")
            
            if model_id not in metrics["models"]:
                metrics["models"][model_id] = {
                    "total_executions": 0,
                    "successful_executions": 0,
                    "failed_executions": 0,
                    "total_execution_time": 0,
                    "average_execution_time": 0
                }
            
            model_metrics = metrics["models"][model_id]
            model_metrics["total_executions"] += 1
            
            if record.get("success", False):
                model_metrics["successful_executions"] += 1
            else:
                model_metrics["failed_executions"] += 1
            
            model_metrics["total_execution_time"] += record.get("execution_time", 0)
            model_metrics["average_execution_time"] = model_metrics["total_execution_time"] / model_metrics["total_executions"]
        
        return metrics

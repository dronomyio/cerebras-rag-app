"""
Model Orchestration - Task Router Module

This module implements a pluggable task routing system for directing tasks
to appropriate specialized AI models based on task requirements.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseTaskRouter:
    """Base class for all task router implementations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base task router.
        
        Args:
            config: Configuration dictionary for the task router
        """
        self.config = config or {}
        logger.info(f"{self.__class__.__name__} initialized")
    
    def route_task(self, task: Dict[str, Any], selected_model: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a task to the appropriate model with formatted inputs.
        
        Args:
            task: Task information
            selected_model: Selected model for the task
            
        Returns:
            Routing information including formatted inputs
        """
        raise NotImplementedError("Subclasses must implement route_task")


class StandardTaskRouter(BaseTaskRouter):
    """
    Standard task router that formats inputs based on model requirements.
    """
    
    def route_task(self, task: Dict[str, Any], selected_model: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a task to the appropriate model with formatted inputs.
        
        Args:
            task: Task information
            selected_model: Selected model for the task
            
        Returns:
            Routing information including formatted inputs
        """
        if not selected_model:
            return {
                "success": False,
                "error": "No model selected for task",
                "task_id": task.get("id")
            }
        
        # Format inputs based on model requirements
        formatted_inputs = self._format_inputs_for_model(task, selected_model)
        
        return {
            "success": True,
            "task_id": task.get("id"),
            "model_id": selected_model.get("id"),
            "model_name": selected_model.get("name"),
            "formatted_inputs": formatted_inputs,
            "routing_time": time.time()
        }
    
    def _format_inputs_for_model(self, task: Dict[str, Any], model: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format task inputs for the specific model.
        
        Args:
            task: Task information
            model: Model information
            
        Returns:
            Formatted inputs for the model
        """
        task_type = task.get("type", "text")
        task_inputs = task.get("inputs", {})
        model_input_format = model.get("input_format", {})
        
        formatted_inputs = {}
        
        # Apply general formatting based on task type
        if task_type == "vision":
            formatted_inputs = self._format_vision_inputs(task_inputs, model_input_format)
        elif task_type == "code":
            formatted_inputs = self._format_code_inputs(task_inputs, model_input_format)
        elif task_type == "math":
            formatted_inputs = self._format_math_inputs(task_inputs, model_input_format)
        elif task_type == "audio":
            formatted_inputs = self._format_audio_inputs(task_inputs, model_input_format)
        else:  # Default to text
            formatted_inputs = self._format_text_inputs(task_inputs, model_input_format)
        
        # Add task description as context
        formatted_inputs["task_description"] = task.get("description", "")
        
        return formatted_inputs
    
    def _format_vision_inputs(self, inputs: Dict[str, Any], input_format: Dict[str, Any]) -> Dict[str, Any]:
        """Format inputs for vision models."""
        formatted = {}
        
        # Handle image inputs
        if "image" in inputs:
            formatted["image"] = inputs["image"]
        elif "image_url" in inputs:
            formatted["image_url"] = inputs["image_url"]
        
        # Add any additional parameters
        if "parameters" in inputs:
            formatted["parameters"] = inputs["parameters"]
        
        return formatted
    
    def _format_code_inputs(self, inputs: Dict[str, Any], input_format: Dict[str, Any]) -> Dict[str, Any]:
        """Format inputs for code models."""
        formatted = {}
        
        # Handle code inputs
        if "code" in inputs:
            formatted["code"] = inputs["code"]
        
        # Add language information
        if "language" in inputs:
            formatted["language"] = inputs["language"]
        
        # Add execution parameters
        if "execution_params" in inputs:
            formatted["execution_params"] = inputs["execution_params"]
        
        return formatted
    
    def _format_math_inputs(self, inputs: Dict[str, Any], input_format: Dict[str, Any]) -> Dict[str, Any]:
        """Format inputs for math models."""
        formatted = {}
        
        # Handle equation inputs
        if "equation" in inputs:
            formatted["equation"] = inputs["equation"]
        
        # Add variables
        if "variables" in inputs:
            formatted["variables"] = inputs["variables"]
        
        # Add calculation parameters
        if "parameters" in inputs:
            formatted["parameters"] = inputs["parameters"]
        
        return formatted
    
    def _format_audio_inputs(self, inputs: Dict[str, Any], input_format: Dict[str, Any]) -> Dict[str, Any]:
        """Format inputs for audio models."""
        formatted = {}
        
        # Handle audio inputs
        if "audio" in inputs:
            formatted["audio"] = inputs["audio"]
        elif "audio_url" in inputs:
            formatted["audio_url"] = inputs["audio_url"]
        
        # Add any additional parameters
        if "parameters" in inputs:
            formatted["parameters"] = inputs["parameters"]
        
        return formatted
    
    def _format_text_inputs(self, inputs: Dict[str, Any], input_format: Dict[str, Any]) -> Dict[str, Any]:
        """Format inputs for text models."""
        formatted = {}
        
        # Handle text inputs
        if "text" in inputs:
            formatted["text"] = inputs["text"]
        elif "prompt" in inputs:
            formatted["prompt"] = inputs["prompt"]
        
        # Add any additional parameters
        if "parameters" in inputs:
            formatted["parameters"] = inputs["parameters"]
        
        return formatted


class AdaptiveTaskRouter(BaseTaskRouter):
    """
    Adaptive task router that adjusts formatting based on past performance.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the adaptive task router.
        
        Args:
            config: Configuration dictionary for the task router
        """
        super().__init__(config)
        self.standard_router = StandardTaskRouter(config)
        self.format_performance = {}  # Track performance of different formatting approaches
    
    def route_task(self, task: Dict[str, Any], selected_model: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a task to the appropriate model with adaptively formatted inputs.
        
        Args:
            task: Task information
            selected_model: Selected model for the task
            
        Returns:
            Routing information including formatted inputs
        """
        if not selected_model:
            return {
                "success": False,
                "error": "No model selected for task",
                "task_id": task.get("id")
            }
        
        # Get model and task identifiers
        model_id = selected_model.get("id")
        task_type = task.get("type", "text")
        
        # Check if we have performance data for this model and task type
        if model_id in self.format_performance and task_type in self.format_performance[model_id]:
            # Use the best performing format approach
            format_approach = self._get_best_format_approach(model_id, task_type)
            formatted_inputs = self._apply_format_approach(task, selected_model, format_approach)
        else:
            # Fall back to standard formatting
            formatted_inputs = self.standard_router._format_inputs_for_model(task, selected_model)
        
        return {
            "success": True,
            "task_id": task.get("id"),
            "model_id": model_id,
            "model_name": selected_model.get("name"),
            "formatted_inputs": formatted_inputs,
            "routing_time": time.time()
        }
    
    def _get_best_format_approach(self, model_id: str, task_type: str) -> str:
        """
        Get the best performing format approach for a model and task type.
        
        Args:
            model_id: Model identifier
            task_type: Type of task
            
        Returns:
            Format approach identifier
        """
        performance_data = self.format_performance.get(model_id, {}).get(task_type, {})
        
        # Find the approach with the highest success rate
        best_approach = "standard"
        best_success_rate = 0
        
        for approach, metrics in performance_data.items():
            success_rate = metrics.get("success_rate", 0)
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_approach = approach
        
        return best_approach
    
    def _apply_format_approach(self, task: Dict[str, Any], model: Dict[str, Any], approach: str) -> Dict[str, Any]:
        """
        Apply a specific formatting approach.
        
        Args:
            task: Task information
            model: Model information
            approach: Format approach identifier
            
        Returns:
            Formatted inputs
        """
        # In a real implementation, this would have multiple formatting approaches
        # For this example, we'll just use the standard formatter
        return self.standard_router._format_inputs_for_model(task, model)
    
    def update_format_performance(self, model_id: str, task_type: str, approach: str, success: bool) -> None:
        """
        Update performance metrics for a formatting approach.
        
        Args:
            model_id: Model identifier
            task_type: Type of task
            approach: Format approach identifier
            success: Whether execution was successful
        """
        # Initialize if needed
        if model_id not in self.format_performance:
            self.format_performance[model_id] = {}
        
        if task_type not in self.format_performance[model_id]:
            self.format_performance[model_id][task_type] = {}
        
        if approach not in self.format_performance[model_id][task_type]:
            self.format_performance[model_id][task_type][approach] = {
                "success_count": 0,
                "failure_count": 0,
                "success_rate": 0
            }
        
        # Update metrics
        metrics = self.format_performance[model_id][task_type][approach]
        
        if success:
            metrics["success_count"] += 1
        else:
            metrics["failure_count"] += 1
        
        # Recalculate success rate
        total_count = metrics["success_count"] + metrics["failure_count"]
        metrics["success_rate"] = metrics["success_count"] / total_count if total_count > 0 else 0


# Factory for creating different types of task routers
class TaskRouterFactory:
    """Factory for creating different types of task routers."""
    
    @staticmethod
    def create_task_router(router_type: str, config: Optional[Dict[str, Any]] = None) -> BaseTaskRouter:
        """
        Create a task router of the specified type.
        
        Args:
            router_type: Type of task router to create
            config: Configuration dictionary for the task router
            
        Returns:
            Task router instance
        """
        if router_type == "standard":
            return StandardTaskRouter(config)
        elif router_type == "adaptive":
            return AdaptiveTaskRouter(config)
        else:
            raise ValueError(f"Unknown task router type: {router_type}")

"""
Model Orchestration - Model Selector Module

This module implements a pluggable model selection system for choosing
appropriate specialized AI models for different tasks.
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

class BaseModelSelector:
    """Base class for all model selector implementations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base model selector.
        
        Args:
            config: Configuration dictionary for the model selector
        """
        self.config = config or {}
        logger.info(f"{self.__class__.__name__} initialized")
    
    def select_models(self, tactical_plan: Dict[str, Any], available_models: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Select appropriate models for each subtask in the tactical plan.
        
        Args:
            tactical_plan: Tactical plan with subtasks
            available_models: Dictionary of available models by type
            
        Returns:
            Dictionary mapping subtask IDs to selected models
        """
        raise NotImplementedError("Subclasses must implement select_models")


class CapabilityBasedModelSelector(BaseModelSelector):
    """
    Model selector that chooses models based on required capabilities.
    """
    
    def select_models(self, tactical_plan: Dict[str, Any], available_models: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Select appropriate models for each subtask based on required capabilities.
        
        Args:
            tactical_plan: Tactical plan with subtasks
            available_models: Dictionary of available models by type
            
        Returns:
            Dictionary mapping subtask IDs to selected models
        """
        selected_models = {}
        
        for subtask in tactical_plan.get("subtasks", []):
            subtask_id = subtask.get("id")
            required_capabilities = subtask.get("required_capabilities", [])
            
            # Match required capabilities to available models
            candidate_models = self._find_candidate_models(required_capabilities, available_models)
            
            # Rank models based on performance history and requirements
            ranked_models = self._rank_models(candidate_models, subtask)
            
            # Select best model
            selected_models[subtask_id] = ranked_models[0] if ranked_models else None
            
            if ranked_models:
                logger.info(f"Selected model {ranked_models[0].get('name', 'unknown')} for subtask {subtask_id}")
            else:
                logger.warning(f"No suitable model found for subtask {subtask_id}")
        
        return selected_models
    
    def _find_candidate_models(self, required_capabilities: List[str], available_models: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Find models that match the required capabilities.
        
        Args:
            required_capabilities: List of capabilities needed
            available_models: Dictionary of available models by type
            
        Returns:
            List of candidate models
        """
        candidates = []
        
        # Flatten all models into a single list for searching
        all_models = []
        for model_type, models in available_models.items():
            all_models.extend(models)
        
        # Find models that have all required capabilities
        for model in all_models:
            model_capabilities = model.get("capabilities", [])
            
            # Check if model has all required capabilities
            if all(cap in model_capabilities for cap in required_capabilities):
                candidates.append(model)
        
        return candidates
    
    def _rank_models(self, candidate_models: List[Dict[str, Any]], subtask: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Rank candidate models based on performance history and requirements.
        
        Args:
            candidate_models: List of candidate models
            subtask: Subtask information
            
        Returns:
            Ranked list of models
        """
        if not candidate_models:
            return []
        
        # In a real implementation, this would use performance metrics
        # For this example, we'll just return the candidates in their original order
        return candidate_models


class PerformanceBasedModelSelector(BaseModelSelector):
    """
    Model selector that chooses models based on past performance.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the performance-based model selector.
        
        Args:
            config: Configuration dictionary for the model selector
        """
        super().__init__(config)
        self.model_performance = {}  # Track performance metrics
    
    def select_models(self, tactical_plan: Dict[str, Any], available_models: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Select appropriate models for each subtask based on past performance.
        
        Args:
            tactical_plan: Tactical plan with subtasks
            available_models: Dictionary of available models by type
            
        Returns:
            Dictionary mapping subtask IDs to selected models
        """
        selected_models = {}
        
        for subtask in tactical_plan.get("subtasks", []):
            subtask_id = subtask.get("id")
            subtask_type = subtask.get("type", "text")
            
            # Get models of the appropriate type
            models_of_type = available_models.get(subtask_type, [])
            
            if not models_of_type:
                logger.warning(f"No models available for type {subtask_type}")
                selected_models[subtask_id] = None
                continue
            
            # Rank models based on performance
            ranked_models = self._rank_models_by_performance(models_of_type, subtask_type)
            
            # Select best model
            selected_models[subtask_id] = ranked_models[0] if ranked_models else None
            
            if ranked_models:
                logger.info(f"Selected model {ranked_models[0].get('name', 'unknown')} for subtask {subtask_id}")
            else:
                logger.warning(f"No suitable model found for subtask {subtask_id}")
        
        return selected_models
    
    def _rank_models_by_performance(self, models: List[Dict[str, Any]], task_type: str) -> List[Dict[str, Any]]:
        """
        Rank models based on past performance.
        
        Args:
            models: List of models to rank
            task_type: Type of task
            
        Returns:
            Ranked list of models
        """
        # Calculate scores for each model
        scored_models = []
        for model in models:
            model_id = model.get("id")
            
            # Get performance history for this model and task type
            performance = self.model_performance.get(model_id, {}).get(task_type, {})
            
            # Calculate score based on success rate and speed
            success_rate = performance.get("success_rate", 0.8)  # Default to 0.8 if no history
            avg_execution_time = performance.get("avg_execution_time", 1.0)  # Default to 1.0 if no history
            
            # Simple scoring formula: success_rate / normalized_time
            # Lower time is better, so we invert it
            normalized_time = max(0.1, min(avg_execution_time, 10.0)) / 10.0
            score = success_rate / normalized_time
            
            scored_models.append((model, score))
        
        # Sort by score in descending order
        scored_models.sort(key=lambda x: x[1], reverse=True)
        
        # Return just the models, not the scores
        return [model for model, _ in scored_models]
    
    def update_model_performance(self, model_id: str, task_type: str, success: bool, execution_time: float) -> None:
        """
        Update performance metrics for a model.
        
        Args:
            model_id: Model identifier
            task_type: Type of task
            success: Whether execution was successful
            execution_time: Execution time in seconds
        """
        # Initialize if needed
        if model_id not in self.model_performance:
            self.model_performance[model_id] = {}
        
        if task_type not in self.model_performance[model_id]:
            self.model_performance[model_id][task_type] = {
                "success_count": 0,
                "failure_count": 0,
                "total_execution_time": 0,
                "execution_count": 0,
                "success_rate": 0,
                "avg_execution_time": 0
            }
        
        # Update metrics
        metrics = self.model_performance[model_id][task_type]
        
        if success:
            metrics["success_count"] += 1
        else:
            metrics["failure_count"] += 1
        
        metrics["total_execution_time"] += execution_time
        metrics["execution_count"] += 1
        
        # Recalculate derived metrics
        total_count = metrics["success_count"] + metrics["failure_count"]
        metrics["success_rate"] = metrics["success_count"] / total_count if total_count > 0 else 0
        metrics["avg_execution_time"] = metrics["total_execution_time"] / metrics["execution_count"] if metrics["execution_count"] > 0 else 0


class HybridModelSelector(BaseModelSelector):
    """
    Model selector that combines capability matching with performance history.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the hybrid model selector.
        
        Args:
            config: Configuration dictionary for the model selector
        """
        super().__init__(config)
        self.capability_selector = CapabilityBasedModelSelector(config)
        self.performance_selector = PerformanceBasedModelSelector(config)
        
        # Weight for balancing capability vs. performance (0-1)
        # Higher values favor capability matching, lower values favor performance
        self.capability_weight = config.get("capability_weight", 0.6)
    
    def select_models(self, tactical_plan: Dict[str, Any], available_models: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Select appropriate models using a hybrid approach.
        
        Args:
            tactical_plan: Tactical plan with subtasks
            available_models: Dictionary of available models by type
            
        Returns:
            Dictionary mapping subtask IDs to selected models
        """
        # Get selections from both approaches
        capability_selections = self.capability_selector.select_models(tactical_plan, available_models)
        performance_selections = self.performance_selector.select_models(tactical_plan, available_models)
        
        # Combine selections
        combined_selections = {}
        
        for subtask in tactical_plan.get("subtasks", []):
            subtask_id = subtask.get("id")
            
            capability_model = capability_selections.get(subtask_id)
            performance_model = performance_selections.get(subtask_id)
            
            # If both approaches selected the same model, use it
            if capability_model and performance_model and capability_model.get("id") == performance_model.get("id"):
                combined_selections[subtask_id] = capability_model
            
            # If only one approach found a model, use it
            elif capability_model and not performance_model:
                combined_selections[subtask_id] = capability_model
            elif performance_model and not capability_model:
                combined_selections[subtask_id] = performance_model
            
            # If both approaches found different models, use weighted decision
            elif capability_model and performance_model:
                # In a real implementation, this would use a more sophisticated scoring system
                # For this example, we'll use the capability weight to decide
                if self.capability_weight >= 0.5:
                    combined_selections[subtask_id] = capability_model
                else:
                    combined_selections[subtask_id] = performance_model
            
            # If neither approach found a model, return None
            else:
                combined_selections[subtask_id] = None
        
        return combined_selections
    
    def update_model_performance(self, model_id: str, task_type: str, success: bool, execution_time: float) -> None:
        """
        Update performance metrics for a model.
        
        Args:
            model_id: Model identifier
            task_type: Type of task
            success: Whether execution was successful
            execution_time: Execution time in seconds
        """
        self.performance_selector.update_model_performance(model_id, task_type, success, execution_time)


# Factory for creating different types of model selectors
class ModelSelectorFactory:
    """Factory for creating different types of model selectors."""
    
    @staticmethod
    def create_model_selector(selector_type: str, config: Optional[Dict[str, Any]] = None) -> BaseModelSelector:
        """
        Create a model selector of the specified type.
        
        Args:
            selector_type: Type of model selector to create
            config: Configuration dictionary for the model selector
            
        Returns:
            Model selector instance
        """
        if selector_type == "capability":
            return CapabilityBasedModelSelector(config)
        elif selector_type == "performance":
            return PerformanceBasedModelSelector(config)
        elif selector_type == "hybrid":
            return HybridModelSelector(config)
        else:
            raise ValueError(f"Unknown model selector type: {selector_type}")

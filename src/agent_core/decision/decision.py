"""
Decision Module for Agent Core

This module implements a pluggable decision-making system that combines
strategic decision-making with model selection intelligence.
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

class BaseDecisionModule:
    """Base class for all decision module implementations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base decision module.
        
        Args:
            config: Configuration dictionary for the decision module
        """
        self.config = config or {}
        self.llm_factory = None  # Will be set by Agent
        logger.info(f"{self.__class__.__name__} initialized")
    
    def set_llm_factory(self, llm_factory: Any) -> None:
        """
        Set the LLM provider factory for the decision module.
        
        Args:
            llm_factory: LLM provider factory
        """
        self.llm_factory = llm_factory
        logger.info(f"LLM factory set for {self.__class__.__name__}")
    
    def make_strategic_decision(self, options: List[Dict[str, Any]], context: str, goal: str) -> Dict[str, Any]:
        """
        Make a high-level strategic decision.
        
        Args:
            options: List of available options
            context: Context information
            goal: User goal
            
        Returns:
            Selected option
        """
        raise NotImplementedError("Subclasses must implement make_strategic_decision")
    
    def select_models_for_plan(self, tactical_plan: Dict[str, Any], model_orchestrator: Any) -> Dict[str, Any]:
        """
        Select appropriate models for a tactical plan.
        
        Args:
            tactical_plan: Tactical plan with subtasks
            model_orchestrator: Model orchestrator
            
        Returns:
            Dictionary mapping subtask IDs to selected models
        """
        raise NotImplementedError("Subclasses must implement select_models_for_plan")


class EnhancedDecisionModule(BaseDecisionModule):
    """
    Enhanced decision module that combines strategic decision-making with model selection.
    Integrates Cerebras RAG's strategic decisions with HuggingGPT's model selection.
    """
    
    def make_strategic_decision(self, options: List[Dict[str, Any]], context: str, goal: str) -> Dict[str, Any]:
        """
        Make a high-level strategic decision.
        
        Args:
            options: List of available options
            context: Context information
            goal: User goal
            
        Returns:
            Selected option
        """
        # Create decision prompt
        decision_template = self._get_strategic_decision_template()
        decision_input = self._format_strategic_input(options, context, goal)
        
        # Generate decision using LLM
        llm_provider = self.llm_factory.get_active_provider()
        decision_output = llm_provider.generate(decision_template.format(input=decision_input))
        
        # Parse and structure the decision
        selected_option = self._parse_strategic_decision(decision_output.get("text", ""), options)
        
        logger.info(f"Made strategic decision: {selected_option.get('description', '')[:50]}...")
        
        return selected_option
    
    def _get_strategic_decision_template(self) -> str:
        """Get the template for strategic decision-making."""
        return """
        You are an AI assistant tasked with making a strategic decision to achieve a user's goal.
        Evaluate the available options and select the most appropriate one.
        
        USER GOAL: {input}
        
        Consider the following factors:
        1. Alignment with the user's goal
        2. Feasibility and likelihood of success
        3. Efficiency and resource requirements
        4. Potential risks and limitations
        
        Provide your decision and a brief explanation of your reasoning.
        """
    
    def _format_strategic_input(self, options: List[Dict[str, Any]], context: str, goal: str) -> str:
        """Format the input for strategic decision-making."""
        options_text = "\n".join([f"Option {i+1}: {option.get('description', '')}" for i, option in enumerate(options)])
        
        return f"""
        GOAL: {goal}
        
        AVAILABLE OPTIONS:
        {options_text}
        
        RELEVANT CONTEXT:
        {context}
        """
    
    def _parse_strategic_decision(self, decision_text: str, options: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Parse the LLM output into a structured decision.
        
        Args:
            decision_text: Raw text output from LLM
            options: List of available options
            
        Returns:
            Selected option
        """
        # In a real implementation, this would use regex or more sophisticated parsing
        # For this example, we'll select the first option if parsing fails
        
        try:
            # Look for option number in the decision text
            for i, option in enumerate(options):
                option_number = i + 1
                if f"Option {option_number}" in decision_text or f"option {option_number}" in decision_text:
                    return option
            
            # If no option number found, look for option description
            for option in options:
                description = option.get("description", "")
                if description and description in decision_text:
                    return option
            
            # Default to first option if no match found
            return options[0] if options else {}
            
        except Exception as e:
            logger.error(f"Error parsing strategic decision: {e}")
            return options[0] if options else {}
    
    def select_models_for_plan(self, tactical_plan: Dict[str, Any], model_orchestrator: Any) -> Dict[str, Any]:
        """
        Select appropriate models for a tactical plan.
        
        Args:
            tactical_plan: Tactical plan with subtasks
            model_orchestrator: Model orchestrator
            
        Returns:
            Dictionary mapping subtask IDs to selected models
        """
        # Delegate model selection to the model orchestrator
        selected_models = model_orchestrator.select_models(tactical_plan)
        
        # Log the selections
        for subtask_id, model in selected_models.items():
            if model:
                logger.info(f"Selected model {model.get('name', 'unknown')} for subtask {subtask_id}")
            else:
                logger.warning(f"No suitable model found for subtask {subtask_id}")
        
        return selected_models


# Factory for creating different types of decision modules
class DecisionModuleFactory:
    """Factory for creating different types of decision modules."""
    
    @staticmethod
    def create_decision_module(module_type: str, config: Optional[Dict[str, Any]] = None) -> BaseDecisionModule:
        """
        Create a decision module of the specified type.
        
        Args:
            module_type: Type of decision module to create
            config: Configuration dictionary for the decision module
            
        Returns:
            Decision module instance
        """
        if module_type == "enhanced":
            return EnhancedDecisionModule(config)
        else:
            raise ValueError(f"Unknown decision module type: {module_type}")

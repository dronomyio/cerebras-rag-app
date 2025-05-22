"""
Hierarchical Planner Module

This module implements a pluggable hierarchical planning system that combines
strategic planning (from Cerebras RAG) with tactical planning and task decomposition
(from HuggingGPT).
"""

import logging
import time
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BasePlanner:
    """Base class for all planner implementations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base planner.
        
        Args:
            config: Configuration dictionary for the planner
        """
        self.config = config or {}
        self.llm_factory = None  # Will be set by Agent
        logger.info(f"{self.__class__.__name__} initialized")
    
    def set_llm_factory(self, llm_factory: Any) -> None:
        """
        Set the LLM provider factory for the planner.
        
        Args:
            llm_factory: LLM provider factory
        """
        self.llm_factory = llm_factory
        logger.info(f"LLM factory set for {self.__class__.__name__}")
    
    def create_plan(self, goal: str, context: Any) -> Dict[str, Any]:
        """
        Create a plan to achieve the given goal.
        
        Args:
            goal: User goal
            context: Context information
            
        Returns:
            Plan dictionary
        """
        raise NotImplementedError("Subclasses must implement create_plan")


class StrategicPlanner(BasePlanner):
    """
    Strategic planner that creates high-level plans for achieving user goals.
    Based on Cerebras RAG's planning approach.
    """
    
    def create_plan(self, goal: str, memory: Any) -> Dict[str, Any]:
        """
        Create a high-level strategic plan to achieve the given goal.
        
        Args:
            goal: User goal
            memory: Memory module for context retrieval
            
        Returns:
            Strategic plan
        """
        # Get relevant context from memory
        context = memory.get_relevant_context(goal)
        
        # Create planning prompt
        plan_template = self._get_strategic_planning_template()
        plan_input = self._format_planning_input(goal, context)
        
        # Generate strategic plan using LLM
        llm_provider = self.llm_factory.get_active_provider()
        plan_output = llm_provider.generate(plan_template.format(input=plan_input))
        
        # Parse and structure the plan
        structured_plan = self._parse_strategic_plan(plan_output.get("text", ""))
        
        logger.info(f"Created strategic plan with {len(structured_plan.get('steps', []))} steps for goal: {goal[:50]}...")
        
        return structured_plan
    
    def _get_strategic_planning_template(self) -> str:
        """Get the template for strategic planning."""
        return """
        You are an AI assistant tasked with creating a strategic plan to achieve a user's goal.
        Break down the goal into logical steps that can be executed sequentially.
        
        USER GOAL: {input}
        
        Create a strategic plan with 3-7 steps. For each step, provide:
        1. A clear description of what needs to be done
        2. The expected outcome of the step
        3. Any dependencies on previous steps
        
        Format your response as a numbered list of steps.
        """
    
    def _format_planning_input(self, goal: str, context: str) -> str:
        """Format the input for strategic planning."""
        return f"""
        GOAL: {goal}
        
        RELEVANT CONTEXT:
        {context}
        """
    
    def _parse_strategic_plan(self, plan_text: str) -> Dict[str, Any]:
        """
        Parse the LLM output into a structured strategic plan.
        
        Args:
            plan_text: Raw text output from LLM
            
        Returns:
            Structured strategic plan
        """
        # In a real implementation, this would use regex or more sophisticated parsing
        # For this example, we'll create a simple structured plan
        
        lines = plan_text.strip().split('\n')
        steps = []
        
        current_step = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this is a new step
            if line[0].isdigit() and '.' in line[:5]:
                if current_step:
                    steps.append(current_step)
                
                step_text = line.split('.', 1)[1].strip()
                current_step = {
                    "id": len(steps) + 1,
                    "description": step_text,
                    "expected_outcome": "",
                    "dependencies": []
                }
            elif current_step and line.lower().startswith("outcome:"):
                current_step["expected_outcome"] = line.split(":", 1)[1].strip()
            elif current_step:
                # Append to the current step description
                current_step["description"] += " " + line
        
        # Add the last step if it exists
        if current_step:
            steps.append(current_step)
        
        return {
            "steps": steps,
            "goal": plan_text.split('\n', 1)[0] if lines else ""
        }


class TacticalPlanner(BasePlanner):
    """
    Tactical planner that decomposes strategic steps into subtasks.
    Based on HuggingGPT's task decomposition approach.
    """
    
    def create_plan(self, strategic_step: Dict[str, Any], memory: Any) -> Dict[str, Any]:
        """
        Create a tactical plan for executing a strategic step.
        
        Args:
            strategic_step: Step from the strategic plan
            memory: Memory module for context retrieval
            
        Returns:
            Tactical plan with subtasks
        """
        # Get relevant context for this specific step
        context = memory.get_relevant_context(strategic_step.get("description", ""))
        
        # Create tactical planning prompt
        tactical_template = self._get_tactical_planning_template()
        tactical_input = self._format_tactical_input(strategic_step, context)
        
        # Generate tactical plan using LLM
        llm_provider = self.llm_factory.get_active_provider()
        tactical_output = llm_provider.generate(tactical_template.format(input=tactical_input))
        
        # Parse and structure the tactical plan
        tactical_plan = self._parse_tactical_plan(tactical_output.get("text", ""), strategic_step)
        
        # Identify required capabilities for each subtask
        for subtask in tactical_plan.get("subtasks", []):
            subtask["required_capabilities"] = self._identify_capabilities(subtask)
        
        logger.info(f"Created tactical plan with {len(tactical_plan.get('subtasks', []))} subtasks for step: {strategic_step.get('description', '')[:50]}...")
        
        return tactical_plan
    
    def _get_tactical_planning_template(self) -> str:
        """Get the template for tactical planning."""
        return """
        You are an AI assistant tasked with breaking down a strategic step into specific subtasks
        that can be executed by specialized AI models.
        
        STRATEGIC STEP: {input}
        
        Break this step down into 2-5 subtasks. For each subtask, specify:
        1. A clear description of the subtask
        2. The type of task (vision, code, math, text, audio)
        3. The expected inputs and outputs
        
        Format your response as a numbered list of subtasks.
        """
    
    def _format_tactical_input(self, strategic_step: Dict[str, Any], context: str) -> str:
        """Format the input for tactical planning."""
        return f"""
        STEP DESCRIPTION: {strategic_step.get('description', '')}
        EXPECTED OUTCOME: {strategic_step.get('expected_outcome', '')}
        
        RELEVANT CONTEXT:
        {context}
        """
    
    def _parse_tactical_plan(self, plan_text: str, strategic_step: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the LLM output into a structured tactical plan.
        
        Args:
            plan_text: Raw text output from LLM
            strategic_step: The strategic step this tactical plan is for
            
        Returns:
            Structured tactical plan with subtasks
        """
        lines = plan_text.strip().split('\n')
        subtasks = []
        
        current_subtask = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this is a new subtask
            if line[0].isdigit() and '.' in line[:5]:
                if current_subtask:
                    subtasks.append(current_subtask)
                
                subtask_text = line.split('.', 1)[1].strip()
                current_subtask = {
                    "id": f"subtask_{strategic_step.get('id', 0)}_{len(subtasks) + 1}",
                    "description": subtask_text,
                    "type": "text",  # Default type
                    "inputs": {},
                    "expected_outputs": {}
                }
            elif current_subtask and "type:" in line.lower():
                task_type = line.lower().split("type:", 1)[1].strip()
                # Map to one of our supported types
                if "vision" in task_type or "image" in task_type:
                    current_subtask["type"] = "vision"
                elif "code" in task_type or "program" in task_type:
                    current_subtask["type"] = "code"
                elif "math" in task_type or "calculation" in task_type:
                    current_subtask["type"] = "math"
                elif "audio" in task_type or "speech" in task_type:
                    current_subtask["type"] = "audio"
                else:
                    current_subtask["type"] = "text"
            elif current_subtask and "input:" in line.lower():
                input_desc = line.split("input:", 1)[1].strip()
                current_subtask["inputs"]["description"] = input_desc
            elif current_subtask and "output:" in line.lower():
                output_desc = line.split("output:", 1)[1].strip()
                current_subtask["expected_outputs"]["description"] = output_desc
            elif current_subtask:
                # Append to the current subtask description
                current_subtask["description"] += " " + line
        
        # Add the last subtask if it exists
        if current_subtask:
            subtasks.append(current_subtask)
        
        return {
            "strategic_step_id": strategic_step.get("id", 0),
            "subtasks": subtasks
        }
    
    def _identify_capabilities(self, subtask: Dict[str, Any]) -> List[str]:
        """
        Identify required capabilities for a subtask.
        
        Args:
            subtask: Subtask information
            
        Returns:
            List of required capabilities
        """
        task_type = subtask.get("type", "")
        description = subtask.get("description", "").lower()
        
        capabilities = []
        
        # Add basic capability based on task type
        if task_type == "vision":
            capabilities.append("image_understanding")
            
            # Add more specific capabilities based on description
            if "detect" in description or "object" in description:
                capabilities.append("object_detection")
            if "classify" in description:
                capabilities.append("image_classification")
            if "segment" in description:
                capabilities.append("image_segmentation")
            if "caption" in description or "describe" in description:
                capabilities.append("image_captioning")
                
        elif task_type == "code":
            capabilities.append("code_processing")
            
            # Add more specific capabilities based on description
            if "execute" in description or "run" in description:
                capabilities.append("code_execution")
            if "generate" in description or "create" in description:
                capabilities.append("code_generation")
            if "complete" in description:
                capabilities.append("code_completion")
            if "debug" in description or "fix" in description:
                capabilities.append("code_debugging")
                
        elif task_type == "math":
            capabilities.append("mathematical_reasoning")
            
            # Add more specific capabilities based on description
            if "equation" in description or "solve" in description:
                capabilities.append("equation_solving")
            if "statistic" in description or "probability" in description:
                capabilities.append("statistical_analysis")
            if "plot" in description or "graph" in description:
                capabilities.append("mathematical_plotting")
                
        elif task_type == "audio":
            capabilities.append("audio_processing")
            
            # Add more specific capabilities based on description
            if "transcribe" in description or "speech to text" in description:
                capabilities.append("speech_to_text")
            if "recognize" in description or "identify" in description:
                capabilities.append("audio_recognition")
            if "generate" in description or "synthesize" in description:
                capabilities.append("speech_synthesis")
        
        return capabilities


class HierarchicalPlanner:
    """
    Hierarchical planner that combines strategic and tactical planning.
    Integrates Cerebras RAG's strategic planning with HuggingGPT's task decomposition.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Hierarchical Planner.
        
        Args:
            config: Configuration dictionary for the planner
        """
        self.config = config or {}
        self.strategic_planner = StrategicPlanner(self.config.get("strategic", {}))
        self.tactical_planner = TacticalPlanner(self.config.get("tactical", {}))
        self.llm_factory = None  # Will be set by Agent
        logger.info("Hierarchical Planner initialized")
    
    def set_llm_factory(self, llm_factory: Any) -> None:
        """
        Set the LLM provider factory for the planner.
        
        Args:
            llm_factory: LLM provider factory
        """
        self.llm_factory = llm_factory
        self.strategic_planner.set_llm_factory(llm_factory)
        self.tactical_planner.set_llm_factory(llm_factory)
        logger.info("LLM factory set for Hierarchical Planner and sub-planners")
    
    def create_strategic_plan(self, goal: str, memory: Any) -> Dict[str, Any]:
        """
        Create a high-level strategic plan to achieve the given goal.
        
        Args:
            goal: User goal
            memory: Memory module for context retrieval
            
        Returns:
            Strategic plan
        """
        return self.strategic_planner.create_plan(goal, memory)
    
    def create_tactical_plan(self, strategic_step: Dict[str, Any], memory: Any) -> Dict[str, Any]:
        """
        Create a tactical plan for executing a strategic step.
        
        Args:
            strategic_step: Step from the strategic plan
            memory: Memory module for context retrieval
            
        Returns:
            Tactical plan with subtasks
        """
        return self.tactical_planner.create_plan(strategic_step, memory)


# Factory for creating different types of planners
class PlannerFactory:
    """Factory for creating different types of planners."""
    
    @staticmethod
    def create_planner(planner_type: str, config: Optional[Dict[str, Any]] = None) -> BasePlanner:
        """
        Create a planner of the specified type.
        
        Args:
            planner_type: Type of planner to create
            config: Configuration dictionary for the planner
            
        Returns:
            Planner instance
        """
        if planner_type == "strategic":
            return StrategicPlanner(config)
        elif planner_type == "tactical":
            return TacticalPlanner(config)
        elif planner_type == "hierarchical":
            return HierarchicalPlanner(config)
        else:
            raise ValueError(f"Unknown planner type: {planner_type}")
